from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.representations.semantic_id.mapper import SemanticIDMapper

class SemanticSequenceAdapter:
    def __init__(
        self,
        mapper: SemanticIDMapper,       # 所有物品级别的转换都委托给它
        *,
        max_seq_len: int = 50,          # 历史序列的最大长度
        min_history_len: int = 1,       # 生成样本的最小历史长度，默认 1
    ) -> None:
        if not isinstance(mapper, SemanticIDMapper):
            raise TypeError("mapper 必须是 SemanticIDMapper")
        if (not isinstance(max_seq_len, int)
            or isinstance(max_seq_len, bool)
            or max_seq_len < 1):
            raise ValueError("max_seq_len 必须是大于 0 的整数")

        if (not isinstance(min_history_len, int)
            or isinstance(min_history_len, bool)
            or min_history_len < 1):
            raise ValueError("min_history_len 必须是大于 0 的整数")

        self.mapper = mapper
        self.max_seq_len = max_seq_len
        self.min_history_len = min_history_len

    @staticmethod
    def _normalize_timestamp(timestamps: pd.Series) -> pd.Series:
        numeric_timestamps = pd.to_numeric(timestamps, errors="coerce")
        if numeric_timestamps.notna().all():
            return numeric_timestamps

        # 非数值格式走 datetime 解析
        datetime_timestamps = pd.to_datetime(timestamps,errors="raise",utc=True,)
        return pd.Series(datetime_timestamps.astype("int64"),index=timestamps.index,)

    """
    输入原始用户交互表，输出按用户 ID 分组、组内按时间升序排列、每个事件携带完整编码信息的结构化数据
    返回结果
{
    # 用户u1的事件序列（按时间从早到晚）
    "u1": [
        # 第一条事件：点击物品101
        {
            "user_id": "u1",
            "item_id": "101",
            "codes": array([12, 34], dtype=int64),  # 多层整数编码数组
            "semantic_id": "12-34",
            "timestamp": 1704067200000000000    # 统一转成int64纳秒级Unix时间戳
        },
        # 第二条事件：点击物品102
        {
            "user_id": "u1",
            "item_id": "102",
            "codes": array([56, 7], dtype=int64),
            "semantic_id": "56-7",
            "timestamp": 1704070800000000000
        }
    ],
    """
    def _prepare_events(self, interactions: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        if "user_id" not in interactions.columns:
            raise ValueError("interactions 缺少 user_id 字段")
        if "timestamp" not in interactions.columns:
            raise ValueError("interactions 缺少 timestamp 字段")

        # 调用方法将将交互表中的 item_id 转换成 code_0、code_1 等字段。
        data = self.mapper.transform_interactions(interactions)

        # user_id 格式标准化
        data["user_id"] = data["user_id"].astype("string").str.strip()
        if data["user_id"].isna().any():
            raise ValueError("user_id 不能包含缺失值")
        # 时间戳标准化,调用上面写的那个方法
        data["timestamp"] = self._normalize_timestamp(data["timestamp"])
        # 全局时序稳定排序，先按 `user_id` 分组，组内按 `timestamp` 升序排列；
        data = data.sort_values(by=["user_id", "timestamp"],kind="stable",).reset_index(drop=True)

        # 准备最终返回的结果
        # key：用户 ID（字符串）
        # value：该用户按时间排序的事件列表，每个元素是一个事件字典
        events_by_user: dict[str, list[dict[str, Any]]] = {}
        for user_id, group in data.groupby("user_id", sort=False): # 按用户分组
            user_events: list[dict[str, Any]] = []
            for _, row in group.iterrows():
                # 提取编码
                codes = np.asarray([int(row[column]) for column in self.mapper.code_columns],dtype=np.int64,)
                # 组装单个事件
                user_events.append({
                    "user_id": str(user_id),
                    "item_id": str(row["item_id"]),
                    "codes": codes,
                    "semantic_id": str(row["semantic_id"]),
                    "timestamp": row["timestamp"],
                })
            events_by_user[str(user_id)] = user_events

        return events_by_user

    """
    构建单个训练样本
    输入「用户 ID + 历史事件列表 + 目标事件」，输出一个标准、完整的序列推荐训练样本，
    包含历史编码、掩码、Token 序列，以及目标端的输入 / 输出 Token 等所有模型训练需要的字段。
    
例：
    # 历史事件（按时间从早到晚）
history = [
    {   "item_id": "101",
        "codes": np.array([12, 34], dtype=np.int64),  # 第0层=12，第1层=34
        "semantic_id": "12-34",
        "timestamp": 1000   },
    {   "item_id": "102",
        "codes": np.array([56, 7], dtype=np.int64),   # 第0层=56，第1层=7
        "semantic_id": "56-7",
        "timestamp": 2000   }  ]

    # 目标事件（下一个要预测的物品）
target = {  "item_id": "103",
            "codes": np.array([9, 42], dtype=np.int64),    # 第0层=9，第1层=42
            "semantic_id": "9-42",
            "timestamp": 3000   }
    
    # 初始化后
数组	                初始值	                        说明
history_codes	    [[0,0], [0,0], [0,0]]	        3 个物品位置，每个位置 2 层编码，全 0（PAD）
history_mask	    [False, False, False]	        全为无效填充

history_tokens	    [0,0,0, 0,0,0, 0,0,0]	        9 个 Token 位置，全 0（PAD）
                                                每 3 个 Token 对应 1 个物品：[编码0, 编码1, 分隔符]
                                                
history_token_mask	[F,F,F, F,F,F, F,F,F]	    全为无效填充

。。。。。。。
填充完成后，开始构建目标端输入输出 Token
    目标输入（解码器输入）：BOS + 编码 Token  [130] + [10, 107] = [130, 10, 107]
    目标输出（训练标签）：编码 Token + EOS    [10, 107] + [131] = [10, 107, 131]
    """
    def _build_example(
        self,
        user_id: str,
        history: list[dict[str, Any]],
        target: dict[str, Any],
    ) -> dict[str, Any] | None:
        if len(history) < self.min_history_len:
            return None

        # 只保留距离目标最近的历史行为，避免超过 max_seq_len。
        history = history[-self.max_seq_len:]

        num_codebooks = self.mapper.num_codebooks

        # 1. 历史编码数组：形状 [最大序列长, 码本层数]，全0（PAD）初始化
        history_codes = np.zeros(
            (self.max_seq_len, num_codebooks),
            dtype=np.int64,)

        # 2. 历史掩码：形状 [最大序列长]，布尔型，全False初始化
        history_mask = np.zeros(self.max_seq_len, dtype=bool)

        # 3. 历史Token序列：每个事件占「层数+1」个Token（编码Token + 物品分隔符），全PAD初始化
        history_tokens = np.full(
            self.max_seq_len * (num_codebooks + 1),
            self.mapper.pad_token_id,
            dtype=np.int64,)

        # 4. 历史Token掩码：对应Token序列的有效标记，全False初始化
        history_token_mask = np.zeros(
            self.max_seq_len * (num_codebooks + 1),
            dtype=bool,)

        # 计算历史在数组中的起始位置：短历史前面留空，右对齐
        history_start = self.max_seq_len - len(history)

        for offset, event in enumerate(history):
            # 当前事件在编码数组中的位置
            sequence_index = history_start + offset
            # 1. 填充编码数组
            history_codes[sequence_index] = event["codes"]
            # 2. 填充掩码，标记为有效
            history_mask[sequence_index] = True

            # 3. 转换当前物品的编码Token
            code_tokens = self.mapper.codes_to_tokens(event["codes"])
            # 计算当前事件在Token序列中的起始位置
            token_start = sequence_index * (num_codebooks + 1)

            # 4. 填充编码Token
            history_tokens[token_start: token_start + num_codebooks] = code_tokens
            # 5. 填充物品分隔符Token
            history_tokens[token_start + num_codebooks] = self.mapper.item_separator_token_id
            # 6. 填充Token掩码，标记这一段为有效
            history_token_mask[token_start: token_start + num_codebooks + 1] = True

        # 目标物品的编码Token
        target_tokens = self.mapper.codes_to_tokens(target["codes"])

        # 目标输入：BOS + 目标编码Token
        target_input_tokens = np.concatenate([
            np.array([self.mapper.bos_token_id], dtype=np.int64),
            target_tokens,])

        # 目标输出：目标编码Token + EOS
        target_output_tokens = np.concatenate([
            target_tokens,
            np.array([self.mapper.eos_token_id], dtype=np.int64),])

        return {
            "user_id": str(user_id),                                    # 用户ID                  "u1"
            "history_item_ids": tuple(e["item_id"] for e in history),   # 历史物品ID元组              ("101", "102"),
            "history_codes": history_codes,                             # 历史多层编码数组          np.array([[0,0], [12,34], [56,7]]),
            "history_mask": history_mask,                               # 历史物品级掩码
            "history_tokens": history_tokens,                           # 历史展平Token序列       np.array([0,0,0, 13,99,129, 57,72,129]),
            "history_token_mask": history_token_mask,                   # 历史Token级掩码        np.array([False,False,False, True,True,True, True,True,True]),
            "target_item_id": target["item_id"],                        # 目标物品ID             "103"
            "target_semantic_id": target["semantic_id"],                # 目标语义ID            "9-42"
            "target_codes": target["codes"].copy(),                     # 目标多层编码            np.array([9, 42]),
            "target_input_tokens": target_input_tokens,                 # 目标输入Token序列       np.array([130, 10, 107]),
            "target_output_tokens": target_output_tokens,               # 目标输出标签Token序列    np.array([10, 107, 131]),
            "target_timestamp": target["timestamp"],                    # 目标行为时间戳           3000
        }

    """
    在训练集内部构造前缀样本。
    例如：
        item_1, item_2, item_3

    构造：
        [item_1] -> item_2
        [item_1, item_2] -> item_3
    """
    def build_training_examples(
        self,
        train_interactions: pd.DataFrame,) -> list[dict[str, Any]]:
        events_by_user = self._prepare_events(train_interactions)

        examples: list[dict[str, Any]] = []
        for user_id, events in events_by_user.items():
            for target_index in range(1, len(events)):
                history = events[:target_index] # 历史序列
                target = events[target_index]   # 目标序列

                example = self._build_example(user_id, history, target,)

                if example is not None:
                    examples.append(example)

        if not examples:
            raise ValueError("没有生成训练样本，请检查用户历史长度")

        return examples

    def build_evaluation_examples(
        self,
        history_interactions: pd.DataFrame,
        target_interactions: pd.DataFrame,
        *,
        include_target_prefix: bool = True,
        ) -> list[dict[str, Any]]:

        history_by_user = self._prepare_events(history_interactions)
        target_by_user = self._prepare_events(target_interactions)

        examples: list[dict[str, Any]] = []
        for user_id, target_events in target_by_user.items():
            history = list(history_by_user.get(user_id, []))

            for target in target_events:
                example = self._build_example(user_id,history,target,)
                if example is not None:
                    examples.append(example)
                if include_target_prefix:
                    history.append(target)
        if not examples:
            raise ValueError("没有生成验证或测试样本，请检查历史数据")
        return examples

    """
    全数据集构建入口
    """
    def build_all_splits(
        self,
        train_interactions: pd.DataFrame,
        valid_interactions: pd.DataFrame,
        test_interactions: pd.DataFrame,
        ) -> dict[str, list[dict[str, Any]]]:

        train_examples = self.build_training_examples(train_interactions)

        valid_examples = self.build_evaluation_examples(
            train_interactions,
            valid_interactions,
            include_target_prefix=True,)

        # 先把训练+验证拼接，作为测试集的历史
        test_history = pd.concat(
            [train_interactions, valid_interactions],
            ignore_index=True,)
        test_examples = self.build_evaluation_examples(
            test_history,
            test_interactions,
            include_target_prefix=True,)

        return {
            "train": train_examples,
            "valid": valid_examples,
            "test": test_examples,  }

"""
将 SemanticSequenceAdapter 生成的样本包装成 PyTorch Dataset。
"""
class SemanticSequenceDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        if not examples:
            raise ValueError("examples 不能为空")

        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            # ========== 模型输入：全部转 PyTorch 张量 ==========
            "history_codes": torch.as_tensor(example["history_codes"], dtype=torch.long),
            "history_mask": torch.as_tensor(example["history_mask"], dtype=torch.bool),
            "history_tokens": torch.as_tensor(example["history_tokens"], dtype=torch.long),
            "history_token_mask": torch.as_tensor(example["history_token_mask"], dtype=torch.bool),
            "target_codes": torch.as_tensor(example["target_codes"], dtype=torch.long),
            "target_input_tokens": torch.as_tensor(example["target_input_tokens"], dtype=torch.long),
            "target_output_tokens": torch.as_tensor(example["target_output_tokens"], dtype=torch.long),

            # ========== 业务元数据：保留原始 Python 类型 ==========
            "user_id": example["user_id"],
            "history_item_ids": example["history_item_ids"],
            "target_item_id": example["target_item_id"],
            "target_semantic_id": example["target_semantic_id"],
            "target_timestamp": example["target_timestamp"],
        }


__all__ = [
    "SemanticSequenceAdapter",
    "SemanticSequenceDataset",
]
