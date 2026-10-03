from __future__ import annotations

from pathlib import Path
from typing import Any
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import re
import yaml
import torch

"""
负责 item_id 与 Semantic ID 之间的双向映射。

输入文件：
    item_semantic_ids.csv

必需字段,放在mapping里面：
    item_id
    code_0
    code_1
    ...
    semantic_id
"""
"""
1. item_id：物品的业务 ID，就是原始数据里的物品编号
2. codes：多层整数编码，比如 `[12, 48, 7]`，是残差量化输出的纯数字数组
3. semantic_id：语义 ID 字符串，比如 `"12-48-7"`，就是把多层编码用 `-` 拼起来的字符串
4. tokens：模型可用的 Token 数组，比如 `[13, 113, 136]`，是给序列推荐模型直接用的整数编号
"""
class SemanticIDMapper:
    def __init__(
        self,
        mapping_path: str | Path,           # 语义 ID 映射 CSV 文件路径
        *,
        mode: str = "advanced",
        codebook_size: int | None = None,   # 单层码本大小，不传则自动从数据中推算
    ) -> None:
        if mode not in {"baseline", "advanced"}:
            raise ValueError("mode 必须是 baseline 或 advanced")

        self.mapping_path = Path(mapping_path)
        self.mode = mode
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"找不到 Semantic ID 映射文件：{self.mapping_path}")

        mapping = pd.read_csv(self.mapping_path)
        if mapping.empty:
            raise ValueError("Semantic ID 映射表不能为空")

        required_columns = {"item_id", "semantic_id"}
        missing_columns = required_columns - set(mapping.columns)
        if missing_columns:
            raise ValueError(f"Semantic ID 映射表缺少字段：{sorted(missing_columns)}")

        mapping["item_id"] = (mapping["item_id"].astype("string").str.strip())
        if mapping["item_id"].isna().any():
            raise ValueError("item_id 不能包含缺失值")
        if mapping["item_id"].duplicated().any():
            raise ValueError("Semantic ID 映射表中的 item_id 不能重复")

        self.code_columns = self._find_code_columns(mapping.columns)

        if not self.code_columns:
            raise ValueError("Semantic ID 映射表中没有找到 code_0、code_1 等编码字段")

        # 从完整的映射表中，只提取所有量化编码列，转换成纯数值的 numpy 二维数组
        raw_codes = mapping[self.code_columns].to_numpy()

        if pd.isna(raw_codes).any():
            raise ValueError("Semantic ID code 不能包含缺失值")

        try:
            codes = raw_codes.astype(np.int64)
        except (TypeError, ValueError) as error:
            raise ValueError("Semantic ID code 必须是整数") from error

        if not np.array_equal(raw_codes, codes):
            raise ValueError("Semantic ID code 必须是整数，不能包含小数")
        if (codes < 0).any():
            raise ValueError("Semantic ID code 不能小于 0")

        # 自动推导并存储残差量化的码本总层数
        self.num_codebooks = len(self.code_columns)

        if codebook_size is None:
            codebook_size = int(codes.max()) + 1
        if codebook_size < 2:
            raise ValueError("codebook_size 必须大于等于 2")
        if (codes >= codebook_size).any():
            raise ValueError("Semantic ID code 超出 codebook_size 范围")
        # 强制将单层码本的容量标准化为 Python 原生整数并存储
        self.codebook_size = int(codebook_size)
        # semantic_id 字段格式标准化
        mapping["semantic_id"] = (mapping["semantic_id"].astype("string").str.strip())
        if mapping["semantic_id"].isna().any():
            raise ValueError("semantic_id 不能包含缺失值")
        # 一行编码 `[12, 48, 7]` 会生成标准字符串 `"12-48-7"`
        expected_semantic_ids = ["-".join(str(int(code)) for code in row)for row in codes]

        actual_semantic_ids = (mapping["semantic_id"].astype(str).tolist())
        if actual_semantic_ids != expected_semantic_ids:
            raise ValueError("semantic_id 与各层 code 拼接结果不一致")

        # 行索引统一对齐
        self.mapping = mapping.reset_index(drop=True)
        self.codes = codes.astype(np.int64)

        # 初始化三个空字典
        self.item_to_codes_: dict[str, tuple[int, ...]] = {}
        self.item_to_semantic_id_: dict[str, str] = {}
        self.semantic_id_to_items_: dict[str, list[str]] = {}

        # 逐行遍历填充
        for row_index, row in self.mapping.iterrows():
            item_id = str(row["item_id"])
            semantic_id = str(row["semantic_id"])
            item_codes = tuple(int(code) for code in self.codes[row_index])

            self.item_to_codes_[item_id] = item_codes
            self.item_to_semantic_id_[item_id] = semantic_id
            self.semantic_id_to_items_.setdefault(semantic_id,[],).append(item_id)

        if self.mode == "advanced":
            duplicated_semantic_ids = {
                semantic_id: item_ids
                for semantic_id, item_ids
                in self.semantic_id_to_items_.items()
                if len(item_ids) > 1}

            if duplicated_semantic_ids:
                examples = list(duplicated_semantic_ids.items())[:5]
                raise ValueError("advanced 模式要求 Semantic ID 唯一，"f"但发现冲突：{examples}")

        # 多层离散编码映射到一套连续、唯一、分层的整数 Token 空间，
        # 同时预留所有特殊 Token，模型可以直接用这套 Token 做输入。
        # baseline 和 advanced 都需要这些字段。
        # 0 专门保留给 PAD
        self.pad_token_id = 0
        # 编码的起始偏移
        self.code_token_offset = 1
        # 物品分隔符Token：所有层编码占满之后的第一个位置
        self.item_separator_token_id = (
                self.code_token_offset
                + self.num_codebooks * self.codebook_size)

        # 序列开始、结束标记
        self.bos_token_id = self.item_separator_token_id + 1
        self.eos_token_id = self.item_separator_token_id + 2

        # 整个词表的总大小
        self.vocab_size = self.eos_token_id + 1

    """
    找到所有的编码列
    必须是 `code_0`、`code_1`、`code_2`... 依次连续，不能缺层、跳号
    """
    @staticmethod # 不需要实例就能调用
    def _find_code_columns(columns: pd.Index,) -> list[str]:
        code_columns: list[tuple[int, str]] = []

        for column in columns:
            match = re.fullmatch(r"code_(\d+)", str(column))
            if match:
                code_columns.append((int(match.group(1)), str(column)))

        code_columns.sort(key=lambda value: value[0])

        # 连续性校验
        indices = [index for index, _ in code_columns]
        expected_indices = list(range(len(indices)))
        if indices != expected_indices:
            raise ValueError(f"Semantic ID code 列不连续：{indices}")

        return [column for _, column in code_columns]

    """
    从标准 Semantic ID 输出目录加载映射。
    目录中应包含：
        item_semantic_ids.csv
        config_snapshot.yaml
    """
    @classmethod
    def from_directory(
        cls,
        semantic_dir: str | Path,
        *,
        mode: str | None = None,
    ) -> "SemanticIDMapper":
        semantic_dir = Path(semantic_dir)
        mapping_path = semantic_dir / "item_semantic_ids.csv"
        snapshot_path = semantic_dir / "config_snapshot.yaml"

        if not snapshot_path.exists():
            raise FileNotFoundError(f"找不到配置快照：{snapshot_path}")
        with snapshot_path.open("r",encoding="utf-8-sig",) as file:
            snapshot = yaml.safe_load(file)

        if not isinstance(snapshot, dict):
            raise ValueError("配置快照必须是 YAML 字典")

        snapshot_mode = str(snapshot.get("quantizer_mode", "advanced"))
        selected_mode = mode or snapshot_mode

        quantizer_config = snapshot.get("quantizer", {})

        if not isinstance(quantizer_config, dict):
            raise ValueError("配置快照中的 quantizer 必须是字典")

        codebook_size = int(quantizer_config.get("codebook_size", 64))

        return cls(mapping_path, mode=selected_mode, codebook_size=codebook_size,)

    @property
    def item_ids(self) -> list[str]:
        return list(self.item_to_codes_.keys())
    @property
    def semantic_ids(self) -> list[str]:
        return list(self.semantic_id_to_items_.keys())

    """
    物品 ID → 多层整数编码数组-------原子方法------
    输入： 一个物品 ID（传字符串、数字都可以）
    输出：numpy 一维 int64 数组，例如 `array([12, 48, 7], dtype=int64)`
    """
    def item_id_to_codes(self, item_id: str,) -> np.ndarray:
        item_id = str(item_id).strip()
        if item_id not in self.item_to_codes_:
            raise KeyError(f"item_id 不存在于 Semantic ID 映射表：{item_id}")
        return np.asarray(self.item_to_codes_[item_id],dtype=np.int64,)
    """
    物品 ID → 语义 ID 字符串 -----原子方法-----
    输入：一个物品 ID
    标准语义 ID 字符串，例如 "12-48-7"
    """
    def item_id_to_semantic_id(self,item_id: str) -> str:
        item_id = str(item_id).strip()
        if item_id not in self.item_to_semantic_id_:
            raise KeyError(f"item_id 不存在于 Semantic ID 映射表：{item_id}")
        return self.item_to_semantic_id_[item_id]

    """
    多层编码 → 语义 ID 字符串
    """
    def codes_to_semantic_id(self,codes: Any,) -> str:
        codes = np.asarray(codes)
        if codes.ndim != 1:
            raise ValueError("codes 必须是一维数组")
        if len(codes) != self.num_codebooks:
            raise ValueError("codes 长度与 num_codebooks 不一致")
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError("codes 必须是整数")
        codes = codes.astype(np.int64)
        if (codes < 0).any() or (codes >= self.codebook_size).any():
            raise ValueError("codes 存在超出码本范围的值")
        return "-".join(str(int(code)) for code in codes)

    """
    语义 ID → 物品 ID
    - 输入：语义 ID 字符串，例如 `"12-48-7"`
    - 输出：对应的物品 ID 字符串
    """
    def semantic_id_to_item_id(
        self,
        semantic_id: str,
        *,
        strict: bool = True,
    ) -> str:
        semantic_id = str(semantic_id).strip()
        item_ids = self.semantic_id_to_items_.get(semantic_id, [])
        if not item_ids:
            raise KeyError(f"Semantic ID 不存在：{semantic_id}")
        if len(item_ids) > 1 and strict:
            raise ValueError(f"Semantic ID 存在冲突，无法唯一还原：{semantic_id} -> {item_ids}")
        return item_ids[0]

    """
    ------------------------组合方法----------------------
    - 输入：多层编码数组 / 列表，例如 `[12, 48, 7]`
    - 输出：对应的物品 ID 字符串
    """
    def codes_to_item_id(self,codes: Any,*,strict: bool = True,) -> str:
        semantic_id = self.codes_to_semantic_id(codes)
        return self.semantic_id_to_item_id(semantic_id,strict=strict,)

    """
    物品 ID → 模型 Token 数组---------组合方法------------
    组合调用item_id_to_codes和item_id_to_semantic_id
    输入：一个物品 ID
    输出：numpy 一维 int64 Token 数组，例如 `array([13, 113, 136], dtype=int64)`
    """
    def item_id_to_tokens(self,item_id: str,) -> np.ndarray:
        codes = self.item_id_to_codes(item_id)
        return self.codes_to_tokens(codes)

    """
    输入编码 `[12, 48, 7]`：
        - 第 0 层 Token：`1 + 0×64 + 12 = 13`
        - 第 1 层 Token：`1 + 1×64 + 48 = 113`
        - 第 2 层 Token：`1 + 2×64 + 7 = 136`
        - 最终输出：`array([13, 113, 136], dtype=int64)`
    """
    def codes_to_tokens(self, codes: Any) -> np.ndarray:
        codes = np.asarray(codes)

        if codes.ndim != 1:
            raise ValueError("codes 必须是一维数组")
        if len(codes) != self.num_codebooks:
            raise ValueError("codes 长度与 num_codebooks 不一致")
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError("codes 必须是整数")
        codes = codes.astype(np.int64)
        if (codes < 0).any() or (codes >= self.codebook_size).any():
            raise ValueError("codes 存在超出码本范围的值")

        # 核心分层转换
        tokens = (self.code_token_offset+ np.arange(self.num_codebooks) * self.codebook_size+ codes)
        return tokens.astype(np.int64)

    """
    将交互表中的 item_id 转换成 code_0、code_1 等字段。
    把整个交互数据表（比如用户点击、收藏、购买记录）里的 `item_id` 批量转换成多层编码和语义ID，
    直接输出模型训练 / 推理可用的完整 DataFrame。
    
    相当于把逐行调用 `item_id_to_codes` + `item_id_to_semantic_id` 的逻辑做了批量封装，
    是对接推荐模型数据流水线的核心工具。
    
    user_id	    item_id
    101	        5
    101	        12
    102	        7
    user_id	    item_id	    code_0	    code_1	    code_2	    semantic_id
    101	        5	        12	        48	        7	        12-48-7
    101	        12	        3	        15	        22	        3-15-22
    102	        7	        45	        2	        59	        45-2-59
    """
    def transform_interactions(self,interactions: pd.DataFrame,) -> pd.DataFrame:
        if not isinstance(interactions, pd.DataFrame):
            raise TypeError("interactions 必须是 pandas.DataFrame")
        if interactions.empty:
            raise ValueError("interactions 不能为空")
        if "item_id" not in interactions.columns:
            raise ValueError("interactions 缺少 item_id 字段")

        result = interactions.copy()
        result["item_id"] = (result["item_id"].astype("string").str.strip())
        if result["item_id"].isna().any():
            raise ValueError("interactions.item_id 不能包含缺失值")

        item_ids = result["item_id"].astype(str).tolist()
        unknown_item_ids = sorted(set(item_ids) - set(self.item_to_codes_))
        if unknown_item_ids:
            raise KeyError("以下 item_id 不在 Semantic ID 映射表中："f"{unknown_item_ids[:10]}")

        codes = np.asarray([self.item_to_codes_[item_id]for item_id in item_ids],dtype=np.int64,)
        # 逐层追加 code_0、code_1... 列
        for code_index, column in enumerate(self.code_columns):
            result[column] = codes[:, code_index]
        # 追加 semantic_id 列
        result["semantic_id"] = [self.item_to_semantic_id_[item_id] for item_id in item_ids]
        return result



"""
    将交互数据转换成 Semantic ID 序列样本。

    训练样本形式：
        history -> target

    输出字段包括：
        history_codes           历史行为物品的多层编码数组
        history_mask            历史序列的有效掩码，标记哪些位置是真实行为、哪些是填充
        history_tokens          历史行为的分层 Token 数组，可直接输入模型
        target_codes            目标物品的多层编码
        target_input_tokens     目标端的输入 Token 序列
        target_output_tokens    目标端的输出标签 Token 序列
"""
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
    "SemanticIDMapper",
    "SemanticSequenceAdapter",
    "SemanticSequenceDataset",
]


