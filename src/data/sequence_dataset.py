from __future__ import annotations
from collections.abc import Mapping
from torch.utils.data import Dataset

import pandas as pd
import torch


"""
为 SASRec 构造序列训练样本。
    输入数据至少需要包含：
        - user_id
        - item_id
        - timestamp
        
    如用户历史：[10, 25, 31, 48]
    自回归生成：
        输入：[10]          目标：25
        输入：[10, 25]      目标：31
        输入：[10, 25, 31]  目标：48
        
    物品索引需要从 1 开始：
        0：padding
        1 ~ num_items：真实物品
    
    序列长度超过50则只保留最后50条
"""
class SASRecSequenceDataset(Dataset):
    def __init__(
        self,interactions: pd.DataFrame,
        *,
        max_seq_len: int = 50,
        sort_by_timestamp: bool = True, # 是否按照时间戳排序
        item_to_index: Mapping[str, int] | None = None,) -> None:

        if not isinstance(interactions, pd.DataFrame):
            raise TypeError("interactions 必须是 pandas.DataFrame")
        if (not isinstance(max_seq_len, int)or isinstance(max_seq_len, bool)):
            raise TypeError("max_seq_len 必须是整数")
        if max_seq_len < 1:
            raise ValueError("max_seq_len 必须大于等于 1")

        required_columns = {"user_id","item_id",}
        if sort_by_timestamp:
            required_columns.add("timestamp")
        missing_columns = (required_columns- set(interactions.columns))
        if missing_columns:
            raise ValueError("interactions 缺少字段："f"{sorted(missing_columns)}")
        if interactions.empty:
            raise ValueError("interactions 不能为空")

        self.max_seq_len = max_seq_len
        self.sort_by_timestamp = sort_by_timestamp

        data = interactions.copy()
        # 检查空值
        if data[["user_id", "item_id"]].isna().any().any():
            raise ValueError("user_id 或 item_id 中存在空值")
        # 统一 ID 类型
        data["user_id"] = (data["user_id"].astype("string").str.strip())
        data["item_id"] = (data["item_id"].astype("string").str.strip())
        if data["user_id"].eq("").any():
            raise ValueError("user_id 中存在空字符串")
        if data["item_id"].eq("").any():
            raise ValueError("item_id 中存在空字符串")

        # 按时间排序
        if sort_by_timestamp:
            data["timestamp"] = pd.to_numeric(data["timestamp"],errors="coerce",)

            if data["timestamp"].isna().any():
                raise ValueError("timestamp 中存在无法转换为数字的值")
            data = data.sort_values(by=["user_id", "timestamp"],kind="stable",)
        else:
            data = data.sort_values(by=["user_id"],kind="stable",)
        data = data.reset_index(drop=True)

        # 构建物品词表
        if item_to_index is None: # 训练集模式
            item_ids = sorted(data["item_id"].unique().tolist())
            self.item_to_index_: dict[str, int] = {item_id: index + 1 for index, item_id in enumerate(item_ids)}
        else:
            self.item_to_index_ = {str(item_id): int(index)for item_id, index in item_to_index.items()}
            if any(index < 1 for index in self.item_to_index_.values()):
                raise ValueError("物品索引必须从 1 开始，0 保留给 padding")
        self.index_to_item_: list[str] = ["<PAD>" for _ in range(max(self.item_to_index_.values(), default=0) + 1)]
        for item_id, index in self.item_to_index_.items():
            if index >= len(self.index_to_item_):
                self.index_to_item_.extend(["<PAD>"] * (index - len(self.index_to_item_) + 1))
            self.index_to_item_[index] = item_id
        self.num_items = len(self.item_to_index_)

        # 构造用户序列
        self.user_sequences_: dict[str, list[int]] = {}
        for user_id, group in data.groupby("user_id", sort=False):
            sequence = []
            for item_id in group["item_id"]:
                item_index = self.item_to_index_.get(str(item_id))
                if item_index is None:
                    continue  # 外部词表时跳过未知物品
                sequence.append(item_index)
            if sequence:
                self.user_sequences_[str(user_id)] = sequence

        # 滑动窗口生成训练样本
        input_sequences: list[list[int]] = []
        target_items: list[int] = []
        for sequence in self.user_sequences_.values():
            if len(sequence) < 2:
                continue  # 至少1个输入+1个目标

            for target_position in range(1, len(sequence)):
                # 取目标位置之前的前缀，最多取 max_seq_len 个
                start_position = max(0, target_position - max_seq_len)
                prefix = sequence[start_position:target_position]

                # 左侧 padding 到固定长度
                padded_prefix = [0] * (max_seq_len - len(prefix)) + prefix

                input_sequences.append(padded_prefix)
                target_items.append(sequence[target_position])

        if not input_sequences:
            raise ValueError("没有足够的用户历史构造训练样本；每个用户至少需要两条交互")
        self.input_ids_ = torch.tensor(input_sequences, dtype=torch.long)
        self.target_ids_ = torch.tensor(target_items, dtype=torch.long)

    def __len__(self) -> int:
        return self.input_ids_.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.input_ids_[index], self.target_ids_[index],)

    # 原始物品 ID（支持字符串或数字格式），转换成模型 Embedding 层能识别的整数索引
    def encode_item(self, item_id: str | int) -> int:
        item_id = str(item_id)
        if item_id not in self.item_to_index_:
            raise KeyError(f"训练集中不存在 item_id={item_id}")
        return self.item_to_index_[item_id]

    # 把模型输出的整数索引，转回业务侧能识别的原始物品字符串 ID
    def decode_item(self, item_index: int) -> str:
        if not isinstance(item_index, int) or isinstance(item_index, bool):
            raise TypeError("item_index 必须是整数")
        if item_index < 0 or item_index >= len(self.index_to_item_):
            raise KeyError(f"不存在 item_index={item_index}")
        return self.index_to_item_[item_index]

    # 根据用户 ID 返回该用户**按时间排序的内部物品索引序列**
    def get_user_sequence(self, user_id: str | int) -> list[int]:
        user_id = str(user_id)
        return list(self.user_sequences_.get(user_id, []))

    def __repr__(self) -> str:
        return (
            "SASRecSequenceDataset("
            f"samples={len(self)}, "
            f"users={len(self.user_sequences_)}, "
            f"items={self.num_items}, "
            f"max_seq_len={self.max_seq_len}"
            ")")



__all__ = [
    "SASRecSequenceDataset",
]
