from __future__ import annotations

import pandas as pd
import numpy as np
import torch

from .sasrec import SASRec


"""
将训练好的 SASRec 包装成项目统一的推荐器接口。
"""
class SASRecRecommender:
    def __init__(
        self,
        model: SASRec,
        item_to_index: dict[str, int],
        index_to_item: list[str],
        *,
        max_seq_len: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.item_to_index = item_to_index
        self.index_to_item = index_to_item
        self.max_seq_len = max_seq_len
        self.device = device
        self._history_by_user: dict[str, np.ndarray] | None = None

    def prepare_history(
            self,
            history_interactions: pd.DataFrame,
    ) -> None:
        """
        只执行一次历史数据预处理，建立：

            user_id -> 按时间排序后的内部 item index 序列
        """
        if not isinstance(history_interactions, pd.DataFrame):
            raise TypeError("history_interactions 必须是 pandas.DataFrame")

        required_columns = {"user_id", "item_id"}
        missing_columns = required_columns - set(history_interactions.columns)
        if missing_columns:
            raise ValueError(
                f"history_interactions 缺少字段：{sorted(missing_columns)}"
            )

        if history_interactions.empty:
            raise ValueError("history_interactions 为空")

        columns = ["user_id", "item_id"]
        if "timestamp" in history_interactions.columns:
            columns.append("timestamp")

        history = history_interactions.loc[:, columns].copy()
        history["user_id"] = (
            history["user_id"].astype("string").str.strip()
        )
        history["item_id"] = (
            history["item_id"].astype("string").str.strip()
        )

        if "timestamp" in history.columns:
            history["timestamp"] = pd.to_numeric(
                history["timestamp"],
                errors="raise",
            )
            history = history.sort_values(
                by=["user_id", "timestamp"],
                kind="stable",
            )

        history["_item_index"] = history["item_id"].map(
            self.item_to_index
        )
        history = history.dropna(subset=["_item_index"])
        history["_item_index"] = history["_item_index"].astype(np.int32)

        self._history_by_user = {
            str(user_id): group["_item_index"].to_numpy(
                dtype=np.int32,
                copy=True,
            )
            for user_id, group in history.groupby(
                "user_id",
                sort=False,
            )
        }

    """
    推理业务阶段
    输入单个用户的 ID 和该用户的历史交互记录表，返回 Top-K 推荐的原始物品 ID 列表。

    数据清洗 → 序列构造 → 模型推理 → 结果过滤 → 索引映射
    """
    def recommend_for_user(
            self,
            user_id: str | int,
            history_interactions: pd.DataFrame | None = None,
            *,
            k: int = 10,
    ) -> list[str]:
        if (not isinstance(k, int) or isinstance(k, bool)or k < 1):
            raise ValueError("k 必须是大于等于 1 的整数")

        user_id = str(user_id)

        if history_interactions is not None:
            self.prepare_history(history_interactions)

        if self._history_by_user is None:
            raise RuntimeError(
                "请先调用 prepare_history() 准备用户历史数据"
            )

        internal_items_array = self._history_by_user.get(user_id)
        if internal_items_array is None or len(internal_items_array) == 0:
            return []

        internal_items = internal_items_array.tolist()

        # 序列截断 + 左填充
        sequence = internal_items[-self.max_seq_len:]
        input_ids = [0] * (self.max_seq_len - len(sequence)) + sequence

        # 构造输入张量
        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        # 模型推理
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)[0].clone()

        # 无效物品屏蔽
        # 屏蔽 padding
        logits[0] = -torch.inf
        # 屏蔽用户已经交互过的物品
        for item_index in set(internal_items):
            logits[item_index] = -torch.inf

        # 可用物品校验 + Top-K 截取
        available_items = int(torch.isfinite(logits).sum().item())
        if available_items == 0:
            return []

        top_k = min(k, available_items)
        top_indices = torch.topk(logits, k=top_k).indices.tolist()

        # 映射回原始物品 ID
        return [self.index_to_item[item_index] for item_index in top_indices]


__all__ = [
    "SASRecRecommender",
]
