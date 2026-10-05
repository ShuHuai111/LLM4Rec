from __future__ import annotations
from scipy.sparse import csr_matrix
from collections.abc import Iterable
from sklearn.preprocessing import normalize

import numpy as np
import  pandas as pd
"""
def fit(interactions) 是 Item-KNN 的核心训练方法,
    接收train.csv至少需要：user_id, item_id
    创建 ID 映射
        self.item_to_index_
        self.index_to_item_
    构造稀疏交互矩阵,矩阵形状[物品数量, 用户数量] csr_matrix(...)
    L2 归一化
    计算物品相似度 similarity_matrix
    建立邻居缓存 self._build_neighbor_cache() 把每个物品最相似的若干物品保存下来
    
def recommend(seen_items=None,k=10,)根据用户已经交互过的物品生成推荐。
    创建得分数组，如果有 1008 个物品，就创建长度为 1008 的数组：[0, 0, 0, ..., 0]
    遍历用户历史 for item_id in seen_item_ids:
    累加相似物品得分
    排除已经看过的物品，只保留正相似度候选
    按得分排序
    
def recommend_for_user() 用户级接口。
    接收 用户 ID,用户历史 DataFrame
"""
class ItemKNNRecommender:
    def __init__(self,neighbor_k: int = 50,) -> None:
        if not isinstance(neighbor_k,int,) or isinstance(neighbor_k, bool):
            raise TypeError("neighbor_k 必须是整数")
        if neighbor_k < 1:
            raise ValueError("neighbor_k 必须大于等于 1")

        self.neighbor_k = neighbor_k

        # 物品ID ↔ 索引映射（字符串ID -> 数字index）
        self.item_to_index_: dict[str, int] = {} # 根据物品原始 ID 快速查到矩阵里的行号
        self.index_to_item_: list[str] = [] # 根据矩阵数字下标还原回原始业务 ID
        # 用户ID ↔ 索引映射
        self.user_to_index_: dict[str, int] = {}
        self.index_to_user_: list[str] = []

        # 稀疏矩阵：物品-用户交互矩阵（行=物品，列=用户，ItemCF常用）
        self.item_user_matrix_: csr_matrix | None = None
        # 物品相似度矩阵，csr稀疏存储
        self.similarity_matrix_: csr_matrix | None = None

        # 分别存当前物品的下标和分数
        self.neighbor_indices_: list[np.ndarray] = []
        self.neighbor_scores_: list[np.ndarray] = []

    """将 ID 输入统一转换为字符串列表。"""
    @staticmethod
    def _to_id_list(values: Iterable[str | int] | str | int | None,) -> list[str]:
        if values is None:
            return []
        if isinstance(values, bytes):
            return [values.decode("utf-8")]
        if isinstance(values, str):
            return [values]
        if not isinstance(values, Iterable):
            return [str(values)]
        return [str(value) for value in values]

    def _check_fitted(self) -> None:
        if self.item_user_matrix_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        """
        使用训练交互数据计算物品相似度。interactions 至少需要包含：
            - user_id
            - item_id
        """
    def fit(self,interactions: pd.DataFrame,) -> "ItemKNNRecommender":
        if not isinstance(interactions,pd.DataFrame,):
            raise TypeError("interactions 必须是 pandas.DataFrame")
        required_columns = {"user_id","item_id",}
        missing_columns = (required_columns - set(interactions.columns))
        if missing_columns:
            raise ValueError("interactions 缺少字段："f"{sorted(missing_columns)}")
        if interactions.empty:
            raise ValueError("interactions 为空")

        data = interactions[["user_id", "item_id"]].copy()
        if data.isna().any().any():
            raise ValueError("user_id 或 item_id 中存在空值")

        data["user_id"] = data["user_id"].astype("string").str.strip()
        data["item_id"] = data["item_id"].astype("string").str.strip()
        if data["user_id"].eq("").any() or data["item_id"].eq("").any():
            raise ValueError("user_id 或 item_id 中存在空字符串")

        # 同一个用户重复交互同一个物品只记为一次
        data = data.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)

        #构建ID ↔ index双向映射
        item_ids = sorted(data["item_id"].unique().tolist())
        user_ids = sorted(data["user_id"].unique().tolist())
        self.item_to_index_ = {item_id: index for index, item_id in enumerate(item_ids)}
        self.index_to_item_ = item_ids
        self.user_to_index_ = {user_id: index for index, user_id in enumerate(user_ids)}
        self.index_to_user_ = user_ids

        # 构造 CSR 物品 - 用户交互矩阵
        # item_to_index_得到物品在交互矩阵中的行下表，user_to_index_得到列下标，一对就是一个坐标
        item_indices = data["item_id"].map(self.item_to_index_).to_numpy(dtype=np.int64)
        user_indices = data["user_id"].map(self.user_to_index_).to_numpy(dtype=np.int64)
        # 把上述坐标位置全部设为1
        values = np.ones(len(data), dtype=np.float32)
        # 构造稀疏矩阵
        item_user_matrix = csr_matrix(
            (values, (item_indices, user_indices)),
            shape=(len(item_ids), len(user_ids)),
            dtype=np.float32,)
        # 去重，合并重复值
        item_user_matrix.sum_duplicates()
        #原地把所有非零元素的值全部强制改成1.0
        item_user_matrix.data[:] = 1.0
        self.item_user_matrix_ = item_user_matrix

        # 对每个物品向量做 L2 归一化
        normalized_matrix = normalize(item_user_matrix,norm="l2",axis=1,copy=True,)
        # 计算# 物品-物品余弦相似度矩阵
        similarity_matrix = (normalized_matrix @ normalized_matrix.T).tocsr()
        # 物品和自身的相似度不参与推荐
        similarity_matrix.setdiag(0)
        similarity_matrix.eliminate_zeros()
        self.similarity_matrix_ = (similarity_matrix)
        self._build_neighbor_cache()
        return self

    """为每个物品保留相似度最高的 Top-N 邻居。"""
    def _build_neighbor_cache(self) -> None:
        self._check_fitted()
        assert self.similarity_matrix_ is not None

        self.neighbor_indices_ = []
        self.neighbor_scores_ = []

        for item_index in range(self.similarity_matrix_.shape[0]):
            row = self.similarity_matrix_.getrow(item_index)
            neighbor_indices = row.indices
            neighbor_scores = row.data

            # 处理「无任何邻居」的冷启动物品，主动存入空数组，保证列表长度和物品总数一一对应，不会出现索引错位
            if len(neighbor_indices) == 0:
                self.neighbor_indices_.append(np.array([], dtype=np.int64))
                self.neighbor_scores_.append(np.array([], dtype=np.float32))
                continue

            # 部分排序，取出相似度最高的k个
            if len(neighbor_indices) > self.neighbor_k:
                selected = np.argpartition(neighbor_scores, -self.neighbor_k)[-self.neighbor_k:]
            else:
                selected = np.arange(len(neighbor_indices))

            # 对 Top-K 内部按相似度降序排序
            order = np.argsort(neighbor_scores[selected], kind="stable")[::-1]
            selected = selected[order]
            # 存入缓存列表
            self.neighbor_indices_.append(neighbor_indices[selected])
            self.neighbor_scores_.append(neighbor_scores[selected])

    """
    根据用户已经交互过的物品生成 Top-K 推荐。
    """
    def recommend(
            self,seen_items: Iterable[str | int] | str | int | None = None,
            *,
            k: int = 10,) -> list[str]:
        self._check_fitted()
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k 必须是整数")
        if k < 1:
            raise ValueError("k 必须大于等于 1")

        seen_item_ids = set(self._to_id_list(seen_items))

        # 初始化推荐分数数组
        scores = np.zeros(len(self.index_to_item_), dtype=np.float32)
        seen_indices: list[int] = []

        for item_id in seen_item_ids:
            item_index = self.item_to_index_.get(item_id)
            # 历史中未出现在训练集的物品直接跳过
            if item_index is None:
                continue

            seen_indices.append(item_index)
            neighbor_indices = (self.neighbor_indices_[item_index])
            neighbor_scores = (self.neighbor_scores_[item_index])
            # 累加每个物品的相似度分数
            scores[neighbor_indices] += (neighbor_scores)

        # 将用户已经看过的物品的分数设为负无穷
        if seen_indices:
            scores[np.asarray(seen_indices, dtype=np.int64)] = -np.inf

        # 筛选有效候选
        candidate_indices = np.flatnonzero(scores > 0)
        if len(candidate_indices) == 0:
            return []

        # 排序并返回 Top-K
        order = np.argsort(-scores[candidate_indices], kind="stable")
        ranked_indices = candidate_indices[order]
        return [self.index_to_item_[index]for index in ranked_indices[:k]]

    """
    根据用户历史交互生成推荐。
    """
    def recommend_for_user(
        self,
        user_id: str | int,
        history_interactions: pd.DataFrame,
        *,
        k: int = 10,) -> list[str]:
        if not isinstance(history_interactions,pd.DataFrame,):
            raise TypeError("history_interactions 必须是pandas.DataFrame")
        required_columns = {"user_id", "item_id"}
        missing_columns = required_columns - set(history_interactions.columns)
        if missing_columns:
            raise ValueError(
                "history_interactions 缺少字段："
                f"{sorted(missing_columns)}")

        user_id = str(user_id)
        user_history = (history_interactions[history_interactions["user_id"].astype("string").eq(user_id)])
        seen_items = (user_history["item_id"].astype("string").tolist())

        return self.recommend(seen_items=seen_items,k=k,)

    """
    返回某个物品最相似的邻居。
    """
    def get_neighbors(self,item_id: str | int,) -> pd.DataFrame:
        self._check_fitted()

        item_id = str(item_id)
        if item_id not in self.item_to_index_:
            raise KeyError(f"训练集中不存在 item_id={item_id}")
        item_index = self.item_to_index_[item_id]
        neighbor_indices = (self.neighbor_indices_[item_index])
        neighbor_scores = (self.neighbor_scores_[item_index])


        return pd.DataFrame(
            {
                "item_id": [self.index_to_item_[index] for index in neighbor_indices],
                "similarity": neighbor_scores,
            })

__all__ = [
    "ItemKNNRecommender",
]





