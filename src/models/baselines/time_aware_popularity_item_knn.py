from __future__ import annotations
from scipy.sparse import csr_matrix
from collections.abc import Iterable
from sklearn.preprocessing import normalize

import numpy as np
import pandas as pd
"""
相对于原始item-KNN，新增时间衰减和流行度惩罚
初始化时新增如下参数：
        time_decay_enabled: bool = False,
        time_half_life_days: float = 90.0,
        popularity_penalty_enabled: bool = False,
        popularity_penalty_alpha: float = 0.25,
        popularity_count_mode: str = "unique_users",
        
fit()新增使用去重前的数据统计流行度：
        events 可以统计真实交互次数；
        unique_users 可以统计不同用户数
        
新增计算时间权重方法_get_time_weight()
    最近一次行为：weight = 1.0
    半衰期之前的行为：weight = 0.5
    两个半衰期之前的行为：weight = 0.25
    
新增流行度惩罚方法_apply_popularity_penalty()
    adjusted_scores = scores / (1+阿尔法×normalized_popuarity)

删除_recommend()

新增_recommend_from_history()
    原recommend：
        读取 item_id
        找到每个物品的邻居
        直接累加相似度
        排除历史物品
        排序
        
    相较于原本的recommed：
        读取 item_id 和 timestamp
        批量计算时间权重
        加权累加相似度
        应用流行度惩罚
        排除历史物品
        排序
修改recommend_for_user()，调用_recommend_from_history()
"""
class TimeAwarePopularityRegularizedItemKNNRecommender:
    def __init__(self,neighbor_k: int = 50,
                 *,
                 time_decay_enabled: bool = False,
                 time_half_life_days: float = 90.0,
                 popularity_penalty_enabled: bool = False,
                 popularity_penalty_alpha: float = 0.25,
                 popularity_count_mode: str = "unique_users",
                 ) -> None:
        if not isinstance(neighbor_k,int,) or isinstance(neighbor_k, bool):
            raise TypeError("neighbor_k 必须是整数")
        if neighbor_k < 1:
            raise ValueError("neighbor_k 必须大于等于 1")
        if not isinstance(time_decay_enabled, bool):
            raise TypeError("time_decay_enabled 必须是 bool")
        if not isinstance(popularity_penalty_enabled, bool):
            raise TypeError("popularity_penalty_enabled 必须是 bool")
        if (isinstance(time_half_life_days, bool)or not isinstance(time_half_life_days, (int, float))):
            raise TypeError("time_half_life_days 必须是数字")
        if (not np.isfinite(time_half_life_days)or time_half_life_days <= 0):
            raise ValueError("time_half_life_days 必须是有限的正数")
        if (isinstance(popularity_penalty_alpha, bool)or not isinstance(popularity_penalty_alpha, (int, float))):
            raise TypeError("popularity_penalty_alpha 必须是数字")
        if (not np.isfinite(popularity_penalty_alpha)or popularity_penalty_alpha < 0):
            raise ValueError("popularity_penalty_alpha 必须是有限的非负数")
        if popularity_count_mode not in {"events", "unique_users"}:
            raise ValueError("popularity_count_mode 必须是 events 或 unique_users")

        self.neighbor_k = neighbor_k

        self.time_decay_enabled = time_decay_enabled
        self.time_half_life_days = float(time_half_life_days)
        self.popularity_penalty_enabled = popularity_penalty_enabled
        self.popularity_penalty_alpha = float(popularity_penalty_alpha)
        self.popularity_count_mode = popularity_count_mode

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

        # 每个物品的训练集流行度
        self.item_popularity_: np.ndarray | None = None
        # 归一化后的物品流行度
        self.normalized_popularity_: np.ndarray | None = None

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
    def fit(self,interactions: pd.DataFrame,) -> "TimeAwarePopularityRegularizedItemKNNRecommender":
        if not isinstance(interactions,pd.DataFrame,):
            raise TypeError("interactions 必须是 pandas.DataFrame")
        required_columns = {"user_id","item_id",}
        missing_columns = (required_columns - set(interactions.columns))
        if missing_columns:
            raise ValueError("interactions 缺少字段："f"{sorted(missing_columns)}")
        if interactions.empty:
            raise ValueError("interactions 为空")

        # raw_data不去重，用于统计流行度
        raw_data = interactions[["user_id", "item_id"]].copy()

        if raw_data.isna().any().any():
            raise ValueError("user_id 或 item_id 中存在空值")

        raw_data["user_id"] = raw_data["user_id"].astype("string").str.strip()
        raw_data["item_id"] = raw_data["item_id"].astype("string").str.strip()
        if raw_data["user_id"].eq("").any() or raw_data["item_id"].eq("").any():
            raise ValueError("user_id 或 item_id 中存在空字符串")
        if raw_data.empty:
            raise ValueError("清洗 user_id 和 item_id 后没有有效数据")
# --------------------------------------------------
# 1. 使用去重前的数据统计物品流行度
# --------------------------------------------------
        # 统计交互记录数
        if self.popularity_count_mode == "events":
            popularity = raw_data.groupby("item_id").size()
        # 统计交互过该物品的不同用户数
        elif self.popularity_count_mode == "unique_users":
            popularity = raw_data.groupby("item_id")["user_id"].nunique()
        else:
            raise ValueError("popularity_count_mode 必须是events 或 unique_users")
# --------------------------------------------------
# 2. 去重后构造物品-用户交互矩阵
# --------------------------------------------------
        # 同一个用户重复交互同一个物品只记为一次
        matrix_data = raw_data.drop_duplicates(subset=["user_id", "item_id"]).reset_index(drop=True)
        if matrix_data.empty:
            raise ValueError("去重后没有有效交互数据")

        #构建ID ↔ index双向映射
        item_ids = sorted(matrix_data["item_id"].unique().tolist())
        user_ids = sorted(matrix_data["user_id"].unique().tolist())
        self.item_to_index_ = {item_id: index for index, item_id in enumerate(item_ids)}
        self.index_to_item_ = item_ids
        self.user_to_index_ = {user_id: index for index, user_id in enumerate(user_ids)}
        self.index_to_user_ = user_ids
# --------------------------------------------------
# 3. 构造物品-用户稀疏交互矩阵
# --------------------------------------------------
        # item_to_index_得到物品在交互矩阵中的行下表，user_to_index_得到列下标，一对就是一个坐标
        item_indices = matrix_data["item_id"].map(self.item_to_index_).to_numpy(dtype=np.int64)
        user_indices = matrix_data["user_id"].map(self.user_to_index_).to_numpy(dtype=np.int64)
        # 把上述坐标位置全部设为1
        values = np.ones(len(matrix_data), dtype=np.float32)
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
# --------------------------------------------------
# 4. 计算物品之间的余弦相似度
# --------------------------------------------------
        # 对每个物品向量做 L2 归一化
        normalized_matrix = normalize(item_user_matrix,norm="l2",axis=1,copy=True,)
        # 计算# 物品-物品余弦相似度矩阵
        similarity_matrix = (normalized_matrix @ normalized_matrix.T).tocsr()
        # 物品和自身的相似度不参与推荐
        similarity_matrix.setdiag(0)
        similarity_matrix.eliminate_zeros()
        self.similarity_matrix_ = (similarity_matrix)
# --------------------------------------------------
# 5. 按物品 ID 顺序保存流行度
# --------------------------------------------------
        self.item_popularity_ = np.array(
            [popularity.get(item_id, 0)for item_id in self.index_to_item_],dtype=np.float32,)
        # 使用 log1p 压缩热门物品的计数差距
        log_popularity = np.log1p(self.item_popularity_)
        max_log_popularity = log_popularity.max()
        if max_log_popularity > 0:
            self.normalized_popularity_ = (log_popularity / max_log_popularity)
        else:
            self.normalized_popularity_ = (np.zeros_like(log_popularity))
# --------------------------------------------------
# 6. 建立每个物品的 Top-K 邻居缓存
# --------------------------------------------------
        self._build_neighbor_cache()
        return self

    """
    根据行为时间和参考时间计算时间衰减权重。
    timestamp:某次用户行为的 Unix 时间戳，单位为秒。
    reference_timestamp:当前用户历史中的最大时间戳。
    返回值：时间权重，范围为 (0, 1]。
    """
    def _get_time_weight(
            self,
            timestamp: float | np.ndarray | pd.Series,
            reference_timestamp: float,) -> float | np.ndarray:

        timestamp_array = np.asarray(timestamp)

        if not self.time_decay_enabled:
            if timestamp_array.ndim == 0:
                return 1.0
            return np.ones(timestamp_array.shape,dtype=np.float32,)

        timestamp_array = np.asarray(timestamp,dtype=np.float64,)
        reference_timestamp = float(reference_timestamp)

        if not np.isfinite(timestamp_array).all():
            raise ValueError("timestamp 必须是有限数值")

        if not np.isfinite(reference_timestamp):
            raise ValueError("reference_timestamp 必须是有限数值")

        # 防止由于数据排序或异常时间导致负时间差
        delta_seconds = np.maximum(0.0,reference_timestamp - timestamp_array,)

        delta_days = delta_seconds / 86400.0

        weights = np.exp2(-delta_days / self.time_half_life_days)

        if timestamp_array.ndim == 0:
            return float(weights)

        return weights.astype(np.float32,copy=False,)

    """
    根据物品流行度调整候选物品分数。流行度越高，分数下降越多。
    adjusted_score =raw_score/ (1 + alpha * normalized_popularity)
    """
    def _apply_popularity_penalty(self,scores: np.ndarray,) -> np.ndarray:
        self._check_fitted()
        if self.normalized_popularity_ is None:
            raise RuntimeError("流行度信息不存在，请先调用 fit()")
        if scores.ndim != 1:
            raise ValueError("scores 必须是一维数组")

        if len(scores) != len(self.index_to_item_):
            raise ValueError("scores 长度必须等于物品数量")

        # 复制一份，避免原地修改外部传入的分数数组
        adjusted_scores = scores.copy()

        # 未启用流行度惩罚时，保持原始分数
        if not self.popularity_penalty_enabled:
            return adjusted_scores

        # 只处理当前有正分数的候选物品
        candidate_indices = np.flatnonzero(
            adjusted_scores > 0)

        if len(candidate_indices) == 0:
            return adjusted_scores

        popularity_values = (
            self.normalized_popularity_[candidate_indices])

        penalty_factor = (1.0+ self.popularity_penalty_alpha * popularity_values)

        adjusted_scores[candidate_indices] = (adjusted_scores[candidate_indices] / penalty_factor)
        return adjusted_scores

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
    根据单个用户的历史交互生成推荐。
    history_interactions 至少需要包含：
        - item_id
    当启用时间衰减时，还需要包含：
        - timestamp
    """
    def _recommend_from_history(
            self,
            history_interactions: pd.DataFrame,
            *,
            k: int = 10,) -> list[str]:
        self._check_fitted()

        if not isinstance(history_interactions,pd.DataFrame,):
            raise TypeError("history_interactions 必须是 pandas.DataFrame")
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k 必须是整数")
        if k < 1:
            raise ValueError("k 必须大于等于 1")

        required_columns = {"item_id",}

        if self.time_decay_enabled:
            required_columns.add("timestamp")

        missing_columns = (required_columns- set(history_interactions.columns))
        if missing_columns:
            raise ValueError("history_interactions 缺少字段："f"{sorted(missing_columns)}")

        if history_interactions.empty:
            return []
        history = history_interactions.copy()
        if history["item_id"].isna().any():
            raise ValueError("history_interactions 的 item_id 中存在空值")
        history["item_id"] = (history["item_id"].astype("string").str.strip())
        if history["item_id"].eq("").any():
            raise ValueError("history_interactions 的 item_id 中存在空字符串")

        # --------------------------------------------------
        # 1. 处理时间信息
        # --------------------------------------------------
        reference_timestamp: float | None = None

        if self.time_decay_enabled:
            history["timestamp"] = pd.to_numeric(history["timestamp"],errors="coerce",)

            if history["timestamp"].isna().any():
                raise ValueError("timestamp 中存在无法转换为数字的值")
            if not np.isfinite(
                    history["timestamp"].to_numpy(dtype=np.float64)).all():
                raise ValueError("timestamp 必须是有限数值")

            # 使用用户全部历史中的最新时间作为参考时间
            reference_timestamp = float(
                history["timestamp"].max())

            # 如果一个用户对同一物品有多次行为，
            # 保留时间上最近的一次行为
            history = history.sort_values("timestamp",kind="stable",)

        # 普通 Item-KNN 中，同一个历史物品只参与一次计算
        history = history.drop_duplicates(subset=["item_id"],keep="last",).reset_index(drop=True)

        # --------------------------------------------------
        # 2. 只保留训练集中出现过的物品
        # --------------------------------------------------
        known_history = history[
            history["item_id"].isin(self.item_to_index_)].copy()

        if known_history.empty:
            return []

        item_indices = (
            known_history["item_id"].map(self.item_to_index_).to_numpy(dtype=np.int64))

        # --------------------------------------------------
        # 3. 批量计算时间权重
        # --------------------------------------------------
        if self.time_decay_enabled:
            assert reference_timestamp is not None

            timestamps = (known_history["timestamp"].to_numpy(dtype=np.float64))

            history_weights = self._get_time_weight(timestamp=timestamps,reference_timestamp=reference_timestamp,)

            history_weights = np.asarray(history_weights,dtype=np.float32,)
        else:
            history_weights = np.ones(len(known_history),dtype=np.float32,)

        # --------------------------------------------------
        # 4. 累加历史物品的邻居分数
        # --------------------------------------------------
        scores = np.zeros(len(self.index_to_item_),dtype=np.float32,)

        for item_index, history_weight in zip(item_indices,history_weights,):
            neighbor_indices = (self.neighbor_indices_[item_index])

            neighbor_scores = (self.neighbor_scores_[item_index])

            scores[neighbor_indices] += (history_weight * neighbor_scores)

        # --------------------------------------------------
        # 5. 应用流行度惩罚
        # --------------------------------------------------
        scores = self._apply_popularity_penalty(scores)

        # --------------------------------------------------
        # 6. 排除用户已经交互过的物品
        # --------------------------------------------------
        seen_indices = np.unique(item_indices)
        scores[seen_indices] = -np.inf

        # --------------------------------------------------
        # 7. 筛选并排序候选物品
        # --------------------------------------------------
        candidate_indices = np.flatnonzero(scores > 0)

        if len(candidate_indices) == 0:
            return []

        order = np.argsort(-scores[candidate_indices],kind="stable",)

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
        if self.time_decay_enabled:
            required_columns.add("timestamp")
        missing_columns = required_columns - set(history_interactions.columns)
        if missing_columns:
            raise ValueError(
                "history_interactions 缺少字段："
                f"{sorted(missing_columns)}")

        user_id = str(user_id)

        user_history = history_interactions[
            history_interactions["user_id"]
            .astype("string")
            .eq(user_id)].copy()

        return self._recommend_from_history(
            history_interactions=user_history,
            k=k,)

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
    "TimeAwarePopularityRegularizedItemKNNRecommender",
]





