from __future__ import annotations
from collections.abc import Iterable

import pandas as pd

"""
基于物品流行度的推荐器。
默认按照训练集中物品出现的次数计算流行度。
    events:
        统计物品出现的交互行数。
    unique_users:
        统计交互过该物品的不同用户数。
        
get_popularity_table() 的输出：
    item_id  popularity
        50       490
        100      399
        181      368
recommend() 的输出：
    ["286","313","318","300","288"]
"""
class PopularityRecommender:
    def __init__(self,count_mode: str = "events",) -> None:
        if count_mode not in {"events","unique_users",}:
            raise ValueError("count_mode 必须是events' 或 'unique_users'")

        self.count_mode = count_mode
        self.popularity_: pd.DataFrame | None = None
        self.ranking_: list[str] = []

# 使用训练集统计物品流行度。这里只能传入 train.csv，不能使用验证集或测试集。
    def fit(self,interactions: pd.DataFrame,) -> "PopularityRecommender":
        if not isinstance(interactions,pd.DataFrame,):
            raise TypeError("interactions 必须是 pandas.DataFrame")

        required_columns = {"user_id","item_id",}
        missing_columns = (required_columns-set(interactions.columns))
        if missing_columns:
            raise ValueError("interactions 缺少字段："f"{sorted(missing_columns)}")
        if interactions.empty:
            raise ValueError("训练交互数据为空")

        data = interactions.copy()
        data["user_id"] = (data["user_id"].astype("string").str.strip())
        data["item_id"] = (data["item_id"].astype("string").str.strip())

        if self.count_mode == "events":
            popularity = (data.groupby("item_id",sort=False,).size().rename("popularity"))
        else:
            popularity = (data.groupby("item_id",sort=False,)["user_id"].nunique().rename("popularity"))

        # 流行度降序排列 kind="stable" 保证相同流行度时结果稳定
        popularity = popularity.sort_values(ascending=False,kind="stable",)
        self.popularity_ = (popularity.reset_index())
        self.ranking_ = (self.popularity_["item_id"].astype("string").tolist())

        return self


        # 为一个用户生成 Top-K 推荐。
        # user_id:当前版本中只用于接口记录，推荐结果不依赖用户画像。
        # seen_items:用户已经交互过的物品。这些物品不会再次推荐。
        # k:返回推荐物品数量。
    def recommend(
        self,
        user_id: str | int | None = None,
        *,
        seen_items: Iterable[str | int] | None = None,
        k: int = 10,) -> list[str]:

        if self.popularity_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k 必须是大于等于 1 的整数")

        if seen_items is None:
            seen_item_ids: set[str] = set()
        else: # 传入已交互物品时，统一转为字符串格式
            seen_item_ids = {str(item_id) for item_id in seen_items}

        recommendations: list[str] = []
        for item_id in self.ranking_:
            if item_id in seen_item_ids:
                continue
            recommendations.append(item_id)
            if len(recommendations) >= k:
                break
        return recommendations

    """
    根据训练集中的用户历史生成推荐。
    """
    def recommend_for_user(
        self,
        user_id: str | int,
        train_interactions: pd.DataFrame,
        *,
        k: int = 10,) -> list[str]:

        if not isinstance(train_interactions,pd.DataFrame,):
            raise TypeError("train_interactions 必须是 pandas.DataFrame")

        user_id = str(user_id)
        user_history = train_interactions[train_interactions["user_id"].astype("string")== user_id]
        seen_items = user_history["item_id"].astype("string").tolist()

        return self.recommend(user_id=user_id,seen_items=seen_items,k=k,)


    # 返回物品流行度表。
    def get_popularity_table(self) -> pd.DataFrame:
        if self.popularity_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        return self.popularity_.copy()
    
__all__ = ["PopularityRecommender",]

