from __future__ import annotations
from typing import Any, Mapping
from collections.abc import Iterable

import math
import pandas as pd

"""检查k是否合法"""
def _validate_k(k: int) -> None:
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k 必须是整数")
    if k < 1:
        raise ValueError("k 必须大于等于 1")

"""把各种不同类型、不同容器格式的物品 ID 输入，统一转换为标准的字符串列表。"""
def _to_item_list(items: Any,) -> list[str]:
    if items is None:
        return []

    # 字符串符合 Iterable 协议，可以被遍历，但遍历字符串得到的是单个字符，所以需要单独处理
    if isinstance(items, (str, bytes)):
        return [str(items)]
    if not isinstance(items, Iterable):
        return [str(items)]

    # 遍历每个元素，统一强制转为字符串，兼容所有可迭代容器：list、tuple、set、pandas.Series、numpy.ndarray 等
    return [str(item) for item in items]

"""去除物品列表中的重复项，并保留原始顺序。"""
def _unique_items(items: Any,) -> list[str]:
    return list(dict.fromkeys(_to_item_list(items))) # 保留原始顺序的去重

"""
Recall@K = 推荐结果命中的真实物品数 / 真实物品总数
"""
def recall_at_k(recommended_items:Any, ground_truth_items:Any, k:int = 10) ->float:
    _validate_k(k)

    recommended = _unique_items(recommended_items)[:k] # 先去重，再截断 Top-K
    ground_truth = set(_to_item_list(ground_truth_items))
    if not ground_truth:
        return 0.0

    hits = len(set(recommended) & ground_truth)
    return hits / len(ground_truth)

def ndcg_at_k(recommended_items: Any,ground_truth_items: Any,k: int = 10,) -> float:
    _validate_k(k)
    recommended = _unique_items(recommended_items)[:k] # 先去重，再截断 Top-K
    ground_truth = set(_to_item_list(ground_truth_items))
    if not ground_truth:
        return 0.0

    dcg = 0.0
    for rank, item_id in enumerate(recommended, start=1):
        if item_id in ground_truth:
            dcg += 1.0 / math.log2(rank+1)

    idea_length = min(k, len(ground_truth))
    idcg = sum(1.0/math.log2(rank+1) for rank in range(1, idea_length+1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

""" 
第一个相关物品排名为 rank 时：
reciprocal_rank = 1 / rank
"""
def mrr_at_k(recommended_items: Any,ground_truth_items: Any,k: int = 10,) -> float:
    _validate_k(k)
    recommended = _unique_items(recommended_items)[:k] # 先去重，再截断 Top-K
    ground_truth = set(_to_item_list(ground_truth_items))
    if not ground_truth:
        return 0.0

    for rank, item_id in enumerate(recommended, start=1):
        if item_id in ground_truth:
            return 1.0/rank
    return 0.0

'''
计算所有用户的平均 Recall@K、NDCG@K 和 MRR@K。
    recommendations:用户到推荐列表的映射,例如：
        {
            "1": ["50", "100", "181"],
            "2": ["127", "174", "300"]
        }
        ground_truth:用户到真实目标物品的映射,例如：
        {
            "1": ["181"],
            "2": ["300"]
        }
'''
def evaluate_ranking(
    recommendations: Mapping[Any, Any],
    ground_truth: Mapping[Any, Any],
    k: int = 10,) -> dict[str, float | int]:

    _validate_k(k)
    if not ground_truth:
        raise ValueError("ground_truth 为空")

    total_recall = 0.0
    total_ndcg = 0.0
    total_mrr = 0.0

    valid_user_count = 0

    normalized_recommendations = {
        str(user_id): predicted_items
        for user_id, predicted_items
        in recommendations.items()
    }

    for user_id, true_items in (ground_truth.items()):
        true_items = _unique_items(true_items)
        if not true_items:
            continue
        predicted_items = normalized_recommendations.get(
            str(user_id),
            [],
        )

        total_recall += recall_at_k(predicted_items,true_items,k,)
        total_ndcg += ndcg_at_k(predicted_items,true_items,k,)
        total_mrr += mrr_at_k(predicted_items,true_items,k,)

        valid_user_count += 1

    if valid_user_count == 0:
        raise ValueError("ground_truth 中没有有效用户")

    return {
        "num_users": valid_user_count,
        f"Recall@{k}": (total_recall / valid_user_count),
        f"NDCG@{k}": (total_ndcg / valid_user_count),
        f"MRR@{k}": (total_mrr / valid_user_count),
        }

"""
history_interactions:
    用于构造用户历史的数据。
target_interactions:
    真实目标数据，例如 valid.csv 或 test.csv。
"""
def evaluate_recommender(
        model: Any,
        history_interactions: pd.DataFrame,
        target_interactions: pd.DataFrame,
        k: int = 10,) -> dict[str, float | int]:
        _validate_k(k)
        required_columns = {"user_id","item_id"}

        for name, dataframe in {
            "history_interactions": history_interactions,
            "target_interactions": target_interactions,}.items():

            if not isinstance(dataframe,pd.DataFrame,):
                raise TypeError(f"{name} 必须是 pandas.DataFrame")

            missing_columns = (required_columns - set(dataframe.columns))
            if missing_columns:
                raise ValueError(f"{name} 缺少字段："f"{sorted(missing_columns)}")
            if dataframe.empty:
                raise ValueError(f"{name} 为空")

        ground_truth: dict[str, list[str]] = {}
        for user_id, group in (target_interactions.groupby("user_id", sort=False)):
            ground_truth[str(user_id)] = (group["item_id"].astype("string").tolist())

        recommendations: dict[str, list[str]] = {}

        prepare_history = getattr(model, "prepare_history", None)
        if callable(prepare_history):
            prepare_history(history_interactions)

        for user_id in ground_truth:
            if callable(prepare_history):
                recommendations[user_id] = (
                    model.recommend_for_user(user_id, k=k)
                )
            else:
                recommendations[user_id] = (
                    model.recommend_for_user(
                        user_id,
                        history_interactions,
                        k=k,
                    )
                )

        return evaluate_ranking(recommendations,ground_truth,k=k,)

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "evaluate_ranking",
    "evaluate_recommender",
]


