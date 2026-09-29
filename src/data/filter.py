from __future__ import annotations
from typing import Any
import pandas as pd

"""
filter.py负责筛选逻辑，包括：
    筛选正向交互，rating >= 4会被视为正向行为，同时如果没有rating字段，会将rating设为1，同时默认为正向交互
    过滤低频用户，统计每个用户的交互数量，低于阈值则不保留，确保用户至少拥有训练序列 + 验证目标 + 测试目标
    过滤低频物品，默认使用item_count_mode: unique_users，也就是统计有多少不同用户交互过这个物品
    迭代过滤直到收敛
    同步过滤物品元数据，只保留实际还存在于交互数据中的物品，减少内存占用
    记录过滤统计
"""


"""检查交互数据的基本结构"""
def _validate_interactions(interactions: pd.DataFrame,) -> None:
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError(
            "interactions 必须是 pandas.DataFrame，"
            f"当前类型为 {type(interactions).__name__}")

    required_columns = {"user_id","item_id",}
    missing_columns = required_columns - set(interactions.columns)
    if missing_columns:
        raise ValueError(
            "interactions 缺少必要字段："
            f"{sorted(missing_columns)}")
    if interactions.empty:
        raise ValueError("interactions 为空")

"""检查物品数据的基本结构。"""
def _validate_items(items: pd.DataFrame) -> None:
    if not isinstance(items, pd.DataFrame):
        raise TypeError(
            "items 必须是 pandas.DataFrame，"
            f"当前类型为 {type(items).__name__}")
    if "item_id" not in items.columns:
        raise ValueError(
            "items 缺少必要字段：item_id")
    if items.empty:
        raise ValueError("items 为空")

"""检查频次阈值是否合法，仅检查是否合法，尚未开始过滤"""
def _validate_threshold(value: int,name: str,) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"{name} 必须是整数，当前类型为 "
            f"{type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} 必须大于等于 1")

"""只保留正向交互"""
def filter_positive_interactions(interactions: pd.DataFrame,) -> pd.DataFrame:
    _validate_interactions(interactions)

    if "is_positive" not in interactions.columns:
        raise ValueError(
            "interactions 缺少 is_positive 字段，"
            "大概率是未执行normalize_interactions()")

    positive_mask = (interactions["is_positive"].fillna(False).astype(bool)) # 空值统一视为负向，不保留。
    positive_interactions = (interactions.loc[positive_mask].copy().reset_index(drop=True))

    if positive_interactions.empty:
        raise ValueError("没有找到正向交互，大概率是评分阈值有问题")
    return positive_interactions

"""
统计物品频次。
    unique_users:统计交互过物品的不同用户数。
    events:统计物品出现的交互行数。
"""
def _get_item_counts(interactions: pd.DataFrame,item_count_mode: str,) -> pd.Series:
    # 按独立用户数统计,按 item_id 分组，对每组的 user_id 做 去重计数
    # 排除同一个用户多次交互的干扰，更能反映物品的真实受众广度和流行度,即常用的 “物品流行度” 口径。
    if item_count_mode == "unique_users":
        return (interactions.groupby("item_id")["user_id"].nunique())

    # 按总交互次数统计，该物品总共产生了多少条交互记录，包含同一个用户的多次交互
    # 反映物品的总行为量和曝光密度，适合做行为量级分析、长尾分布统计
    if item_count_mode == "events":
        return (interactions.groupby("item_id").size())

    # 传入不在枚举内的模式字符串时，直接抛出明确错误，并给出合法取值
    raise ValueError(
        "item_count_mode 必须是 'unique_users' 或 'events'"
    )

"""
迭代过滤低频用户和低频物品。
    每轮执行：
        1. 删除交互次数不足的用户；
        2. 删除交互频次不足的物品；
        3. 重新统计；
        4. 直到数据不再变化。
"""
def iterative_filter_interactions(
    interactions: pd.DataFrame,
    *,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    item_count_mode: str = "unique_users",
    max_rounds: int = 100,) -> tuple[pd.DataFrame, dict[str, Any]]:

    _validate_interactions(interactions)
    _validate_threshold(min_user_interactions,"min_user_interactions",)
    _validate_threshold(min_item_interactions,"min_item_interactions",)
    _validate_threshold(max_rounds,"max_rounds",)
    if item_count_mode not in {"unique_users","events",}:
        raise ValueError(
            "item_count_mode 必须是 'unique_users' 或 'events'"
        )

    current = interactions.copy()
    initial_stats = {
        "num_interactions": int(len(current)),
        "num_users": int(current["user_id"].nunique()),
        "num_items": int(current["item_id"].nunique()),
    }
    history: list[dict[str, int]] = [] #列表容器，记录每一轮迭代的前后数据，用于追踪完整过滤过程。
    converged = False # 收敛标记，用于判断是否提前终止循环。

    for round_index in range(1, max_rounds + 1):
        before_stats = {
            "num_interactions": int(len(current)),
            "num_users": int(current["user_id"].nunique()),
            "num_items": int(current["item_id"].nunique()),}

        # 统计每个用户的交互数量
        user_counts = (current.groupby("user_id").size()) # 按用户分组统计每人的总交互次数
        keep_user_ids = user_counts[user_counts >= min_user_interactions].index # 保留交互数 ≥ 阈值的活跃用户
        current = current[current["user_id"].isin(keep_user_ids)].copy()

        # 统计每个物品的频次
        item_counts = _get_item_counts(current,item_count_mode,)
        keep_item_ids = item_counts[item_counts >= min_item_interactions].index
        current = current[current["item_id"].isin(keep_item_ids)].copy()

        # 记录本轮结束后的统计
        after_stats = {
            "num_interactions": int(len(current)),
            "num_users": int(current["user_id"].nunique()),
            "num_items": int(current["item_id"].nunique()),}
        history.append({
            "round": round_index,
            "before_interactions": before_stats["num_interactions"],
            "after_interactions": after_stats["num_interactions"],
            "before_users": before_stats["num_users"],
            "after_users": after_stats["num_users"],
            "before_items": before_stats["num_items"],
            "after_items": after_stats["num_items"],})

        if before_stats == after_stats:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"过滤在 {max_rounds} 轮内没有收敛")
    if current.empty:
        raise ValueError(
            "过滤后没有剩余交互，请降低频次阈值")

    current = current.reset_index(drop=True)
    # 最终状态统计
    final_stats = {
        "num_interactions": int(len(current)),
        "num_users": int(current["user_id"].nunique()),
        "num_items": int(current["item_id"].nunique()),}
    stats: dict[str, Any] = {
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "item_count_mode": item_count_mode,
        "initial": initial_stats,
        "final": final_stats,
        "num_rounds": len(history),
        "history": history,}
    return current, stats

"""
总函数， 过滤交互数据和物品元数据。
    返回：
        1. 过滤后的交互数据；
        2. 过滤后的物品数据；
        3. 过滤统计信息。
"""
def filter_dataset(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    *,
    positive_only: bool = True,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    item_count_mode: str = "unique_users",
    max_rounds: int = 100,) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        dict[str, Any],
    ]:

    _validate_interactions(interactions)
    _validate_items(items)

    # 默认只用正向交互
    if positive_only:
        filtered_interactions = filter_positive_interactions(interactions)
    else:
        filtered_interactions = interactions.copy()

    # k-core 迭代稠密过滤
    filtered_interactions, stats = iterative_filter_interactions(
        filtered_interactions,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        item_count_mode=item_count_mode,
        max_rounds=max_rounds,)

    # 校验过滤后的交互中所有 item_id，都能在物品元数据中找到对应记录
    interaction_item_ids = set(filtered_interactions["item_id"])
    metadata_item_ids = set(items["item_id"])
    missing_item_ids = sorted(interaction_item_ids - metadata_item_ids)
    if missing_item_ids:
        raise ValueError(
            "以下 item_id 在交互数据中存在，"
            "但在物品元数据中不存在："
            f"{missing_item_ids[:10]}")

    # 只保留在过滤后的交互中实际出现过的物品，删掉所有没有交互的冷门物品元数据。
    filtered_items = (items[items["item_id"].isin(interaction_item_ids)].copy().reset_index(drop=True))

    # 把正向筛选开关、最终物品元数据数量补充到统计字典中，让统计信息覆盖全流程配置与结果。
    stats["positive_only"] = positive_only
    stats["num_metadata_items"] = int(len(filtered_items))

    return (filtered_interactions, filtered_items, stats)

__all__ = [
    "filter_positive_interactions",
    "iterative_filter_interactions",
    "filter_dataset",
]
