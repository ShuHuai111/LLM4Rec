from __future__ import annotations
from typing import Any

import pandas as pd

'''
按照用户行为时间顺序，
把每个用户的历史交互拆成训练集、验证集和测试集。
'''

REQUIRED_COLUMNS = (
    "user_id",
    "item_id",
    "timestamp",
)

def _validate_interactions(interactions: pd.DataFrame,) -> None:
    """检查交互数据的基本结构。"""
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError(
            "interactions 必须是 pandas.DataFrame，"
            f"当前类型为 {type(interactions).__name__}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in interactions.columns]
    if missing_columns:
        raise ValueError("interactions 缺少必要字段："f"{missing_columns}")
    if interactions.empty:
        raise ValueError("interactions 为空，无法进行时间切分")
    if interactions[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("user_id、item_id、timestamp 中存在空值")

def _check_item_coverage(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,) -> tuple[list[str], list[str]]:
    """
    检查验证集和测试集中的物品是否在训练集中出现过。

    返回：
    - valid_unseen_items
    - test_unseen_items
    """

    train_item_ids = set(train["item_id"])
    valid_unseen_items = sorted(set(valid["item_id"]) - train_item_ids)
    test_unseen_items = sorted(set(test["item_id"]) - train_item_ids)

    return (valid_unseen_items,test_unseen_items,)

def leave_last_two_split(
    interactions: pd.DataFrame,
    *,
    require_train_item_coverage: bool = True,) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict[str, Any]]:
    """
    使用 leave-last-two 策略进行时间切分。

    对每个用户：

    - 最后一条交互进入测试集；
    - 倒数第二条交互进入验证集；
    - 其余交互进入训练集。

    每个用户至少需要 3 条交互。
    """
    _validate_interactions(interactions)

    # 复制数据，避免修改原始 DataFrame
    ordered = interactions.copy()
    # 对ordered表新增 _source_order 记录原始行号，作为兜底排序键
    ordered["_source_order"] = range(len(ordered))
    # 先按用户分组，同用户内按时间升序（从早到晚），时间戳相同时按原始文件顺序排列。
    ordered = (ordered.sort_values(
            by=["user_id", "timestamp", "_source_order"],
            ascending=[True, True, True],
            kind="stable",).reset_index(drop=True))

    # 用户交互数合法性校验
    user_counts = ordered.groupby("user_id").size()
    users_with_insufficient_history = user_counts[user_counts < 3]
    if not users_with_insufficient_history.empty:
        examples = users_with_insufficient_history.head(10).to_dict()
        raise ValueError(
            "以下用户的交互数量少于 3，无法进行 leave-last-two 切分。"f"示例：{examples}")

    # 从后往前编号：
    # 最后一条       -> 0，测试集
    # 倒数第二条     -> 1，验证集
    # 倒数第三条及之前 -> >= 2，训练集
    # groupby(..., sort=False) 保留之前的用户和时间排序结果。
    # cumcount(ascending=False) 从后往前给每个用户的交互编号：
    # 最后一条为 0，倒数第二条为 1，更早的记录大于等于 2。
    reverse_position = ordered.groupby("user_id", sort=False).cumcount(ascending=False)
    # 用编号做掩码筛选，一步完成三个集合的划分
    test = ordered[reverse_position == 0].copy()
    valid = ordered[reverse_position == 1].copy()
    train = ordered[reverse_position >= 2].copy()

    # 删除内部辅助字段 _source_order，避免污染输出数据。重置为从 0 开始的连续行索引，方便后续使用。
    train = train.drop(columns=["_source_order"]).reset_index(drop=True)
    valid = valid.drop(columns=["_source_order"]).reset_index(drop=True)
    test = test.drop(columns=["_source_order"]).reset_index(drop=True)

    # 检查三个集合中的用户是否一致
    train_users = set(train["user_id"])
    valid_users = set(valid["user_id"])
    test_users = set(test["user_id"])
    if not (train_users == valid_users == test_users):
        raise ValueError("train、valid、test 中的用户集合不一致")

    # 检查每个用户是否恰好拥有一条验证和测试记录
    valid_user_counts = (valid.groupby("user_id").size())
    test_user_counts = (test.groupby("user_id").size())
    if not (valid_user_counts == 1).all():
        raise ValueError("验证集中存在用户拥有多条或零条记录")
    if not (test_user_counts == 1).all():
        raise ValueError("测试集中存在用户拥有多条或零条记录")

    # 检查时间顺序，杜绝数据泄露
    train_last_timestamp = (train.groupby("user_id")["timestamp"].max())
    valid_timestamp = (valid.set_index("user_id")["timestamp"])
    test_timestamp = (test.set_index("user_id")["timestamp"])
    invalid_temporal_order = ((train_last_timestamp > valid_timestamp)| (valid_timestamp > test_timestamp))
    if invalid_temporal_order.any():
        invalid_users = invalid_temporal_order[invalid_temporal_order].index.tolist()
        raise ValueError("发现时间顺序错误，示例用户：" f"{invalid_users[:10]}")

    # 检查验证集和测试集物品是否在训练集中出现（冷启动检查）
    valid_unseen_items, test_unseen_items = _check_item_coverage(train, valid, test)
    if require_train_item_coverage:
        if valid_unseen_items:
            raise ValueError("验证集中存在训练阶段未出现的物品，" f"示例：{valid_unseen_items[:10]}")
        if test_unseen_items:
            raise ValueError("测试集中存在训练阶段未出现的物品，" f"示例：{test_unseen_items[:10]}")

    # 构建完整统计信息并返回
    stats: dict[str, Any] = {
        "strategy": "leave_last_two",
        "num_users": int(len(train_users)),
        "num_interactions": int(len(ordered)),
        "train_interactions": int(len(train)),
        "valid_interactions": int(len(valid)),
        "test_interactions": int(len(test)),
        "min_train_interactions": int(train.groupby("user_id").size().min()),
        "valid_users": int(valid["user_id"].nunique()),
        "test_users": int(test["user_id"].nunique()),
        "valid_unseen_item_count": int(len(valid_unseen_items)),
        "test_unseen_item_count": int(len(test_unseen_items)),
        "valid_unseen_item_examples": valid_unseen_items[:10],
        "test_unseen_item_examples": test_unseen_items[:10],}

    return train, valid, test, stats

# 一个更加通用的接口
def split_dataset(
    interactions: pd.DataFrame,
    *,
    strategy: str = "leave_last_two",
    require_train_item_coverage: bool = True,
) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,dict[str, Any],]:
    """根据 strategy 调用对应的时间切分策略。"""
    if strategy == "leave_last_two":
        return leave_last_two_split(interactions,
            require_train_item_coverage=(require_train_item_coverage),
        )

    raise ValueError(
        "暂不支持的切分策略："
        f"{strategy}")

__all__ = [
    "leave_last_two_split",
    "split_dataset",
]
