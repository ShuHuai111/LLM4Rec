from __future__ import  annotations
import math
import pandas as pd
from collections.abc import Sequence
INTERACTION_REQUIRED_COLUMNS = ("user_id", "item_id", "timestamp",)
ITEM_REQUIRED_COLUMNS = ("item_id",)

"""
交互数据：
    输入数据：                  
            user_id  item_id  rating  timestamp   
            196      242      3       881250949
            186      302      3       891717742
            22       377      1       87888711
            
    输出数据的标准化工作：如果数据没有 rating，代码会自动补充： rating = 1.0；is_positive = True
            user_id     item_id    rating       timestamp       is_positive
                1        168        5.0         874965478           True
                1        172        5.0         874965478           True
                1        165        5.0         874965518           True

物品数据：
    输入数据：
            item_id         title             release_date  imdb_url      genres
            1            Toy Story (1995)     01-Jan-1995   ...           Animation|Children's|Comedy
            2            GoldenEye (1995)     01-Jan-1995   ...           Action|Adventure|Thriller
            
    对物品数据的标准化工作：
            item_id 统一为字符串；
            title 和 genres 的缺失值转换为空字符串；
            检查 item_id 是否重复；
            其他列保留。
"""

"""
检查输入对象和字段。
isinstance(dataframe, pd.DataFrame)检查输入是不是 DataFrame
检查字段是否存在
"""
def _require_columns(
        dataframe:pd.DataFrame,
        required_columns:Sequence[str],
        dataframe_name:str) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{dataframe_name}必须是 pandas.DataFrame,"
                        f"当前类型为{type(dataframe).__name__}")
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{dataframe_name}缺少必要字段："
                        f"{','.join(missing_columns)}")

"""
统一处理用户 ID 和物品 ID:
    转换为字符串,去除首尾空格
    invalid_mask = normalized.isna() | normalized.eq("")检查缺失值或空字符串
"""
def _normalize_id_column(
        series:pd.Series,
        column_name:str,
        dataframe_name:str) -> pd.Series:# 统一将ID转换成字符串
    normalized = series.astype("string").str.strip()
    invalid_mask = normalized.isna() | normalized.eq("")
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        raise ValueError(
            f"{dataframe_name}.{column_name}中存在"
            f"{invalid_count}个空id"
        )
    return normalized


"""
把不同形式的时间统一为Unix秒级时间戳
"""
def _normalize_timestamp(series: pd.Series) -> pd.Series:
    # datetime64 底层存储的是纳秒，不能先走 pd.to_numeric()
    # 否则会把纳秒误当成秒。
    if not pd.api.types.is_datetime64_any_dtype(series):
        numeric_timestamp = pd.to_numeric(series, errors="coerce")

        # 如果整列都可以转换为数字，则认为原始数据已经是时间戳
        if numeric_timestamp.notna().all():
            if not numeric_timestamp.map(math.isfinite).all():
                raise ValueError("timestamp 中存在无穷大或非法数值")

            # 时间戳应该是整数秒，不允许 1.5 这种小数时间戳
            if numeric_timestamp.mod(1).ne(0).any():
                raise ValueError("数值型 timestamp 必须是整数")

            return numeric_timestamp.astype("int64")

    # 否则尝试解析为日期字符串或 datetime 类型
    parsed_timestamp = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed_timestamp.isna().any():
        invalid_count = int(parsed_timestamp.isna().sum())
        raise ValueError(
            f"timestamp 中存在 {invalid_count} 个无法解析的时间值"
        )
    # pandas datetime 内部是纳秒，转换为 Unix 秒
    return (
        parsed_timestamp.astype("int64") // 1_000_000_000
    ).astype("int64")


"""
交互数据标准化的主函数
    检查字段,至少有:user_id, item_id, timestamp, rating 是可选的。
    复制DataFrame, 避免直接修改调用方原来的 DataFrame。
    清洗 ID
    清洗时间
    处理评分
    生成 is_positive
    如果原始数据没有 rating,处理隐式反馈
    可选去重
    按用户和时间排序
    重置索引
"""
def normalize_interactions(
    interactions: pd.DataFrame,
    *,
    positive_rating_threshold: float | None = 4.0,
    rating_min: float | None = 1.0,
    rating_max: float | None = 5.0,
    deduplicate: bool = False,
) -> pd.DataFrame:
    """
    标准化用户交互数据。
    输入至少包含：
    - user_id
    - item_id
    - timestamp

    rating 是可选字段：
    - 显式评分数据：保留并转换 rating；
    - 隐式反馈数据：自动补充 rating=1.0。

    返回字段至少包含：
    - user_id
    - item_id
    - rating
    - timestamp
    - is_positive

    参数
    ----
    positive_rating_threshold:
        评分大于等于该值时，is_positive=True。
        如果设置为 None，则所有交互都视为正向交互。

    rating_min / rating_max:
        显式评分的合法范围。
        对 MovieLens 100K 来说通常是 1 到 5。

    deduplicate:
        是否删除完全相同的交互记录。
        默认 False，避免误删真实的重复行为。
    """
    _require_columns(interactions, INTERACTION_REQUIRED_COLUMNS,"interactions",)
    if interactions.empty:
        raise ValueError("interactions 为空，无法进行标准化")

    normalized = interactions.copy()
    normalized["user_id"] = _normalize_id_column(
        normalized["user_id"],
        "user_id",
        "interactions",
    )
    normalized["item_id"] = _normalize_id_column(
        normalized["item_id"],
        "item_id",
        "interactions",
    )
    normalized["timestamp"] = _normalize_timestamp(
        normalized["timestamp"]
    )

    if "rating" in normalized.columns:
        rating = pd.to_numeric(
            normalized["rating"],
            errors="coerce",)

        if rating.isna().any():
            invalid_count = int(rating.isna().sum())
            raise ValueError(
                f"rating 中存在 {invalid_count} 个无法转换为数值的值")

        if not rating.map(math.isfinite).all():
            raise ValueError("rating 中存在无穷大或非法数值")

        if rating_min is not None:
            invalid_min = rating < rating_min
            if invalid_min.any():
                raise ValueError(
                    f"rating 中存在小于 {rating_min} 的值")

        if rating_max is not None:
            invalid_max = rating > rating_max
            if invalid_max.any():
                raise ValueError(
                    f"rating 中存在大于 {rating_max} 的值")

        normalized["rating"] = rating.astype("float32")
        if positive_rating_threshold is None:
            normalized["is_positive"] = True
        else:
            normalized["is_positive"] = (
                normalized["rating"] >= positive_rating_threshold)
    else:
        # 隐式反馈数据没有 rating 时，统一视为一次正向行为
        normalized["rating"] = pd.Series(
            1.0,
            index=normalized.index,
            dtype="float32",
        )
        normalized["is_positive"] = True

    if deduplicate:
        normalized = normalized.drop_duplicates(keep="first").reset_index(drop=True)

    # 保存原始行顺序，用于 timestamp 相同时保持确定性
    normalized["_source_order"] = range(len(normalized))

    normalized = normalized.sort_values(
        by=["user_id", "timestamp", "_source_order"],
        ascending=[True, True, True],
        kind="stable",
    )

    normalized = normalized.drop(
        columns=["_source_order"]
    ).reset_index(drop=True)

    return normalized

"""
处理物品元数据的主函数：
    检查 item_id，物品表至少需要item_id
    _normalize_id_column(...)统一物品 ID
    检查重复物品，所有重复项都标记为 True，发现重复后，会抛出错误。
    处理文本字段 for column in ("title", "genres")
"""
def normalize_items(items: pd.DataFrame,) -> pd.DataFrame:
    _require_columns(items,ITEM_REQUIRED_COLUMNS,"items",)

    if items.empty:
        raise ValueError("items 为空，无法进行标准化")

    normalized = items.copy()

    normalized["item_id"] = _normalize_id_column(
        normalized["item_id"],
        "item_id",
        "items",)

    duplicated_mask = normalized["item_id"].duplicated(keep=False)

    if duplicated_mask.any():
        duplicated_ids = (normalized.loc[duplicated_mask,"item_id",].drop_duplicates().tolist())
        raise ValueError(
            "items 中存在重复 item_id，示例："
            f"{duplicated_ids[:10]}")

    # 常见文本元数据统一为空字符串，而不是 NaN
    for column in ("title", "genres"):
        if column in normalized.columns:
            normalized[column] = (normalized[column].astype("string").fillna("").str.strip())
    return normalized.reset_index(drop=True)


"""
整合了处理交互数据和物品数据的函数
"""
def normalize_dataset(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    *,
    positive_rating_threshold: float | None = 4.0,
    rating_min: float | None = 1.0,
    rating_max: float | None = 5.0,
    deduplicate: bool = False,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    同时标准化交互数据和物品数据，并检查引用关系。

    最终会确保：
        interactions 中出现的每个 item_id，
        都能够在 items 中找到对应的物品元数据。
    """
    normalized_interactions = normalize_interactions(interactions,
                                                     positive_rating_threshold=positive_rating_threshold,
                                                     rating_min=rating_min,
                                                     rating_max=rating_max,
                                                     deduplicate=deduplicate)
    normalized_items = normalize_items(items)
    interaction_item_ids = set(normalized_interactions["item_id"].tolist())
    metadata_item_ids = set(normalized_items["item_id"].tolist())
    missing_item_ids = sorted(interaction_item_ids - metadata_item_ids)

    if missing_item_ids:
        raise ValueError(
            "以下 item_id 在交互数据中出现，"
            "但在物品元数据中不存在："
            f"{missing_item_ids[:10]}"
        )
    return normalized_interactions, normalized_items

"""
表示对外提供的函数
"""
__all__ = [
    "normalize_interactions",
    "normalize_items",
    "normalize_dataset",
]
