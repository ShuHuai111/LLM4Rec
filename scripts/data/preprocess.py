from __future__ import annotations

from typing import Any
from pathlib import Path
import sys
import json
import argparse
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.datasets.adapters.movielens_100k import MovieLens100KAdapter
from src.datasets.adapters.movielens_1m import MovieLens1MAdapter
from src.preprocessing.filter import filter_dataset
from src.preprocessing.normalize import normalize_dataset
from src.preprocessing.split import split_dataset

"""
将配置中的路径解析为绝对路径。
"""
def resolve_path(value: str | Path,project_root: Path,) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path

"""读取 YAML 配置文件。"""
def load_config(config_path: str | Path,) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}")

    with config_path.open("r",encoding="utf-8-sig",) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("YAML 配置文件必须解析为字典")

    return config

"""
    根据配置创建数据适配器。
    当前支持 MovieLens 100K 和 MovieLens 1M。
    后续扩展时可以用字典代替if，避免代码臃肿
"""
def build_adapter(config: dict[str, Any],):
    adapter_name = config.get(
        "adapter",
        "movielens_100k",)

    if adapter_name == "movielens_100k":
        raw_data_dir = resolve_path(
            config["raw_data_dir"],
            PROJECT_ROOT,)
        return MovieLens100KAdapter(
            raw_data_dir)

    if adapter_name == "movielens_1m":
        raw_data_dir = resolve_path(
            config["raw_data_dir"],
            PROJECT_ROOT,
        )
        return MovieLens1MAdapter(raw_data_dir)

    raise ValueError(f"暂不支持的数据适配器：{adapter_name}")

"""统计交互数据和物品数据的基本规模。"""
def summarize_data(interactions,items,) -> dict[str, int]:
    return {
        "num_interactions": int(len(interactions)),
        "num_users": int(interactions["user_id"].nunique()),
        "num_items_in_interactions": int(interactions["item_id"].nunique()),
        "num_items_in_metadata": int(len(items)),}

"""
将配置和统计信息转换为可以写入 JSON 的对象。
"""
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item)for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item)for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item)for item in value]
    return value

"""保存 DataFrame 为 CSV 文件。"""
def save_csv(dataframe,path: Path,) -> None:
    path.parent.mkdir(parents=True,exist_ok=True,) # 自动创建目标目录
    dataframe.to_csv(path,index=False,encoding="utf-8",)

def main(config_path: str | Path,) -> None:
    config = load_config(config_path)
    adapter = build_adapter(config)

    print("开始读取原始数据...")
    interactions, items = adapter.load()

    raw_stats = summarize_data(interactions,items,)
    print(f"原始交互：{raw_stats['num_interactions']}")
    print(f"原始用户：{raw_stats['num_users']}")
    print(f"原始物品：{raw_stats['num_items_in_metadata']}")

    # 第一步：标准化
    print("开始标准化数据...")
    normalized_interactions, normalized_items = (
        normalize_dataset(
            interactions,
            items,
            positive_rating_threshold=config.get("positive_rating_threshold",4,),
            rating_min=config.get("rating_min",1,),
            rating_max=config.get("rating_max",5,),
        )
    )
    normalized_stats = summarize_data(normalized_interactions,normalized_items,)

    # 第二步：过滤
    print("开始过滤低频用户和物品...")
    filtered_interactions, filtered_items, filter_stats = (
        filter_dataset(
            normalized_interactions,
            normalized_items,
            positive_only=config.get("positive_only",True,),
            min_user_interactions=config.get("min_user_interactions",5,),
            min_item_interactions=config.get("min_item_interactions",5,),
            item_count_mode=config.get("item_count_mode","unique_users",),
            max_rounds=config.get("max_filter_rounds",100,),
        )
    )

    # 第三步：时间切分
    print("开始进行时间切分...")
    train, valid, test, split_stats = (
        split_dataset(
            filtered_interactions,
            strategy=config.get("split_strategy","leave_last_two",),
            require_train_item_coverage=config.get("require_train_item_coverage",True,),
        )
    )

    # 输出路径处理与文件保存
    processed_dir = resolve_path(config["processed_data_dir"], PROJECT_ROOT)
    processed_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = resolve_path(config.get("interactions_path",processed_dir / "interactions.csv",),PROJECT_ROOT,)
    items_path = resolve_path(config.get("items_path",processed_dir / "items.csv",),PROJECT_ROOT,)
    train_path = processed_dir / "train.csv"
    valid_path = processed_dir / "valid.csv"
    test_path = processed_dir / "test.csv"
    stats_path = processed_dir / "stats.json"

    # 保存标准化后的完整数据
    save_csv(normalized_interactions,interactions_path,)
    save_csv(normalized_items,items_path,)
    # 保存切分后的数据
    save_csv(train,train_path,)
    save_csv(valid,valid_path,)
    save_csv(test,test_path,)

    all_stats = {
        "project_name": config.get("project_name","GenRec",),
        "dataset": config.get("dataset","unknown",),
        "raw": raw_stats,
        "normalized": normalized_stats,
        "filter": filter_stats,
        "split": split_stats,
        "outputs": {
            "interactions": interactions_path,
            "items": items_path,
            "train": train_path,
            "valid": valid_path,
            "test": test_path,
            "stats": stats_path,},}

    with stats_path.open("w",encoding="utf-8",) as file:
        json.dump(to_jsonable(all_stats),file,ensure_ascii=False,indent=2,)

    print()
    print("预处理完成")
    print(f"训练集：{train_path}")
    print(f"验证集：{valid_path}")
    print(f"测试集：{test_path}")
    print(f"统计信息：{stats_path}")
    print()
    print(json.dumps(to_jsonable(all_stats),ensure_ascii=False,indent=2,))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenRec 数据预处理入口")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="YAML 配置文件路径",)
    args = parser.parse_args()
    main(args.config)
