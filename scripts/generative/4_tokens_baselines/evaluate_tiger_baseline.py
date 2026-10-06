from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_ranking
from src.models.generative.tiger.recommender import TigerRecommender
from src.models.generative.tiger.trainer import set_seed
from src.representations.semantic_id.mapper import SemanticIDMapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the isolated TIGER_baseline recommender.")

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ml-100k-tiger-baseline.yaml",
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="TIGER checkpoint. Overrides tiger.checkpoint_path.",
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=None,
        help="Semantic ID directory. Overrides semantic_id.output_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Evaluation output directory. Overrides tiger.result_dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Evaluation device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Ranking cutoff. Overrides the root top_k setting.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="Beam width used by constrained decoding.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Logit temperature used during decoding.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Optional deterministic user limit for a smoke evaluation.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("YAML 根节点必须是字典")
    return config


def load_interactions(config: dict[str, Any]) -> tuple[pd.DataFrame, ...]:
    processed_dir = resolve_path(config.get("processed_data_dir", "data/processed/ml-100k"))
    paths = {split: processed_dir / f"{split}.csv" for split in ("train", "valid", "test")
}

    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("缺少预处理文件：\n" + "\n".join(missing_paths))

    dataframes = tuple(pd.read_csv(paths[split]) for split in ("train", "valid", "test"))
    required_columns = {"user_id", "item_id", "timestamp"}
    for split, dataframe in zip(("train", "valid", "test"), dataframes):
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(f"{split}.csv 缺少字段：{sorted(missing_columns)}")
        if dataframe.empty:
            raise ValueError(f"{split}.csv 为空")

    return dataframes

"""
构造正确答案
把目标交互表转换成：user_id → 这个用户真正应该被推荐的 item_id 列表
输入：
    user_id	    item_id	    timestamp
        1	    120	        300
        2	    300	        500
        3	    81	        700
输出：
    {
    "1": ["120"],
    "2": ["300"],
    "3": ["81"],
}
"""
def build_ground_truth(
    target_interactions: pd.DataFrame, # 表示目标交互表，真实目标。验证阶段：target_interactions = valid 测试阶段：target_interactions = test
    *,                                 # *之后的参数只能接受关键字传参，之前的可以位置传参
    max_users: int | None = None,
) -> dict[str, list[str]]:

    # 提取必要字段。从目标交互表中只保留user_id 和 item_id
    data = target_interactions.loc[:, ["user_id", "item_id"]].copy()
    # 统一 ID 格式。astype是DataFrame实例的自带方法，strip是原生方法
    data["user_id"] = data["user_id"].astype("string").str.strip()
    data["item_id"] = data["item_id"].astype("string").str.strip()
    # pandas库中DataFrame实例的自带方法。删除缺失值，如果user_id或item_id中某一行存在缺失，直接不要这一行。
    data = data.dropna(subset=["user_id", "item_id"])

    # 按用户分组并构造字典
    ground_truth = {
        str(user_id): group["item_id"].astype(str).tolist()         # 按用户分组放在一起
        for user_id, group in data.groupby("user_id", sort=False)
    }
    """
    列表推导式得到的结果
    {
    "1": ["120", "121"],
    "2": ["300"],
    "3": ["81", "90"],
    }
    """
    # 限制用户数量
    if max_users is not None:
        if not isinstance(max_users, int) or isinstance(max_users, bool):
            raise TypeError("max_users 必须是整数或 None")
        if max_users < 1:
            raise ValueError("max_users 必须大于 0")
        # 如果限制了用户数量，就进行截断，取[:max_users]
        ground_truth = dict(list(ground_truth.items())[:max_users])

    if not ground_truth:
        raise ValueError("测试目标中没有有效用户")
    return ground_truth

"""
评估一个数据集
给定一份历史数据和一份目标数据，让推荐器为每个目标用户生成推荐，然后统一计算指标。
"""
def evaluate_split(
    recommender: TigerRecommender,
    history_interactions: pd.DataFrame,     # 模型可以看到的过去行为
    target_interactions: pd.DataFrame,      # 用于评分的真实目标
    *,
    top_k: int,
    beam_width: int,
    temperature: float,
    max_users: int | None,
    split_name: str,
) -> tuple[dict[str, float | int], dict[str, list[str]], float]:
    # 构造正确答案，传入的是target_interactions
    ground_truth = build_ground_truth(target_interactions, max_users=max_users,)

    # 只建立一次用户历史索引，单用户推荐不再扫描整张 DataFrame。如：
    # "1": ("101", "205", "330"),
    # "10": ("72", "91"),
    recommender.prepare_history(history_interactions)

    # 初始化推荐结果字典
    recommendations: dict[str, list[str]] = {}
    start_time = time.perf_counter()        # 测量运行时间的计时器
    total_users = len(ground_truth)

    # 遍历所有目标用户
    for index, user_id in enumerate(ground_truth, start=1):
        recommendations[user_id] = recommender.recommend_for_user(
            user_id,
            k=top_k,
            beam_width=beam_width,
            temperature=temperature,
        )

        # 输出进度，如test: 100/938 users evaluated (31.0s)
        if index == 1 or index % 100 == 0 or index == total_users:
            elapsed = time.perf_counter() - start_time
            print(f"{split_name}: {index}/{total_users} users evaluated "f"({elapsed:.1f}s)")

    metrics = evaluate_ranking(
        recommendations=recommendations,
        ground_truth=ground_truth,
        k=top_k,
    )
    # 重新计算总耗时
    elapsed = time.perf_counter() - start_time

    return metrics, recommendations, elapsed


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> int:
    # 读取参数
    args = parse_args()
    # 读取配置
    config = load_config(args.config)
    # 设置随机种子
    set_seed(int(config.get("seed", 42)))
    # 读取交互表，从 YAML 中取出
    train, valid, test = load_interactions(config)

    # 读取 Semantic ID 配置区块
    semantic_config = config.get("semantic_id", {})
    if not isinstance(semantic_config, dict):
        raise ValueError("semantic_id 必须是 YAML 字典")
    semantic_dir = resolve_path(
        args.semantic_dir if args.semantic_dir is not None
        else semantic_config.get("output_dir","data/processed/ml-100k/semantic_id/advanced",))
    # 读取 TIGER 配置区块
    tiger_config = config.get("tiger", {})
    if not isinstance(tiger_config, dict):
        raise ValueError("tiger 必须是 YAML 字典")

    checkpoint_path = resolve_path(
        args.checkpoint_path if args.checkpoint_path is not None
        else tiger_config.get("checkpoint_path","checkpoints/ml-100k/tiger_best.pt",))

    output_dir = resolve_path(
        args.output_dir if args.output_dir is not None
        else tiger_config.get("result_dir", "results/ml-100k/tiger"))

    top_k = int(args.top_k if args.top_k is not None else config.get("top_k", 10))
    beam_width = int(
        args.beam_width if args.beam_width is not None
        else tiger_config.get("beam_width", 32))
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    if beam_width < 1:
        raise ValueError("beam_width 必须大于 0")
    if args.temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    mapper = SemanticIDMapper.from_directory(semantic_dir)
    recommender = TigerRecommender.from_checkpoint(
        checkpoint_path,
        mapper,
        device=args.device or config.get("device"),
        beam_width=beam_width,
    )

    print("=" * 60)
    print("TIGER ranking evaluation")
    print("=" * 60)
    print(f"semantic_dir: {semantic_dir}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"device: {recommender.device}")
    print(f"top_k: {top_k}")
    print(f"beam_width: {beam_width}")
    print(f"temperature: {args.temperature}")

    validation_metrics, validation_recommendations, validation_seconds = evaluate_split(
        recommender,
        train,
        valid,
        top_k=top_k,
        beam_width=beam_width,
        temperature=args.temperature,
        max_users=args.max_users,
        split_name="valid",
    )

    test_history = pd.concat([train, valid], ignore_index=True)
    test_metrics, test_recommendations, test_seconds = evaluate_split(
        recommender,
        test_history,
        test,
        top_k=top_k,
        beam_width=beam_width,
        temperature=args.temperature,
        max_users=args.max_users,
        split_name="test",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "validation_recommendations.json", validation_recommendations)
    save_json(output_dir / "test_recommendations.json", test_recommendations)

    result = {
        "model": "TIGER_baseline",
        "dataset": config.get("dataset", "unknown"),
        "device": str(recommender.device),
        "top_k": top_k,
        "beam_width": beam_width,
        "temperature": args.temperature,
        "semantic_dir": str(semantic_dir),
        "checkpoint_path": str(checkpoint_path),
        "data": {
            "train_rows": len(train),
            "validation_rows": len(valid),
            "test_rows": len(test),
            "validation_history_rows": len(train),
            "test_history_rows": len(test_history),
            "validation_users_evaluated": validation_metrics["num_users"],
            "test_users_evaluated": test_metrics["num_users"],
            "max_users": args.max_users,
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "runtime_seconds": {
            "validation": validation_seconds,
            "test": test_seconds,
        },
    }
    save_json(output_dir / "evaluation.json", result)

    snapshot = copy.deepcopy(config)
    snapshot["tiger_evaluation"] = {
        "semantic_dir": str(semantic_dir),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir),
        "top_k": top_k,
        "beam_width": beam_width,
        "temperature": args.temperature,
        "device": str(recommender.device),
        "max_users": args.max_users,
    }
    with (output_dir / "evaluation_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, allow_unicode=True, sort_keys=False)

    print("TIGER evaluation completed")
    print(f"results_dir: {output_dir}")
    print(f"validation: {validation_metrics}")
    print(f"test: {test_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
