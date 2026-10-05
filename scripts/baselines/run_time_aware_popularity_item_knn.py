from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baselines.time_aware_popularity_item_knn import (
    TimeAwarePopularityRegularizedItemKNNRecommender,
)
from src.evaluation.metrics import evaluate_recommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the time-aware and popularity-regularized "
            "Item-KNN model."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("配置文件必须解析为 YAML 字典")

    return config


def load_data(config: dict[str, Any]) -> tuple[pd.DataFrame, ...]:
    processed_dir = resolve_path(
        config.get(
            "processed_data_dir",
            "data/processed/ml-100k",
        )
    )

    paths = {
        "train": processed_dir / "train.csv",
        "valid": processed_dir / "valid.csv",
        "test": processed_dir / "test.csv",
    }

    missing_paths = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "缺少预处理文件：\n"
            + "\n".join(missing_paths)
        )

    return tuple(
        pd.read_csv(paths[name])
        for name in ("train", "valid", "test")
    )


def write_html(
    output_path: Path,
    result: dict[str, Any],
) -> None:
    validation = result["validation"]
    test = result["test"]

    metric_names = [
        key
        for key in validation
        if key != "num_users"
    ]

    rows = []
    for metric_name in metric_names:
        rows.append(
            "<tr>"
            f"<td>{metric_name}</td>"
            f"<td>{validation[metric_name]:.6f}</td>"
            f"<td>{test[metric_name]:.6f}</td>"
            "</tr>"
        )

    model_config = result["model_config"]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Time-aware Popularity-regularized Item-KNN Results</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; min-width: 700px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 12px; }}
    th {{ background: #f6f8fa; }}
    code {{ background: #f6f8fa; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Time-aware Popularity-regularized Item-KNN</h1>
  <p>Dataset: {result["dataset"]}</p>
  <p>Top-K: {result["top_k"]}</p>
  <p>Model configuration:</p>
  <pre>{json.dumps(model_config, ensure_ascii=False, indent=2)}</pre>
  <table>
    <thead>
      <tr><th>Metric</th><th>Validation</th><th>Test</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    config = load_config(config_path)
    train, valid, test = load_data(config)

    model_config = config.get(
        "time_aware_item_knn",
        {},
    )
    if not isinstance(model_config, dict):
        raise ValueError(
            "time_aware_item_knn 配置必须是 YAML 字典"
        )

    top_k = int(config.get("top_k", 10))

    model = TimeAwarePopularityRegularizedItemKNNRecommender(
        neighbor_k=int(
            model_config.get("neighbor_k", 50)
        ),
        time_decay_enabled=bool(
            model_config.get(
                "time_decay_enabled",
                False,
            )
        ),
        time_half_life_days=float(
            model_config.get(
                "time_half_life_days",
                90,
            )
        ),
        popularity_penalty_enabled=bool(
            model_config.get(
                "popularity_penalty_enabled",
                False,
            )
        ),
        popularity_penalty_alpha=float(
            model_config.get(
                "popularity_penalty_alpha",
                0.25,
            )
        ),
        popularity_count_mode=str(
            model_config.get(
                "popularity_count_mode",
                "unique_users",
            )
        ),
    )

    model.fit(train)

    validation_metrics = evaluate_recommender(
        model=model,
        history_interactions=train,
        target_interactions=valid,
        k=top_k,
    )

    test_history = pd.concat(
        [train, valid],
        ignore_index=True,
    )

    test_metrics = evaluate_recommender(
        model=model,
        history_interactions=test_history,
        target_interactions=test,
        k=top_k,
    )

    resolved_model_config = {
        "neighbor_k": model.neighbor_k,
        "time_decay_enabled": model.time_decay_enabled,
        "time_half_life_days": model.time_half_life_days,
        "popularity_penalty_enabled": (
            model.popularity_penalty_enabled
        ),
        "popularity_penalty_alpha": (
            model.popularity_penalty_alpha
        ),
        "popularity_count_mode": (
            model.popularity_count_mode
        ),
    }

    result = {
        "model": (
            "time_aware_popularity_regularized_item_knn"
        ),
        "dataset": config.get("dataset", "unknown"),
        "top_k": top_k,
        "model_config": resolved_model_config,
        "data": {
            "train_rows": len(train),
            "validation_rows": len(valid),
            "test_rows": len(test),
            "train_users": train["user_id"].nunique(),
            "train_items": train["item_id"].nunique(),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }

    result_dir = resolve_path(
        model_config.get(
            "result_dir",
            "results/ml-100k/time-aware-popularity-item-knn",
        )
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "results.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (result_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset": config.get("dataset"),
                "top_k": top_k,
                "time_aware_item_knn": resolved_model_config,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    write_html(result_dir / "results.html", result)

    print(
        "Time-aware Popularity-regularized Item-KNN "
        "completed"
    )
    print(f"results_dir: {result_dir}")
    print(f"validation: {validation_metrics}")
    print(f"test: {test_metrics}")


if __name__ == "__main__":
    main()
