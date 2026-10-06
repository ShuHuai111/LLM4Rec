from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


EXPERIMENTS = [
    {
        "id": "E1",
        "name": "100K 同数据集 TIGER_baseline",
        "dataset": "ml-100k",
        "mode": "baseline_paper_4code",
        "semantic_dir": "data/processed/ml-100k/semantic_id/TIGER_baseline",
        "sequence_dir": "data/processed/ml-100k/semantic_sequences/TIGER_baseline",
        "result_dir": "results/ml-100k/tiger/TIGER_baseline",
    },
    {
        "id": "E2",
        "name": "100K 同数据集 Advanced TIGER",
        "dataset": "ml-100k",
        "mode": "advanced_3code",
        "semantic_dir": "data/processed/ml-100k/semantic_id/advanced",
        "sequence_dir": "data/processed/ml-100k/semantic_sequences/advanced",
        "result_dir": "results/ml-100k/tiger/advanced_3code",
    },
    {
        "id": "E3",
        "name": "1M 同数据集 TIGER_baseline",
        "dataset": "ml-1m",
        "mode": "baseline_paper_4code",
        "semantic_dir": "data/processed/ml-1m/semantic_id/TIGER_baseline",
        "sequence_dir": "data/processed/ml-1m/semantic_sequences/TIGER_baseline",
        "result_dir": "results/ml-1m/tiger/TIGER_baseline",
    },
    {
        "id": "E4",
        "name": "1M 同数据集 Advanced TIGER",
        "dataset": "ml-1m",
        "mode": "advanced_3code",
        "semantic_dir": "data/processed/ml-1m/semantic_id/advanced",
        "sequence_dir": "data/processed/ml-1m/semantic_sequences/advanced",
        "result_dir": "results/ml-1m/tiger/advanced_3code",
    },
    {
        "id": "E5",
        "name": "1M 编码规则迁移到 100K TIGER_baseline",
        "dataset": "ml-100k",
        "mode": "cross_dataset_baseline_paper_4code",
        "semantic_dir": "data/processed/ml-100k/semantic_id/TIGER_cross_from_ml-1m_baseline",
        "sequence_dir": "data/processed/ml-100k/semantic_sequences/TIGER_cross_from_ml-1m_baseline",
        "result_dir": "results/ml-100k/tiger/TIGER_cross_from_ml-1m_baseline",
    },
    {
        "id": "E6",
        "name": "1M 编码规则迁移到 100K Advanced TIGER",
        "dataset": "ml-100k",
        "mode": "cross_dataset_advanced_3code",
        "semantic_dir": "data/processed/ml-100k/semantic_id/TIGER_cross_from_ml-1m_advanced",
        "sequence_dir": "data/processed/ml-100k/semantic_sequences/TIGER_cross_from_ml-1m_advanced",
        "result_dir": "results/ml-100k/tiger/TIGER_cross_from_ml-1m_advanced",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点不是对象：{path}")
    return value


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def collect_experiment(spec: dict[str, str]) -> dict[str, Any]:
    semantic_dir = PROJECT_ROOT / spec["semantic_dir"]
    result_dir = PROJECT_ROOT / spec["result_dir"]
    evaluation = load_json(result_dir / "evaluation.json")
    training = load_json(result_dir / "results.json")
    stats = load_json(semantic_dir / "assignment_stats.json")

    recommendation_stats: dict[str, Any] = {}
    for split in ("validation", "test"):
        recommendation_path = result_dir / f"{split}_recommendations.json"
        recommendations = load_json(recommendation_path)
        lengths = [len(items) for items in recommendations.values()]
        recommendation_stats[f"{split}_avg_recommendations"] = (
            sum(lengths) / len(lengths) if lengths else 0.0
        )
        recommendation_stats[f"{split}_users_below_top_k"] = sum(
            length < 10 for length in lengths
        )

    prefix = stats.get("prefix_quantization", stats.get("ordinary_quantization", {}))
    final = stats.get("final_semantic_ids", stats.get("final_assignment", {}))
    return {
        **spec,
        "semantic_code_length": stats.get("semantic_code_length", 3),
        "prefix_collision_rate": prefix.get("collision_rate"),
        "full_collision_rate": final.get("collision_rate"),
        "reconstruction_mse": final.get("reconstruction_mse", prefix.get("reconstruction_mse")),
        "fit_source": stats.get("fit_source", "target_dataset"),
        "train_samples": training.get("train_samples"),
        "valid_samples": training.get("valid_samples"),
        "best_epoch": training.get("best_epoch"),
        "best_valid_loss": training.get("best_valid_loss"),
        "beam_width": evaluation.get("beam_width"),
        "top_k": evaluation.get("top_k"),
        "valid_users": evaluation.get("validation", {}).get("num_users"),
        "test_users": evaluation.get("test", {}).get("num_users"),
        "valid_recall": evaluation.get("validation", {}).get("Recall@10"),
        "valid_ndcg": evaluation.get("validation", {}).get("NDCG@10"),
        "valid_mrr": evaluation.get("validation", {}).get("MRR@10"),
        "test_recall": evaluation.get("test", {}).get("Recall@10"),
        "test_ndcg": evaluation.get("test", {}).get("NDCG@10"),
        "test_mrr": evaluation.get("test", {}).get("MRR@10"),
        "valid_seconds": evaluation.get("runtime_seconds", {}).get("validation"),
        "test_seconds": evaluation.get("runtime_seconds", {}).get("test"),
        **recommendation_stats,
        "semantic_dir": relative(semantic_dir),
        "sequence_dir": relative(PROJECT_ROOT / spec["sequence_dir"]),
        "result_dir": relative(result_dir),
    }


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_html(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("实验", "name"),
        ("数据集", "dataset"),
        ("Semantic ID", "mode"),
        ("长度", "semantic_code_length"),
        ("前三码碰撞率", "prefix_collision_rate"),
        ("完整 ID 碰撞率", "full_collision_rate"),
        ("Valid Recall@10", "valid_recall"),
        ("Valid NDCG@10", "valid_ndcg"),
        ("Valid MRR@10", "valid_mrr"),
        ("Test Recall@10", "test_recall"),
        ("Test NDCG@10", "test_ndcg"),
        ("Test MRR@10", "test_mrr"),
        ("Test 平均推荐数", "test_avg_recommendations"),
        ("Test 少于 10 条用户", "test_users_below_top_k"),
        ("最佳 epoch", "best_epoch"),
        ("Beam width", "beam_width"),
    ]
    header = "".join(f"<th>{html.escape(label)}</th>" for label, _ in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(format_value(row[key]))}</td>"
            for _, key in columns
        )
        body.append(f"<tr><td>{html.escape(row['id'])}</td>{cells}</tr>")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>TIGER 六组实验结果</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
h1 {{ margin-bottom: 8px; }}
p {{ color: #4b5563; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: right; white-space: nowrap; }}
th {{ background: #eef2ff; text-align: center; }}
td:nth-child(2), td:nth-child(3), td:nth-child(4) {{ text-align: left; }}
tr:nth-child(even) {{ background: #f9fafb; }}
.note {{ margin-top: 18px; font-size: 13px; }}
</style>
</head>
<body>
<h1>TIGER 六组实验结果</h1>
<p>完整 valid/test 用户评估；训练 3 epoch；batch size 32；beam width 16；原始三码 baseline 未纳入本表。</p>
<table><thead><tr><th>ID</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>
<p class="note">前三码碰撞率用于描述量化前缀；完整 ID 碰撞率用于描述最终可解码 Semantic ID。</p>
</body>
</html>
"""


def main() -> None:
    rows = [collect_experiment(spec) for spec in EXPERIMENTS]
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tiger_six_experiments.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "tiger_six_experiments.html").write_text(
        build_html(rows), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
