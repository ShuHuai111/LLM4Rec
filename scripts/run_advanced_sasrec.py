from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.advanced_sasrec import (
    SASRec,
    SASRecRecommender,
    SASRecTrainer,
    set_seed,
)
from src.data.sequence_dataset import SASRecSequenceDataset
from src.evaluation.metrics import evaluate_recommender


VARIANTS = (
    {
        "name": "relu_learnable",
        "label": "ReLU + Learnable Position",
        "ffn_type": "relu",
        "position_encoding": "learnable",
    },
    {
        "name": "swiglu_learnable",
        "label": "SwiGLU + Learnable Position",
        "ffn_type": "swiglu",
        "position_encoding": "learnable",
    },
    {
        "name": "relu_alibi",
        "label": "ReLU + ALiBi",
        "ffn_type": "relu",
        "position_encoding": "alibi",
    },
    {
        "name": "swiglu_alibi",
        "label": "SwiGLU + ALiBi",
        "ffn_type": "swiglu",
        "position_encoding": "alibi",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four Advanced SASRec ablations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device in YAML.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

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
            "缺少预处理文件：\n" + "\n".join(missing_paths)
        )

    train, valid, test = (
        pd.read_csv(paths[name])
        for name in ("train", "valid", "test")
    )

    required_columns = {"user_id", "item_id", "timestamp"}
    for name, dataframe in {
        "train": train,
        "valid": valid,
        "test": test,
    }.items():
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"{name}.csv 缺少字段：{sorted(missing_columns)}"
            )
        if dataframe.empty:
            raise ValueError(f"{name}.csv 为空")

    return train, valid, test


def _sorted_user_items(
    dataframe: pd.DataFrame,
) -> dict[str, list[str]]:
    data = dataframe.copy()
    data["user_id"] = data["user_id"].astype("string").str.strip()
    data["item_id"] = data["item_id"].astype("string").str.strip()
    data["timestamp"] = pd.to_numeric(
        data["timestamp"],
        errors="raise",
    )
    data = data.sort_values(
        by=["user_id", "timestamp"],
        kind="stable",
    )

    return {
        str(user_id): group["item_id"].astype("string").tolist()
        for user_id, group in data.groupby(
            "user_id",
            sort=False,
        )
    }


def build_next_item_dataset(
    history_interactions: pd.DataFrame,
    target_interactions: pd.DataFrame,
    *,
    item_to_index: dict[str, int],
    max_seq_len: int,
) -> TensorDataset:
    history_by_user = _sorted_user_items(history_interactions)

    targets = target_interactions.copy()
    targets["user_id"] = targets["user_id"].astype("string").str.strip()
    targets["item_id"] = targets["item_id"].astype("string").str.strip()
    targets["timestamp"] = pd.to_numeric(
        targets["timestamp"],
        errors="raise",
    )
    targets = targets.sort_values(
        by=["user_id", "timestamp"],
        kind="stable",
    )

    input_sequences: list[list[int]] = []
    target_items: list[int] = []

    for user_id, group in targets.groupby(
        "user_id",
        sort=False,
    ):
        history = list(
            history_by_user.get(str(user_id), [])
        )

        for row in group.itertuples(index=False):
            target_item = str(row.item_id)
            target_index = item_to_index.get(target_item)

            if target_index is None:
                continue

            history_indices = [
                item_to_index[item_id]
                for item_id in history
                if item_id in item_to_index
            ]

            if not history_indices:
                continue

            prefix = history_indices[-max_seq_len:]
            padded_prefix = [
                0
            ] * (max_seq_len - len(prefix)) + prefix

            input_sequences.append(padded_prefix)
            target_items.append(target_index)
            history.append(target_item)

    if not input_sequences:
        raise ValueError("无法构造一步预测数据集")

    return TensorDataset(
        torch.tensor(input_sequences, dtype=torch.long),
        torch.tensor(target_items, dtype=torch.long),
    )


def write_variant_results(
    result_dir: Path,
    result: dict[str, Any],
    history: list[dict[str, float]],
    config_snapshot: dict[str, Any],
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (result_dir / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (result_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(
            config_snapshot,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def write_comparison_html(
    output_path: Path,
    results: list[dict[str, Any]],
) -> None:
    def make_rows(split: str) -> str:
        rows = []
        for result in results:
            metrics = result[split]
            rows.append(
                "<tr>"
                f"<td>{html.escape(result['label'])}</td>"
                f"<td>{result['best_epoch']}</td>"
                f"<td>{metrics['Recall@10']:.6f}</td>"
                f"<td>{metrics['NDCG@10']:.6f}</td>"
                f"<td>{metrics['MRR@10']:.6f}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Advanced SASRec 消融实验</title>
  <style>
    body {{ margin: 0; padding: 32px; background: #f5f7fb; color: #1f2937; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }}
    main {{ max-width: 1120px; margin: auto; }}
    section {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 4px 14px rgba(15,23,42,.05); }}
    h1 {{ margin: 0 0 8px; }}
    h2 {{ margin: 0 0 16px; font-size: 20px; }}
    p {{ color: #57606a; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    th {{ background: #f9fafb; color: #57606a; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
  </style>
</head>
<body>
<main>
  <section>
    <h1>Advanced SASRec 消融实验</h1>
    <p>比较 SwiGLU 和 ALiBi 对 SASRec 的独立影响与组合效果。</p>
    <p>指标：Recall@10、NDCG@10、MRR@10；验证用户数和测试用户数均为 938。</p>
  </section>
  <section>
    <h2>验证集</h2>
    <table>
      <thead><tr><th>模型</th><th>最佳 epoch</th><th>Recall@10</th><th>NDCG@10</th><th>MRR@10</th></tr></thead>
      <tbody>{make_rows("validation")}</tbody>
    </table>
  </section>
  <section>
    <h2>测试集</h2>
    <table>
      <thead><tr><th>模型</th><th>最佳 epoch</th><th>Recall@10</th><th>NDCG@10</th><th>MRR@10</th></tr></thead>
      <tbody>{make_rows("test")}</tbody>
    </table>
  </section>
</main>
</body>
</html>
"""

    output_path.write_text(html_text, encoding="utf-8")


def run_variant(
    variant: dict[str, str],
    *,
    config: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    model_config: dict[str, Any],
    result_root: Path,
    checkpoint_root: Path,
    requested_device: str | None,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)

    max_seq_len = int(config.get("max_seq_len", 50))
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 0))
    learning_rate = float(config.get("learning_rate", 1e-3))

    train_dataset = SASRecSequenceDataset(
        train,
        max_seq_len=max_seq_len,
    )

    valid_dataset = build_next_item_dataset(
        train,
        valid,
        item_to_index=train_dataset.item_to_index_,
        max_seq_len=max_seq_len,
    )

    pin_memory = str(requested_device).startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    embedding_dim = int(model_config.get("embedding_dim", 64))
    num_heads = int(model_config.get("num_heads", 2))
    num_layers = int(model_config.get("num_layers", 2))
    feed_forward_dim = int(
        model_config.get("feed_forward_dim", 128)
    )
    dropout = float(model_config.get("dropout", 0.2))
    weight_decay = float(model_config.get("weight_decay", 1e-5))
    grad_clip_norm = float(
        model_config.get("grad_clip_norm", 5.0)
    )
    patience = int(model_config.get("patience", 3))

    model = SASRec(
        num_items=train_dataset.num_items,
        max_seq_len=max_seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feed_forward_dim=feed_forward_dim,
        dropout=dropout,
        ffn_type=variant["ffn_type"],
        position_encoding=variant["position_encoding"],
    )

    result_dir = result_root / variant["name"]
    checkpoint_path = checkpoint_root / f"{variant['name']}_best.pt"

    trainer = SASRecTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=requested_device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        grad_clip_norm=grad_clip_norm,
        patience=patience,
        checkpoint_path=checkpoint_path,
    )

    print(f"开始训练 {variant['label']}...")
    history = trainer.fit()

    recommender = SASRecRecommender(
        model=model,
        item_to_index=train_dataset.item_to_index_,
        index_to_item=train_dataset.index_to_item_,
        max_seq_len=max_seq_len,
        device=trainer.device,
    )

    validation_metrics = evaluate_recommender(
        model=recommender,
        history_interactions=train,
        target_interactions=valid,
        k=int(config.get("top_k", 10)),
    )

    test_history = pd.concat(
        [train, valid],
        ignore_index=True,
    )
    test_metrics = evaluate_recommender(
        model=recommender,
        history_interactions=test_history,
        target_interactions=test,
        k=int(config.get("top_k", 10)),
    )

    result = {
        "model": "advanced_sasrec",
        "variant": variant["name"],
        "label": variant["label"],
        "dataset": config.get("dataset", "unknown"),
        "device": str(trainer.device),
        "top_k": int(config.get("top_k", 10)),
        "best_epoch": trainer.best_epoch,
        "best_validation_loss": trainer.best_metric,
        "model_config": {
            "max_seq_len": max_seq_len,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "feed_forward_dim": feed_forward_dim,
            "dropout": dropout,
            "ffn_type": variant["ffn_type"],
            "position_encoding": variant["position_encoding"],
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "grad_clip_norm": grad_clip_norm,
            "patience": patience,
        },
        "data": {
            "train_rows": len(train),
            "validation_rows": len(valid),
            "test_rows": len(test),
            "train_samples": len(train_dataset),
            "validation_samples": len(valid_dataset),
            "train_users": int(train["user_id"].nunique()),
            "train_items": train_dataset.num_items,
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }

    write_variant_results(
        result_dir=result_dir,
        result=result,
        history=history,
        config_snapshot={
            "variant": variant,
            "model_config": result["model_config"],
        },
    )

    print(f"{variant['label']} 完成")
    print(f"validation: {validation_metrics}")
    print(f"test: {test_metrics}")

    del model, trainer, recommender
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path.resolve())

    seed = int(config.get("seed", 42))
    set_seed(seed)
    train, valid, test = load_data(config)

    advanced_config = config.get(
        "advanced_sasrec",
        config.get("sasrec", {}),
    )
    if not isinstance(advanced_config, dict):
        raise ValueError("advanced_sasrec 配置必须是 YAML 字典")

    epochs = int(
        args.epochs
        if args.epochs is not None
        else config.get("epochs", 10)
    )
    requested_device = (
        args.device
        if args.device is not None
        else config.get("device", None)
    )

    result_root = resolve_path(
        advanced_config.get(
            "result_dir",
            "results/ml-100k/advanced-sasrec",
        )
    )
    checkpoint_root = resolve_path(
        advanced_config.get(
            "checkpoint_dir",
            "checkpoints/ml-100k/advanced-sasrec",
        )
    )
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    results = [
        run_variant(
            variant,
            config=config,
            train=train,
            valid=valid,
            test=test,
            model_config=advanced_config,
            result_root=result_root,
            checkpoint_root=checkpoint_root,
            requested_device=requested_device,
            epochs=epochs,
            seed=seed,
        )
        for variant in VARIANTS
    ]

    (result_root / "comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_comparison_html(
        result_root / "comparison.html",
        results,
    )

    print("Advanced SASRec 四组对照完成")
    print(f"results_dir: {result_root}")


if __name__ == "__main__":
    main()
