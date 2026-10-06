from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.representations.semantic_id.advanced_quantizer import (
    ResidualVectorQuantizer as AdvancedQuantizer,
)
from src.representations.semantic_id.quantizer import (
    ResidualVectorQuantizer as BasicQuantizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate isolated TIGER Semantic IDs.")
    parser.add_argument("--semantic-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "advanced"), required=True)
    return parser.parse_args()


def collision_stats(codes: np.ndarray) -> dict[str, int | float]:
    ids = ["-".join(str(int(value)) for value in row) for row in codes]
    total = len(ids)
    unique = len(set(ids))
    return {
        "total_items": total,
        "unique_ids": unique,
        "collision_items": total - unique,
        "collision_rate": (total - unique) / total if total else 0.0,
    }


def main() -> int:
    args = parse_args()
    semantic_dir = args.semantic_dir.resolve()
    mapping_path = semantic_dir / "item_semantic_ids.csv"
    snapshot_path = semantic_dir / "config_snapshot.yaml"
    codebooks_path = semantic_dir / "codebooks.npz"
    embeddings_path = semantic_dir / "item_embeddings.npy"

    for path in (mapping_path, snapshot_path, codebooks_path, embeddings_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少文件：{path}")

    with snapshot_path.open("r", encoding="utf-8-sig") as handle:
        snapshot = yaml.safe_load(handle)
    if not isinstance(snapshot, dict):
        raise ValueError("config_snapshot.yaml 必须是字典")

    quantizer_cls = BasicQuantizer if args.mode == "baseline" else AdvancedQuantizer
    quantizer = quantizer_cls.load(codebooks_path)
    mapping = pd.read_csv(mapping_path)
    if mapping.empty:
        raise ValueError("映射表为空")
    if mapping["item_id"].astype("string").duplicated().any():
        raise ValueError("item_id 存在重复")

    code_columns = [f"code_{index}" for index in range(quantizer.num_codebooks)]
    missing = [column for column in code_columns if column not in mapping.columns]
    if missing:
        raise ValueError(f"缺少量化 code 列：{missing}")

    expected_length = 4 if args.mode == "baseline" else quantizer.num_codebooks
    all_code_columns = sorted(
        [column for column in mapping.columns if column.startswith("code_")],
        key=lambda value: int(value.split("_")[1]),
    )
    if len(all_code_columns) != expected_length:
        raise ValueError(
            f"Semantic code 列数量错误：实际 {len(all_code_columns)}，期望 {expected_length}"
        )

    codes = mapping[all_code_columns].to_numpy(dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= quantizer.codebook_size):
        raise ValueError("存在超出 codebook_size 范围的 code")

    expected_ids = ["-".join(str(int(value)) for value in row) for row in codes]
    actual_ids = mapping["semantic_id"].astype(str).tolist()
    if actual_ids != expected_ids:
        raise ValueError("semantic_id 与 code 列拼接结果不一致")

    prefix_codes = codes[:, : quantizer.num_codebooks]
    prefix_stats = collision_stats(prefix_codes)
    final_stats = collision_stats(codes)
    if final_stats["collision_items"] != 0:
        raise ValueError("完整 Semantic ID 仍然存在冲突")

    embeddings = np.load(embeddings_path)
    reconstruction_mse = quantizer.reconstruction_error(embeddings, prefix_codes)
    result: dict[str, Any] = {
        "semantic_dir": str(semantic_dir),
        "mode": args.mode,
        "quantizer_num_codebooks": quantizer.num_codebooks,
        "semantic_code_length": len(all_code_columns),
        "prefix": {**prefix_stats, "reconstruction_mse": reconstruction_mse},
        "full": final_stats,
        "passed": True,
    }
    (semantic_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
