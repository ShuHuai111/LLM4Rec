from __future__ import annotations

from pathlib import Path
from typing import Any

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_id.advanced_quantizer import (
    ResidualVectorQuantizer as AdvancedQuantizer,
)
from src.semantic_id.item_encoder import ItemTextEncoder
from src.semantic_id.quantizer import (
    ResidualVectorQuantizer as BasicQuantizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated item Semantic IDs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "advanced"),
        default=None,
        help="Semantic ID mode. Defaults to semantic_id.quantizer_mode.",
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=None,
        help="Semantic ID directory. Overrides the configured output directory.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    with config_path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("配置文件必须解析为 YAML 字典")
    return config


def normalize_ids(dataframe: pd.DataFrame, dataframe_name: str) -> pd.DataFrame:
    if "item_id" not in dataframe.columns:
        raise ValueError(f"{dataframe_name} 缺少 item_id 字段")

    data = dataframe.copy()
    data["item_id"] = data["item_id"].astype("string").str.strip()
    if data["item_id"].isna().any():
        raise ValueError(f"{dataframe_name}.item_id 不能包含缺失值")
    return data


def parse_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    result = values.map({"true": True, "false": False})
    if result.isna().any():
        raise ValueError(f"{column_name} 必须只包含 true 或 false")
    return result.astype(bool)


def get_output_dir(
    *,
    config: dict[str, Any],
    mode: str,
    semantic_dir: Path | None,
) -> Path:
    if semantic_dir is not None:
        return resolve_path(semantic_dir)

    semantic_config = config.get("semantic_id", {})
    configured_mode = semantic_config.get("quantizer_mode", "advanced")
    configured_output = semantic_config.get("output_dir")

    if mode == configured_mode and configured_output is not None:
        return resolve_path(configured_output)

    dataset = str(config.get("dataset", "unknown"))
    return PROJECT_ROOT / "data" / "processed" / dataset / "semantic_id" / mode


def get_quantizer(mode: str, codebooks_path: Path) -> Any:
    quantizer_class = (
        AdvancedQuantizer if mode == "advanced" else BasicQuantizer
    )
    return quantizer_class.load(codebooks_path)


def validate(mode: str, output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    required_files = {
        "mapping": output_dir / "item_semantic_ids.csv",
        "embeddings": output_dir / "item_embeddings.npy",
        "encoder": output_dir / "item_encoder.pkl",
        "codebooks": output_dir / "codebooks.npz",
        "stats": output_dir / "assignment_stats.json",
        "snapshot": output_dir / "config_snapshot.yaml",
    }

    missing = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Semantic ID 产物缺少文件：\n" + "\n".join(missing)
        )

    processed_dir = resolve_path(
        config.get("processed_data_dir", "data/processed/ml-100k")
    )
    items_path = resolve_path(
        config.get("items_path", processed_dir / "items.csv")
    )
    train_path = processed_dir / "train.csv"

    items = normalize_ids(pd.read_csv(items_path), "items")
    train = normalize_ids(pd.read_csv(train_path), "train")
    mapping = pd.read_csv(required_files["mapping"])

    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(not items["item_id"].duplicated().any(), "items.csv 的 item_id 存在重复")

    mapping = normalize_ids(mapping, "item_semantic_ids")
    check(
        not mapping["item_id"].duplicated().any(),
        "item_semantic_ids.csv 的 item_id 存在重复",
    )
    check(
        set(mapping["item_id"].astype(str)) == set(items["item_id"].astype(str)),
        "Semantic ID 映射没有覆盖完全相同的物品集合",
    )
    check(
        len(mapping) == len(items),
        f"映射行数与物品数不一致：{len(mapping)} != {len(items)}",
    )

    code_columns_with_numbers: list[tuple[int, str]] = []
    for column in mapping.columns:
        match = re.fullmatch(r"code_(\d+)", column)
        if match:
            code_columns_with_numbers.append((int(match.group(1)), column))

    code_columns_with_numbers.sort()
    code_indices = [index for index, _ in code_columns_with_numbers]
    expected_indices = list(range(len(code_indices)))
    check(
        code_indices == expected_indices,
        f"Semantic ID code 列不连续：{code_indices}",
    )
    check(bool(code_columns_with_numbers), "缺少 code_0、code_1 等编码列")

    code_columns = [column for _, column in code_columns_with_numbers]
    codes = mapping[code_columns].to_numpy()
    check(not pd.isna(codes).any(), "编码列包含缺失值")

    try:
        numeric_codes = codes.astype(np.int64)
        check(
            np.array_equal(codes, numeric_codes),
            "编码列包含非整数值",
        )
    except (TypeError, ValueError):
        numeric_codes = np.zeros((len(mapping), len(code_columns)), dtype=np.int64)
        errors.append("编码列无法转换为整数")

    semantic_ids = mapping.get("semantic_id")
    check(semantic_ids is not None, "缺少 semantic_id 列")

    if semantic_ids is not None and len(code_columns) > 0:
        expected_semantic_ids = [
            "-".join(str(int(code)) for code in row)
            for row in numeric_codes
        ]
        check(
            semantic_ids.astype(str).tolist() == expected_semantic_ids,
            "semantic_id 列与各层 code 拼接结果不一致",
        )

    quantizer = get_quantizer(mode, required_files["codebooks"])
    check(quantizer.is_fitted, "保存的量化器未处于 fitted 状态")
    check(
        numeric_codes.shape[1] == quantizer.num_codebooks,
        "编码列数量与量化器 num_codebooks 不一致",
    )
    check(
        bool((numeric_codes >= 0).all())
        and bool((numeric_codes < quantizer.codebook_size).all()),
        "存在超出码本范围的 code",
    )

    embeddings = np.load(required_files["embeddings"])
    check(
        embeddings.ndim == 2,
        f"物品向量必须是二维数组，实际形状为 {embeddings.shape}",
    )
    check(
        embeddings.shape[0] == len(mapping),
        "物品向量行数与 Semantic ID 映射行数不一致",
    )
    check(
        embeddings.shape[1] == quantizer.embedding_dim_,
        "物品向量维度与量化器 embedding_dim 不一致",
    )
    check(bool(np.isfinite(embeddings).all()), "物品向量包含 NaN 或 Inf")

    collision_stats = quantizer.collision_statistics(numeric_codes)
    reconstruction_mse = quantizer.reconstruction_error(
        embeddings.astype(np.float32),
        numeric_codes,
    )

    encoder = ItemTextEncoder.load(required_files["encoder"])
    check(encoder.is_fitted, "保存的 ItemTextEncoder 未处于 fitted 状态")
    check(
        encoder.output_dim_ == embeddings.shape[1],
        "编码器输出维度与物品向量维度不一致",
    )

    train_item_ids = set(train["item_id"].astype(str))
    expected_train_flags = mapping["item_id"].astype(str).isin(train_item_ids)
    check("is_train_item" in mapping.columns, "缺少 is_train_item 列")
    if "is_train_item" in mapping.columns:
        actual_train_flags = parse_boolean_series(
            mapping["is_train_item"],
            "is_train_item",
        )
        check(
            actual_train_flags.tolist() == expected_train_flags.tolist(),
            "is_train_item 标记与 train.csv 不一致",
        )

    with required_files["stats"].open("r", encoding="utf-8") as file:
        saved_stats = json.load(file)
    with required_files["snapshot"].open("r", encoding="utf-8-sig") as file:
        snapshot = yaml.safe_load(file)

    check(
        saved_stats.get("quantizer_mode") == mode,
        "assignment_stats.json 中的 quantizer_mode 不一致",
    )
    check(
        snapshot.get("quantizer_mode") == mode,
        "config_snapshot.yaml 中的 quantizer_mode 不一致",
    )
    if mode == "advanced":
        check(
            collision_stats["collision_rate"] == 0.0,
            "advanced 模式仍然存在 Semantic ID 冲突",
        )

    if errors:
        raise ValueError("\n".join(f"- {error}" for error in errors))

    return {
        "mode": mode,
        "output_dir": project_relative_path(output_dir),
        "total_items": len(mapping),
        "train_items": int(expected_train_flags.sum()),
        "train_item_coverage": len(train_item_ids & set(items["item_id"].astype(str)))
        / len(train_item_ids),
        "code_shape": list(numeric_codes.shape),
        "embedding_shape": list(embeddings.shape),
        "codebook_size": quantizer.codebook_size,
        "num_codebooks": quantizer.num_codebooks,
        "collision_rate": collision_stats["collision_rate"],
        "unique_semantic_ids": collision_stats["unique_semantic_ids"],
        "reconstruction_mse": reconstruction_mse,
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)

    semantic_config = config.get("semantic_id", {})
    if not isinstance(semantic_config, dict):
        raise ValueError("semantic_id 必须是 YAML 字典")

    mode = str(args.mode or semantic_config.get("quantizer_mode", "advanced"))
    if mode not in {"baseline", "advanced"}:
        raise ValueError("mode 必须是 baseline 或 advanced")

    output_dir = get_output_dir(
        config=config,
        mode=mode,
        semantic_dir=args.semantic_dir,
    )
    summary = validate(mode, output_dir, config)

    print("Semantic ID 验证通过")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Semantic ID 验证失败：{error}")
        raise SystemExit(1)
