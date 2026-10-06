from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.representations.semantic_id.advanced_quantizer import (
    ResidualVectorQuantizer as AdvancedQuantizer,
)
from src.representations.semantic_id.encoder import ItemTextEncoder
from src.representations.semantic_id.quantizer import (
    ResidualVectorQuantizer as BasicQuantizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build isolated TIGER Semantic IDs, including the paper-style baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "ml-100k-tiger-baseline.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "advanced"),
        default="baseline",
        help="baseline adds a deterministic fourth disambiguation code; advanced uses unique assignment.",
    )
    parser.add_argument(
        "--fit-semantic-dir",
        type=Path,
        default=None,
        help="Optional fitted Semantic ID directory. When supplied, encoder and codebooks are transferred.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--branch-width", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_config(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置文件必须解析为 YAML 字典")
    return config


def load_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    processed_dir = resolve_path(config.get("processed_data_dir", "data/processed/ml-100k"))
    items_path = resolve_path(config.get("items_path", processed_dir / "items.csv"))
    train_path = processed_dir / "train.csv"
    valid_path = processed_dir / "valid.csv"
    for path in (items_path, train_path, valid_path):
        if not path.is_file():
            raise FileNotFoundError(f"找不到数据文件：{path}")

    items = pd.read_csv(items_path)
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)
    for name, dataframe in (("items", items), ("train", train), ("valid", valid)):
        if "item_id" not in dataframe.columns:
            raise ValueError(f"{name} 缺少 item_id 字段")
        dataframe["item_id"] = dataframe["item_id"].astype("string").str.strip()
        if dataframe["item_id"].isna().any():
            raise ValueError(f"{name}.item_id 不能包含缺失值")
    if items["item_id"].duplicated().any():
        raise ValueError("items.item_id 不能重复")
    return items, train, items_path, train_path, valid_path


def make_encoder(config: dict[str, Any], seed: int) -> ItemTextEncoder:
    semantic_config = config.get("semantic_id", {})
    encoder_config = semantic_config.get("encoder", {})
    if not isinstance(encoder_config, dict):
        raise ValueError("semantic_id.encoder 必须是字典")
    text_columns = tuple(encoder_config.get("text_columns", ["title", "genres"]))
    ngram_range = encoder_config.get("ngram_range", [1, 2])
    if len(ngram_range) != 2:
        raise ValueError("ngram_range 必须有两个元素")
    return ItemTextEncoder(
        n_components=encoder_config.get("n_components", 32),
        text_columns=text_columns,
        max_features=encoder_config.get("max_features", 20000),
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        min_df=encoder_config.get("min_df", 1),
        max_df=encoder_config.get("max_df", 1.0),
        lowercase=bool(encoder_config.get("lowercase", True)),
        normalize_embeddings=bool(encoder_config.get("normalize_embeddings", True)),
        random_state=seed,
    )


def make_quantizer(mode: str, config: dict[str, Any], seed: int) -> Any:
    semantic_config = config.get("semantic_id", {})
    quantizer_config = semantic_config.get("quantizer", {})
    if not isinstance(quantizer_config, dict):
        raise ValueError("semantic_id.quantizer 必须是字典")
    kwargs = {
        "codebook_size": int(quantizer_config.get("codebook_size", 64)),
        "num_codebooks": int(quantizer_config.get("num_codebooks", 3)),
        "n_init": int(quantizer_config.get("n_init", 10)),
        "max_iter": int(quantizer_config.get("max_iter", 300)),
        "random_state": seed,
        "distance_batch_size": int(quantizer_config.get("distance_batch_size", 4096)),
    }
    if mode == "baseline":
        return BasicQuantizer(**kwargs)
    return AdvancedQuantizer(**kwargs)


def collision_statistics(codes: np.ndarray) -> dict[str, int | float]:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError("codes 必须是二维数组")
    ids = ["-".join(str(int(value)) for value in row) for row in codes]
    total = len(ids)
    unique = len(set(ids))
    return {
        "total_items": total,
        "unique_semantic_ids": unique,
        "collision_items": total - unique,
        "collision_rate": (total - unique) / total if total else 0.0,
    }


def append_disambiguation_code(
    item_ids: list[str],
    prefix_codes: np.ndarray,
    *,
    capacity: int,
) -> tuple[np.ndarray, dict[str, int | float]]:
    prefix_codes = np.asarray(prefix_codes, dtype=np.int64)
    if prefix_codes.ndim != 2 or prefix_codes.shape[1] != 3:
        raise ValueError("论文式 TIGER baseline 要求量化器输出形状为 [N, 3]")
    if len(item_ids) != prefix_codes.shape[0]:
        raise ValueError("item_ids 与 codes 行数不一致")

    groups: dict[tuple[int, int, int], list[int]] = {}
    for row_index, row in enumerate(prefix_codes):
        groups.setdefault(tuple(int(value) for value in row), []).append(row_index)

    final_codes = np.empty((len(item_ids), 4), dtype=np.int64)
    collision_groups = 0
    max_group_size = 0
    collision_items = 0

    for prefix, row_indices in groups.items():
        ordered_indices = sorted(row_indices, key=lambda index: str(item_ids[index]))
        group_size = len(ordered_indices)
        max_group_size = max(max_group_size, group_size)
        if group_size > 1:
            collision_groups += 1
            collision_items += group_size - 1
        if group_size > capacity:
            raise ValueError(
                f"前三码冲突组大小 {group_size} 超过 code_3 容量 {capacity}：{prefix}"
            )
        for disambiguation, row_index in enumerate(ordered_indices):
            final_codes[row_index, :3] = prefix
            final_codes[row_index, 3] = disambiguation

    return final_codes, {
        "prefix_collision_groups": collision_groups,
        "prefix_collision_items": collision_items,
        "max_prefix_collision_group_size": max_group_size,
        "disambiguation_capacity": capacity,
        "ordering": "normalized_item_id_lexicographic",
    }


def build_mapping(
    item_ids: list[str],
    codes: np.ndarray,
    train_item_ids: set[str],
) -> pd.DataFrame:
    codes = np.asarray(codes, dtype=np.int64)
    result = pd.DataFrame({"item_id": item_ids})
    for level in range(codes.shape[1]):
        result[f"code_{level}"] = codes[:, level]
    result["semantic_id"] = [
        "-".join(str(int(value)) for value in row) for row in codes
    ]
    result["is_train_item"] = [item_id in train_item_ids for item_id in item_ids]
    return result


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config).resolve()
    config = load_config(config_path)
    dataset = str(config.get("dataset", "unknown"))
    seed = int(config.get("seed", 42))
    items, train, items_path, train_path, valid_path = load_data(config)

    item_ids = items["item_id"].astype(str).tolist()
    train_item_ids = set(train["item_id"].astype(str).tolist())
    train_items = items[items["item_id"].isin(train_item_ids)].reset_index(drop=True)
    if train_items.empty:
        raise ValueError("训练集中没有对应的物品元数据")

    source_dir = resolve_path(args.fit_semantic_dir) if args.fit_semantic_dir else None
    if source_dir is None:
        encoder = make_encoder(config, seed)
        print(f"[{dataset}/{args.mode}] 拟合 ItemTextEncoder...")
        encoder.fit(train_items)
        train_embeddings = encoder.transform(train_items)
        all_embeddings = encoder.transform(items)
        quantizer = make_quantizer(args.mode, config, seed)
        print(f"[{dataset}/{args.mode}] 拟合量化器...")
        quantizer.fit(train_embeddings)
        fit_source = "target_dataset"
    else:
        encoder_path = source_dir / "item_encoder.pkl"
        codebooks_path = source_dir / "codebooks.npz"
        if not encoder_path.is_file() or not codebooks_path.is_file():
            raise FileNotFoundError(f"拟合目录缺少 item_encoder.pkl 或 codebooks.npz：{source_dir}")
        encoder = ItemTextEncoder.load(encoder_path)
        quantizer_cls = BasicQuantizer if args.mode == "baseline" else AdvancedQuantizer
        quantizer = quantizer_cls.load(codebooks_path)
        all_embeddings = encoder.transform(items)
        fit_source = relative_path(source_dir)
        print(f"[{dataset}/{args.mode}] 复用编码规则：{source_dir}")

    if quantizer.num_codebooks != 3:
        raise ValueError(f"TIGER baseline 实验要求 3 个量化 codebook，实际为 {quantizer.num_codebooks}")

    ordinary_codes = quantizer.encode(all_embeddings)
    prefix_collision = collision_statistics(ordinary_codes)

    semantic_config = config.get("semantic_id", {})
    beam_width = int(
        args.beam_width
        if args.beam_width is not None
        else semantic_config.get("beam_width", 128)
    )
    branch_width = int(
        args.branch_width
        if args.branch_width is not None
        else semantic_config.get("branch_width", 16)
    )

    if args.mode == "baseline":
        final_codes, disambiguation_stats = append_disambiguation_code(
            item_ids,
            ordinary_codes,
            capacity=quantizer.codebook_size,
        )
    else:
        final_codes, disambiguation_stats = quantizer.encode_unique(
            all_embeddings,
            beam_width=beam_width,
            branch_width=branch_width,
        )

    final_collision = collision_statistics(final_codes)
    reconstruction_mse = quantizer.reconstruction_error(all_embeddings, ordinary_codes)

    output_config = semantic_config.get(
        "output_dir",
        f"data/processed/{dataset}/semantic_id/TIGER_baseline",
    )
    output_dir = resolve_path(args.output_dir or output_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = output_dir / "item_semantic_ids.csv"
    embeddings_path = output_dir / "item_embeddings.npy"
    encoder_path = output_dir / "item_encoder.pkl"
    codebooks_path = output_dir / "codebooks.npz"
    stats_path = output_dir / "assignment_stats.json"
    snapshot_path = output_dir / "config_snapshot.yaml"

    mapping = build_mapping(item_ids, final_codes, train_item_ids)
    mapping.to_csv(mapping_path, index=False)
    np.save(embeddings_path, all_embeddings)
    encoder.save(encoder_path)
    quantizer.save(codebooks_path)

    stats = {
        "dataset": dataset,
        "experiment": "TIGER_baseline" if args.mode == "baseline" else "TIGER_cross_advanced",
        "quantizer_mode": args.mode,
        "semantic_code_length": int(final_codes.shape[1]),
        "quantizer_num_codebooks": int(quantizer.num_codebooks),
        "seed": seed,
        "fit_source": fit_source,
        "data": {
            "items_path": relative_path(items_path),
            "train_path": relative_path(train_path),
            "valid_path": relative_path(valid_path),
            "total_items": len(items),
            "train_items": len(train_items),
        },
        "encoder": {
            "output_dim": encoder.output_dim_,
            "text_columns": list(encoder.text_columns_ or ()),
        },
        "quantizer": {
            "codebook_size": quantizer.codebook_size,
            "num_codebooks": quantizer.num_codebooks,
            "embedding_dim": quantizer.embedding_dim_,
        },
        "prefix_quantization": {
            **prefix_collision,
            "reconstruction_mse": reconstruction_mse,
        },
        "final_semantic_ids": {
            **final_collision,
            "reconstruction_mse": reconstruction_mse,
            **disambiguation_stats,
            "beam_width": beam_width,
            "branch_width": branch_width,
        },
        "outputs": {
            "mapping": relative_path(mapping_path),
            "embeddings": relative_path(embeddings_path),
            "encoder": relative_path(encoder_path),
            "codebooks": relative_path(codebooks_path),
        },
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    snapshot = {
        "dataset": dataset,
        "seed": seed,
        "quantizer_mode": args.mode,
        "semantic_code_length": int(final_codes.shape[1]),
        "quantizer_num_codebooks": int(quantizer.num_codebooks),
        "fit_source": fit_source,
        "config_file": relative_path(config_path),
        "data": {
            "items_path": relative_path(items_path),
            "train_path": relative_path(train_path),
            "valid_path": relative_path(valid_path),
        },
        "quantizer": {
            "codebook_size": quantizer.codebook_size,
            "num_codebooks": quantizer.num_codebooks,
            "n_init": quantizer.n_init,
            "max_iter": quantizer.max_iter,
            "distance_batch_size": quantizer.distance_batch_size,
        },
        "disambiguation": {
            "enabled": args.mode == "baseline",
            "capacity": quantizer.codebook_size,
            "ordering": "normalized_item_id_lexicographic",
        },
        "advanced_assignment": {
            "beam_width": beam_width,
            "branch_width": branch_width,
        },
        "output_dir": relative_path(output_dir),
    }
    snapshot_path.write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print("TIGER Semantic ID 构建完成")
    print(f"dataset: {dataset}")
    print(f"mode: {args.mode}")
    print(f"semantic_code_length: {final_codes.shape[1]}")
    print(f"prefix_collision_rate: {prefix_collision['collision_rate']:.6f}")
    print(f"final_collision_rate: {final_collision['collision_rate']:.6f}")
    print(f"reconstruction_mse: {reconstruction_mse:.8f}")
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
