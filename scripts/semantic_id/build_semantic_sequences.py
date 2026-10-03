"""
调用adapter.py，处理并保存数据
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml
import sys
import json
import argparse
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_id.adapter import (
    SemanticIDMapper,
    SemanticSequenceAdapter,)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Semantic ID sequence datasets.")

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
        help="Semantic ID mode.",
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=None,
        help="Semantic ID directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for sequence datasets.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum history sequence length.",
    )
    parser.add_argument(
        "--min-history-len",
        type=int,
        default=None,
        help="Minimum history length.",
    )
    return parser.parse_args()

def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path

def project_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(
            PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())

def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}")

    with config_path.open("r",encoding="utf-8-sig",) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("配置文件必须解析为 YAML 字典")

    return config

def get_mode(
    config: dict[str, Any],
    requested_mode: str | None,) -> str:

    semantic_config = config.get("semantic_id",{},)

    if not isinstance(semantic_config, dict):
        raise ValueError("semantic_id 必须是 YAML 字典")

    mode = str(requested_mode
        or semantic_config.get(
            "quantizer_mode",
            "advanced",))

    if mode not in {"baseline", "advanced"}:
        raise ValueError("mode 必须是 baseline 或 advanced")
    return mode

def get_semantic_dir(
    config: dict[str, Any],
    mode: str,
    semantic_dir: Path | None,
) -> Path:
    if semantic_dir is not None:
        return resolve_path(semantic_dir)

    semantic_config = config.get("semantic_id",{},)

    configured_mode = str(
        semantic_config.get(
            "quantizer_mode",
            "advanced",))

    configured_output_dir = semantic_config.get("output_dir")

    if (mode == configured_mode and configured_output_dir is not None):
        return resolve_path(configured_output_dir)

    dataset = str(config.get("dataset","unknown",))

    return (PROJECT_ROOT/ "data"/ "processed"/ dataset/ "semantic_id"/ mode)

def get_sequence_output_dir(
    config: dict[str, Any],
    mode: str,
    output_dir: Path | None,) -> Path:
    if output_dir is not None:
        return resolve_path(output_dir)

    sequence_config = config.get("semantic_sequence",{},)

    if not isinstance(sequence_config, dict):
        raise ValueError("semantic_sequence 必须是 YAML 字典")

    configured_output_dir = sequence_config.get("output_dir")

    if configured_output_dir is not None:
        return resolve_path(configured_output_dir)

    dataset = str(config.get("dataset","unknown",))

    return (PROJECT_ROOT / "data" / "processed" / dataset / "semantic_sequences" / mode)

def load_interactions(
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    processed_dir = resolve_path(
        config.get("processed_data_dir","data/processed/ml-100k",))

    split_names = ("train","valid","test",)

    result: dict[str, pd.DataFrame] = {}

    for split_name in split_names:
        split_path = processed_dir / f"{split_name}.csv"

        if not split_path.exists():
            raise FileNotFoundError(f"找不到 {split_name} 文件：{split_path}")

        dataframe = pd.read_csv(split_path)

        required_columns = {"user_id","item_id","timestamp",}

        missing_columns = (required_columns- set(dataframe.columns))
        if missing_columns:
            raise ValueError(f"{split_name}.csv 缺少字段："f"{sorted(missing_columns)}")

        if dataframe.empty:
            raise ValueError(f"{split_name}.csv 为空")
        result[split_name] = dataframe

    return result

"""
批量堆叠成完整的多维 numpy 数组。
它主要用于数据集持久化保存、批量向量化计算等场景
"""
def stack_numeric_fields(examples: list[dict[str, Any]],) -> dict[str, np.ndarray]:
    if not examples:
        raise ValueError("不能保存空样本集")

    array_fields = (
        "history_codes",
        "history_mask",
        "history_tokens",
        "history_token_mask",
        "target_codes",
        "target_input_tokens",
        "target_output_tokens",
    )
    arrays: dict[str, np.ndarray] = {}

    for field in array_fields:
        values = [example[field] for example in examples]
        arrays[field] = np.stack(values,axis=0,)
    return arrays

"""
堆叠后数据集的维度一致性校验函数
一般每个字段都是[样本数, 历史物品数, 码本层数]或[样本数, 历史物品数] [样本数, 历史Token总长度]
"""
def validate_split_arrays(
    split_name: str,
    arrays: dict[str, np.ndarray],
    *,
    max_seq_len: int,
    num_codebooks: int,
    ) -> None:

    expected_history_codes_shape = (arrays["history_codes"].shape[0],max_seq_len,num_codebooks,)
    if (arrays["history_codes"].shape != expected_history_codes_shape):
        raise ValueError(f"{split_name}.history_codes 形状错误："f"{arrays['history_codes'].shape}")

    expected_history_mask_shape = (arrays["history_codes"].shape[0], max_seq_len,)
    if (arrays["history_mask"].shape!= expected_history_mask_shape):
        raise ValueError(f"{split_name}.history_mask 形状错误："f"{arrays['history_mask'].shape}")

    expected_target_codes_shape = (arrays["history_codes"].shape[0],num_codebooks,)
    if (arrays["target_codes"].shape!= expected_target_codes_shape):
        raise ValueError(f"{split_name}.target_codes 形状错误："f"{arrays['target_codes'].shape}")

    expected_token_length = (max_seq_len * (num_codebooks + 1))

    if (arrays["history_tokens"].shape!= (arrays["history_codes"].shape[0],expected_token_length,)):
        raise ValueError(f"{split_name}.history_tokens 形状错误："f"{arrays['history_tokens'].shape}")

    if (arrays["history_token_mask"].shape!= (arrays["history_codes"].shape[0],expected_token_length,)):
        raise ValueError(f"{split_name}.history_token_mask 形状错误："f"{arrays['history_token_mask'].shape}")

    if (arrays["target_input_tokens"].shape!= (arrays["history_codes"].shape[0],num_codebooks + 1,)):
        raise ValueError(f"{split_name}.target_input_tokens 形状错误："f"{arrays['target_input_tokens'].shape}")

    if (arrays["target_output_tokens"].shape!= (arrays["history_codes"].shape[0],num_codebooks + 1,)):
        raise ValueError(f"{split_name}.target_output_tokens 形状错误："f"{arrays['target_output_tokens'].shape}")

# 单数据集的持久化保存函数
def save_split(
    output_dir: Path,
    split_name: str,
    examples: list[dict[str, Any]],) -> dict[str, Any]:
    arrays = stack_numeric_fields(examples)
    output_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(output_path,**arrays,)

    return {"file": project_relative_path(output_path),
            "count": len(examples),
            "arrays": {field: list(array.shape) for field, array in arrays.items()},}

def convert_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value

"""
**提取所有非数值型的业务元数据**，组装成结构化字典。
这些字段不适合存入 `.npz` 文件（numpy 对字符串存储效率低、加载麻烦），
单独提取出来后可以存为 JSON 文件，和 `.npz` 数值文件配套使用。
"""
def build_split_metadata(examples: list[dict[str, Any]],) -> dict[str, Any]:
    return {
        "count": len(examples),
        "user_ids": [str(example["user_id"])for example in examples],
        "history_item_ids": [[str(item_id) for item_id in example["history_item_ids"]]for example in examples],
        "target_item_ids": [str(example["target_item_id"])for example in examples],
        "target_semantic_ids": [str(example["target_semantic_id"])for example in examples],
        "target_timestamps": [convert_json_value(example["target_timestamp"])for example in examples],}

def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)

    dataset = str(config.get("dataset", "unknown",))
    seed = int(config.get("seed", 42,))
    mode = get_mode(config, args.mode,)

    semantic_dir = get_semantic_dir(config,mode,args.semantic_dir,)

    output_dir = get_sequence_output_dir(config,mode,args.output_dir,)
    output_dir.mkdir(parents=True,exist_ok=True,)

    sequence_config = config.get("semantic_sequence",{},)
    if not isinstance(sequence_config, dict):
        raise ValueError("semantic_sequence 必须是 YAML 字典")

    max_seq_len = int(args.max_seq_len if args.max_seq_len is not None
        else sequence_config.get("max_seq_len", config.get("max_seq_len",50,),))

    min_history_len = int(args.min_history_len if args.min_history_len is not None
        else sequence_config.get("min_history_len",1,))

    if max_seq_len < 1:
        raise ValueError("max_seq_len 必须大于 0")
    if min_history_len < 1:
        raise ValueError("min_history_len 必须大于 0")

    print("开始加载 Semantic ID 映射...")

    # 实例化SemanticIDMapper并打包参数
    mapper = SemanticIDMapper.from_directory(semantic_dir,mode=mode,)

    print(f"semantic_items: {len(mapper.item_ids)}")
    print(f"num_codebooks: {mapper.num_codebooks}")
    print(f"codebook_size: {mapper.codebook_size}")
    print(f"vocab_size: {mapper.vocab_size}")

    interactions = load_interactions(config)
    adapter = SemanticSequenceAdapter(
        mapper,
        max_seq_len=max_seq_len,
        min_history_len=min_history_len,)

    print("开始构造 Semantic ID 序列...")
    # 一次性构建全部分集样本
    splits = adapter.build_all_splits(
        interactions["train"],
        interactions["valid"],
        interactions["test"],)
    split_metadata: dict[str, Any] = {}
    split_summary: dict[str, Any] = {}

    for split_name in ("train","valid","test",):
        examples = splits[split_name]

        # 1. 数值字段批量堆叠
        arrays = stack_numeric_fields(examples)
        # 2. 维度一致性校验
        validate_split_arrays(
            split_name,
            arrays,
            max_seq_len=max_seq_len,
            num_codebooks=mapper.num_codebooks,)
        # 3. 保存压缩数值文件
        split_summary[split_name] = save_split(output_dir,split_name,examples, )
        # 4. 提取业务元数据
        split_metadata[split_name] = (build_split_metadata(examples))

    metadata = {
        "dataset": dataset,
        "seed": seed,
        "mode": mode,
        "semantic_dir": project_relative_path(semantic_dir),
        "output_dir": project_relative_path(output_dir),
        "max_seq_len": max_seq_len,
        "min_history_len": min_history_len,
        "num_codebooks": mapper.num_codebooks,
        "codebook_size": mapper.codebook_size,
        "vocab_size": mapper.vocab_size,
        "special_tokens": { "pad_token_id": mapper.pad_token_id,
                            "item_separator_token_id": (mapper.item_separator_token_id),
                            "bos_token_id": mapper.bos_token_id,
                            "eos_token_id": mapper.eos_token_id,},
        "files": split_summary,
        "splits": split_metadata,}

    metadata_path = output_dir / "metadata.json"

    metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,),encoding="utf-8",)

    snapshot = {
        "dataset": dataset,
        "seed": seed,
        "mode": mode,
        "config_file": project_relative_path(config_path),
        "semantic_dir": project_relative_path(semantic_dir),
        "output_dir": project_relative_path(output_dir),
        "max_seq_len": max_seq_len,
        "min_history_len": min_history_len,
        "num_codebooks": mapper.num_codebooks,
        "codebook_size": mapper.codebook_size,}
    snapshot_path = output_dir / "config_snapshot.yaml"

    snapshot_path.write_text(yaml.safe_dump(snapshot,allow_unicode=True,sort_keys=False,),encoding="utf-8",)

    print("Semantic ID 序列构建完成")
    print(f"dataset: {dataset}")
    print(f"mode: {mode}")
    print(f"max_seq_len: {max_seq_len}")
    print(f"output_dir: {output_dir}")

    for split_name, summary in split_summary.items():
        print(f"{split_name}: " f"{summary['count']} examples")

    return 0
"""
最终产物
output_dir/
├── train.npz              # 训练集数值数组（模型输入）
├── valid.npz              # 验证集数值数组
├── test.npz               # 测试集数值数组
├── metadata.json          # 全局完整元数据说明书
└── config_snapshot.yaml   # 核心配置快照（用于复现/对齐）
"""
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            f"Semantic ID 序列构建失败：{error}"
        )
        raise SystemExit(1)
