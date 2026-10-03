from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import sys
import argparse
import yaml

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_id.item_encoder import ItemTextEncoder
from src.semantic_id.quantizer import (ResidualVectorQuantizer as BasicQuantizer,)
from src.semantic_id.advanced_quantizer import (ResidualVectorQuantizer as AdvancedQuantizer,)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build item Semantic IDs.")

    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
        help="YAML configuration path.",)

    parser.add_argument(
        "--quantizer-mode",
        choices=("baseline", "advanced"),
        default=None,
        help="Quantizer mode. Overrides YAML configuration.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Semantic ID output directory.",
    )

    parser.add_argument(
        "--codebook-size",
        type=int,
        default=None,
        help="Override codebook size.",
    )

    parser.add_argument(
        "--num-codebooks",
        type=int,
        default=None,
        help="Override number of codebooks.",
    )

    parser.add_argument(
        "--beam-width",
        type=int,
        default=None,
        help="Override advanced quantizer beam width.",
    )

    parser.add_argument(
        "--branch-width",
        type=int,
        default=None,
        help="Override advanced quantizer branch width.",
    )
    return parser.parse_args()

def resolve_path(path_value: str | Path,) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path

"""
优先保存项目相对路径，避免配置快照绑定到某台机器的绝对路径。
"""
def project_relative_path(path: Path,) -> str:
    try:
        relative_path = path.resolve().relative_to(PROJECT_ROOT.resolve())
        return relative_path.as_posix()
    except ValueError:
        return str(path.resolve())


def load_config(config_path: Path,) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    with config_path.open("r",encoding="utf-8-sig",) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("配置文件必须解析为 YAML 字典")

    return config

"""
物品 ID 字段标准化校验工具函数
"""
def normalize_item_ids(
    dataframe: pd.DataFrame,
        *,
        dataframe_name: str,
        require_unique: bool = False,
    ) -> pd.DataFrame:
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{dataframe_name} 必须是 pandas.DataFrame")
    if "item_id" not in dataframe.columns:
        raise ValueError(f"{dataframe_name} 缺少 item_id 字段")
    if dataframe.empty:
        raise ValueError(f"{dataframe_name} 为空")

    data = dataframe.copy()
    data["item_id"] = (data["item_id"].astype("string").str.strip())

    if data["item_id"].isna().any():
        raise ValueError(f"{dataframe_name}.item_id 不能包含缺失值")
    if require_unique and data["item_id"].duplicated().any():
        raise ValueError(f"{dataframe_name}.item_id 不能重复")
    return data


"""
返回一个长度为 6 的元组 (物品表, 训练集表, 验证集表, 物品文件路径, 训练文件路径, 验证文件路径)
"""
def load_data(
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Path,
    Path,
    Path,
]:

    processed_dir = resolve_path(config.get("processed_data_dir", "data/processed/ml-100k",))
    items_path = resolve_path(config.get("items_path",processed_dir / "items.csv",))

    train_path = processed_dir / "train.csv"
    valid_path = processed_dir / "valid.csv"

    if not items_path.exists():
        raise FileNotFoundError(f"找不到物品文件：{items_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"找不到训练文件：{train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"找不到验证文件：{valid_path}")

    items = pd.read_csv(items_path)
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)

    items = normalize_item_ids(
        items,
        dataframe_name="items",
        require_unique=True,
    )
    train = normalize_item_ids(train, dataframe_name="train")
    valid = normalize_item_ids(valid, dataframe_name="valid")

    return (items, train, valid, items_path, train_path, valid_path,)

"""
语义ID构建流水线的结果组装函数，
负责将物品主键、多层数值编码、字符串语义 ID、训练集标记四类信息整合为一张结构化的 pandas 结果表
"""
def build_semantic_id_table(
    *,
    item_ids: list[str],        # 物品 ID 列表，顺序与编码数组的行一一对应
    codes: np.ndarray,          # RVQ 输出的二维整数编码数组，形状`[物品数, 码本层数]`
    train_item_ids: set[str],   # 训练集包含的物品 ID 集合，用于打训练集标记
    quantizer: Any,             # 量化器实例，要求实现`semantic_id_strings`接口，用于生成字符串语义 ID
    ) -> pd.DataFrame:

    if codes.ndim != 2:
        raise ValueError("codes 必须是二维数组")
    if len(item_ids) != codes.shape[0]:
        raise ValueError("item_ids 数量与 codes 行数不一致")

    # 按层数生成 `code_0`、`code_1`、`code_2`... 格式的列名,每层编码单独一列
    code_columns = [f"code_{index}"for index in range(codes.shape[1])]

    # 初始化编码表,把二维编码数组转为 DataFrame，每列对应一层的数值编码。
    result = pd.DataFrame(codes,columns=code_columns,)
    # 插入主键列.将物品 ID 列插入到表格的**第 0 列（最左侧）**，作为主键放在最前面
    result.insert(0,"item_id", item_ids,)

    # 生成字符串语义 ID 列.将多层数值编码拼接为完整的字符串语义 ID（如 `"12-48-7"`），作为最终的语义主键。
    result["semantic_id"] = (quantizer.semantic_id_strings(codes))
    # 后续可以按训练集 / 验证集拆分，分别统计冲突率、重构误差，评估量化器的泛化能力，判断是否过拟合。
    result["is_train_item"] = [item_id in train_item_ids for item_id in item_ids]

    return result

"""
文本特征提取环节的工厂
"""
def make_encoder(
    semantic_config: dict[str, Any],
    seed: int,
    ) -> ItemTextEncoder:
    encoder_config = semantic_config.get("encoder",{},)
    if not isinstance(encoder_config, dict):
        raise ValueError("semantic_id.encoder 必须是 YAML 字典")

    # 指定用物品表中的哪些列作为文本特征来生成嵌入。默认使用 `title`（标题）和 `genres`（类型）
    text_columns = encoder_config.get("text_columns", ["title", "genres"])
    if not isinstance(text_columns, (list, tuple)):
        raise TypeError("text_columns 必须是列表或元组")

    # 指定用物品表中的哪些列作为文本特征来生成嵌入。默认使用 `title`（标题）和 `genres`（类型）
    text_columns = encoder_config.get("text_columns", ["title", "genres"])
    if not isinstance(text_columns, (list, tuple)):
        raise TypeError("text_columns 必须是列表或元组")

    # 控制文本特征的切分粒度...提高切分粒度能不能降低碰撞率？
    ngram_range = encoder_config.get("ngram_range", [1, 2])
    if (not isinstance(ngram_range, (list, tuple))or len(ngram_range) != 2):
        raise ValueError("ngram_range 必须是长度为 2 的列表或元组")

    return ItemTextEncoder(
        n_components=encoder_config.get("n_components", 32),
        text_columns=tuple(text_columns),
        max_features=encoder_config.get("max_features", 20000),
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        min_df=encoder_config.get("min_df", 1),
        max_df=encoder_config.get("max_df", 1.0),
        lowercase=bool(encoder_config.get("lowercase", True)),
        normalize_embeddings=bool(encoder_config.get("normalize_embeddings", True)),
        random_state=seed,
    )

"""
向量量化编码环节的工厂
"""
def make_quantizer(
    mode: str,
    semantic_config: dict[str, Any],
    seed: int,
    codebook_size: int | None = None,
    num_codebooks: int | None = None,
    ) -> Any:
    quantizer_config = semantic_config.get("quantizer", {})
    if not isinstance(quantizer_config, dict):
        raise ValueError("semantic_id.quantizer 必须是 YAML 字典")

    common_kwargs = {
        "codebook_size": int(
            codebook_size
            if codebook_size is not None
            else quantizer_config.get("codebook_size", 64)
        ),
        "num_codebooks": int(
            num_codebooks
            if num_codebooks is not None
            else quantizer_config.get("num_codebooks", 3)
        ),
        "n_init": int(quantizer_config.get("n_init", 10)),
        "max_iter": int(quantizer_config.get("max_iter", 300)),
        "random_state": seed,
        "distance_batch_size": int(quantizer_config.get("distance_batch_size", 4096)),
    }

    if mode == "baseline":
        return BasicQuantizer(**common_kwargs)
    if mode == "advanced":
        return AdvancedQuantizer(**common_kwargs)
    raise ValueError(f"不支持的 quantizer mode：{mode}")

def main() -> None:
    args = parse_args()

    config_path = args.config

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config_path = config_path.resolve()
    config = load_config(config_path)

    dataset = str(config.get("dataset", "unknown",))
    seed = int(config.get("seed", 42))

    semantic_config = config.get("semantic_id",{},)
    if not isinstance(semantic_config, dict):
        raise ValueError("semantic_id 必须是 YAML 字典")

    mode = (args.quantizer_mode or semantic_config.get("quantizer_mode",
                                                       "advanced",))
    if mode not in {
        "baseline",
        "advanced",}:
        raise ValueError("quantizer_mode 必须是 baseline 或 advanced")

    (items, train, valid, items_path, train_path, valid_path,) = load_data(config)
    all_item_ids = (items["item_id"].astype(str).tolist())
    all_item_id_set = set(all_item_ids)

    # 检查训练集中的所有物品 ID，是否都存在于物品元数据表中。缺失则直接报错并列出前 10 个缺失 ID
    train_item_ids = set(train["item_id"].astype(str).tolist())
    missing_train_items = (train_item_ids - all_item_id_set)
    if missing_train_items:
        raise ValueError("以下训练物品不在 items.csv 中："f"{sorted(missing_train_items)[:10]}")

    train_items = items[items["item_id"].isin(train_item_ids)].reset_index(drop=True)
    if train_items.empty:
        raise ValueError("没有找到训练物品对应的元数据")

    # 调用文本特征提取环节的工厂
    encoder = make_encoder(semantic_config,seed,)       # 实例化
    print("开始拟合 ItemTextEncoder...")
    encoder.fit(train_items)                            # 真正调用：检查、拼接物品文本、拟合 TF-IDF、拟合 TruncatedSVD、保存拟合结果
    train_embeddings = encoder.transform(train_items)   # 使用已经学习好的编码规则，生成训练物品向量：
    all_embeddings = encoder.transform(items)           # 全部物品的低维向量

    # 调用向量量化编码环节的工厂
    quantizer = make_quantizer(
        mode,
        semantic_config,
        seed,
        codebook_size=args.codebook_size,
        num_codebooks=args.num_codebooks,
    )
    print("开始拟合 Semantic ID quantizer...")
    quantizer.fit(train_embeddings)

    ordinary_codes = quantizer.encode(all_embeddings)
    ordinary_collision = (quantizer.collision_statistics(ordinary_codes))
    ordinary_mse = (quantizer.reconstruction_error(all_embeddings,ordinary_codes,))

    beam_width = int(args.beam_width
        if args.beam_width is not None
        else semantic_config.get("beam_width",32,))

    branch_width = int(
        args.branch_width
        if args.branch_width is not None
        else semantic_config.get("branch_width",8,))

    if mode == "advanced":
        print("开始执行全局唯一 Semantic ID 分配...")
        final_codes, assignment_stats = (
            quantizer.encode_unique(
                all_embeddings,
                beam_width=beam_width,
                branch_width=branch_width,
            ))
    else:
        final_codes = ordinary_codes

        assignment_stats = {
            **ordinary_collision,
            "assigned_items": len(items),
            "unassigned_items": 0,
            "mean_reconstruction_error": (ordinary_mse),
            "max_reconstruction_error": (ordinary_mse),
            "mean_error_increase": 0.0,
            "max_error_increase": 0.0,
            "beam_width": 1,
            "branch_width": 1,
        }

    final_collision = (quantizer.collision_statistics(final_codes))
    final_mse = (quantizer.reconstruction_error(all_embeddings,final_codes,))

    if mode == "advanced":
        if final_collision["collision_rate"] != 0.0:
            raise RuntimeError("Advanced Quantizer 没有生成唯一 Semantic ID")

    output_config = semantic_config.get("output_dir",
        (Path("data") / "processed" / dataset/ "semantic_id"/ mode),)

    if args.output_dir is not None:
        output_config = args.output_dir

    output_dir = resolve_path(output_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    item_ids = items["item_id"].astype(str).tolist()
    mapping = build_semantic_id_table(
        item_ids=item_ids,
        codes=final_codes,
        train_item_ids=train_item_ids,
        quantizer=quantizer,
    )

    mapping_path = output_dir / "item_semantic_ids.csv"  # 语义ID映射表
    embeddings_path = output_dir / "item_embeddings.npy"  # 全量物品嵌入向量
    encoder_path = output_dir / "item_encoder.pkl"  # 训练好的文本编码器
    codebooks_path = output_dir / "codebooks.npz"  # 训练好的量化码本
    stats_path = output_dir / "assignment_stats.json"  # 运行统计指标
    snapshot_path = output_dir / "config_snapshot.yaml"  # 本次运行配置快照

    mapping.to_csv(mapping_path, index=False)  # 导出CSV，不存行索引
    np.save(embeddings_path, all_embeddings)  # 保存numpy格式嵌入
    encoder.save(encoder_path)  # 序列化保存文本编码器
    quantizer.save(codebooks_path)  # 压缩保存量化码本

    stats = {
        "dataset": dataset,
        "quantizer_mode": mode,
        "seed": seed,
        "data": {"items_path": project_relative_path(items_path),
                "train_path": project_relative_path(train_path),
                "valid_path": project_relative_path(valid_path),
                "total_items": len(items),
                "train_items": len(train_items),
                "train_item_coverage": (len(train_item_ids & all_item_id_set) / len(train_item_ids)),
                },
        "encoder": {"output_dim": encoder.output_dim_,
                    "text_columns": list(encoder.text_columns_ or ()),
                    "feature_count": len(encoder.feature_names_ or ()),
                },
        "quantizer": {  "codebook_size": quantizer.codebook_size,
                        "num_codebooks": quantizer.num_codebooks,
                        "embedding_dim": quantizer.embedding_dim_,},
        "ordinary_quantization": {
            **ordinary_collision,
            "reconstruction_mse": ordinary_mse,},
        "final_assignment": {
            **assignment_stats,
            "reconstruction_mse": final_mse,},
        "outputs": {"mapping": project_relative_path(mapping_path),
                    "embeddings": project_relative_path(embeddings_path),
                    "encoder": project_relative_path(encoder_path),
                    "codebooks": project_relative_path(codebooks_path),},}

    stats_path.write_text(
        json.dumps(
            stats,
            ensure_ascii=False,
            indent=2,),
        encoding="utf-8",)

    snapshot = {
        "dataset": dataset,
        "seed": seed,
        "quantizer_mode": mode,
        "config_file": project_relative_path(config_path),
        "data": {   "items_path": project_relative_path(items_path),
                    "train_path": project_relative_path(train_path),
                    "valid_path": project_relative_path(valid_path),},
        "encoder": {"n_components": encoder.n_components,
                    "text_columns": list(encoder.text_columns),
                    "max_features": encoder.max_features,
                    "ngram_range": list(encoder.ngram_range),
                    "min_df": encoder.min_df,
                    "max_df": encoder.max_df,
                    "lowercase": encoder.lowercase,
                    "normalize_embeddings": (encoder.normalize_embeddings),},
        "quantizer": {  "codebook_size": quantizer.codebook_size,
                        "num_codebooks": quantizer.num_codebooks,
                        "n_init": quantizer.n_init,
                        "max_iter": quantizer.max_iter,
                        "distance_batch_size": (quantizer.distance_batch_size),},
        "advanced_assignment": {"beam_width": beam_width,
                                "branch_width": branch_width,},
        "output_dir": project_relative_path(output_dir),}

    snapshot_path.write_text(
            yaml.safe_dump(
            snapshot,
            allow_unicode=True,
            sort_keys=False,),
        encoding="utf-8",)

    print("Semantic ID 构建完成")
    print(f"quantizer_mode: {mode}")
    print(f"total_items: {len(items)}")
    print("train_item_coverage: "f"{stats['data']['train_item_coverage']:.4f}")
    print("ordinary_collision_rate: "f"{ordinary_collision['collision_rate']:.6f}")
    print("final_collision_rate: "f"{final_collision['collision_rate']:.6f}")
    print("final_reconstruction_mse: "f"{final_mse:.8f}")
    print(f"output_dir: {output_dir}")

if __name__ == "__main__":
    main()
