from __future__ import annotations
from pathlib import Path
from typing import Any
import sys
import argparse
import yaml
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.sasrec import (
    SASRec,
    SASRecRecommender,
    SASRecTrainer,
    set_seed,
)
from src.data.sequence_dataset import SASRecSequenceDataset
from src.evaluation.metrics import evaluate_recommender

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the SASRec sequential recommender.")
    # 配置文件路径
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
        help="Path to the YAML configuration file.",)
    # 运行设备覆盖
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device in YAML, for example cuda or cpu.",)
    # 训练轮次覆盖
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs.",)
    return parser.parse_args()

def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path

def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{config_path}")
    with config_path.open("r",encoding="utf-8-sig",) as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件必须解析为 YAML 字典")
    return config

def load_data(config: dict[str, Any],) -> tuple[pd.DataFrame, ...]:
    processed_dir = resolve_path(config.get(
            "processed_data_dir", # 自定义路径
            "data/processed/ml-100k", # 默认路径
        ))

    paths = {
        "train": processed_dir / "train.csv",
        "valid": processed_dir / "valid.csv",
        "test": processed_dir / "test.csv",
    }

    missing_paths = [str(path) for path in paths.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("缺少预处理文件：\n" + "\n".join(missing_paths))

    train, valid, test = (pd.read_csv(paths[name]) for name in ("train", "valid", "test"))

    required_columns = {"user_id", "item_id", "timestamp",}
    for name, dataframe in {
        "train": train,
        "valid": valid,
        "test": test,
    }.items():
        missing_columns = (required_columns - set(dataframe.columns))
        if missing_columns:
            raise ValueError(f"{name}.csv 缺少字段："f"{sorted(missing_columns)}")
        if dataframe.empty:
            raise ValueError(f"{name}.csv 为空")

    return train, valid, test

"""
把原始的用户 - 物品交互明细表，转换成「每个用户 → 按时间升序排列的物品 ID 序列」的字典结构
"""
def _sorted_user_items(dataframe: pd.DataFrame,) -> dict[str, list[str]]:
    data = dataframe.copy()
    data["user_id"] = (data["user_id"].astype("string").str.strip())
    data["item_id"] = (data["item_id"].astype("string").str.strip())
    data["timestamp"] = pd.to_numeric(data["timestamp"],errors="raise",)

    data = data.sort_values(by=["user_id", "timestamp"], kind="stable",)

    return {
        str(user_id): (group["item_id"].astype("string").tolist())
        for user_id, group in data.groupby(
            "user_id",
            sort=False,)
        }

"""
    构造验证或测试用的一步预测数据集。

    对每个目标交互：
        输入：目标交互发生前的用户历史
        目标：当前交互中的 item_id
"""
def build_next_item_dataset(
    history_interactions: pd.DataFrame,
    target_interactions: pd.DataFrame,
    *,
    item_to_index: dict[str, int],
    max_seq_len: int,) -> TensorDataset:

    history_by_user = _sorted_user_items(history_interactions)
    targets = target_interactions.copy()
    targets["user_id"] = (targets["user_id"].astype("string").str.strip())
    targets["item_id"] = (targets["item_id"].astype("string").str.strip())
    targets["timestamp"] = pd.to_numeric(targets["timestamp"],errors="raise",)
    targets = targets.sort_values(by=["user_id", "timestamp"],kind="stable",)

    # 两个列表分别存储所有样本的输入序列和对应的目标物品索引
    input_sequences: list[list[int]] = []
    target_items: list[int] = []

    # 逐用户逐目标生成样本
    for user_id, group in targets.groupby("user_id",sort=False,):
        user_id = str(user_id)
        history = list(history_by_user.get(user_id, [],))
        # 遍历当前用户的每条目标交互，取出目标物品 ID，映射为模型内部的整数索引
        for row in group.itertuples(index=False):
            # 取出目标物品 ID，映射为模型内部的整数索引
            target_item = str(row.item_id)
            target_index = item_to_index.get(target_item)
            if target_index is None:
                continue
            # 把用户的历史物品列表全部转换成内部整数索引，同时过滤掉不在索引表里的历史物品
            history_indices = [item_to_index[item_id] for item_id in history if item_id in item_to_index]
            if not history_indices:
                continue

            # 序列截断与左填充
            prefix = history_indices[-max_seq_len:]
            padded_prefix = ([0] * (max_seq_len - len(prefix)) + prefix)

            # 存入样本，递进更新历史
            input_sequences.append(padded_prefix)
            target_items.append(target_index)
            # 支持一个用户拥有多个按时间排列的目标交互。
            history.append(target_item)

    if not input_sequences:
        raise ValueError("无法构造一步预测数据集")

    return TensorDataset(
        torch.tensor(
            input_sequences,
            dtype=torch.long,),
        torch.tensor(
            target_items,
            dtype=torch.long,
        ),)

"""
保存把单次实验的「核心指标结论、训练过程曲线、完整配置快照」三类数据
"""
def write_results(
    result_dir: Path, # 结果保存的根目录，所有文件都会写入这个目录下
    result: dict[str, Any],
    history: list[dict[str, float]],
    config_snapshot: dict[str, Any],  # 本次实验的完整配置快照，包含模型结构、训练超参、数据路径等所有配置，用于复现实验、对比实验差异
) -> None:
    result_dir.mkdir(parents=True,exist_ok=True,)

    (result_dir / "results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,),encoding="utf-8",)

    (result_dir / "training_history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2,),encoding="utf-8",)

    (result_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config_snapshot,allow_unicode=True,sort_keys=False,),encoding="utf-8",)

def main() -> None:
    args = parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    config = load_config(config_path)

    set_seed(int(config.get("seed", 42)))

    train, valid, test = load_data(config)

    max_seq_len = int(config.get("max_seq_len",50,))
    top_k = int(config.get("top_k",10,))
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 0)) # 线程数
    learning_rate = float(config.get("learning_rate", 1e-3))
    epochs = int(args.epochs if args.epochs is not None else config.get("epochs", 20))
    requested_device = (args.device if args.device is not None else config.get("device", None))

    # 从配置的sasrec字段中，提取模型结构参数和训练超参数
    sasrec_config = config.get("sasrec", {})
    if not isinstance(sasrec_config, dict):
        raise ValueError("sasrec 配置必须是 YAML 字典")
    embedding_dim = int(sasrec_config.get("embedding_dim", 64))
    num_heads = int(sasrec_config.get("num_heads", 2))
    num_layers = int(sasrec_config.get("num_layers", 2))
    feed_forward_dim = int(sasrec_config.get("feed_forward_dim", 128))
    dropout = float(sasrec_config.get("dropout", 0.2))
    weight_decay = float(sasrec_config.get("weight_decay", 1e-5))
    grad_clip_norm = float(sasrec_config.get("grad_clip_norm", 5.0))
    patience = int(sasrec_config.get("patience", 3))

    # 数据集构建
    train_dataset = SASRecSequenceDataset(train, max_seq_len=max_seq_len,)
    valid_dataset = build_next_item_dataset(
        history_interactions=train,
        target_interactions=valid,
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

    # 实例化模型
    model = SASRec(
        num_items=train_dataset.num_items,
        max_seq_len=max_seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feed_forward_dim=feed_forward_dim,
        dropout=dropout,
    )

    # 路径解析
    result_dir = resolve_path(sasrec_config.get("result_dir", "results/ml-100k/sasrec"))
    checkpoint_path = resolve_path(sasrec_config.get("checkpoint_path", "checkpoints/ml-100k/sasrec_best.pt"))

    # 训练器构建，调用 src/baselines/sasrec/trainer中的SASRecTrainer，将其实例化
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

    print("开始训练 SASRec...")
    history = trainer.fit()

    # 封装为业务推荐器，（上文的类）
    recommender = SASRecRecommender(
        model=model,
        item_to_index=train_dataset.item_to_index_,
        index_to_item=train_dataset.index_to_item_,
        max_seq_len=max_seq_len,
        device=trainer.device,
    )

    print("开始评估验证集...")
    validation_metrics = evaluate_recommender(
        model=recommender,
        history_interactions=train,
        target_interactions=valid,
        k=top_k,
    )

    # 测试阶段要用训练集 + 验证集
    test_history = pd.concat(
        [train, valid],
        ignore_index=True,)
    print("开始评估测试集...")
    test_metrics = evaluate_recommender(
        model=recommender,
        history_interactions=test_history,
        target_interactions=test,
        k=top_k,
    )

    # 组装完整结果字典
    result = {
        "model": "sasrec",
        "dataset": config.get("dataset", "unknown"),
        "device": str(trainer.device),
        "top_k": top_k,
        "best_epoch": trainer.best_epoch,
        "best_validation_loss": trainer.best_metric,
        "model_config": {
            "max_seq_len": max_seq_len,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "feed_forward_dim": feed_forward_dim,
            "dropout": dropout,
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
            "train_items": train_dataset.num_items,},

        "validation": validation_metrics,
        "test": test_metrics,
        }

    config_snapshot = {
        "dataset": config.get("dataset","unknown",),
        "top_k": top_k,
        "sasrec": result["model_config"],
    }

    write_results(
        result_dir=result_dir,
        result=result,
        history=history,
        config_snapshot=config_snapshot,
    )

    print("SASRec completed")
    print(f"results_dir: {result_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"validation: {validation_metrics}")
    print(f"test: {test_metrics}")

if __name__ == "__main__":
    main()







