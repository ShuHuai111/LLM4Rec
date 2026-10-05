from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.generative.tiger.dataset import TigerSequenceDataset
from src.models.generative.tiger.tiger import TIGER, TigerConfig
from src.models.generative.tiger.trainer import TigerTrainer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TIGER-style Semantic ID generator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
        help="YAML configuration path.",
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=None,
        help="Semantic ID sequence directory. Overrides the configured path.",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "advanced"),
        default=None,
        help="Semantic ID mode used to select the default sequence directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory. Overrides tiger.result_dir.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint path. Overrides tiger.checkpoint_path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size. Overrides tiger.batch_size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. Overrides tiger.epochs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. Overrides the root num_workers setting.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit for a short smoke run.",
    )
    parser.add_argument(
        "--max-valid-samples",
        type=int,
        default=None,
        help="Optional validation sample limit for a short smoke run.",
    )
    return parser.parse_args()

"""
把相对路径转换成绝对路径。
"""
def resolve_path(path: str | Path, *, base_dir: Path = PROJECT_ROOT) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve()

"""
确认配置文件存在；
使用 YAML 读取；
确认根节点是字典；
确认存在 tiger 配置区块。
"""
def load_config(config_path: Path) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("YAML 根节点必须是字典")
    if not isinstance(config.get("tiger"), dict):
        raise ValueError("配置中缺少 tiger 字典")
    return config

"""
确定train.npz 和 valid.npz 到底在哪个目录
路径优先级是：
    1. 命令行 --sequence-dir
    2. semantic_sequence.output_dir
    3. 根据 dataset 和 mode 自动构造
"""
def get_sequence_dir(
    config: dict[str, Any],
    mode: str | None,
    sequence_dir: Path | None,
) -> Path:
    if sequence_dir is not None:
        return resolve_path(sequence_dir)

    semantic_config = config.get("semantic_id", {})
    if not isinstance(semantic_config, dict):
        raise ValueError("semantic_id 必须是 YAML 字典")

    selected_mode = mode or str(semantic_config.get("quantizer_mode", "advanced"))
    if selected_mode not in {"baseline", "advanced"}:
        raise ValueError("Semantic ID mode 必须是 baseline 或 advanced")

    sequence_config = config.get("semantic_sequence", {})
    if isinstance(sequence_config, dict):
        configured_output = sequence_config.get("output_dir")
        if configured_output:
            return resolve_path(configured_output)

    dataset = str(config.get("dataset", "unknown"))
    return (
        PROJECT_ROOT
        / "data"
        / "processed"
        / dataset
        / "semantic_sequences"
        / selected_mode
    ).resolve()

"""
如果不传样本限制,则直接返回完整 Dataset。
如果传：max_samples=100 ，则Subset(dataset, range(100))
"""
def select_dataset(
    dataset: TigerSequenceDataset,
    max_samples: int | None,
) -> TigerSequenceDataset | Subset[TigerSequenceDataset]:
    if max_samples is None:
        return dataset
    if not isinstance(max_samples, int) or isinstance(max_samples, bool):
        raise TypeError("max_samples 必须是整数或 None")
    if max_samples < 1:
        raise ValueError("max_samples 必须大于 0")
    return Subset(dataset, range(min(max_samples, len(dataset))))

"""
解决配置接入问题的关键。
"""
def build_model_kwargs(tiger_config: dict[str, Any]) -> dict[str, Any]:
    model_fields = set(TigerConfig.__annotations__)
    return {
        key: tiger_config[key]
        for key in model_fields
        if key in tiger_config
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> int:
    # 读取命令行参数,例如得到:args.device = "cpu";args.epochs = 1等
    args = parse_args()

    # 读取配置并复制一份
    config = load_config(args.config)
    tiger_config = copy.deepcopy(config["tiger"])

    # 设置随机种子
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # 加载 Dataset，先得到路径，然后分别加载train_dataset和valid_dataset
    sequence_dir = get_sequence_dir(config, args.mode, args.sequence_dir)
    train_dataset = TigerSequenceDataset.from_directory(sequence_dir, "train")
    valid_dataset = TigerSequenceDataset.from_directory(sequence_dir, "valid")
    # 加一个中间层，调用前面的方法，确认一下有没有size限制
    train_data = select_dataset(train_dataset, args.max_train_samples)
    valid_data = select_dataset(valid_dataset, args.max_valid_samples)

    # 确定训练参数，优先命令行参数，然后YAML 配置
    batch_size = (args.batch_size if args.batch_size is not None
        else int(tiger_config.get("batch_size", config.get("batch_size", 32))))

    epochs = (args.epochs if args.epochs is not None
        else int(tiger_config.get("epochs", config.get("epochs", 10))))

    num_workers = (args.num_workers if args.num_workers is not None
        else int(config.get("num_workers", 0)))
    # 校验参数
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0")
    if epochs < 1:
        raise ValueError("epochs 必须大于 0")
    if num_workers < 0:
        raise ValueError("num_workers 不能小于 0")

    requested_device = args.device or config.get("device")
    resolved_device = torch.device(requested_device if requested_device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    pin_memory = resolved_device.type == "cuda" and torch.cuda.is_available()

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,               # 训练集每轮打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 创建模型
    model = TIGER(**build_model_kwargs(tiger_config))

    output_dir = resolve_path(
        args.output_dir if args.output_dir is not None
        else tiger_config.get("result_dir", PROJECT_ROOT / "results" / str(config.get("dataset", "unknown")) / "tiger",))

    checkpoint_path = resolve_path(args.checkpoint_path if args.checkpoint_path is not None
        else tiger_config.get("checkpoint_path",PROJECT_ROOT/ "checkpoints"/ str(config.get("dataset", "unknown"))/ "tiger_best.pt",))

    # 将命令行覆盖项写入配置快照，确保结果可以追溯。
    snapshot_config = copy.deepcopy(config)
    snapshot_config.setdefault("tiger", {})["batch_size"] = batch_size
    snapshot_config.setdefault("tiger", {})["epochs"] = epochs
    snapshot_config["runtime"] = {
        "sequence_dir": str(sequence_dir),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "device": str(resolved_device),
        "num_workers": num_workers,
        "max_train_samples": args.max_train_samples,
        "max_valid_samples": args.max_valid_samples,
    }

    trainer = TigerTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=resolved_device,
        learning_rate=float(tiger_config.get("learning_rate", 1e-3)),
        weight_decay=float(tiger_config.get("weight_decay", 1e-5)),
        epochs=epochs,
        grad_clip_norm=tiger_config.get("grad_clip_norm", 5.0),
        patience=int(tiger_config.get("patience", 3)),
        checkpoint_path=checkpoint_path,
        run_config=snapshot_config,
    )

    print("=" * 60)
    print("TIGER training")
    print("=" * 60)
    print(f"sequence_dir: {sequence_dir}")
    print(f"train_samples: {len(train_data)}")
    print(f"valid_samples: {len(valid_data)}")
    print(f"batch_size: {batch_size}")
    print(f"epochs: {epochs}")
    print(f"device: {trainer.device}")
    print(f"checkpoint_path: {checkpoint_path}")

    history = trainer.fit()

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            snapshot_config,
            handle,
            allow_unicode=True,
            sort_keys=False,
        )

    save_json(output_dir / "training_history.json", history)
    save_json(
        output_dir / "results.json",
        {
            "dataset": config.get("dataset"),
            "sequence_dir": str(sequence_dir),
            "train_samples": len(train_data),
            "valid_samples": len(valid_data),
            "device": str(trainer.device),
            "best_epoch": trainer.best_epoch,
            "best_valid_loss": trainer.best_metric,
            "checkpoint_path": str(checkpoint_path),
            "model_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
    )

    print(f"results_dir: {output_dir}")
    print(f"best_epoch: {trainer.best_epoch}")
    print(f"best_valid_loss: {trainer.best_metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
