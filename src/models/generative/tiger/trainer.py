from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed 必须是整数")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""训练 TIGER 模型的 trainer。

    Dataset 的一个 batch 至少需要包含以下字段：

    - ``history_tokens``: [B, H]
    - ``history_token_mask``: [B, H]
    - ``target_input_tokens``: [B, T]
    - ``target_output_tokens``: [B, T]

    训练器负责 next-token 交叉熵、优化器、验证损失、早停和 checkpoint。
    Semantic ID 的位置约束属于生成解码阶段，不在这里修改 logits。
"""
class TigerTrainer:

    _REQUIRED_FIELDS = (
        "history_tokens",
        "history_token_mask",
        "target_input_tokens",
        "target_output_tokens",
    )

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        *,
        device: str | torch.device | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 10,
        grad_clip_norm: float | None = 5.0,
        patience: int = 3,
        checkpoint_path: str | Path | None = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model 必须是 torch.nn.Module")
        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader 必须是 torch.utils.data.DataLoader")
        if valid_loader is not None and not isinstance(valid_loader, DataLoader):
            raise TypeError("valid_loader 必须是 DataLoader 或 None")
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
            raise TypeError("learning_rate 必须是数值")
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool):
            raise TypeError("weight_decay 必须是数值")
        if weight_decay < 0:
            raise ValueError("weight_decay 不能小于 0")
        if not isinstance(epochs, int) or isinstance(epochs, bool):
            raise TypeError("epochs 必须是整数")
        if epochs < 1:
            raise ValueError("epochs 必须大于等于 1")
        if grad_clip_norm is not None:
            if not isinstance(grad_clip_norm, (int, float)) or isinstance(grad_clip_norm, bool):
                raise TypeError("grad_clip_norm 必须是数值或 None")
            if grad_clip_norm <= 0:
                raise ValueError("grad_clip_norm 必须大于 0 或设置为 None")
        if not isinstance(patience, int) or isinstance(patience, bool):
            raise TypeError("patience 必须是整数")
        if patience < 0:
            raise ValueError("patience 不能小于 0")

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = epochs
        self.grad_clip_norm = (None if grad_clip_norm is None else float(grad_clip_norm))
        self.patience = patience
        self.device = self._resolve_device(device)
        self.model.to(self.device)

        # AdamW优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,)

        self.checkpoint_path = (Path(checkpoint_path) if checkpoint_path is not None else None)
        self.run_config = dict(run_config) if run_config is not None else None
        self.history: list[dict[str, float | int | None]] = []
        self.best_epoch: int | None = None
        self.best_metric: float | None = None

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA 不可用，TIGER 训练将自动切换到 CPU。")
            return torch.device("cpu")
        return resolved_device

    """
    解包
    从 Dataset 返回的字典中取出训练需要的四个字段
    """
    @classmethod
    def _unpack_batch(cls, batch: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("TIGER batch 必须是包含字段的 Mapping")

        missing = [field for field in cls._REQUIRED_FIELDS if field not in batch]
        if missing:
            raise KeyError(f"batch 缺少字段：{', '.join(missing)}")

        values = tuple(batch[field] for field in cls._REQUIRED_FIELDS)
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError("TIGER batch 的输入字段必须是 torch.Tensor")

        return values  # type: ignore[return-value]

    """
    解包 -> 移动到设备顺便转换dtype -> 检查形状
    """
    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            history_tokens,
            history_token_mask,
            target_input_tokens,
            target_output_tokens,
        ) = self._unpack_batch(batch)

        history_tokens = history_tokens.to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,)
        history_token_mask = history_token_mask.to(
            device=self.device,
            dtype=torch.bool,
            non_blocking=True,)
        target_input_tokens = target_input_tokens.to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,)
        target_output_tokens = target_output_tokens.to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,)

        if history_tokens.ndim != 2:
            raise ValueError("history_tokens 必须是二维张量，"f"实际形状为 {tuple(history_tokens.shape)}")
        if history_token_mask.shape != history_tokens.shape:
            raise ValueError("history_token_mask 必须与 history_tokens 形状一致")
        if target_input_tokens.ndim != 2:
            raise ValueError("target_input_tokens 必须是二维张量，"f"实际形状为 {tuple(target_input_tokens.shape)}")
        if target_output_tokens.shape != target_input_tokens.shape:
            raise ValueError("target_output_tokens 必须与 target_input_tokens 形状一致")

        batch_size = history_tokens.shape[0]
        if target_input_tokens.shape[0] != batch_size:
            raise ValueError("TIGER batch 中各字段的 batch size 不一致")

        return (
            history_tokens,
            history_token_mask,
            target_input_tokens,
            target_output_tokens,
        )

    def _compute_loss(
        self,
        history_tokens: Tensor,
        history_token_mask: Tensor,
        target_input_tokens: Tensor,
        target_output_tokens: Tensor,
    ) -> Tensor:
        if not hasattr(self.model, "compute_loss"):
            raise AttributeError("model 必须提供 compute_loss() 方法")

        loss = self.model.compute_loss(
            history_tokens=history_tokens,
            history_token_mask=history_token_mask,
            target_input_tokens=target_input_tokens,
            target_output_tokens=target_output_tokens,)

        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise ValueError("model.compute_loss() 必须返回标量 Tensor")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"检测到非有限 loss：{loss.item()}")
        return loss

    """训练一个 epoch，返回按样本数加权的平均 loss。"""
    def train_epoch(self) -> float:
        # 切换训练模式
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            # 解包、移动到设备、转换数据类型、检查
            (
                history_tokens,
                history_token_mask,
                target_input_tokens,
                target_output_tokens,
            ) = self._move_batch(batch)

            # 清空梯度
            self.optimizer.zero_grad(set_to_none=True)
            # 计算loss
            loss = self._compute_loss(
                history_tokens,
                history_token_mask,
                target_input_tokens,
                target_output_tokens,
            )
            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_norm,
                )
            # 更新参数
            self.optimizer.step()

            # 统计，用于计算平均loss
            batch_size = int(history_tokens.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("train_loader 为空")
        return total_loss / total_samples

    """计算验证集或其他数据集的平均 loss。"""
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> float:

        if not isinstance(data_loader, DataLoader):
            raise TypeError("data_loader 必须是 DataLoader")

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for batch in data_loader:
            (
                history_tokens,
                history_token_mask,
                target_input_tokens,
                target_output_tokens,
            ) = self._move_batch(batch)
            loss = self._compute_loss(
                history_tokens,
                history_token_mask,
                target_input_tokens,
                target_output_tokens,
            )

            batch_size = int(history_tokens.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("data_loader 为空")
        return total_loss / total_samples

    """保存模型、优化器、训练历史和实验配置。"""
    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        train_loss: float,
        valid_loss: float | None,
    ) -> None:

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": float(train_loss),
            "valid_loss": (None if valid_loss is None else float(valid_loss)),
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "device": str(self.device),
            "trainer_config": {
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
                "grad_clip_norm": self.grad_clip_norm,
                "patience": self.patience,},
            "run_config": self.run_config,
        }
        torch.save(checkpoint, checkpoint_path)

    """加载模型 checkpoint。"""
    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:

        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,)

        if "model_state_dict" not in checkpoint:
            raise ValueError("checkpoint 缺少 model_state_dict")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if isinstance(checkpoint.get("history"), list):
            self.history = checkpoint["history"]
        self.best_epoch = checkpoint.get("best_epoch")
        self.best_metric = checkpoint.get("best_metric")
        return checkpoint

    """完整训练流程，默认根据验证 loss 选择最佳模型。"""
    def fit(self) -> list[dict[str, float | int | None]]:

        best_metric = float("inf")
        best_state_dict: dict[str, Tensor] | None = None
        best_epoch: int | None = None
        bad_epochs = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()

            if self.valid_loader is not None:
                valid_loss = self.evaluate(self.valid_loader)
                current_metric = valid_loss
            else:
                valid_loss = None
                current_metric = train_loss

            record: dict[str, float | int | None] = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "valid_loss": (
                    None if valid_loss is None else float(valid_loss)),}

            self.history.append(record)

            message = (
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"train_loss={train_loss:.6f}")

            if valid_loss is not None:
                message += f" | valid_loss={valid_loss:.6f}"
            print(message)

            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch
                bad_epochs = 0
                best_state_dict = copy.deepcopy(self.model.state_dict())

                self.best_epoch = best_epoch
                self.best_metric = best_metric

                if self.checkpoint_path is not None:
                    self.save_checkpoint(
                        self.checkpoint_path,
                        epoch=epoch,
                        train_loss=train_loss,
                        valid_loss=valid_loss,)
            else:
                bad_epochs += 1

            if (self.valid_loader is not None and bad_epochs >= self.patience):
                print(f"Early stopping：连续 {self.patience} 个 epoch 验证集没有改善。")
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.best_epoch = best_epoch
        self.best_metric = best_metric
        return self.history


TIGERTrainer = TigerTrainer


__all__ = [
    "set_seed",
    "TigerTrainer",
    "TIGERTrainer",
]
