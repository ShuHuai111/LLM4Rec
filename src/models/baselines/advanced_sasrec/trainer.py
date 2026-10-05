from __future__ import annotations

import torch
import numpy as np
import random
import copy

from typing import Any
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from collections.abc import Mapping
def set_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed 必须是整数")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""
SASRec 训练器。
每个 batch 应该包含：
    input_ids:  [B, L]
    target_ids: [B]
其中：
    input_ids  是用户历史序列
    target_ids 是下一个真实物品
"""
class SASRecTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader | None = None,
        *,
        device: str | torch.device | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 20,
        grad_clip_norm: float | None = 5.0, # 梯度裁剪的范数阈值
        patience: int = 3, # 早停耐心值
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model 必须是 torch.nn.Module")
        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader 必须是 torch.utils.data.DataLoader")
        if valid_loader is not None and not isinstance(valid_loader, DataLoader):
            raise TypeError("valid_loader 必须是 DataLoader 或 None")
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须大于 0")
        if weight_decay < 0:
            raise ValueError("weight_decay 不能小于 0")
        if not isinstance(epochs, int) or isinstance(epochs, bool):
            raise TypeError("epochs 必须是整数")
        if epochs < 1:
            raise ValueError("epochs 必须大于等于 1")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm 必须大于 0 或设置为 None")
        if patience < 0:
            raise ValueError("patience 不能小于 0")

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.patience = patience

        # 设备自动适配
        self.device = self._resolve_device(device)
        self.model.to(self.device)

        # 损失函数与优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), # 代表优化模型里所有可训练参数
            lr=learning_rate,
            weight_decay=weight_decay,)

        # 早停与检查点机制
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None)
        self.history: list[dict[str, float]] = []
        self.best_epoch: int | None = None
        self.best_metric: float | None = None

    @staticmethod
    def _resolve_device(device: str | torch.device | None,) -> torch.device:
        # 未指定设备：自动优先选择显卡
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 指定了设备：解析 + 可用性校验
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA 不可用，训练将自动切换到 CPU。")
            return torch.device("cpu")

        return resolved_device

    """
    批次数据格式适配工具
            把「元组」或「字典」两种常见打包方式的批次数据，
            统一解包成标准的 `(input_ids, target_ids)` 张量对
            
    兼容以下 batch 格式：
            1. (input_ids, target_ids)
            2. {"input_ids": ..., "target_ids": ...}
    """
    @staticmethod
    def _unpack_batch(batch: Any,) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, Mapping):
            if "input_ids" not in batch:
                raise KeyError("batch 中缺少 input_ids")
            if "target_ids" not in batch:
                raise KeyError("batch 中缺少 target_ids")

            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]

        # 支持长度为 2 的元组或列表,直接按位置解包，第一个元素是输入序列，第二个是目标标签。
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            input_ids, target_ids = batch

        # 非法格式拦截
        else:
            raise TypeError(
                "batch 必须是 (input_ids, target_ids)或包含这两个字段的 Mapping")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids 必须是 torch.Tensor")
        if not isinstance(target_ids, torch.Tensor):
            raise TypeError("target_ids 必须是 torch.Tensor")
        return input_ids, target_ids

    """解析 batch 并移动到训练设备。"""
    def _move_batch(self,batch: Any,) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_ids = self._unpack_batch(batch)

        # non_blocking=True异步执行，不用阻塞 CPU 的后续计算
        input_ids = input_ids.to(device=self.device, dtype=torch.long, non_blocking=True,)
        target_ids = target_ids.to(device=self.device, dtype=torch.long, non_blocking=True,)

        # 形状是 `[batch_size, sequence_length]`
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids 必须是二维张量，实际形状为 {tuple(input_ids.shape)}")
        if target_ids.ndim != 1:
            raise ValueError(f"target_ids 必须是一维张量，实际形状为 {tuple(target_ids.shape)}")
        if input_ids.shape[0] != target_ids.shape[0]:
            raise ValueError("input_ids 和 target_ids 的 batch size 不一致")
        if (target_ids <= 0).any():
            raise ValueError("target_ids 必须是大于 0 的物品索引，0 只用于 padding")
        return input_ids, target_ids

    """执行一次前向传播并计算交叉熵损失。"""
    def _compute_loss(self,input_ids: torch.Tensor,target_ids: torch.Tensor,) -> torch.Tensor:
        # 模型前向传播，得到预测得分
        logits = self.model(input_ids)

        # 交叉熵损失要求预测值必须是二维张量[批次大小，类别]
        if logits.ndim != 2:
            raise ValueError(f"模型输出必须是二维 logits，实际形状为 {tuple(logits.shape)}")
        if logits.shape[0] != target_ids.shape[0]:
            raise ValueError("logits 和 target_ids 的 batch size 不一致")
        if target_ids.max().item() >= logits.shape[1]:
            raise ValueError("target_ids 中存在超出模型物品数量范围的索引")

        # 计算交叉熵损失
        loss = self.criterion(logits, target_ids)

        # 会同时排除 `NaN`（非数值）、`inf`（正无穷）、`-inf`（负无穷）三种异常值
        if not torch.isfinite(loss):
            raise FloatingPointError(f"检测到非有限 loss：{loss.item()}")
        return loss

    """训练一个 epoch，返回平均训练损失。"""
    def train_epoch(self) -> float:
        # 切换为训练模式
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            # 一次性完成「格式解包→设备迁移→类型强制→全维度校验」
            input_ids, target_ids = self._move_batch(batch)

            # 清空优化器中所有参数的历史梯度
            # 默认写法zero_grad()是把梯度张量的值全部置为0
            # set_to_none=True是直接把梯度张量置为 `None`，释放对应内存，计算速度更快
            self.optimizer.zero_grad(set_to_none=True)

            loss = self._compute_loss(input_ids=input_ids, target_ids=target_ids)

            loss.backward()

            # 可选的梯度裁剪
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm,)

            # 参数更新
            self.optimizer.step()

            # 累计损失与样本数
            batch_size = input_ids.shape[0]
            total_loss += loss.detach().item()*batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("train_loader 为空")

        # 返回本轮平均训练损失
        return total_loss / total_samples

    """在验证集或其他数据集上计算平均损失。"""
    @torch.no_grad() # PyTorch 的上下文装饰器，在整个方法执行期间全局禁用梯度计算
    def evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        for batch in data_loader:
            input_ids, target_ids = self._move_batch(batch)
            loss = self._compute_loss(input_ids=input_ids, target_ids=target_ids)

            batch_size = input_ids.shape[0]
            total_loss += loss.detach().item()*batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("data_loader 为空")
        return total_loss / total_samples

    """
    保存模型、优化器和训练状态。
    将当前轮次的模型权重、优化器状态、训练指标完整序列化保存到磁盘，形成一个检查点文件
    """
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

        # 组装检查点字典
        checkpoint = {
            "epoch": epoch, # 记录当前训练到第几轮
            "model_state_dict": self.model.state_dict(), # 模型的所有权重参数
            "optimizer_state_dict": self.optimizer.state_dict(), # 优化器的状态（动量、累积梯度等）
            "train_loss": train_loss, # 记录当前轮的指标
            "valid_loss": valid_loss,
            "device": str(self.device), # 记录训练时的设备
        }

        torch.save(checkpoint, checkpoint_path)

    """加载模型检查点。"""
    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

        checkpoint = torch.load(checkpoint_path,map_location=self.device,)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    """
    完整训练流程。
    
    如果提供 valid_loader：
        根据 valid_loss 保存最佳模型并进行 Early Stopping。
        
    如果没有提供 valid_loader：
        根据 train_loss 保存最佳模型，但不进行真正的验证。
    """
    def fit(self) -> list[dict[str, float]]:
        best_metric = float('inf')
        best_state_dict: dict[str, torch.Tensor] | None = None
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

            record: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),}
            if valid_loss is not None:
                record["valid_loss"] = float(valid_loss)
            self.history.append(record)

            print(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "f"train_loss={train_loss:.6f}"
                + (f" | valid_loss={valid_loss:.6f}"if valid_loss is not None else ""))

            if current_metric < best_metric:
                best_metric = current_metric
                best_epoch = epoch
                bad_epochs = 0

                best_state_dict = copy.deepcopy(self.model.state_dict())

                if self.checkpoint_path is not None:
                    self.save_checkpoint(
                        self.checkpoint_path,
                        epoch=epoch,
                        train_loss=train_loss,
                        valid_loss=valid_loss,
                    )
            else:
                bad_epochs += 1

            if (self.valid_loader is not None and bad_epochs >= self.patience):
                print(
                    f"Early stopping：连续 {self.patience} 个 epoch "
                    "验证集没有改善。")
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
        self.best_epoch = best_epoch
        self.best_metric = best_metric

        return self.history

    """
    专门用于预测服务场景,根据输入序列返回所有物品的预测分数。
    输入：
        input_ids: [B, L]
    输出：
        logits: [B, num_items + 1]
    """
    @torch.no_grad()
    def predict_logits(
            self,
            input_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        input_ids = input_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        return self.model(input_ids)






