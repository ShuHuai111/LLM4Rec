from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


"""
输入文件：data/processed/ml-100k/semantic_sequences/advanced/train.npz
        data/processed/ml-100k/semantic_sequences/advanced/valid.npz
        data/processed/ml-100k/semantic_sequences/advanced/test.npz

每个.npz 文件包含 7 个数组：
        字段	                形状	            类型	            作用                    例子
        history_codes	    [N, 50, 3]	    int64	    历史物品的 Semantic ID      [[0, 0, 0],  # padding....[12,  7, 41], [ 3,  8, 22]]
        history_mask	    [N, 50]	        bool	    历史物品位置是否有效          [False,False,.....True,True,]
        history_tokens	    [N, 200]	    int64	    展平后的历史 token          [PAD, PAD, PAD, PAD,.....code_0, code_1, code_2, ITEM_SEPARATOR,]
    history_token_mask	    [N, 200]	    bool	    历史 token 是否有效
        target_codes	    [N, 3]	        int64	    目标物品的 Semantic ID
        target_input_tokens	[N, 4]	        int64	    解码器输入
    target_output_tokens	[N, 4]	        int64	    训练标签                    [code_0, code_1, code_2, EOS]
"""

_ARRAY_FIELDS = (
    "history_codes",
    "history_mask",
    "history_tokens",
    "history_token_mask",
    "target_codes",
    "target_input_tokens",
    "target_output_tokens",)

"""
读取 build_semantic_sequences.py 生成的单个 split。

每条样本包含：
- history_tokens：左侧 padding 后的历史 Semantic ID token 序列；
- history_token_mask：历史 token 的有效位置；
- target_input_tokens：BOS + 目标 Semantic ID token；
- target_output_tokens：目标 Semantic ID token + EOS。

Dataset 只负责读取和校验数据，不负责把数据移动到 GPU。
设备迁移应在训练循环中完成。
"""
class TigerSequenceDataset(Dataset):
    def __init__(self, npz_path: str | Path) -> None:
        self.npz_path = Path(npz_path)
        if not self.npz_path.is_file():
            raise FileNotFoundError(f"TIGER 序列文件不存在：{self.npz_path}")

        # 得到七个字段的字典
        arrays = self._load_arrays(self.npz_path)
        # 校验字典
        self._validate_arrays(arrays)

        # 复制到可写的连续数组，避免np.load返回的只读数组造成torch.from_numpy 的共享内存警告。
        self._arrays: dict[str, Tensor] = {
            "history_codes": self._to_long_tensor(arrays["history_codes"]),
            "history_mask": self._to_bool_tensor(arrays["history_mask"]),
            "history_tokens": self._to_long_tensor(arrays["history_tokens"]),
            "history_token_mask": self._to_bool_tensor(arrays["history_token_mask"]),
            "target_codes": self._to_long_tensor(arrays["target_codes"]),
            "target_input_tokens": self._to_long_tensor(arrays["target_input_tokens"]),
            "target_output_tokens": self._to_long_tensor(arrays["target_output_tokens"]),}

        # 保存结构信息
        self.num_samples = int(self._arrays["history_codes"].shape[0])
        self.max_seq_len = int(self._arrays["history_codes"].shape[1])
        self.num_codebooks = int(self._arrays["history_codes"].shape[2])
        self.history_token_length = int(self._arrays["history_tokens"].shape[1])
        self.target_token_length = int(self._arrays["target_input_tokens"].shape[1])

    """
    打开 .npz 文件 -> 检查7个必需字段是否存在 -> 读取数组 -> 返回字典
    """
    @staticmethod
    def _load_arrays(npz_path: Path) -> dict[str, np.ndarray]:
        with np.load(npz_path, allow_pickle=False) as data:
            missing = [field for field in _ARRAY_FIELDS if field not in data]
            if missing:
                raise ValueError(f"{npz_path} 缺少字段：{', '.join(missing)}")
            return {field: np.asarray(data[field]) for field in _ARRAY_FIELDS}

    """
    用于history_codes、history_tokens、target_codes、target_input_tokens、target_output_tokens
    """
    @staticmethod
    def _to_long_tensor(array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.int64).copy())

    """
    用于history_mask;history_token_mask
    """
    @staticmethod
    def _to_bool_tensor(array: np.ndarray) -> Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.bool_).copy())

    """
    校验七个字段是否完全正确
    """
    @staticmethod
    def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
        history_codes = arrays["history_codes"]
        history_mask = arrays["history_mask"]
        history_tokens = arrays["history_tokens"]
        history_token_mask = arrays["history_token_mask"]
        target_codes = arrays["target_codes"]
        target_input_tokens = arrays["target_input_tokens"]
        target_output_tokens = arrays["target_output_tokens"]

        if history_codes.ndim != 3:
            raise ValueError("history_codes 必须是三维数组 [N, max_seq_len, num_codebooks]")

        num_samples, max_seq_len, num_codebooks = history_codes.shape
        if num_samples == 0:
            raise ValueError("TIGER 数据集不能为空")
        if max_seq_len < 1 or num_codebooks < 1:
            raise ValueError("max_seq_len 和 num_codebooks 必须大于 0")
        # 预期[N, 50]
        expected_history_mask = (num_samples, max_seq_len)
        if history_mask.shape != expected_history_mask:
            raise ValueError("history_mask 形状错误："f"期望 {expected_history_mask}，实际 {history_mask.shape}")

        expected_history_tokens = (num_samples,max_seq_len * (num_codebooks + 1),)
        if history_tokens.shape != expected_history_tokens:
            raise ValueError("history_tokens 形状错误："f"期望 {expected_history_tokens}，实际 {history_tokens.shape}")
        if history_token_mask.shape != expected_history_tokens:
            raise ValueError("history_token_mask 形状错误："f"期望 {expected_history_tokens}，实际 {history_token_mask.shape}")

        expected_target_codes = (num_samples, num_codebooks)
        if target_codes.shape != expected_target_codes:
            raise ValueError("target_codes 形状错误："f"期望 {expected_target_codes}，实际 {target_codes.shape}")

        expected_target_tokens = (num_samples, num_codebooks + 1)
        if target_input_tokens.shape != expected_target_tokens:
            raise ValueError("target_input_tokens 形状错误："f"期望 {expected_target_tokens}，实际 {target_input_tokens.shape}")
        if target_output_tokens.shape != expected_target_tokens:
            raise ValueError("target_output_tokens 形状错误："f"期望 {expected_target_tokens}，实际 {target_output_tokens.shape}")

        for name, array in arrays.items():
            if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
                raise TypeError(f"{name} 必须是数值或布尔数组，实际为 {array.dtype}")

        for name in ("history_codes", "history_tokens", "target_codes",
                     "target_input_tokens", "target_output_tokens"):
            if np.any(arrays[name] < 0):
                raise ValueError(f"{name} 不能包含负数 token 或 code")

    """
    简化路径构造
    从序列目录读取 train、valid 或 test split。
    """
    @classmethod
    def from_directory(cls,directory: str | Path,split: str,) -> "TigerSequenceDataset":
        if split not in {"train", "valid", "test"}:
            raise ValueError("split 必须是 train、valid 或 test")
        return cls(Path(directory) / f"{split}.npz")

    def __len__(self) -> int:
        return self.num_samples

    """
    返回指定下标的一条样本。每个字段都返回对应样本的 Tensor
    """
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("index 必须是整数")
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError("Dataset index 超出范围")

        return {field: tensor[index] for field, tensor in self._arrays.items()}

    """返回 Dataset 的结构信息，便于构造模型配置。"""
    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.npz_path),
            "num_samples": self.num_samples,
            "max_seq_len": self.max_seq_len,
            "num_codebooks": self.num_codebooks,
            "history_token_length": self.history_token_length,
            "target_token_length": self.target_token_length,
        }

    """
    封装函数
    简化 Dataset 加载入口。
    """
def load_tiger_split(sequence_dir: str | Path,split: str,) -> TigerSequenceDataset:
    return TigerSequenceDataset.from_directory(sequence_dir, split)

__all__ = [
    "TigerSequenceDataset",
    "load_tiger_split",
]
