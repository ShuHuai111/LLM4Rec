from __future__ import annotations

from pathlib import Path
from typing import Any

import re

import numpy as np
import pandas as pd
import yaml

class SemanticIDMapper:
    def __init__(
        self,
        mapping_path: str | Path,           # 语义 ID 映射 CSV 文件路径
        *,
        mode: str = "advanced",
        codebook_size: int | None = None,   # 单层码本大小，不传则自动从数据中推算
    ) -> None:
        if mode not in {"baseline", "advanced"}:
            raise ValueError("mode 必须是 baseline 或 advanced")

        self.mapping_path = Path(mapping_path)
        self.mode = mode
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"找不到 Semantic ID 映射文件：{self.mapping_path}")

        mapping = pd.read_csv(self.mapping_path)
        if mapping.empty:
            raise ValueError("Semantic ID 映射表不能为空")

        required_columns = {"item_id", "semantic_id"}
        missing_columns = required_columns - set(mapping.columns)
        if missing_columns:
            raise ValueError(f"Semantic ID 映射表缺少字段：{sorted(missing_columns)}")

        mapping["item_id"] = (mapping["item_id"].astype("string").str.strip())
        if mapping["item_id"].isna().any():
            raise ValueError("item_id 不能包含缺失值")
        if mapping["item_id"].duplicated().any():
            raise ValueError("Semantic ID 映射表中的 item_id 不能重复")

        self.code_columns = self._find_code_columns(mapping.columns)

        if not self.code_columns:
            raise ValueError("Semantic ID 映射表中没有找到 code_0、code_1 等编码字段")

        # 从完整的映射表中，只提取所有量化编码列，转换成纯数值的 numpy 二维数组
        raw_codes = mapping[self.code_columns].to_numpy()

        if pd.isna(raw_codes).any():
            raise ValueError("Semantic ID code 不能包含缺失值")

        try:
            codes = raw_codes.astype(np.int64)
        except (TypeError, ValueError) as error:
            raise ValueError("Semantic ID code 必须是整数") from error

        if not np.array_equal(raw_codes, codes):
            raise ValueError("Semantic ID code 必须是整数，不能包含小数")
        if (codes < 0).any():
            raise ValueError("Semantic ID code 不能小于 0")

        # 自动推导并存储残差量化的码本总层数
        self.num_codebooks = len(self.code_columns)

        if codebook_size is None:
            codebook_size = int(codes.max()) + 1
        if codebook_size < 2:
            raise ValueError("codebook_size 必须大于等于 2")
        if (codes >= codebook_size).any():
            raise ValueError("Semantic ID code 超出 codebook_size 范围")
        # 强制将单层码本的容量标准化为 Python 原生整数并存储
        self.codebook_size = int(codebook_size)
        # semantic_id 字段格式标准化
        mapping["semantic_id"] = (mapping["semantic_id"].astype("string").str.strip())
        if mapping["semantic_id"].isna().any():
            raise ValueError("semantic_id 不能包含缺失值")
        # 一行编码 `[12, 48, 7]` 会生成标准字符串 `"12-48-7"`
        expected_semantic_ids = ["-".join(str(int(code)) for code in row)for row in codes]

        actual_semantic_ids = (mapping["semantic_id"].astype(str).tolist())
        if actual_semantic_ids != expected_semantic_ids:
            raise ValueError("semantic_id 与各层 code 拼接结果不一致")

        # 行索引统一对齐
        self.mapping = mapping.reset_index(drop=True)
        self.codes = codes.astype(np.int64)

        # 初始化三个空字典
        self.item_to_codes_: dict[str, tuple[int, ...]] = {}
        self.item_to_semantic_id_: dict[str, str] = {}
        self.semantic_id_to_items_: dict[str, list[str]] = {}

        # 逐行遍历填充
        for row_index, row in self.mapping.iterrows():
            item_id = str(row["item_id"])
            semantic_id = str(row["semantic_id"])
            item_codes = tuple(int(code) for code in self.codes[row_index])

            self.item_to_codes_[item_id] = item_codes
            self.item_to_semantic_id_[item_id] = semantic_id
            self.semantic_id_to_items_.setdefault(semantic_id,[],).append(item_id)

        if self.mode == "advanced":
            duplicated_semantic_ids = {
                semantic_id: item_ids
                for semantic_id, item_ids
                in self.semantic_id_to_items_.items()
                if len(item_ids) > 1}

            if duplicated_semantic_ids:
                examples = list(duplicated_semantic_ids.items())[:5]
                raise ValueError("advanced 模式要求 Semantic ID 唯一，"f"但发现冲突：{examples}")

        # 多层离散编码映射到一套连续、唯一、分层的整数 Token 空间，
        # 同时预留所有特殊 Token，模型可以直接用这套 Token 做输入。
        # baseline 和 advanced 都需要这些字段。
        # 0 专门保留给 PAD
        self.pad_token_id = 0
        # 编码的起始偏移
        self.code_token_offset = 1
        # 物品分隔符Token：所有层编码占满之后的第一个位置
        self.item_separator_token_id = (
                self.code_token_offset
                + self.num_codebooks * self.codebook_size)

        # 序列开始、结束标记
        self.bos_token_id = self.item_separator_token_id + 1
        self.eos_token_id = self.item_separator_token_id + 2

        # 整个词表的总大小
        self.vocab_size = self.eos_token_id + 1

    """
    找到所有的编码列
    必须是 `code_0`、`code_1`、`code_2`... 依次连续，不能缺层、跳号
    """
    @staticmethod # 不需要实例就能调用
    def _find_code_columns(columns: pd.Index,) -> list[str]:
        code_columns: list[tuple[int, str]] = []

        for column in columns:
            match = re.fullmatch(r"code_(\d+)", str(column))
            if match:
                code_columns.append((int(match.group(1)), str(column)))

        code_columns.sort(key=lambda value: value[0])

        # 连续性校验
        indices = [index for index, _ in code_columns]
        expected_indices = list(range(len(indices)))
        if indices != expected_indices:
            raise ValueError(f"Semantic ID code 列不连续：{indices}")

        return [column for _, column in code_columns]

    """
    从标准 Semantic ID 输出目录加载映射。
    目录中应包含：
        item_semantic_ids.csv
        config_snapshot.yaml
    """
    @classmethod
    def from_directory(
        cls,
        semantic_dir: str | Path,
        *,
        mode: str | None = None,
    ) -> "SemanticIDMapper":
        semantic_dir = Path(semantic_dir)
        mapping_path = semantic_dir / "item_semantic_ids.csv"
        snapshot_path = semantic_dir / "config_snapshot.yaml"

        if not snapshot_path.exists():
            raise FileNotFoundError(f"找不到配置快照：{snapshot_path}")
        with snapshot_path.open("r",encoding="utf-8-sig",) as file:
            snapshot = yaml.safe_load(file)

        if not isinstance(snapshot, dict):
            raise ValueError("配置快照必须是 YAML 字典")

        snapshot_mode = str(snapshot.get("quantizer_mode", "advanced"))
        selected_mode = mode or snapshot_mode

        quantizer_config = snapshot.get("quantizer", {})

        if not isinstance(quantizer_config, dict):
            raise ValueError("配置快照中的 quantizer 必须是字典")

        codebook_size = int(quantizer_config.get("codebook_size", 64))

        return cls(mapping_path, mode=selected_mode, codebook_size=codebook_size,)

    @property
    def item_ids(self) -> list[str]:
        return list(self.item_to_codes_.keys())
    @property
    def semantic_ids(self) -> list[str]:
        return list(self.semantic_id_to_items_.keys())

    """
    物品 ID → 多层整数编码数组-------原子方法------
    输入： 一个物品 ID（传字符串、数字都可以）
    输出：numpy 一维 int64 数组，例如 `array([12, 48, 7], dtype=int64)`
    """
    def item_id_to_codes(self, item_id: str,) -> np.ndarray:
        item_id = str(item_id).strip()
        if item_id not in self.item_to_codes_:
            raise KeyError(f"item_id 不存在于 Semantic ID 映射表：{item_id}")
        return np.asarray(self.item_to_codes_[item_id],dtype=np.int64,)
    """
    物品 ID → 语义 ID 字符串 -----原子方法-----
    输入：一个物品 ID
    标准语义 ID 字符串，例如 "12-48-7"
    """
    def item_id_to_semantic_id(self,item_id: str) -> str:
        item_id = str(item_id).strip()
        if item_id not in self.item_to_semantic_id_:
            raise KeyError(f"item_id 不存在于 Semantic ID 映射表：{item_id}")
        return self.item_to_semantic_id_[item_id]

    """
    多层编码 → 语义 ID 字符串
    """
    def codes_to_semantic_id(self,codes: Any,) -> str:
        codes = np.asarray(codes)
        if codes.ndim != 1:
            raise ValueError("codes 必须是一维数组")
        if len(codes) != self.num_codebooks:
            raise ValueError("codes 长度与 num_codebooks 不一致")
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError("codes 必须是整数")
        codes = codes.astype(np.int64)
        if (codes < 0).any() or (codes >= self.codebook_size).any():
            raise ValueError("codes 存在超出码本范围的值")
        return "-".join(str(int(code)) for code in codes)

    """
    语义 ID → 物品 ID
    - 输入：语义 ID 字符串，例如 `"12-48-7"`
    - 输出：对应的物品 ID 字符串
    """
    def semantic_id_to_item_id(
        self,
        semantic_id: str,
        *,
        strict: bool = True,
    ) -> str:
        semantic_id = str(semantic_id).strip()
        item_ids = self.semantic_id_to_items_.get(semantic_id, [])
        if not item_ids:
            raise KeyError(f"Semantic ID 不存在：{semantic_id}")
        if len(item_ids) > 1 and strict:
            raise ValueError(f"Semantic ID 存在冲突，无法唯一还原：{semantic_id} -> {item_ids}")
        return item_ids[0]

    """
    ------------------------组合方法----------------------
    - 输入：多层编码数组 / 列表，例如 `[12, 48, 7]`
    - 输出：对应的物品 ID 字符串
    """
    def codes_to_item_id(self,codes: Any,*,strict: bool = True,) -> str:
        semantic_id = self.codes_to_semantic_id(codes)
        return self.semantic_id_to_item_id(semantic_id,strict=strict,)

    """
    物品 ID → 模型 Token 数组---------组合方法------------
    组合调用item_id_to_codes和item_id_to_semantic_id
    输入：一个物品 ID
    输出：numpy 一维 int64 Token 数组，例如 `array([13, 113, 136], dtype=int64)`
    """
    def item_id_to_tokens(self,item_id: str,) -> np.ndarray:
        codes = self.item_id_to_codes(item_id)
        return self.codes_to_tokens(codes)

    """
    输入编码 `[12, 48, 7]`：
        - 第 0 层 Token：`1 + 0×64 + 12 = 13`
        - 第 1 层 Token：`1 + 1×64 + 48 = 113`
        - 第 2 层 Token：`1 + 2×64 + 7 = 136`
        - 最终输出：`array([13, 113, 136], dtype=int64)`
    """
    def codes_to_tokens(self, codes: Any) -> np.ndarray:
        codes = np.asarray(codes)

        if codes.ndim != 1:
            raise ValueError("codes 必须是一维数组")
        if len(codes) != self.num_codebooks:
            raise ValueError("codes 长度与 num_codebooks 不一致")
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError("codes 必须是整数")
        codes = codes.astype(np.int64)
        if (codes < 0).any() or (codes >= self.codebook_size).any():
            raise ValueError("codes 存在超出码本范围的值")

        # 核心分层转换
        tokens = (self.code_token_offset+ np.arange(self.num_codebooks) * self.codebook_size+ codes)
        return tokens.astype(np.int64)

    """
    将交互表中的 item_id 转换成 code_0、code_1 等字段。
    把整个交互数据表（比如用户点击、收藏、购买记录）里的 `item_id` 批量转换成多层编码和语义ID，
    直接输出模型训练 / 推理可用的完整 DataFrame。
    
    相当于把逐行调用 `item_id_to_codes` + `item_id_to_semantic_id` 的逻辑做了批量封装，
    是对接推荐模型数据流水线的核心工具。
    
    user_id	    item_id
    101	        5
    101	        12
    102	        7
    user_id	    item_id	    code_0	    code_1	    code_2	    semantic_id
    101	        5	        12	        48	        7	        12-48-7
    101	        12	        3	        15	        22	        3-15-22
    102	        7	        45	        2	        59	        45-2-59
    """
    def transform_interactions(self,interactions: pd.DataFrame,) -> pd.DataFrame:
        if not isinstance(interactions, pd.DataFrame):
            raise TypeError("interactions 必须是 pandas.DataFrame")
        if interactions.empty:
            raise ValueError("interactions 不能为空")
        if "item_id" not in interactions.columns:
            raise ValueError("interactions 缺少 item_id 字段")

        result = interactions.copy()
        result["item_id"] = (result["item_id"].astype("string").str.strip())
        if result["item_id"].isna().any():
            raise ValueError("interactions.item_id 不能包含缺失值")

        item_ids = result["item_id"].astype(str).tolist()
        unknown_item_ids = sorted(set(item_ids) - set(self.item_to_codes_))
        if unknown_item_ids:
            raise KeyError("以下 item_id 不在 Semantic ID 映射表中："f"{unknown_item_ids[:10]}")

        codes = np.asarray([self.item_to_codes_[item_id]for item_id in item_ids],dtype=np.int64,)
        # 逐层追加 code_0、code_1... 列
        for code_index, column in enumerate(self.code_columns):
            result[column] = codes[:, code_index]
        # 追加 semantic_id 列
        result["semantic_id"] = [self.item_to_semantic_id_[item_id] for item_id in item_ids]
        return result



"""
    将交互数据转换成 Semantic ID 序列样本。

    训练样本形式：
        history -> target

    输出字段包括：
        history_codes           历史行为物品的多层编码数组
        history_mask            历史序列的有效掩码，标记哪些位置是真实行为、哪些是填充
        history_tokens          历史行为的分层 Token 数组，可直接输入模型
        target_codes            目标物品的多层编码
        target_input_tokens     目标端的输入 Token 序列
        target_output_tokens    目标端的输出标签 Token 序列
"""


__all__ = [
    "SemanticIDMapper",
]
