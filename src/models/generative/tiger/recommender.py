from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from src.representations.semantic_id.mapper import SemanticIDMapper

from .tiger import TIGER, TigerConfig

"""使用训练好的 TIGER 模型生成 Semantic ID 推荐结果。

推荐器与 ``TigerTrainer`` 分开：Trainer 只负责训练和保存 checkpoint，
本类负责线上或离线推理阶段的历史构造、约束解码和 item_id 映射。

生成过程遵循固定的 Semantic ID 结构：

- 第 0 个目标 token 只能来自 codebook 0；
- 第 1 个目标 token 只能来自 codebook 1；
- 依次类推；
- 最后一个目标 token 强制为 EOS。

这样可以避免模型生成 PAD、BOS、分隔符或错误 codebook 的 token。
"""
class TigerRecommender:
    def __init__(
        self,
        model: TIGER,
        mapper: SemanticIDMapper,                   # Semantic ID 映射器 item_id ↔ codes ↔ Semantic ID ↔ token
        *,
        max_seq_len: int | None = None,
        device: str | torch.device | None = None,
        beam_width: int = 32,                       # Beam Search 每一步保留多少条候选路径。
    ) -> None:
        if not isinstance(model, TIGER):
            raise TypeError("model 必须是 TIGER")
        if not isinstance(mapper, SemanticIDMapper):
            raise TypeError("mapper 必须是 SemanticIDMapper")
        if not isinstance(beam_width, int) or isinstance(beam_width, bool):
            raise TypeError("beam_width 必须是整数")
        if beam_width < 1:
            raise ValueError("beam_width 必须大于 0")

        item_stride = mapper.num_codebooks + 1
        if model.target_token_length != mapper.num_codebooks + 1:
            raise ValueError("模型 target_token_length 必须等于 num_codebooks + 1，"
                            f"实际为{model.target_token_length}和{mapper.num_codebooks + 1}")

        if max_seq_len is None:
            if model.max_history_tokens % item_stride != 0:
                raise ValueError("model.max_history_tokens 不能被每个物品的 token 数整除")
            max_seq_len = model.max_history_tokens // item_stride

        if not isinstance(max_seq_len, int) or isinstance(max_seq_len, bool):
            raise TypeError("max_seq_len 必须是整数或 None")
        if max_seq_len < 1:
            raise ValueError("max_seq_len 必须大于 0")
        expected_history_tokens = max_seq_len * item_stride
        if model.max_history_tokens != expected_history_tokens:
            raise ValueError("max_seq_len 与模型的 max_history_tokens 不匹配："
                            f"期望 {expected_history_tokens}，"
                            f"实际 {model.max_history_tokens}")

        if mapper.vocab_size != model.vocab_size:
            raise ValueError("Semantic ID 映射表与模型词表大小不一致："
                            f"mapper={mapper.vocab_size}, model={model.vocab_size}")
        if mapper.pad_token_id != model.pad_token_id:
            raise ValueError("mapper.pad_token_id 与 model.pad_token_id 不一致")
        if mapper.bos_token_id != model.bos_token_id:
            raise ValueError("mapper.bos_token_id 与 model.bos_token_id 不一致")
        if mapper.eos_token_id != model.eos_token_id:
            raise ValueError("mapper.eos_token_id 与 model.eos_token_id 不一致")

        self.model = model
        self.mapper = mapper
        self.max_seq_len = max_seq_len
        self.beam_width = beam_width
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

        # 只在 prepare_history() 中建立一次，避免每个用户推荐时重新扫描整张表。
        self._history_by_user: dict[str, tuple[str, ...]] | None = None

    """
    决定模型运行在 CPU 还是 GPU。
    """
    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            print("CUDA 不可用，TIGER 推荐器自动切换到 CPU。")
            return torch.device("cpu")
        return resolved

    """
    从 Trainer 保存的 checkpoint 构造推荐器。
    """
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        mapper: SemanticIDMapper,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
        max_seq_len: int | None = None,
        device: str | torch.device | None = None,
        beam_width: int = 32,
    ) -> "TigerRecommender":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

        resolved_device = cls._resolve_device(device)

        # 读取 checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=resolved_device,
            weights_only=False,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint 必须是字典")

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint 缺少 model_state_dict")

        if model_kwargs is None:
            run_config = checkpoint.get("run_config")
            tiger_config: Mapping[str, Any] = {}
            if isinstance(run_config, Mapping) and isinstance(
                run_config.get("tiger"), Mapping
            ):
                tiger_config = run_config["tiger"]

            model_kwargs = {
                field.name: tiger_config[field.name]
                for field in fields(TigerConfig)
                if field.name in tiger_config
            }

        allowed_fields = {field.name for field in fields(TigerConfig)}
        filtered_kwargs = {
            key: value
            for key, value in dict(model_kwargs).items()
            if key in allowed_fields
        }

        # 推荐器会从里面提取模型参数，重新创建同样结构的 TIGER
        model = TIGER(**filtered_kwargs)
        # 把 checkpoint 中保存的参数加载到新模型里
        model.load_state_dict(state_dict)

        # 最终得到可以直接推荐的 TigerRecommender
        return cls(
            model=model,
            mapper=mapper,
            max_seq_len=max_seq_len,
            device=resolved_device,
            beam_width=beam_width,
        )

    @staticmethod
    def _normalize_id(value: Any) -> str:
        if value is None or pd.isna(value):
            raise ValueError("user_id 或 item_id 不能缺失")
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("user_id 或 item_id 不能是空字符串")
        return normalized

    """
    预先构造 ``user_id -> 按时间排列的 item_id 序列`` 索引。

    这一步只遍历一次交互表。之后的单用户推荐只访问字典中的一条
    用户历史，不会反复筛选整张 DataFrame。
    """
    def prepare_history(self,history_interactions: pd.DataFrame,) -> "TigerRecommender":
        if not isinstance(history_interactions, pd.DataFrame):
            raise TypeError("history_interactions 必须是 pandas.DataFrame")

        required_columns = {"user_id", "item_id"}
        missing_columns = required_columns - set(history_interactions.columns)
        if missing_columns:
            raise ValueError(f"history_interactions 缺少字段：{sorted(missing_columns)}")
        if history_interactions.empty:
            raise ValueError("history_interactions 不能为空")

        columns = ["user_id", "item_id"]
        if "timestamp" in history_interactions.columns:
            columns.append("timestamp")

        data = history_interactions.loc[:, columns].copy()
        data["user_id"] = data["user_id"].map(self._normalize_id)
        data["item_id"] = data["item_id"].map(self._normalize_id)

        if "timestamp" in data.columns:
            numeric_timestamp = pd.to_numeric(data["timestamp"], errors="coerce")
            if numeric_timestamp.isna().any():
                datetime_timestamp = pd.to_datetime(data["timestamp"], errors="raise", utc=True)
                data["timestamp"] = datetime_timestamp.astype("int64")
            else:
                data["timestamp"] = numeric_timestamp
            data = data.sort_values(by=["user_id", "timestamp"],kind="stable",)

        known_items = set(self.mapper.item_to_codes_)
        unknown_items = sorted(set(data["item_id"]) - known_items)
        if unknown_items:
            # 新数据可能包含 Semantic ID 生成时不存在的物品，推荐时跳过它们。
            data = data[data["item_id"].isin(known_items)]

        self._history_by_user = {
            str(user_id): tuple(group["item_id"].tolist())
            for user_id, group in data.groupby("user_id", sort=False)
        }
        return self

    """
    输入：history_items = ["101", "205", "330"]
    输出：history_tokens  history_token_mask
    """
    def _build_history_tensors(self,history_items: Iterable[str | int],) -> tuple[Tensor, Tensor]:
        # 把输入的物品 ID 统一转换成字符串
        items = [str(item).strip() for item in history_items]
        # 删除未知物品
        items = [item for item in items if item in self.mapper.item_to_codes_]
        # 截取最近的历史
        items = items[-self.max_seq_len:]
        # 计算每个物品占多少个 token，一般是4个
        item_stride = self.mapper.num_codebooks + 1
        # 初始化历史 token 数组
        history_tokens = np.full(
            self.model.max_history_tokens,
            self.mapper.pad_token_id,
            dtype=np.int64,
        )
        # 初始化有效位置掩码
        history_token_mask = np.zeros(
            self.model.max_history_tokens,
            dtype=np.bool_,
        )
        # 计算左侧填充位置 如 max_seq_len = 5 实际历史长度 = 2； start_index = 5 - 2 = 3
        start_index = self.max_seq_len - len(items)
        # 遍历每个历史物品
        for offset, item_id in enumerate(items):
            sequence_index = start_index + offset               # 计算物品所在的槽位
            token_start = sequence_index * item_stride          # 计算 token 起始位置
            item_tokens = self.mapper.item_id_to_tokens(item_id)# 获取物品的 Semantic ID token
            # history_tokens[12:15] = [13, 113, 136]
            history_tokens[token_start: token_start + self.mapper.num_codebooks] = (item_tokens)
            # 写入分隔符
            history_tokens[token_start + self.mapper.num_codebooks] = (self.mapper.item_separator_token_id)
            # 标记有效 token
            history_token_mask[token_start: token_start + item_stride] = True

        return (torch.from_numpy(history_tokens).unsqueeze(0).to(self.device),
                torch.from_numpy(history_token_mask).unsqueeze(0).to(self.device),)

    """
    返回当前生成位置允许出现的 token。
    针对生成的非法token
    """
    def _allowed_token_ids(self, step: int) -> Tensor:  # step 表示当前正在生成目标序列的第几个 token。
        if step < self.mapper.num_codebooks:
            start = (self.mapper.code_token_offset + step * self.mapper.codebook_size)
            end = start + self.mapper.codebook_size
            return torch.arange(start, end, dtype=torch.long, device=self.device)

        if step == self.mapper.num_codebooks:
            return torch.tensor(
                [self.mapper.eos_token_id],
                dtype=torch.long,
                device=self.device,
            )

        raise ValueError(f"非法的生成位置：{step}")

    """
    在合法 Semantic ID token 空间内执行 beam search。
    """
    @torch.no_grad()
    def _beam_search(
        self,
        history_tokens: Tensor,
        history_token_mask: Tensor,
        *,
        beam_width: int,
        temperature: float,
    ) -> list[tuple[tuple[int, ...], float]]:
        beams: list[tuple[tuple[int, ...], float]] = [((self.mapper.bos_token_id,), 0.0)]

        for step in range(self.model.target_token_length):
            # 把当前 Beam 变成张量
            prefixes = torch.tensor(
                [prefix for prefix, _ in beams],
                dtype=torch.long,
                device=self.device,
            )
            # 为每条 Beam 复制历史输入
            batch_history = history_tokens.expand(len(beams), -1)
            batch_history_mask = history_token_mask.expand(len(beams), -1)

            # 调用 TIGER 模型
            logits = self.model(
                batch_history,
                batch_history_mask,
                prefixes,
                allow_partial_target=True,
            )[:, -1, :]
            logits = logits / temperature

            # 获取当前位置允许的 token，调用上面的方法
            allowed = self._allowed_token_ids(step)
            # 只保留合法 token 的 logits
            allowed_logits = logits.index_select(dim=1, index=allowed)
            # 转换成 log probability，连乘后概率很小，取对数换成连加
            log_probs = torch.log_softmax(allowed_logits, dim=-1)
            # 确定每条路径扩展多少个候选
            local_width = min(beam_width, int(allowed.numel()))
            # 每条 Beam 取局部 Top-K
            top_log_probs, top_positions = torch.topk(
                log_probs,
                k=local_width,
                dim=-1,
            )
            # 扩展每条路径
            expanded: list[tuple[tuple[int, ...], float]] = []      # 这个列表保存当前步骤产生的所有子路径。
            # 遍历当前所有 Beam。
            for beam_index, (prefix, score) in enumerate(beams):
                # 为每条路径扩展局部 Top-K 个 token。
                for candidate_index in range(local_width):
                    token_id = int(allowed[top_positions[beam_index, candidate_index]])
                    # 累加路径得分
                    candidate_score = score + float(top_log_probs[beam_index, candidate_index].item())
                    expanded.append((prefix + (token_id,), candidate_score))

            # 全局排序并剪枝
            expanded.sort(key=lambda value: value[1], reverse=True)
            beams = expanded[:beam_width]

        return beams

    def _decode_beam(
        self,
        prefix: tuple[int, ...],
    ) -> tuple[str, tuple[int, ...]] | None:
        if len(prefix) != self.model.target_token_length + 1:
            return None
        if prefix[0] != self.mapper.bos_token_id:
            return None
        if prefix[-1] != self.mapper.eos_token_id:
            return None

        code_tokens = prefix[1:-1]
        codes: list[int] = []
        for level, token_id in enumerate(code_tokens):
            start = (
                self.mapper.code_token_offset
                + level * self.mapper.codebook_size
            )
            end = start + self.mapper.codebook_size
            if not start <= token_id < end:
                return None
            codes.append(token_id - start)

        try:
            item_id = self.mapper.codes_to_item_id(
                np.asarray(codes, dtype=np.int64),
                strict=True,
            )
        except (KeyError, ValueError):
            return None
        return item_id, tuple(codes)

    @torch.no_grad()
    def recommend(
        self,
        seen_items: Iterable[str | int] | str | int,
        *,
        k: int = 10,
        beam_width: int | None = None,
        temperature: float = 1.0,
    ) -> list[str]:
        """根据一个用户的有序历史物品生成 Top-K 推荐。"""
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k 必须是大于等于 1 的整数")
        if beam_width is None:
            beam_width = max(self.beam_width, k * 4)
        if not isinstance(beam_width, int) or isinstance(beam_width, bool):
            raise TypeError("beam_width 必须是整数或 None")
        if beam_width < 1:
            raise ValueError("beam_width 必须大于 0")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise TypeError("temperature 必须是数值")
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")

        if isinstance(seen_items, (str, int)):
            ordered_items = [str(seen_items)]
        else:
            ordered_items = [str(item) for item in seen_items]
        known_history = [
            item.strip()
            for item in ordered_items
            if item.strip() in self.mapper.item_to_codes_
        ]
        seen_set = set(known_history)

        history_tokens, history_token_mask = self._build_history_tensors(known_history)
        beams = self._beam_search(
            history_tokens,
            history_token_mask,
            beam_width=beam_width,
            temperature=float(temperature),
        )

        recommendations: list[str] = []
        returned: set[str] = set()
        for prefix, _score in beams:
            decoded = self._decode_beam(prefix)
            if decoded is None:
                continue
            item_id, _codes = decoded
            if item_id in seen_set or item_id in returned:
                continue
            recommendations.append(item_id)
            returned.add(item_id)
            if len(recommendations) >= k:
                break
        return recommendations

    def recommend_for_user(
        self,
        user_id: str | int,
        history_interactions: pd.DataFrame | None = None,
        *,
        k: int = 10,
        beam_width: int | None = None,
        temperature: float = 1.0,
    ) -> list[str]:
        """按用户 ID 推荐；DataFrame 只需在第一次调用时传入。"""
        if history_interactions is not None:
            self.prepare_history(history_interactions)
        if self._history_by_user is None:
            raise RuntimeError("请先调用 prepare_history() 准备用户历史数据")

        normalized_user_id = self._normalize_id(user_id)
        history = self._history_by_user.get(normalized_user_id, ())
        if not history:
            return []
        return self.recommend(
            history,
            k=k,
            beam_width=beam_width,
            temperature=temperature,
        )


__all__ = ["TigerRecommender"]
