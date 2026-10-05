from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

"""
用户历史 Semantic ID token -> 因果 Transformer -> 自回归预测目标物品的 Semantic ID token
训练时,
    输入：BOS    + code_0 + code_1 + code_2
    标签：code_0 + code_1 + code_2 + EOS
"""

"""
配置容器，用来保存模型超参数。
"""
@dataclass(frozen=True)
class TigerConfig:
    vocab_size: int = 196               # token 总数 ；3×64 = 192，加上各种特殊符号
    max_history_tokens: int = 200       # 历史 token；最大长度3个Semantic ID code + 1个分隔符 = 4个token，序列最长50
    target_token_length: int = 4        # 目标 token 长度；BOS + 3 个目标 code = 4 个 token
    embedding_dim: int = 64             # token 向量维度
    num_heads: int = 2                  # 注意力头数量
    num_layers: int = 2                 # Transformer 层数
    feed_forward_dim: int = 128         # 前馈网络隐藏层维度
    dropout: float = 0.1                # dropout 比例
    pad_token_id: int = 0               # padding token
    bos_token_id: int | None = 194      # 序列开始 token
    eos_token_id: int | None = 195      # 序列结束 token


class TIGER(nn.Module):
    """
    初始化部分:
    1. 检查配置
    2. 创建 token embedding
    3. 创建位置 embedding
    4. 创建 Transformer
    5. 创建输出层
    6. 创建因果 attention mask
    """
    def __init__(
        self,
        *,
        vocab_size: int = 196,
        max_history_tokens: int = 200,
        target_token_length: int = 4,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        feed_forward_dim: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int | None = 194,
        eos_token_id: int | None = 195,
    ) -> None:
        super().__init__()

        self._validate_constructor_arguments(
            vocab_size=vocab_size,
            max_history_tokens=max_history_tokens,
            target_token_length=target_token_length,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            feed_forward_dim=feed_forward_dim,
            dropout=dropout,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        self.vocab_size = vocab_size
        self.max_history_tokens = max_history_tokens
        self.target_token_length = target_token_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.feed_forward_dim = feed_forward_dim
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # 创建位置编码，这里先与基础版SASRec同步，额外用一个位置向量
        self.max_position_embeddings = (max_history_tokens + target_token_length)
        self.position_embedding = nn.Embedding(
            num_embeddings=self.max_position_embeddings,
            embedding_dim=embedding_dim,
        )

        # 创建 token embedding。作用是把离散 token 转换成连续向量。输入：[12, 7, 41] ；输出：[embedding(12), embedding(7), embedding(41)]
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,      # 词表大小196
            embedding_dim=embedding_dim,    # 向量embedding(idx)维度64
            padding_idx=pad_token_id,       # 设定token0为专门的 padding token，不参与梯度更新
        )

        # Transformer编码器，还需要去详细看它的结构
        # 当前使用gelu
        # Pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,              # 隐藏维度，每个 Token 向量的长度
            nhead=num_heads,                    # 多头自注意力的头数
            dim_feedforward=feed_forward_dim,   # 前馈网络（FFN）的中间层维度
            dropout=dropout,                    # Dropout 概率
            activation="gelu",                  # FFN 的激活函数
            batch_first=True,                   # 输入张量维度顺序为 [batch_size, seq_len, d_model]
            norm_first=True,                    # 开启预归一化（Pre-LN）结构
        )
        # 实现多层encoder堆叠
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,        # 上面的基础编码器单元
            num_layers=num_layers,              # 编码器堆叠层数
            norm=nn.LayerNorm(embedding_dim),   # 编码器最终输出的归一化层，用了pre-LN,后面最好再跟一层
            enable_nested_tensor=False,         # PyTorch 的 NestedTensor 用于加速带 padding 的注意力计算，但兼容性一般。
        )
        # 对 Transformer 输出做最后一次归一化，让输出分布更稳定
        self.output_norm = nn.LayerNorm(embedding_dim)
        # 作用是把隐藏向量转换成每个 token 的预测分数。输入：[B, 204, 64] 输出：[B, 204, 196]
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

        # 输入输出 embedding 权重共享
        self.lm_head.weight = self.token_embedding.weight

        # 因果掩码矩阵
        causal_mask = torch.triu(
            torch.ones(
                self.max_position_embeddings,
                self.max_position_embeddings,
                dtype=torch.bool,),diagonal=1,)

        self.register_buffer(
            "causal_mask",
            causal_mask,
            persistent=False,
        )

        self.reset_parameters()

    """
    查模型初始化参数是否合法。
    """
    @staticmethod
    def _validate_constructor_arguments(
        *,
        vocab_size: int,
        max_history_tokens: int,
        target_token_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        feed_forward_dim: int,
        dropout: float,
        pad_token_id: int,
        bos_token_id: int | None,
        eos_token_id: int | None,
    ) -> None:
        integer_arguments = {
            "vocab_size": vocab_size,
            "max_history_tokens": max_history_tokens,
            "target_token_length": target_token_length,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "feed_forward_dim": feed_forward_dim,
        }
        for name, value in integer_arguments.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} 必须是整数")
            if value < 1:
                raise ValueError(f"{name} 必须大于 0")

        if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool):
            raise TypeError("pad_token_id 必须是整数")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim 必须能被 num_heads 整除")
        if not isinstance(dropout, (float, int)) or isinstance(dropout, bool):
            raise TypeError("dropout 必须是数值")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间")

        for name, token_id in (("bos_token_id", bos_token_id),("eos_token_id", eos_token_id),):
            if token_id is not None:
                if not isinstance(token_id, int) or isinstance(token_id, bool):
                    raise TypeError(f"{name} 必须是整数或 None")
                if not 0 <= token_id < vocab_size:
                    raise ValueError(f"{name} 必须位于 [0, vocab_size) 内")

        if not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id 必须位于 [0, vocab_size) 内")

    """
    初始化 token 和位置 embedding。
    使用均值为 0、标准差为 0.02 的正态分布初始化
    将 PAD 对应的 embedding 清零
    """
    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.pad_token_id].zero_()

    """
    校验每次 forward 的输入。
    """
    def _validate_inputs(
        self,
        history_tokens: Tensor,
        history_token_mask: Tensor,
        target_input_tokens: Tensor,
        *,
        allow_partial_target: bool = False,
    ) -> None:
        if history_tokens.ndim != 2:
            raise ValueError("history_tokens 必须是二维张量 [B, H]")
        if history_token_mask.shape != history_tokens.shape:
            raise ValueError("history_token_mask 必须与 history_tokens 形状一致")
        if target_input_tokens.ndim != 2:
            raise ValueError("target_input_tokens 必须是二维张量 [B, T]")
        if history_tokens.shape[0] != target_input_tokens.shape[0]:
            raise ValueError("history_tokens 和 target_input_tokens 的 batch 不一致")

        target_length = target_input_tokens.shape[1]
        if target_length < 1 or target_length > self.target_token_length:
            raise ValueError("target_input_tokens 长度错误："f"合法范围是 [1, {self.target_token_length}]，"f"实际 {target_length}")
        if not allow_partial_target and target_length != self.target_token_length:
            raise ValueError("训练模式下 target_input_tokens 必须使用完整长度："f"{self.target_token_length}")

        for name, tokens in (("history_tokens", history_tokens),("target_input_tokens", target_input_tokens),):
            integer_dtypes = (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,)
            if tokens.dtype not in integer_dtypes:
                raise TypeError(f"{name} 必须是整数张量")
            if torch.any(tokens < 0) or torch.any(tokens >= self.vocab_size):
                raise ValueError(f"{name} 包含越界 token")

        if history_token_mask.dtype != torch.bool:
            raise TypeError("history_token_mask 必须是 torch.bool")

    """
    核心编码流程
    编码历史和目标输入组成的因果前缀。
    """
    def _encode_prefix(
        self,
        history_tokens: Tensor,         # [B, 200]
        history_token_mask: Tensor,
        target_input_tokens: Tensor,    # [B, 4]
        *,
        allow_partial_target: bool = False,
    ) -> Tensor:
        # 调用前面的方法，校验输入
        self._validate_inputs(
            history_tokens,
            history_token_mask,
            target_input_tokens,
            allow_partial_target=allow_partial_target,)

        # 拼接历史和目标输入 dim=1 表示沿序列长度维度拼接。
        # 如history_tokens: [32, 200],target_input_tokens: [32,4]
        # 得到prefix_tokens: [32, 204]
        # 两个拼接时[PAD, PAD, PAD, PAD, 13, 72, 170, SEP, BOS, 6, 76, 159]
        prefix_tokens = torch.cat([history_tokens, target_input_tokens],dim=1,)

        # 拼接掩码（这只是目标位置有效掩码，还有因果掩码）
        # 历史有效性由 history_token_mask 决定[F, F, F, F, T, T, T, T]
        # 目标输入全部是有效位置，因此使用：torch.ones(...)[T, T, T, T]
        # 拼接后[F, F, F, F, T, T, T, T, T, T, T, T]形状与 prefix_tokens 完全一致
        prefix_mask = torch.cat([history_token_mask,
                                    torch.ones(
                                        target_input_tokens.shape,
                                        dtype=torch.bool,
                                        device=target_input_tokens.device,),],
                                    dim=1,)

        sequence_length = prefix_tokens.shape[1]
        if sequence_length > self.max_position_embeddings:
            raise ValueError("历史和目标 token 总长度超过模型上限："f"{sequence_length} > {self.max_position_embeddings}")

        # 设置位置编码的形状，由总长度确定，
        # 如果总长度是 12 position_ids = [0, 1, 2, ..., 11]
        # 如果总长度是 204 position_ids = [0, 1, 2, ..., 203]
        position_ids = torch.arange(sequence_length,device=prefix_tokens.device,)

        # 位置embedding和token embedding相加，然后进行dropout
        hidden_states = self.token_embedding(prefix_tokens) # 查表[32, 204] ->[32, 204, 64]
        hidden_states = hidden_states + self.position_embedding(position_ids)[None, :, :] # 加入位置向量，利用广播机制
        hidden_states = F.dropout(hidden_states,p=self.dropout,training=self.training,)

        # 左侧 padding 与因果 mask 同时使用时，最前面的 PAD 查询位置
        # 可能出现“所有 key 都被屏蔽”的情况，softmax 会产生 NaN。
        # 这里构造 batch/head 级别的联合 mask：真实位置不能关注 PAD，
        # PAD 查询位置保留自身这一条可见边，保证注意力分布始终有效。
        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        attention_mask = causal_mask[None, :, :] | (~prefix_mask[:, None, :])

        padding_query_self = (
            (~prefix_mask)[:, :, None]
            & torch.eye(
                sequence_length,
                dtype=torch.bool,
                device=prefix_tokens.device,
            )[None, :, :]
        )
        attention_mask = attention_mask & ~padding_query_self
        attention_mask = (
            attention_mask[:, None, :, :]
            .expand(-1, self.num_heads, -1, -1)
            .reshape(
                prefix_tokens.shape[0] * self.num_heads,
                sequence_length,
                sequence_length,
            )
        )

        # 经过 transformer 模型
        hidden_states = self.transformer(
            hidden_states,
            mask=attention_mask,
        )

        # 输出[B, 204, 64]
        return self.output_norm(hidden_states)

    """
    标准前向传播函数
    标准前向传播函数
    """
    def forward(
        self,
        history_tokens: Tensor,
        history_token_mask: Tensor,
        target_input_tokens: Tensor,
        *,
        return_full_logits: bool = False,
        allow_partial_target: bool = False,
    ) -> Tensor:

        # 经过前面的编码流程 ，得到[32, 204, 64]
        hidden_states = self._encode_prefix(
            history_tokens,
            history_token_mask,
            target_input_tokens,
            allow_partial_target=allow_partial_target,)

        # [B, L, 64] ->[B, L, 196] 长度为 196 的向量表示每个token的分数
        logits = self.lm_head(hidden_states)

        if return_full_logits:
            return logits

        # 只返回目标位置
        return logits[:, -target_input_tokens.shape[1] :, :]

    """计算 teacher-forcing 的交叉熵损失。"""
    def compute_loss(
        self,
        history_tokens: Tensor,
        history_token_mask: Tensor,
        target_input_tokens: Tensor,
        target_output_tokens: Tensor,
    ) -> Tensor:

        if target_output_tokens.shape != target_input_tokens.shape:
            raise ValueError("target_output_tokens 必须与 target_input_tokens 形状一致")

        if target_output_tokens.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,):
            raise TypeError("target_output_tokens 必须是整数张量")

        if torch.any(target_output_tokens < 0) or torch.any(target_output_tokens >= self.vocab_size):
            raise ValueError("target_output_tokens 包含越界 token")

        logits = self.forward(
            history_tokens,
            history_token_mask,
            target_input_tokens,)

        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_output_tokens.reshape(-1).long(),
            ignore_index=self.pad_token_id,
        )

    """
    推理阶段自回归生成 Semantic ID。
    当前：
        没有 Semantic ID 结构约束。理论上可以生成词表中任意 token
        没有 Beam Search
    """
    @torch.no_grad()
    def generate(
        self,
        history_tokens: Tensor,
        history_token_mask: Tensor,
        *,
        max_new_tokens: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        do_sample: bool = False,
    ) -> Tensor:

        if history_tokens.ndim != 2:
            raise ValueError("history_tokens 必须是二维张量 [B, H]")
        if history_token_mask.shape != history_tokens.shape:
            raise ValueError("history_token_mask 必须与 history_tokens 形状一致")
        if max_new_tokens is None:
            max_new_tokens = self.target_token_length
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise TypeError("max_new_tokens 必须是整数")
        if max_new_tokens < 1 or max_new_tokens > self.target_token_length:
            raise ValueError("max_new_tokens 必须位于 [1, target_token_length] 内")
        if temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        if top_k is not None and (top_k < 1 or top_k > self.vocab_size):
            raise ValueError("top_k 必须位于 [1, vocab_size] 内")

        # 托底，防止没传bos和eos
        bos_token_id = self.bos_token_id if bos_token_id is None else bos_token_id
        eos_token_id = self.eos_token_id if eos_token_id is None else eos_token_id
        if bos_token_id is None or eos_token_id is None:
            raise ValueError("generate 需要 bos_token_id 和 eos_token_id")
        if not 0 <= bos_token_id < self.vocab_size:
            raise ValueError("bos_token_id 越界")
        if not 0 <= eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id 越界")

        # 初始化生成序列，如btach为2，则generated =[[194, 194]],194为bos编号
        generated = torch.full(
            (history_tokens.shape[0], 1),
            fill_value=bos_token_id,
            dtype=torch.long,
            device=history_tokens.device,)
        # 初始时finished = [[False],[False]],表示尚未生成eos
        finished = torch.zeros(
            history_tokens.shape[0],
            dtype=torch.bool,
            device=history_tokens.device,)

        # 生成循环
        for _ in range(max_new_tokens):
            # 调用前向传播allow_partial_target=True
            logits = self.forward(
                history_tokens,
                history_token_mask,
                generated,
                allow_partial_target=True,)

            # 下一个token可能性的列表，temperate控制自信程度
            next_logits = logits[:, -1, :] / temperature

            # 先取概率最高的top-k个
            if top_k is not None:
                top_values, top_indices = torch.topk(
                    next_logits,
                    k=top_k,
                    dim=-1,)
                # 表示直接取概率最高的，还是按概率采样
                if do_sample:
                    probabilities = torch.softmax(top_values, dim=-1)
                    sampled = torch.multinomial(probabilities, num_samples=1)
                    next_tokens = top_indices.gather(1, sampled).squeeze(1)
                else:
                    next_tokens = top_indices[:, 0]

            # 没有top-k，全量采样，可能采到一些功能性token
            elif do_sample:
                probabilities = torch.softmax(next_logits, dim=-1)
                next_tokens = torch.multinomial(
                    probabilities,
                    num_samples=1,).squeeze(1)
            # 直接取最高概率
            else:
                next_tokens = torch.argmax(next_logits, dim=-1)

            # condition 为 True  → 选择 a
            # condition 为 False → 选择 b
            next_tokens = torch.where(finished, torch.full_like(next_tokens, eos_token_id),next_tokens,)

            generated = torch.cat([generated, next_tokens[:, None]], dim=1,)

            finished = finished | (next_tokens == eos_token_id)

            # 整个 batch 中的用户都已经生成 EOS，就提前退出循环。
            if bool(torch.all(finished)):
                break

        return generated


# 更直观的别名，方便在 trainer 中使用。
TigerModel = TIGER


__all__ = [
    "TigerConfig",
    "TIGER",
    "TigerModel",
]
