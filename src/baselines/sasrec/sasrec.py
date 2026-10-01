from __future__ import annotations
from torch import nn
import torch
import math

"""
一个因果自注意力模块：
    LayerNorm
    -> Self-Attention
    -> 残差连接
    -> LayerNorm
    -> Feed-Forward
    -> 残差连接
"""

"""
to_do:
    MHA改成MLA?
    ReLU改成 GeLU? OR SWiGLU?
    位置嵌入层改成 AliBi or RoPE? (正余弦没必要，原论文试过)
    
    位置嵌入position_embedding未参与缩放 item_embedding(input_ids) * sqrt(D) + position_embedding:
        答：item_embedding比position_embedding小了大约一个sqrt(D)的量级
"""
class SASRecBlock(nn.Module):
    def __init__(# 创建一层注意力模块所需的网络层。
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int, # 前馈网络的中间升维维度(多少多少倍)
        dropout: float,) -> None:

        # 初始化 PyTorch 的 nn.Module,这样后面创建的层会被正确注册
        super().__init__()
        # 对每个位置的 D 维特征做归一化,[B, L, D] → [B, L, D],形状不变。它有可学习参数，帮助稳定进入注意力层的特征分布。
        self.attention_norm = nn.LayerNorm(embedding_dim)
        # 执行多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,) # 输入张量的维度顺序为 `[批次, 序列长度, 嵌入维度]`

        self.attention_dropout = nn.Dropout(dropout)

        self.feed_forward_norm = nn.LayerNorm(embedding_dim)

        # 64 → 128 → 64
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim), # 第一个线性层：将维度从 `embedding_dim` 升维到 `feed_forward_dim`，扩展特征空间
            nn.ReLU(),
            nn.Dropout(dropout), # 中间层正则化
            nn.Linear(feed_forward_dim, embedding_dim), # 将维度降回 `embedding_dim`，保证输出和输入维度一致，支持残差连接
            nn.Dropout(dropout),)

    """让序列特征经过一层注意力和前馈网络"""
    def forward(self,
            hidden_states: torch.Tensor, # [B, L, D]当前序列特征
            attention_mask: torch.Tensor, # [B × num_heads, L, L]哪些位置禁止关注
            padding_mask: torch.Tensor,  # [B, L]哪些位置是 padding
                ) -> torch.Tensor:

        # Pre-LN 归一化，稳定输入分布
        normalized_states = self.attention_norm(hidden_states)

        # 返回注意力计算后的输出特征张量和注意力权重矩阵，后者没用，丢弃省显存
        attention_output, _ = self.attention(
            query=normalized_states,
            key=normalized_states,
            value=normalized_states, # 用注意力权重矩阵算出来的结果，多头分割
            attn_mask=attention_mask,
            need_weights=False,)

        # 只正则化子层增量，残差通路保持完整干净
        hidden_states = (hidden_states + self.attention_dropout(attention_output))
        # padding_mask先扩展成3维，然后利用广播机制将所有掩码位为True的位置置0
        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1),0.0,)

        hidden_states = (hidden_states # 残差
                    + self.feed_forward(self.feed_forward_norm(hidden_states))) # FFN

        return hidden_states.masked_fill(padding_mask.unsqueeze(-1),0.0,)

    """
    作用：创建完整模型，并检查配置是否合法。
    把物品内部索引转换成特征向量：
        输入：[B, L]
        输出：[B, L, D]
    训练时使用：
        CrossEntropyLoss(logits, target_ids)
    """
class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        *,
        max_seq_len: int = 50,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        feed_forward_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # 用一张表把参数整理下来
        integer_parameters = {
            "num_items": num_items,
            "max_seq_len": max_seq_len,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "feed_forward_dim": feed_forward_dim,}

        for name, value in integer_parameters.items():
            if (not isinstance(value, int)or isinstance(value, bool)):
                raise TypeError(f"{name} 必须是整数")
            if value < 1:
                raise ValueError(f"{name} 必须大于等于 1")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim 必须能够被 num_heads 整除")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 范围内")

        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # 索引 0 保留给 padding
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,)

        # 位置嵌入层
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim,)

        self.embedding_dropout = nn.Dropout(dropout)

        # 多层因果自注意力块堆叠
        self.blocks = nn.ModuleList(
            [SASRecBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                feed_forward_dim=feed_forward_dim,
                dropout=dropout,)
                for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(embedding_dim)
        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02,)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02,)

        with torch.no_grad():
            self.item_embedding.weight[0].zero_()


    def _build_attention_mask(self,padding_mask: torch.Tensor,) -> torch.Tensor:
        batch_size, length = padding_mask.shape
        # 禁止当前位置看到未来位置。生成通用的因果上三角掩码，`True` 的位置代表掩盖
        causal_mask = torch.triu(
            torch.ones(length, length,dtype=torch.bool, device=padding_mask.device,),
            diagonal=1,)
        # 合并 padding 约束（无效位置屏蔽）
        attention_mask = (causal_mask.unsqueeze(0) | padding_mask.unsqueeze(1))

        # 左侧 padding 的查询可能没有任何可见位置。
        # 允许这些查询看自身，避免全掩码产生 NaN。
        # 随后将这些 padding 位置的输出归零。

        # 定位 padding 位置的对角线
        diagonal_mask = torch.eye(length, dtype=torch.bool, device=padding_mask.device,).unsqueeze(0)
        # 找出「本身是 padding 位置」且「是对角线（自己对自己）」的位置，也就是 padding 位置自己关注自己的那个点
        padding_self_positions = (padding_mask.unsqueeze(-1) & diagonal_mask)
        # 放开这些位置的屏蔽
        attention_mask = (attention_mask & ~padding_self_positions)

        # 扩展为多头格式，把批次维度按注意力头数重复
        return attention_mask.repeat_interleave(self.num_heads,dim=0,)

    """
    将物品序列编码为每个位置的隐藏状态,即把物品 ID 序列转换为每个位置的上下文特征
        输入：[B, L]
        输出：[B, L, D] （[batch_size, sequence_length, embedding_dim]）
    
    每个位置对应一个深度特征向量，
    这个向量里编码了「当前物品 + 它所有历史行为」的综合信息，越靠后的位置，融合的历史信息越多。
    
    如某个用户：
     [0.0,  0.0,  0.0,  0.0],    # 第0位：padding位，强制全0，无意义
     [0.12, -0.34, 0.51, -0.22], # 第1位：物品2编码后的特征
     [0.31, -0.58, 0.76, -0.35], # 第2位：物品5编码后的特征（融合了物品2的信息）
     [0.45, -0.72, 0.91, -0.48]  # 第3位：物品7编码后的特征（融合了2、5、7的全部信息）

    """
    def encode_sequence(self, input_ids: torch.Tensor,) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids 必须是二维张量")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids 必须使用 torch.long 类型")
        batch_size, length = input_ids.shape
        if batch_size < 1 or length < 1:
            raise ValueError("input_ids 不能为空")
        if length > self.max_seq_len:
            raise ValueError("输入序列长度超过 max_seq_len")
        padding_mask = input_ids.eq(0)
        if padding_mask.all(dim=1).any():
            raise ValueError("每条序列至少需要一个非 padding 物品")

        position_ids = torch.arange(length, device=input_ids.device,).unsqueeze(0)
        # 物品嵌入和位置嵌入相加
        hidden_states = (
                self.item_embedding(input_ids)
                * math.sqrt(self.embedding_dim)
                + self.position_embedding(position_ids))

        hidden_states = self.embedding_dropout(hidden_states)

        hidden_states = hidden_states.masked_fill(padding_mask.unsqueeze(-1),0.0,)

        attention_mask = self._build_attention_mask(padding_mask)

        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                padding_mask=padding_mask,)

        hidden_states = self.final_norm(hidden_states)

        return hidden_states.masked_fill(padding_mask.unsqueeze(-1),0.0,)

    """
    根据最后一个有效位置预测下一个物品。
    """
    def forward(self,input_ids: torch.Tensor,) -> torch.Tensor:
        hidden_states = self.encode_sequence(input_ids)
        batch_size, length = input_ids.shape

        # 生成 `[0, 1, ..., length-1]` 的位置序号
        positions = torch.arange(length,device=input_ids.device,).unsqueeze(0).expand(batch_size, -1)

        # 屏蔽 padding 位置，取最大值得到最后有效位
        last_positions = positions.masked_fill(input_ids.eq(0),-1,).max(dim=1).values

        # 批量取出最后一个位置的特征
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        sequence_features = hidden_states[batch_indices, last_positions,]

        # 计算用户兴趣向量和所有物品嵌入向量的内积相似度**，相似度越高，得分越高，用户下一个交互该物品的概率越大。
        logits = (sequence_features @ self.item_embedding.weight.T)

        # 屏蔽 padding 位
        logits = logits.clone()
        logits[:, 0] = -torch.inf

        return logits

__all__ = [
    "SASRec",
]





