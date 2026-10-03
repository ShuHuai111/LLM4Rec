from __future__ import annotations

from pathlib import Path
from sklearn.cluster import KMeans

import numpy as np

"""
目前只用了title + genres
"""

"""
残差向量量化器（RVQ）
Residual Vector Quantizer，使用多层 K-Means 将连续向量转换为离散编码。

量化过程：

    embedding
        -> code_0 + residual_0
        -> code_1 + residual_1
        -> code_2 + residual_2

最终得到：

    [code_0, code_1, code_2]
"""
class ResidualVectorQuantizer:
    def __init__(
        self,
        *,
        codebook_size: int = 64,        #单个码本的大小，也就是每层有多少个聚类中心（码本向量）。
        num_codebooks: int = 3,         # 码本的层数
        n_init: int = 10,               # 单层 K-Means 的初始化次数。K-Means 结果受初始中心影响，多次初始化选最优结果，避免陷入局部最优。
        max_iter: int = 300,            # 单层 K-Means 的最大迭代次数，控制训练收敛的上限，防止无限迭代。
        random_state: int = 42,         # 随机种子
        distance_batch_size: int = 4096,# 距离计算批大小
    ) -> None:
        if (not isinstance(codebook_size, int) or isinstance(codebook_size, bool) or codebook_size < 2):
            raise ValueError("codebook_size 必须是大于等于 2 的整数")
        if (not isinstance(num_codebooks, int) or isinstance(num_codebooks, bool) or num_codebooks < 1):
            raise ValueError("num_codebooks 必须是大于等于 1 的整数")
        if (not isinstance(n_init, int)or isinstance(n_init, bool)or n_init < 1):
            raise ValueError("n_init 必须是大于等于 1 的整数")
        if (not isinstance(max_iter, int)or isinstance(max_iter, bool)or max_iter < 1):
            raise ValueError("max_iter 必须是大于等于 1 的整数")
        if (not isinstance(random_state, int)or isinstance(random_state, bool)):
            raise TypeError("random_state 必须是整数")
        if (not isinstance(distance_batch_size, int)or isinstance(distance_batch_size, bool)or distance_batch_size < 1):
            raise ValueError("distance_batch_size 必须是大于等于 1 的整数")

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.distance_batch_size = distance_batch_size

        self.codebooks_: list[np.ndarray] | None = None
        self.embedding_dim_: int | None = None
        self.reconstruction_error_: float | None = None

    """
    检查并标准化输入向量。
    输入形状：
        [num_items, embedding_dim]
    """
    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray,) -> np.ndarray:
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维数组，形状应为 [num_items, embedding_dim]")
        if embeddings.shape[0] < 1:
            raise ValueError("embeddings 至少需要包含一个物品")
        if embeddings.shape[1] < 1:
            raise ValueError("embedding_dim 必须大于 0")

        embeddings = np.asarray(embeddings,dtype=np.float32,)
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings 不能包含 NaN 或 Inf")
        return embeddings

    """
    检查量化器是否已经拟合。
    """
    def _validate_fitted(self) -> None:
        if self.codebooks_ is None:
            raise RuntimeError("量化器尚未 fit")
        if self.embedding_dim_ is None:
            raise RuntimeError("量化器 embedding_dim 尚未初始化")
        if len(self.codebooks_) != self.num_codebooks:
            raise RuntimeError("codebooks 数量与 num_codebooks 不一致")

    """
    为每个输入向量寻找最近的 codebook 向量。
    使用分批计算，避免一次性构造过大的距离矩阵。
    """
    def _nearest_code_indices(self,vectors: np.ndarray,codebook: np.ndarray,) -> np.ndarray:
        vectors = self._validate_embeddings(vectors)
        if codebook.ndim != 2:
            raise ValueError("codebook 必须是二维数组")

        if vectors.shape[1] != codebook.shape[1]:
            raise ValueError("输入向量维度与 codebook 维度不一致")

        # 提前分配好和输入向量数量一致的空数组，数据类型为 int64
        num_vectors = vectors.shape[0]  # 输入向量的总个数，如输入的 `vectors` 是 100 行 32 列的二维数组，那num_vectors=100
        codes = np.empty(num_vectors,dtype=np.int64,) # 一个整数数组，长度和输入向量的个数一样，存「每个输入向量对应的最近码本编号」

        # 提前计算码本中每个中心向量的 L2 范数平方（每个向量自身元素平方和），得到形状为 `[codebook_size]` 的一维数组。
        # 这个值是固定的，后续每个批次都可以复用，不用重复计算，大幅减少冗余运算。
        codebook = np.asarray(codebook, dtype=np.float32) # 形状是 `[码本大小, 向量维度]`，形状是 `(64, 32)`，里面有 64 个 32 维的向量
        # 假设 0 号模板向量是 `[1, 2, 3]`，那它的平方和就是 `1² + 2² + 3² = 14`，对应 `codebook_norm[0] = 14`；
        codebook_norm = np.sum(codebook * codebook, axis=1)

        # 分批计算最近索引，按批次大小，把所有向量切成一小段一小段，循环处理每一段
        for start in range(0, num_vectors, self.distance_batch_size):
            end = min(start + self.distance_batch_size, num_vectors)
            # 切出当前批次的向量
            batch = vectors[start:end]

            # 对批次里每个向量，计算自身所有元素的平方和（L2 范数平方），和之前提前算好的
            batch_norm = np.sum(batch * batch, axis=1, keepdims=True)
            # 批量算所有距离平方
            distances = (batch_norm - 2.0 * batch @ codebook.T+ codebook_norm[None, :])
            # 找最近的模板编号，填进结果
            codes[start:end] = np.argmin(distances, axis=1)

        return codes

    """
    拟合多层 residual codebook。
    输入全量的嵌入向量，按照「逐层拟合残差」的逻辑，训练出每一层的码本。
    训练完成后，量化器就可以用来把任意新向量转换成对应的多层离散编码。
    参数：
        embeddings：
            shape = [num_items, embedding_dim]
    返回：
        self
    """
    def fit(self,embeddings: np.ndarray,) -> ResidualVectorQuantizer:
        embeddings = self._validate_embeddings(embeddings)
        num_items, embedding_dim = embeddings.shape

        if num_items < self.codebook_size:
            raise ValueError("物品数量不能小于 codebook_size："f"{num_items} < {self.codebook_size}")

        self.embedding_dim_ = embedding_dim
        self.codebooks_ = []
        residual = embeddings.copy()

        # 逐层训练码本
        for level in range(self.num_codebooks):
            kmeans = KMeans(
                n_clusters=self.codebook_size,          # 聚类数 = 码本大小，比如 64，这一层最终会得到 64 个聚类中心
                n_init=self.n_init,                     # 初始化次数
                max_iter=self.max_iter,                 # 最大迭代次数
                random_state=self.random_state + level, # 每层的随机种子不一样，避免每层码本趋同
                algorithm="lloyd",                      # 标准的 K-Means 算法
            )

            kmeans.fit(residual)
            # 提取当前层的码本与标签
            codebook = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
            codes = np.asarray(kmeans.labels_, dtype=np.int64)

            quantized = codebook[codes]
            self.codebooks_.append(codebook)
            residual = residual - quantized

        # 计算最终重构误差
        self.reconstruction_error_ = (float(np.mean(residual * residual)))

        return self

    """
    供推理使用，码本复用训练好的
    将连续向量转换为 Semantic ID code。
    返回：
        shape = [num_items, num_codebooks]
        dtype = int64
    """
    def encode(self,embeddings: np.ndarray,) -> np.ndarray:
        self._validate_fitted()
        embeddings = self._validate_embeddings(embeddings)
        if embeddings.shape[1] != self.embedding_dim_:
            raise ValueError("输入 embedding_dim 与量化器不一致："f"{embeddings.shape[1]} != "f"{self.embedding_dim_}")
        assert self.codebooks_ is not None

        # 复制输入向量作为初始残差
        residual = embeddings.copy()
        # 提前创建好空的二维结果数组
        codes = np.empty((embeddings.shape[0], self.num_codebooks),dtype=np.int64,)

        for level, codebook in enumerate(self.codebooks_):
            # 找当前层的最近码本索引，调用之前的 `_nearest_code_indices` 工具
            level_codes = self._nearest_code_indices(residual, codebook)
            # 存入结果数组
            codes[:, level] = level_codes
            # 更新残差，传给下一层
            residual = residual - codebook[level_codes]

        return codes

    """
    将 Semantic ID code 重构为连续向量。
    
    输入形状：
        [num_items, num_codebooks]
        
    输出形状：
        [num_items, embedding_dim]
    """
    def decode(self,codes: np.ndarray,) -> np.ndarray:
        self._validate_fitted()
        if not isinstance(codes, np.ndarray):
            codes = np.asarray(codes)
        if codes.ndim != 2:
            raise ValueError("codes 必须是二维数组，形状应为 [num_items, num_codebooks]")
        if codes.shape[1] != self.num_codebooks:
            raise ValueError(
                "codes 的列数与 num_codebooks 不一致："f"{codes.shape[1]} != "f"{self.num_codebooks}")
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError("codes 必须是整数数组")
        assert self.codebooks_ is not None
        assert self.embedding_dim_ is not None
        codes = np.asarray(codes, dtype=np.int64)

        # 创建一个全 0 的二维数组，行数 = 输入编码的样本数，列数 = 训练时的向量维度。
        num_items = codes.shape[0]
        reconstructed = np.zeros((num_items, self.embedding_dim_),dtype=np.float32,)
        # 逐层累加重构
        for level, codebook in enumerate(self.codebooks_):
            level_codes = codes[:, level]
            if np.any(level_codes < 0) or np.any(level_codes >= self.codebook_size):
                raise ValueError(f"第 {level} 层 code 超出范围，合法范围是 [0, {self.codebook_size - 1}]")
            reconstructed += codebook[level_codes]

        return reconstructed

    """
    拟合量化器并直接生成 Semantic ID。
    便捷式组合方法，「训练 + 编码」
    """
    def fit_encode(self,embeddings: np.ndarray,) -> np.ndarray:
        self.fit(embeddings)
        return self.encode(embeddings)

    """
    量化精度评估工具，计算「原始向量」和「编码后再重构的向量」之间的均方误差（MSE）
    """
    def reconstruction_error(self,embeddings: np.ndarray,codes: np.ndarray | None = None,) -> float:
        embeddings = self._validate_embeddings(embeddings)

        if codes is None:
            codes = self.encode(embeddings)

        reconstructed = self.decode(codes)
        if reconstructed.shape != embeddings.shape:
            raise ValueError("重构向量形状与原始向量不一致")

        error = np.mean((embeddings - reconstructed) ** 2)

        return float(error)

    """
    把 `encode` 输出的二维整数编码数组，转换成用分隔符拼接的字符串形式语义 ID。
        例如：[12, 48, 7] -> "12-48-7"
    """
    def semantic_id_strings(self,codes: np.ndarray,separator: str = "-",) -> list[str]:
        if not isinstance(separator, str):
            raise TypeError("separator 必须是字符串")
        if separator == "":
            raise ValueError("separator 不能是空字符串")

        if not isinstance(codes, np.ndarray):
            codes = np.asarray(codes)
        if codes.ndim != 2:
            raise ValueError("codes 必须是二维数组")
        if codes.shape[1] != self.num_codebooks:
            raise ValueError("codes 的列数与 num_codebooks 不一致")
        if not np.issubdtype(codes.dtype, np.integer):
            raise TypeError("codes 必须是整数数组")

        return [separator.join(str(int(code))for code in row) for row in codes]

    """
    统计 Semantic ID 冲突情况。
    """
    def collision_statistics(self,codes: np.ndarray,) -> dict[str, int | float]:
        # 转换为字符串语义 ID
        semantic_ids = self.semantic_id_strings(codes)

        total_items = len(semantic_ids)  # 总样本数量
        unique_ids = len(set(semantic_ids))  # 去重后的唯一语义ID数量
        collision_items = total_items - unique_ids  # 发生冲突的样本总数

        collision_rate = (collision_items / total_items if total_items > 0 else 0.0)

        return {
            "total_items": total_items,
            "unique_semantic_ids": unique_ids,
            "collision_items": collision_items,
            "collision_rate": collision_rate,}

    """
    保存量化器和 codebook。
    """
    def save(self,path: str | Path,) -> None:
        self._validate_fitted()
        assert self.codebooks_ is not None
        assert self.embedding_dim_ is not None

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 原本 `self.codebooks_` 是一个 Python 列表，每个元素是一层的码本（二维数组，形状 `[codebook_size, embedding_dim]`）。
        # 用 `np.stack` 在第 0 维堆叠，变成一个**三维数组**，形状为 `[num_codebooks, codebook_size, embedding_dim]`，
        # 也就是「层数 × 码本大小 × 向量维度」。
        codebooks = np.stack(self.codebooks_, axis=0)

        # 把所有数据打包压缩保存到一个 `.npz` 文件里，每个数据都有对应的键名，方便加载时按名读取。
        np.savez_compressed(
            output_path,
            codebooks=codebooks,
            codebook_size=np.asarray(self.codebook_size, dtype=np.int64),
            num_codebooks=np.asarray(self.num_codebooks, dtype=np.int64),
            embedding_dim=np.asarray(self.embedding_dim_, dtype=np.int64),
            n_init=np.asarray(self.n_init, dtype=np.int64),
            max_iter=np.asarray(self.max_iter, dtype=np.int64),
            random_state=np.asarray(self.random_state, dtype=np.int64),
        )

    """
    加载已经保存的量化器。
    """
    @classmethod
    def load(cls,path: str | Path,) -> ResidualVectorQuantizer:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"找不到量化器文件：{input_path}")

        with np.load(input_path, allow_pickle=False) as data:
            codebooks = np.asarray(data["codebooks"], dtype=np.float32)
            codebook_size = int(data["codebook_size"])
            num_codebooks = int(data["num_codebooks"])
            embedding_dim = int(data["embedding_dim"])
            n_init = int(data["n_init"])
            max_iter = int(data["max_iter"])
            random_state = int(data["random_state"])

        if codebooks.ndim != 3:
            raise ValueError("保存的 codebooks 必须是三维数组")
        if codebooks.shape[0] != num_codebooks:
            raise ValueError("保存的 codebooks 层数不一致")
        if codebooks.shape[1] != codebook_size:
            raise ValueError("保存的 codebook 大小不一致")
        if codebooks.shape[2] != embedding_dim:
            raise ValueError("保存的 embedding_dim 不一致")

        # 创建量化器实例，用读取到的所有配置参数，调用类构造函数，创建一个全新的量化器对象。
        quantizer = cls(
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )

        # 还原拟合状态。把三维码本数组，拆成「每层一个二维数组」的列表形式，和训练时 `self.codebooks_` 的结构完全一致。
        quantizer.codebooks_ = [codebooks[level].copy() for level in range(num_codebooks)]
        quantizer.embedding_dim_ = embedding_dim
        quantizer.reconstruction_error_ = None

        return quantizer

    """quanzer
    判断量化器是否已经完成拟合。
    """
    @property
    def is_fitted(self) -> bool:
        return (self.codebooks_ is not None
                and self.embedding_dim_ is not None
                and len(self.codebooks_) == self.num_codebooks)

__all__ = [
    "ResidualVectorQuantizer",
]













