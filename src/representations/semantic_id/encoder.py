from __future__ import annotations

from typing import Iterable
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import pandas as pd
import numpy as np
import pickle
"""
使用物品文本和类别元数据生成物品向量。

    默认输入字段：
        item_id
        title
        genres

    可选输入字段：
        release_date
        video_release_date
        imdb_url

    典型流程：
        items DataFrame
            -> 文本拼接
            -> TF-IDF
            -> TruncatedSVD
            -> L2 归一化
            -> item embeddings
"""
"""
TF-IDF 向量化:
    TF（词频）：某个词在当前这个物品的文本里出现的频率
    IDF（逆文档频率）：衡量一个词在所有物品里的普遍程度。一个词在越多物品里出现，它的区分度越低,权重越低。
    TF-IDF = TF × IDF
"""
"""
文本预处理与拼接
.......
.....

遍历所有物品的文本，分词后根据 `min_df`（过滤低频词）、`max_df`（过滤高频通用词）、
`max_features`（保留 TopN 个词）过滤，最终形成固定大小的词典，比如默认 20000 个词。

对每个物品的文本，计算词典中每个词的 TF-IDF 值，形成一个高维稀疏向量.
维度等于词典大小（比如 20000 维），大部分位置都是 0，因为一个物品的文本只会包含少量词。

`ngram_range=(1,2)` 表示同时提取「单个词」和「连续两个词的词组」作为特征。
比如 `toy story` 会作为一个二元词组特征，比单个词更能捕捉局部语义。

......
......

TruncatedSVD 降维：把高维稀疏向量压成稠密向量
L2 归一化，最终向量的模长等于 1。
"""
class ItemTextEncoder:
    def __init__(
        self,
        *,
        n_components: int | None = 32,                      # **最终输出的嵌入维度**，即 SVD 降维后的目标维度。设为 `None` 则跳过 SVD 降维，直接返回 TF-IDF 原始特征。
        text_columns: Iterable[str] = ("title","genres",),  # 用于生成特征的文本列名列表，默认使用物品标题和类型。可扩展加入简介、标签、描述等更多文本字段。
        max_features: int | None = 20000,                   # TF-IDF 的最大特征词数量，按文档词频排序保留 TopN
        ngram_range: tuple[int, int] = (1, 2),              # 词组拆分范围，默认一元词 + 二元词组。比如 `(1,2)` 会同时提取单个词和连续两个词的短语特征，捕捉局部语义。
        min_df: int | float = 1,                            # 最小文档频率，低于该阈值的词会被过滤。
        max_df: int | float = 1.0,                          # 最大文档频率，高于该阈值的词会被过滤，
        lowercase: bool = True,                             # 全部转成小写，避免 `Toy` 和 `toy` 被识别成两个不同的词
        normalize_embeddings: bool = True,                  # 是否对最终输出向量做 L2 归一化。
        random_state: int = 42,                             # 随机种子，保证 SVD 降维结果可复现。
    ) -> None:
        if (n_components is not None and (
                not isinstance(n_components, int)
                or isinstance(n_components, bool)
                or n_components < 1)):
            raise ValueError("n_components 必须是大于等于 1 的整数或 None")
        if (not isinstance(ngram_range, tuple)
            or len(ngram_range) != 2
            or ngram_range[0] < 1
            or ngram_range[1] < ngram_range[0]):
            raise ValueError("ngram_range 必须是类似 (1, 2) 的合法二元组")
        if (not isinstance(random_state, int)
            or isinstance(random_state, bool)):
            raise TypeError("random_state 必须是整数")

        # 配置参数，初始化传入，永久保留
        self.n_components = n_components
        self.text_columns = tuple(text_columns)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.lowercase = lowercase
        self.normalize_embeddings = normalize_embeddings
        self.random_state = random_state

        # 拟合后状态变量（初始为 None，调用`fit()`后才赋值）
        self.vectorizer_: TfidfVectorizer | None = None  # 拟合完成的TF-IDF向量化器
        self.svd_: TruncatedSVD | None = None  # 拟合完成的SVD降维器
        self.text_columns_: tuple[str, ...] | None = None  # 实际生效的文本列
        self.feature_names_: list[str] | None = None  # TF-IDF的特征词列表（词典）
        self.output_dim_: int | None = None  # 最终输出的嵌入维度

    """
    物品数据进入编码器前的统一入口校验器
    做全维度的输入合法性检查，同时标准化 `item_id` 字段格式，
    从根源上避免脏数据导致后续文本向量化、降维等流程出现晦涩、难排查的错误。
    .....
    数据类型校验
    必填字段校验
    空数据集校验
    
    item_id 字段标准化
    ID 合法性校验
        缺失值校验
        重复值校验
    """
    @staticmethod
    def _validate_dataframe(items: pd.DataFrame,) -> pd.DataFrame:
        if not isinstance(items, pd.DataFrame):
            raise TypeError("items 必须是 pandas.DataFrame")
        if "item_id" not in items.columns:
            raise ValueError("items 必须包含 item_id 字段")
        if items.empty:
            raise ValueError("items 不能为空")

        # copy, 丢弃原始行索引，重置为从 0 开始的连续整数索引。避免原始数据的索引不连续、索引重复等问题，
        data = items.copy().reset_index(drop=True)
        # item_id 字段标准化
        data["item_id"] = (data["item_id"].astype("string").str.strip())
        if data["item_id"].isna().any():
            raise ValueError("item_id 不能包含缺失值")

        if data["item_id"].duplicated().any():
            duplicated_ids = (data.loc[data["item_id"].duplicated(),"item_id",].astype(str).tolist())
            raise ValueError("item_id 不能重复，重复值示例："f"{duplicated_ids[:5]}")

        return data

    """
    拼接文本
    """
    def _build_texts(self,items: pd.DataFrame,) -> list[str]:
        if self.text_columns_ is None:
            raise RuntimeError("编码器尚未完成 fit")

        text_parts: list[pd.Series] = []
        for column in self.text_columns_:
            if column in items.columns:
                values = (items[column].fillna("").astype("string").str.strip())
            else: # 如果当前列不存在，生成一个和输入数据行数一致、全为空字符串的 Series，保证和其他列的行数对齐
                values = pd.Series("", index=items.index, dtype="string",)

            # 添加字段名，避免不同字段中的相同词完全混淆
            values = (column + " " + values.astype(str))
            text_parts.append(values)

        if not text_parts:
            raise ValueError("没有可用于编码的文本字段")

        # 取出第一列的文本作为基准，后续每一列都用空格分隔拼接上去。空格分隔
        combined = text_parts[0].copy()
        for values in text_parts[1:]:
            combined = combined + " " + values

        # 最终清洗
        texts = (combined.fillna("").astype(str).str.strip().tolist())
        if not any(texts):
            raise ValueError("物品文本字段全部为空")

        return texts

    """
    在物品表上拟合 TF-IDF 和 SVD。
    
    参数：
        items：至少包含 item_id 和一个文本字段。
    
    返回：self
    """
    def fit(self,items: pd.DataFrame,) -> ItemTextEncoder:
        data = self._validate_dataframe(items)

        # 筛选可用列表
        available_columns = tuple(column for column in self.text_columns if column in data.columns)
        if not available_columns:
            raise ValueError(
                "输入数据中不存在任何可用文本字段，"f"期望字段：{self.text_columns}")

        self.text_columns_ = available_columns

        # 调用内部文本构建方法，构建标准化文本
        texts = self._build_texts(data)

        # 拟合 TF-IDF 向量化器
        self.vectorizer_ = TfidfVectorizer(
            lowercase=self.lowercase,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        tfidf_matrix = self.vectorizer_.fit_transform(texts)

        if tfidf_matrix.shape[1] == 0:
            raise ValueError("TF-IDF 没有生成任何特征，请检查物品文本内容")

        self.feature_names_ = (self.vectorizer_.get_feature_names_out().tolist())

        # 拟合 TruncatedSVD 降维
        self.svd_ = None
        if self.n_components is not None:
            max_components = min(tfidf_matrix.shape[0] - 1,tfidf_matrix.shape[1] - 1,)

            if max_components >= 1:
                actual_components = min(self.n_components, max_components,)
                self.svd_ = TruncatedSVD(
                    n_components=actual_components,
                    random_state=self.random_state,
                )
                self.svd_.fit(tfidf_matrix)

        if self.svd_ is not None:
            self.output_dim_ = int(self.svd_.n_components)
        else:
            self.output_dim_ = int(tfidf_matrix.shape[1])
        return self


    """
    使用已经拟合的编码器生成物品向量。
    
    返回：
        shape = [num_items, output_dim]
        dtype = float32
    """
    def transform(self,items: pd.DataFrame,) -> np.ndarray:
        if self.vectorizer_ is None:
            raise RuntimeError("请先调用 fit()")
        if self.text_columns_ is None:
            raise RuntimeError("编码器文本字段尚未初始化")

        data = self._validate_dataframe(items)
        texts = self._build_texts(data)
        tfidf_matrix = self.vectorizer_.transform(texts)

        if self.svd_ is not None:
            embeddings = self.svd_.transform(tfidf_matrix)
        else:
            embeddings = tfidf_matrix.toarray()

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.normalize_embeddings:
            embeddings = normalize(embeddings, norm="l2", axis=1, copy=False,)
        embeddings = np.asarray(embeddings, dtype=np.float32,)

        if not np.isfinite(embeddings).all():
            raise ValueError("生成的物品向量包含 NaN 或 Inf")

        return embeddings

    """
    拟合编码器并直接生成物品向量。
    """
    def fit_transform(self,items: pd.DataFrame,) -> np.ndarray:
        self.fit(items)
        return self.transform(items)

    """
    在底层 `fit/transform` 能力的基础上，
    额外打包返回和向量严格行对齐的物品 ID 列表，省去调用方手动匹配 ID 与向量的步骤
    """

    def encode_with_ids(
            self,
            items: pd.DataFrame,
            *,
            fit: bool = False,
    ) -> tuple[list[str], np.ndarray]:
        data = self._validate_dataframe(items)
        item_ids = (data["item_id"].astype(str).tolist())
        if fit:
            embeddings = self.fit_transform(data)
        else:
            embeddings = self.transform(data)

        return item_ids, embeddings

    """
    保存已经拟合的编码器。
    """
    def save(self,path: str | Path,) -> None:
        if self.vectorizer_ is None:
            raise RuntimeError("编码器尚未 fit，不能保存")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True,exist_ok=True,)

        with output_path.open("wb") as file:
            pickle.dump(self, file)
    """
    负责从磁盘文件加载之前保存的编码器，还原为完整可用的 `ItemTextEncoder` 实例
    """
    @classmethod
    def load(cls, path: str | Path,) -> ItemTextEncoder:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"找不到编码器文件：{input_path}")

        with input_path.open("rb") as file:
            encoder = pickle.load(file)

        if not isinstance(encoder, cls):
            raise TypeError("保存文件中的对象不是 ItemTextEncoder")
        return encoder

    """
    判断编码器是否已经完成拟合。
    """
    @property
    def is_fitted(self) -> bool:
        return (
                self.vectorizer_ is not None
                and self.text_columns_ is not None
                and self.output_dim_ is not None
        )

__all__ = [
    "ItemTextEncoder",
]
