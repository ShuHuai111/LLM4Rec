from __future__ import annotations

from pathlib import Path

import pandas as pd


class MovieLens1MAdapter:
    """将 MovieLens 1M 原始文件转换为项目统一的两张表。"""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

        self.interactions_path = self.root_dir / "ratings.dat"
        self.items_path = self.root_dir / "movies.dat"

        self._check_required_files()

    def _check_required_files(self) -> None:
        required_files = [
            self.interactions_path,
            self.items_path,
        ]

        missing_files = [
            str(path)
            for path in required_files
            if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "MovieLens 1M 缺少以下文件:\n"
                + "\n".join(missing_files)
            )

    def load_interactions(self) -> pd.DataFrame:
        """读取 ratings.dat，输出统一的用户交互表。"""
        columns = [
            "user_id",
            "item_id",
            "rating",
            "timestamp",
        ]

        interactions = pd.read_csv(
            self.interactions_path,
            sep=r"::",
            engine="python",
            header=None,
            names=columns,
            encoding="latin-1",
            dtype={
                "user_id": "string",
                "item_id": "string",
                "rating": "int8",
                "timestamp": "int64",
            },
        )

        if interactions.empty:
            raise ValueError("ratings.dat 为空")

        if interactions[columns].isnull().any().any():
            raise ValueError("ratings.dat 中存在空值")

        invalid_ratings = ~interactions["rating"].between(1, 5)
        if invalid_ratings.any():
            raise ValueError(
                "ratings.dat 中存在不在 1 到 5 范围内的评分。"
            )

        return (
            interactions
            .sort_values(
                by=["user_id", "timestamp"],
                ascending=[True, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    def load_items(self) -> pd.DataFrame:
        """读取 movies.dat，并输出与 100K 一致的物品字段。"""
        columns = [
            "item_id",
            "title",
            "genres",
        ]

        items = pd.read_csv(
            self.items_path,
            sep=r"::",
            engine="python",
            header=None,
            names=columns,
            encoding="latin-1",
            dtype="string",
            keep_default_na=False,
        )

        if items.empty:
            raise ValueError("movies.dat 为空")

        if items[columns].isnull().any().any():
            raise ValueError("movies.dat 中存在空值")

        if items["item_id"].duplicated().any():
            raise ValueError("movies.dat 中存在重复 item_id。")

        # 1M 没有 100K 中的日期和 IMDb 字段，用空字符串补齐统一接口。
        items["release_date"] = ""
        items["video_release_date"] = ""
        items["imdb_url"] = ""

        return items[
            [
                "item_id",
                "title",
                "release_date",
                "video_release_date",
                "imdb_url",
                "genres",
            ]
        ].reset_index(drop=True)

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        interactions = self.load_interactions()
        items = self.load_items()

        interaction_item_ids = set(
            interactions["item_id"].unique()
        )
        metadata_item_ids = set(items["item_id"].unique())
        missing_metadata = interaction_item_ids - metadata_item_ids

        if missing_metadata:
            raise ValueError(
                "以下交互中的 item_id 没有对应的物品元数据："
                f"{sorted(missing_metadata)[:10]}"
            )

        return interactions, items


__all__ = ["MovieLens1MAdapter"]
