from __future__ import annotations

from pathlib import Path

import pandas as pd


class MovieLens100KAdapter:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

        self.interactions_path = self.root_dir / "u.data"
        self.items_path = self.root_dir / "u.item"
        self.genres_path = self.root_dir / "u.genre"

        self._check_required_files()

    def _check_required_files(self) -> None:
        required_files = [self.interactions_path, self.items_path, self.genres_path]

        missing_files = [str(path) for path in required_files if not path.exists()]

        if missing_files:
            raise FileNotFoundError("MovieLens 100K 缺少以下文件:\n" + "\n".join(missing_files))

    def _load_genre_mapping(self) -> dict[int, str]:
        """
            输入：Action|1
                Adventure|2
                unknown|0
            输出：{0: 'unknown', 1: 'Action', 2: 'Adventure', ..., 18: 'Western'}
        """
        genre_mapping: dict[int, str] = {}
        with self.genres_path.open("r", encoding = "latin-1") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("|")

                if len(parts) != 2:
                    continue

                genre_name, genre_id = parts

                if not genre_id.isdigit():
                    continue
                genre_mapping[int(genre_id)] = genre_name
        return dict(sorted(genre_mapping.items(), key=lambda item: item[0]))


    def load_interactions(self) -> pd.DataFrame:
        # 读取u.data，四个字段为user_id", "item_id", "rating", "timestamp
        """
            输入：196\t242\t3\t881250949
            输出：user_id	item_id	      rating	timestamp
                1	        1	           5	    874965758
                1	        2	           3	    876893171
        """
        columns = ["user_id", "item_id", "rating", "timestamp"]
        interactions = pd.read_csv(
            self.interactions_path,
            sep="\t",
            header=None,
            names=columns,
            dtype={
                "user_id": "string",
                "item_id": "string",
                "rating": "int8",
                "timestamp": "int64"
            },
        )

        if interactions.empty:
            raise ValueError("u.data为空")
        if interactions[columns].isnull().any().any():
            raise ValueError("u.data中存在空值")

        invalid_ratings = ~interactions["rating"].between(1, 5)
        if invalid_ratings.any():
            raise ValueError("u.data 中存在不在 1 到 5 范围内的评分。")

        interactions = interactions.sort_values(
            by=["user_id", "timestamp"],
            ascending=[True, True]
        ).reset_index(drop=True)

        return interactions

    def load_items(self) -> pd.DataFrame:
        # 电影ID|电影标题|上映日期|录像带发行日期|IMDB链接|流派0标记|流派1标记|...|流派18标记
        """
            输出：item_id	   title	           genres
                    1	   Toy Story (1995)      Animation|Children's|Comedy
                    2	   Jumanji (1995)	    Adventure|Children's|Fantasy
        """
        genre_mapping = self._load_genre_mapping()
        genre_names = [genre_mapping[index] for index in sorted(genre_mapping)]

        base_columns = [
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
        ]
        columns = base_columns + [f"genre_{index}" for index in range(len(genre_names))]
        genre_columns = [f"genre_{index}" for index in range(len(genre_names))]

        items = pd.read_csv(
            self.items_path,
            sep="|",
            header=None,
            names=columns,
            encoding="latin-1",
            dtype="string",
            keep_default_na=False,
            engine="python"
        )

        if items.empty:
            raise ValueError("u.item为空")

        def build_genres(row: pd.Series):
            genres = []

            for index, column in enumerate(genre_columns):
                if row[column] == "1":
                    genres.append(genre_names[index])
            return "|".join(genres)

        items["genres"] = items.apply(build_genres, axis=1)
        items = items[
            [
                "item_id",
                "title",
                "release_date",
                "video_release_date",
                "imdb_url",
                "genres",
            ]
        ]
        if items["item_id"].duplicated().any():
            raise ValueError("u.item 中存在重复 item_id。")

        return items.reset_index(drop=True)

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 返回用户行为和物品信息
        interactions = self.load_interactions()
        items = self.load_items()

        interaction_item_ids = set(interactions["item_id"].unique())
        metadata_item_ids = set(items["item_id"].unique())

        missing_metadata = (interaction_item_ids - metadata_item_ids)

        if missing_metadata:
            raise ValueError(
                "以下交互中的 item_id 没有对应的物品元数据："
                f"{sorted(missing_metadata)[:10]}"
            )

        return interactions, items
