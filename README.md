# GenRec：基于 Semantic ID 的生成式推荐

## 1. 项目简介

GenRec 是一个面向推荐算法与搜索算法岗位的生成式推荐项目。项目使用 MovieLens 100K 作为流程验证数据集，目标是从传统推荐基线逐步实现 Semantic ID 和 TIGER-style 自回归生成式推荐。

当前主线：

    用户历史行为
        -> Semantic ID 序列
        -> 自回归生成模型
        -> 生成下一个物品的 Semantic ID
        -> 映射回 item_id
        -> Top-K 推荐

项目重点不是把通用聊天 LLM 接到推荐系统后面，而是研究如何把物品表示成离散 Semantic ID，并让生成模型直接生成物品表示。

## 2. 当前完成情况

已完成：

- Python 虚拟环境与依赖检查；
- MovieLens 100K 数据适配；
- 交互数据标准化、过滤和时间切分；
- Popularity、Item-KNN、Time-aware / Popularity-regularized Item-KNN；
- SASRec 与 Advanced SASRec 对照实验；
- Recall@K、NDCG@K、MRR@K 等评估指标；
- Item 文本编码器；
- Residual Vector Quantizer 与 Advanced Residual Vector Quantizer；
- Item ID 与 Semantic ID 的双向映射；
- 训练、验证、测试 Semantic ID 序列数据；
- TIGER-style Semantic ID 自回归生成模型、训练器和推荐器；
- TIGER 约束 Beam Search 解码与测试集评估；
- 结果 JSON、HTML 表格和配置快照。

尚未完成：

- HSTU-lite；
- API 或网页 Demo。

## 3. 环境

推荐使用 Python 3.10 和项目虚拟环境。

PowerShell：

    cd D:\LLM\LLM4Rec
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python scripts/tools/check_env.py

当前开发环境曾验证过：

- Python 3.10.10；
- PyTorch 2.6.0；
- CUDA build 12.6；
- NVIDIA GTX 1650，约 4GB 显存。

4GB 显存适合运行 MovieLens 100K 的小规模实验。训练更大的生成模型时，可以将配置中的 device 改为 cpu，或迁移到云端 GPU。

## 4. 目录结构

    GenRec/
    ├── configs/
    │   └── base.yaml                       # 统一实验配置
    ├── data/
    │   ├── raw/                            # 原始数据，不提交 Git
    │   └── processed/                      # 预处理和 Semantic ID 数据，不提交 Git
    ├── results/                            # 实验结果、HTML 表格和配置快照
    ├── checkpoints/                        # 模型权重，不提交 Git
    ├── scripts/
    │   ├── data/
    │   │   └── preprocess.py               # 数据预处理总入口
    │   ├── baselines/
    │   │   ├── run_item_knn.py
    │   │   ├── run_sasrec.py
    │   │   ├── run_advanced_sasrec.py
    │   │   └── run_time_aware_popularity_item_knn.py
    │   ├── semantic_id/
    │   │   ├── build_semantic_ids.py       # 生成物品 Semantic ID
    │   │   ├── validate_semantic_ids.py    # 验证冲突率和重构误差
    │   │   └── build_semantic_sequences.py # 构造序列训练数据
    │   ├── generative/
    │   │   ├── run_tiger.py                # 训练 TIGER
    │   │   └── evaluate_tiger.py           # 评估 TIGER
    │   └── tools/
    │       └── check_env.py
    ├── src/
    │   ├── datasets/
    │   │   └── adapters/                   # 数据集适配器
    │   ├── preprocessing/                  # 标准化、过滤、切分
    │   ├── representations/
    │   │   └── semantic_id/                # 文本编码、量化、映射
    │   ├── sequences/                      # Semantic ID 序列适配
    │   ├── models/
    │   │   ├── baselines/                  # 推荐模型和基线
    │   │   │   ├── sasrec/
    │   │   │   └── advanced_sasrec/
    │   │   └── generative/
    │   │       └── tiger/                  # TIGER 模型、训练器和推荐器
    │   └── evaluation/                     # 评估指标
    ├── tests/
    ├── requirements.txt
    ├── 项目方案.md
    ├── 当前项目进展.md
    └── 今日学习记录.md

------------------------------------------------------------------------------------
datasets/adapters是数据集适配器，只负责“读取和统一格式”，不负责低频过滤、数据切分或生成 Semantic ID。
    如MovieLens 100K 原始文件有：u.data   u.item   u.user
    适配器负责： 读取这些文件；解析字段；统一列名；处理编码和分隔符；输出统一的 interactions 和 items 表。

统一用户 ID、物品 ID、时间戳的数据类型；
处理缺失值、非法记录和编码问题；
保留原始字段，便于后续分析；

输出统一结构，例如：
    interactions:
    user_id | item_id | rating | timestamp

    items:
    item_id | title | genres | release_date
-----------------------------------------------------------------------------------
preprocessing是通用数据集预处理，包括：标准化、过滤、切分，
    主要工作：字段标准化；评分过滤；低频用户和低频物品过滤；按时间排序；划分训练集、验证集和测试集。
标准化:
    -去重与非法记录处理
    -评分过滤
    -低频用户/物品过滤
    -按时间排序
    -按用户划分 train / valid / test
低频过滤通常需要迭代执行，因为删除低频用户后，某些物品可能又变成低频物品。
-----------------------------------------------------------------------------------
representations负责“如何表示物品”，
    如：title + genres-> ItemTextEncoder --------> 
        连续向量 embedding -> Residual Vector Quantizer -> Semantic ID
    结果：item_id = 42  Semantic ID = [12, 7, 41]


src/representations/semantic_id/，是当前真正的 Semantic ID 实现目录。
src/semantic_id/是为旧代码和旧模型文件保留的兼容层

-encoder.py：把物品文本转换成连续向量。
    -流程：title + genres -> TF-IDF -> TruncatedSVD -> L2 归一化 -> item embedding
    -输入：item_id, title, genres
    -输出：item_id -> [0.12, -0.03, ...]

-quantizer.py：基础版残差向量量化器。
    -流程：embedding -> codebook 0 -> 残差 -> codebook 1 -> 残差 -> codebook 2 -> Semantic ID
    -输入：[0.12, -0.03, ...] 向量
    -输出：[12, 7, 41] 三层表示
-advanced_quantizer.py：改进版残差向量量化器
    -当前改进：Beam Search 候选 Semantic ID，为一个物品保留多个候选编码。

-mapper.py：管理 item_id 和 Semantic ID 之间的映射。
    -输入：item_id,     code_0,     code_1,     code_2,     semantic_id
            1,          12,         7,          41,          12-7-41
    -功能： 加载 Semantic ID 映射文件；检查 code 是否完整、合法；检查 semantic_id 是否与各层 code 一致；
            item_id -> codes；item_id -> semantic_id；semantic_id -> item_id；codes -> token IDs；
            将交互表转换成带 Semantic ID 的交互表；添加 PAD、BOS、EOS 和分隔符 token。
-----------------------------------------------------------------------------------------------
sequences：负责“把数据组织成模型输入序列”
    用户 A:
    item_1 -> [12, 7, 41]
    item_5 -> [3, 8, 22]
    item_9 -> [17, 4, 6]
==>
    输入序列:
    [
    [12, 7, 41],
    [3, 8, 22],
    [17, 4, 6]
    ]
-----------------------------------------------------------------------------------------------------
models负责定义和训练推荐模型
---------------------------------------------
evaluation负责统一计算
-------------------------------------------------------------------

## 5. 数据处理流程
原始数据
  |
  v
datasets/adapters
读取不同数据集并统一字段
  |
  v
preprocessing
标准化、过滤、按时间切分
  |
  +----------------------+
  |                      |
  v                      v
interactions             items
用户行为表                物品元数据表
                             |
                             v
                    representations
                    文本向量与 Semantic ID
                             |
                             v
                       sequences
              历史行为 -> 预测目标 -> 模型输入
                             |
                             v
                         models
                   SASRec / TIGER 等
                             |
                             v
                       evaluation
                    Recall / NDCG / MRR
    ----------------------------------------------------------------
预处理总脚本：scripts/data/preprocess.py
预处理流程： 标准化 -> 过滤 -> 时间切分 -> 保存文件
预处理生成结果： 
    interactions.csv    标准化后的完整交互表        user_id | item_id | rating | timestamp
    items.csv           标准化后的物品元数据表      tem_id | title | release_date |........
    train.csv           训练集用户行为             user_id | item_id | rating | timestamp
    valid.csv           验证集用户行为
    test.csv            测试集用户行为
    stats.json          记录每个阶段的数据规模

preprocess.py中的方法:
    -PROJECT_ROOT = Path(__file__).resolve().parents[2] 向上两层后得到根目录
    -resolve_path:把配置中的相对路径转换为绝对路径。
    -load_config:是读取 YAML 配置文件,检查结果是否为字典,返回配置字典
    -build_adapter:根据 YAML 中的 adapter 字段创建适配器,如MovieLens100KAdapter(raw_data_dir)
    -summarize_data：统计当前数据规模
    -to_jsonable：把一些不能直接写入 JSON 的 Python 对象转换成可序列化形式。如：Path -> 字符串
    -save_csv：保存 DataFrame，自动创建父目录，使用 UTF-8 编码；不保存 pandas 默认索引。

preprocess.py的main()流程：
    -读取配置：config = load_config(config_path)
    -创建适配器：adapter = build_adapter(config)，（调用如MovieLens100KAdapter）
    -读取原始数据：interactions, items = adapter.load()
    -标准化：调用normalize.py.py中的normalize_dataset
    -过滤低频用户和物品:调用fliter.py中的filter_dataset
    -按时间切分：调用split.py中的split_dataset
    -确定输出路径：processed_dir = resolve_path(config["processed_data_dir"],PROJECT_ROOT,)
    -保存数据：  save_csv(normalized_interactions, interactions_path)
                save_csv(normalized_items, items_path)
                save_csv(train, train_path)
                save_csv(valid, valid_path)
                save_csv(test, test_path)
        注：interactions.csv和items.csv保存的是只经过标准化，未经过低频过滤和切分的数据
            train、valid、test保存的是经过完整处理的数据
    ---------------------------------------------------------------------------
调用关系：
            scripts/data/preprocess.py
                    |
                    v
            MovieLens100KAdapter
                    |
                    v
            interactions, items
                    |
                    v
            normalize_dataset()
                    |
                    v
            filter_dataset()
                    |
                    v
            split_dataset()
                    |
                    v
            保存 CSV 和统计信息
    ---------------------------------------------------------------------
MovieLens100KAdapter 把 MovieLens 100K 的原始文件转换成项目统一使用的两个表：
    interactions：用户行为表
    items：物品信息表

MovieLens100KAdapter中的方法：
    _check_required_files:检查u.data、u.item、u.genre是否存在
    _load_genre_mapping：读取u.genre类别编号到类别名称的映射，输出：
            {0: 'unknown', 1: 'Action', 2: 'Adventure', ..., 18: 'Western'}
    load_interactions：读取 u.data，生成用户行为表。
    load_items：u.item，并将 19 个类别标记转换成一个 genres 字符串字段。
                读取类别映射，构造原始列名，读取 u.item，构造 genres
    load：对外暴露的总入口，
        依次调用：interactions = self.load_interactions()
              和：items = self.load_items()
        对外返回：interactions：用户行为表
                  items：物品信息表
    -------------------------------------------------------------------
normalize.py的把adapter输出的两张表变成项目内部统一格式：
    输入示例：
        user_id | item_id | rating | timestamp
        " u1 "  | " i1 "  | "5"    | "2020-01-01"
        "u1"    | "i2"    | "3"    | "2020-01-02"
    输出示例：
        user_id | item_id | rating | timestamp  | is_positive
        u1      | i1      | 5.0    | 1577836800 | True
        u1      | i2      | 3.0    | 1577923200 | False

normalize.py中的方法：
    _require_columns：检查输入是否为 DataFrame以及必要字段是否存在
    _normalize_id_column：转换为 pandas 的字符串类型并删除首尾空格，检查空 ID
    _normalize_timestamp：作用是把不同形式的时间统一成Unix 秒级时间戳
    normalize_interactions：
        -检查必要字段
        -检查空表
        -复制 DataFrame
        -标准化 user_id和item_id
        -标准化时间戳
        -处理显式评分、检查评分范围、统一评分类型、生成 is_positive
        -处理隐式反馈数据，输入表没有 rating 列，默认rating = 1.0，is_positive = True
        -可选去重，只删除完全相同的行
        -排序：先按 user_id，再按 timestamp、最后按原始行顺序
    normalize_items：
        -检查必要字段   _require_columns(items,ITEM_REQUIRED_COLUMNS,"items",)
        -检查空表
        -复制原始表： normalized = items.copy()
        -标准化 item_id： normalized["item_id"] = _normalize_id_column(...)
        -检查重复 ID
        -处理文本字段：.astype("string").fillna("").str.strip()
        -返回物品表
    normalize_dataset：两张表的统一入口
    ------------------------------------------------------------
filter.py 负责从标准化后的数据中筛选出适合训练推荐模型的交互，并同步筛选物品元数据。
    流程：
        -标准化后的 interactions
        -可选：只保留正向交互
        -迭代过滤低频用户和低频物品
        -检查 item_id 是否都有元数据
        -同步过滤 items

filter.py中的方法：
    _validate_interactions：检查输入类型、必要字段、是否为空
    _validate_items：检查物品表DataFrame、存在 item_id、不能是空表
    _validate_threshold：检查频次阈值是否合法。
    filter_positive_interactions：保留正向交互，表必须先经过normalize.py（因为需要is_positive）
    _get_item_counts：统计每个物品的频次，用于判断物品是否低频。多种模式，如交互用户数或交互总次数
    iterative_filter_interactions：迭代式的 k-core 过滤：
        -检查参数
        -复制数据
        -记录初始统计
        -开始迭代，过滤低频用户、过滤低频物品、记录变化
        -判断是否收敛
    filter_dataset：对外提供的总入口
        -检查两张表： _validate_interactions(interactions) _validate_items(items)
        -是否只保留正向交互： 是否使用filter_positive_interactions
        -执行迭代低频过滤： filtered_interactions, stats = (iterative_filter_interactions(...))
        -检查物品元数据完整性
        -同步过滤物品表
    ---------------------------------------------------------------
split.py 的作用是按照用户行为的时间顺序，把过滤后的交互表拆成：train.csv、valid.csv、test.csv
当前只有一种策略：leave_last_two

split.py中的方法：
    _validate_interactions：检查必要字段、检查空表、检查关键字段中的空值
    _check_item_coverage：检查验证集和测试集中的物品是否在训练集出现过。这个检查用于识别物品冷启动。
    leave_last_two_split：真正执行切分的函数
    split_dataset：这是对外统一入口
--------------------------------------------------------------------------------------

## 6. 基线实验

### 6.1 Popularity

按训练集中的交互次数或独立用户数统计物品流行度，向所有用户推荐相同的热门物品，作为最简单的非个性化基线。

### 6.2 Item-KNN

根据用户历史交互构造物品共现关系，使用余弦相似度查找相似物品，再累加历史物品对候选物品的贡献。

### 6.3 Advanced Item-KNN

在 Item-KNN 基础上加入：

- 时间衰减：近期行为权重更高；
- 流行度惩罚：降低过热门物品的分数。

### 6.4 SASRec

使用 Transformer 自注意力建模用户行为序列，预测用户下一个可能交互的物品。它是生成式推荐主线前的重要序列建模基线。

### 6.5 TIGER-style 生成式推荐

TIGER 使用物品的 Semantic ID 作为离散目标序列。模型根据用户历史 Semantic ID，自回归生成目标物品的多层 code，再映射回原始 `item_id`。

运行示例：

    python scripts/generative/run_tiger.py --config configs/base.yaml --device cuda --batch-size 8 --epochs 10
    python scripts/generative/evaluate_tiger.py --config configs/base.yaml --device cuda

TIGER 的训练结果默认保存在：

    results/ml-100k/tiger/
    checkpoints/ml-100k/tiger_best.pt

其他基线运行示例：

    python scripts/baselines/run_item_knn.py --config configs/base.yaml
    python scripts/baselines/run_sasrec.py --config configs/base.yaml
    python scripts/baselines/run_advanced_sasrec.py --config configs/base.yaml

实验结果默认保存在：

    results/ml-100k/

其中包括 JSON 结果、训练历史、配置快照和 HTML 对比表。

## 7. Semantic ID 流程

Semantic ID 流程将物品的标题和类型等文本字段编码成稠密向量，再使用残差向量量化器将向量转换成短离散码。

    items.csv
        -> ItemTextEncoder
        -> item embedding
        -> Residual Vector Quantizer
        -> Semantic ID，例如 [12, 7, 41]
        -> item_id <-> Semantic ID 映射
        -> Semantic ID 序列

生成 Semantic ID：

    python scripts/semantic_id/build_semantic_ids.py --config configs/base.yaml --quantizer-mode advanced

验证 Semantic ID：

    python scripts/semantic_id/validate_semantic_ids.py --config configs/base.yaml --mode advanced

构造序列数据：

    python scripts/semantic_id/build_semantic_sequences.py --config configs/base.yaml --mode advanced

主要检查指标：

- collision_rate：不同物品映射到相同 Semantic ID 的比例；
- unique_semantic_ids：唯一 Semantic ID 数量；
- reconstruction_mse：量化向量与原始向量之间的重构误差；
- 训练、验证、测试序列数量和形状。

当前 Advanced Semantic ID 实验已经在 MovieLens 100K 上生成了 3 段编码、码本大小 64 的 Semantic ID，并完成了零冲突验证。

TIGER 使用这些 Semantic ID 构造历史序列和目标序列：

    历史 Semantic ID 序列
        -> TIGER Transformer
        -> codebook 0 token
        -> codebook 1 token
        -> codebook 2 token
        -> EOS
        -> item_id

## 8. 评估指标

项目主要使用 Top-K 排序指标：

- Recall@K：真实目标是否出现在前 K 个推荐中；
- NDCG@K：考虑目标位置的折损排序质量；
- MRR@K：真实目标首次出现位置的倒数。

所有模型应使用相同的数据切分、目标定义和 K 值进行比较。

## 9. 配置说明

统一配置位于：

    configs/base.yaml

配置包含：

- 数据路径；
- 预处理参数；
- Item-KNN 参数；
- SASRec 参数；
- Advanced SASRec 参数；
- Semantic ID 编码器和量化器参数；
- 序列长度、batch size、训练轮数；
- 结果目录和 checkpoint 路径。

修改配置后，应保留实验结果目录中的 config_snapshot.yaml，这样可以追溯实验使用的参数。

## 10. 可复现实验建议

建议每次实验遵循以下顺序：

    1. 检查虚拟环境
    2. 执行数据预处理
    3. 构建 Semantic ID
    4. 验证 Semantic ID
    5. 构建 Semantic ID 序列
    6. 训练或运行模型
    7. 查看 JSON 和 HTML 结果
    8. 保存配置快照
    9. 提交代码和必要的实验结果

数据文件、模型权重、日志和 Python 缓存已通过 .gitignore 排除。代码、配置、结果摘要和文档可以提交到 Git。

## 11. 后续开发路线

下一阶段建议按以下顺序推进：

1. 优化 TIGER 的候选补全策略，减少少量用户推荐不足 Top-K 的情况；
2. 增加生成错误分析，例如非法 Semantic ID、重复候选和热门物品偏置；
3. 完成 TIGER 与 Popularity、Item-KNN、Advanced Item-KNN、SASRec 的统一对比；
4. 在需要时实现 HSTU-lite 或加入检索增强模块；
5. 最后再封装 API 或网页 Demo。

## 12. 项目定位

简历中可以将项目概括为：

> 构建一个基于 Semantic ID 的生成式推荐系统：完成 MovieLens 100K 数据处理、Popularity/Item-KNN/SASRec 基线，使用文本编码器和残差向量量化器生成离散物品 Semantic ID，并构造可供 TIGER-style 自回归模型训练的序列数据。

项目结构重构说明见 REMADE.md。

## 13. 当前 TIGER 评估结果

当前 TIGER 实验使用 Advanced Semantic ID、3 个 codebook、每个 codebook 大小为 64，模型使用 2 层 Transformer，`embedding_dim=64`，训练 `batch_size=8`，解码 `beam_width=32`。

| 数据集 | Recall@10 | NDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| Validation | 0.160981 | 0.084800 | 0.061869 |
| Test | 0.141791 | 0.067361 | 0.044839 |

评估边界为：

    验证集：train -> valid
    测试集：train + valid -> test

详细结果位于：

    results/ml-100k/tiger/evaluation.json
    results/ml-100k/tiger/evaluation_config.yaml
    results/ml-100k/tiger/test_recommendations.json

所有主要模型的汇总对比见：

    results/ml-100k/baseline_comparison.html
