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
- 结果 JSON、HTML 表格和配置快照。

尚未完成：

- TIGER-style 自回归生成模型；
- 基于 Semantic ID 的生成式训练与评估；
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
    │   │   └── baselines/                  # 推荐模型和基线
    │   │       ├── sasrec/
    │   │       └── advanced_sasrec/
    │   └── evaluation/                     # 评估指标
    ├── tests/
    ├── requirements.txt
    ├── 项目方案.md
    ├── 当前项目进展.md
    └── 今日学习记录.md

src/semantic_id/ 中保留了少量兼容层，用于兼容旧代码和已经保存的 pickle 文件；新的实现位于 src/representations/semantic_id/ 和 src/sequences/。

## 5. 数据处理流程

原始 MovieLens 数据先经过适配器转换为统一格式，再执行标准化、正向交互过滤、低频用户/物品过滤和时间切分。

    原始 u.data / u.item / u.user
        -> MovieLens100KAdapter
        -> interactions.csv + items.csv
        -> normalize
        -> filter
        -> leave-last-two split
        -> train.csv / valid.csv / test.csv

统一交互字段：

| 字段 | 含义 |
| --- | --- |
| user_id | 用户 ID |
| item_id | 物品 ID |
| rating | 评分 |
| timestamp | Unix 时间戳 |

执行预处理：

    python scripts/data/preprocess.py --config configs/base.yaml

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

运行示例：

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

1. 实现 TIGER-style Semantic ID 自回归生成模型；
2. 使用 train.npz 训练模型，使用 valid.npz 选择 checkpoint；
3. 在 test.npz 上报告 Recall@K、NDCG@K 和 MRR@K；
4. 与 Popularity、Item-KNN、Advanced Item-KNN、SASRec 对比；
5. 增加生成错误分析，例如非法 Semantic ID、重复候选和热门物品偏置；
6. 在需要时实现 HSTU-lite 或加入检索增强模块；
7. 最后再封装 API 或网页 Demo。

## 12. 项目定位

简历中可以将项目概括为：

> 构建一个基于 Semantic ID 的生成式推荐系统：完成 MovieLens 100K 数据处理、Popularity/Item-KNN/SASRec 基线，使用文本编码器和残差向量量化器生成离散物品 Semantic ID，并构造可供 TIGER-style 自回归模型训练的序列数据。

REMADE.md 是当前项目说明文件；如果希望使用 GitHub 常见的默认展示文件名，可以将它重命名为 README.md。
