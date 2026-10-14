# Edge Dataset Artifacts

中文 | [English](README.en.md)

为 xrag 复现提供经过核对的预构建索引。GitHub 保存下载工具、版本清单和说明，
[Hugging Face](https://huggingface.co/datasets/argsLiang/edge-dataset-artifacts)
保存大文件。下载后可以直接加载索引，跳过耗时的离线构建。

当前发布 **v1，共 8 项产物，压缩后约 6.60 GiB**。
每项可以独立下载；下载固定到 HF commit，并校验压缩包和解压后每个文件的 SHA-256。

## 目录结构

```text
edge-dataset-artifacts/
├── README.md                 # 中文使用说明
├── README.en.md              # English guide
├── requirements.txt          # 下载工具依赖
├── scripts/                  # 下载、核对、打包、发布和运行入口
│   ├── artifactctl.py
│   ├── run_prebuilt.py
│   ├── audit.py
│   ├── rebuild_diskann.py
│   └── publish.py
├── manifests/manifest.json   # 发布版本、下载路径及文件校验值
├── config/sources.json       # 维护者使用的本地产物来源配置
├── docs/
│   ├── REPRODUCING.md         # 版本选择、数据来源与复现限制
│   └── DATASET_CARD.md        # HF 数据集介绍
└── tests/                    # 下载、安装和配置验证测试
```

## 包含哪些产物

| 数据集 | 原始 corpus | MiniRAG-graph | Flat-index | DiskANN-index |
|---|---|---|---|---|
| LihuaWorld | 425 篇周记录 TXT + QA | 完整覆盖 425 篇的历史 4B 版本 | 425 条向量 | 不备份 |
| ConcurrentQA | 从 HF 获取 | 4 月 30 日 4B 版本，2,743 篇 | 2,743 条向量 | 重建的匹配版本，2,743 篇 |
| ConcurrentQA-Extended | 从源数据重建 | 不备份 | 991,397 条向量，991,376 篇不同正文 | 991,376 篇 |

版本选择依据是目标语料覆盖、索引一致性及构建时间。旧 ConcurrentQA DiskANN 对应
train split，已用已有 embedding 重建匹配 test split 的版本。Extended Flat 的 21 条
重复正文保留原样，具体记录见 [复现说明](docs/REPRODUCING.md)。

不包含 SQLite corpus 数据库、embedding 缓存、日志、模型权重，或单独导出的
ConcurrentQA / Extended 原始语料。索引内部用于返回检索结果的正文仍然保留。

## 下载与校验

下载工具需要 Python 3.10+；运行 xrag 需要另外安装 xrag 的依赖。

```bash
git clone https://github.com/lar0129/edge-dataset-artifacts.git
python -m pip install -r edge-dataset-artifacts/requirements.txt

python edge-dataset-artifacts/scripts/artifactctl.py list
python edge-dataset-artifacts/scripts/artifactctl.py download --dataset concurrentqa --root /path/to/xrag
python edge-dataset-artifacts/scripts/artifactctl.py verify --dataset concurrentqa --root /path/to/xrag
```

`--root` 是准备运行 xrag 的仓库根目录，工具会将文件安装到其 `storage/` 或 `data/`。

| 选择方式 | 示例 |
|---|---|
| 指定单项产物 | `--artifact concurrentqa-flat` |
| 指定数据集 | `--dataset concurrentqa` |
| 指定索引类型 | `--kind graph` |
| 下载全部 | `--all` |

同一选项可以重复使用，不同选项之间取交集。公开下载不需要登录。
工具使用 HF 缓存和断点续传，可用 `--cache-dir /large/disk/hf-cache` 指定缓存位置，
或通过 `HF_ENDPOINT` 选择兼容镜像。

已存在且校验一致的产物会跳过下载；发现本地文件不一致时会报错，不覆盖已有目录。
如果 `data/LiHuaWorld` 中仍是 Git LFS 指针，请先拉取原始文件或自行移走该目录。
安装先写入临时目录，验证通过后再整体移入目标位置。磁盘空间需同时容纳压缩包缓存
和解压后的文件，`list` 命令可查看各包大小。

FAISS / LEANN 的元数据包含 pickle 文件。下载工具只校验文件，不执行 pickle；
检索程序加载时会反序列化，因此应使用可信发布版本。

## 跳过索引重建运行

xrag 原有入口会先执行离线流程，缺少 SQLite 缓存时可能重新编码语料。
本项目提供 `run_prebuilt.py`，用于验证预构建索引后直接进入在线流程。
在已安装依赖的 **xrag 仓库根目录**执行：

```bash
python /path/to/edge-dataset-artifacts/scripts/run_prebuilt.py \
  --artifact concurrentqa-flat \
  --config configs/pipelines/eta/qwen/agentic_rag_answer_tool.yml --check-only

python /path/to/edge-dataset-artifacts/scripts/run_prebuilt.py \
  --artifact concurrentqa-flat \
  --config configs/pipelines/eta/qwen/agentic_rag_answer_tool.yml
```

- `--check-only` 只检查文件及配置，无需下载数据集或启动模型服务。
- 去掉该参数后，会加载 QA、核对正文和文档 ID，然后执行在线查询。
- `--evaluate` 额外运行配置中的评估器，可能调用 LLM。
- DiskANN 使用 `agentic_rag_leann_diskann.yml`；图索引使用 `mini_rag.yml`。
  这些配置位于 xrag 的 `configs/pipelines/eta/qwen/`。
- 查询仍需按 xrag 的方式配置 embedding / LLM 服务。
  Compact DiskANN 会在查询时重算部分向量，需要匹配的 embedding 模型。

运行入口根据清单设置数据集及 embedding 参数，并核对索引配置哈希。
Extended 查询只加载基础 ConcurrentQA 的 QA 证据，无需重建百万篇扩展语料。

## 维护与扩展

在 xrag 仓库中，这套工具位于 `artifacts/`。核对和重建请使用 xrag 的 Python 环境：

```bash
python artifacts/scripts/audit.py
python artifacts/scripts/publish.py pack
python artifacts/scripts/publish.py upload
python artifacts/scripts/publish.py github --tag
python -m unittest discover -s artifacts/tests -p 'test_*.py' -v
```

`config/sources.json` 定义数据集、索引类型、本地来源目录、文件选择规则、安装位置及
参考 corpus。新增同类型产物只需扩展配置；新增存储格式时再补充对应核对逻辑。
公开清单会移除本地来源路径。

`audit` 验证语料覆盖、映射、向量和图引用；`pack` 可跳过已完成的包；`upload` 发布到 HF；
`github` 只发布明确列出的代码、清单和文档。首次发布前需创建带初始 README 的 GitHub 仓库。

发布新数据版本时修改 `config/sources.json` 的 `release`，重新核对和发布。
只更新代码或文档时使用 `publish.py github`，不加 `--tag`，保持历史版本标签不变。
维护者核对程序会加载本地 pickle，请勿用于不可信的索引文件。

## 复现边界与数据条款

发布前已验证产物结构、语料覆盖、Flat 自检查询和 DiskANN 原生加载，但尚未执行完整
LLM 问答评测。历史模型 revision、部分构建参数并未完整记录，不能保证从源数据重建出
逐字节相同的索引。要使用相同产物，请下载清单中固定的发布版本。

Extended 由 ConcurrentQA、CMU Enron 邮件及 Wikipedia 抽样构成，并非单个现成 HF 数据集。
数据来源 revision、重建规则、历史版本差异和上游条款见 [REPRODUCING.md](docs/REPRODUCING.md)。
