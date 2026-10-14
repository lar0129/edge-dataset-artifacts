# Edge Dataset Artifacts

[中文版](README.md) | English

Audited prebuilt indexes for xrag. GitHub stores the downloader, release manifest,
and reproduction instructions; [Hugging Face](https://huggingface.co/datasets/argsLiang/edge-dataset-artifacts)
stores eight independently downloadable archives. Downloads are pinned to an
immutable Hub commit and verified with SHA-256 at both archive and file level.

## Repository layout

```text
edge-dataset-artifacts/
├── README.md                 # Chinese guide
├── README.en.md              # English guide
├── requirements.txt
├── scripts/                  # Download, audit, publish, and run tools
├── manifests/manifest.json   # Pinned release and integrity metadata
├── config/sources.json       # Maintainer build recipe
├── docs/                     # Reproduction notes and HF dataset card
└── tests/                    # Offline verification tests
```

## Download

Python 3.10+ is sufficient for downloading; running xrag requires its own environment.

```bash
git clone https://github.com/lar0129/edge-dataset-artifacts.git
python -m pip install -r edge-dataset-artifacts/requirements.txt
python edge-dataset-artifacts/scripts/artifactctl.py list
python edge-dataset-artifacts/scripts/artifactctl.py download --dataset concurrentqa --root /path/to/xrag
python edge-dataset-artifacts/scripts/artifactctl.py verify --dataset concurrentqa --root /path/to/xrag
```

Select one artifact with `--artifact concurrentqa-flat`, a family with
`--kind graph`, or explicitly use `--all`. Repeated filters are supported.
Downloads use the Hugging Face cache and support resuming through the Hub client.
Use `--cache-dir /large/disk/hf-cache` to choose its location. Public downloads do
not require login. `HF_ENDPOINT` can select a compatible Hub mirror.

The installer verifies existing matching files and skips downloading them.
It refuses to overwrite a different existing directory: use a clean checkout or
move that directory aside yourself. This includes Git LFS pointer files in an
existing `data/LiHuaWorld` directory. Extraction is staged in the target filesystem,
checks an explicit file allowlist, rejects links/path traversal, and only exposes
the final directory after validation. Budget space for both the compressed archive
in the cache and the uncompressed index. `list` reports compressed sizes.

FAISS/LEANN metadata includes Python pickle files. Install only a trusted release;
the downloader verifies bytes without executing them, while the retrieval runtime
deserializes these files.

## Included artifacts

| Dataset | Corpus | MiniRAG graph | Flat | DiskANN |
|---|---|---|---|---|
| LihuaWorld | 425 week TXT files + QA | Complete 425-document historical 4B graph | 425 vectors | Not distributed |
| ConcurrentQA | Reconstruct from HF | Complete April 30 4B graph, 2,743 documents | 2,743 vectors | Rebuilt to match 2,743 documents |
| ConcurrentQA-Extended | Reconstruct from sources | Not distributed | 991,397 vectors / 991,376 distinct texts | 991,376 passages |

See [REPRODUCING.md](docs/REPRODUCING.md) for exact choices, known differences,
data provenance, dependencies, and configuration. `manifests/manifest.json` contains the
complete inventory, hashes, install paths, validation results, and pinned revision.
Corpus SQLite databases, embeddings caches, logs, model weights, and standalone
ConcurrentQA/Extended corpus dumps are not included. Text embedded in index
docstores is included because retrieval needs it.

## Use without rebuilding

The ordinary xrag workflow runs its offline stage before loading indexes and can
encode the corpus again when its SQLite cache is absent. Use the provided runner
to go straight to the online stage. From an installed xrag checkout:

```bash
python /path/to/edge-dataset-artifacts/scripts/run_prebuilt.py \
  --artifact concurrentqa-flat \
  --config configs/pipelines/eta/qwen/agentic_rag_answer_tool.yml --check-only

python /path/to/edge-dataset-artifacts/scripts/run_prebuilt.py \
  --artifact concurrentqa-flat \
  --config configs/pipelines/eta/qwen/agentic_rag_answer_tool.yml
```

The runner sets the dataset and embedding settings from the manifest, verifies
files and index configuration hashes, loads only the base QA evidence for Extended,
and checks evidence-text coverage against the release. Configure model endpoints
in xrag as usual. `--evaluate` also runs the configured evaluators, which may call
LLMs. `--check-only` needs no model endpoint or dataset download.

For DiskANN use `agentic_rag_leann_diskann.yml`; for graphs use `mini_rag.yml`.
The artifact selects the graph working directory automatically. Config filenames
above refer to `configs/pipelines/eta/qwen/` in the audited xrag checkout.

## Maintainers

The same files live under `artifacts/` in the xrag checkout. Run these commands from
that checkout, using its Python environment for audit/rebuild:

```bash
python artifacts/scripts/audit.py
python artifacts/scripts/publish.py pack
python artifacts/scripts/publish.py upload
python artifacts/scripts/publish.py github --tag
python -m unittest discover -s artifacts/tests -p 'test_*.py' -v
```

`config/sources.json` is a local build recipe: each artifact declares a dataset, kind,
source directory, file globs, install path, and reference corpus. Audit fails on
corpus mismatch, missing mappings, invalid vectors, or broken graph references.
The public manifest omits local source directories. `pack` is deterministic and
resumes completed packages. `upload` uses the configured HF account; `github` uses
`gh` and publishes only an explicit allowlist of tooling/documentation files.
Create the public GitHub repository with an initial README before `github`.

For a new release, change `release` in `config/sources.json`, audit, package, and publish
again. Keep published tags immutable. Adding another dataset using an existing
index type only needs a new source recipe; a new storage format needs its auditor.
Do not audit untrusted pickle files: the audit deliberately loads local FAISS and
LEANN metadata. The downloader itself never deserializes pickle.

For tooling or documentation updates to an existing release, use `publish.py github`
without `--tag`. This updates the default branch without changing the published tag
or reuploading the artifact archives.
