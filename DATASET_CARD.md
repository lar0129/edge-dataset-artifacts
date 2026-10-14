---
language:
- en
task_categories:
- text-retrieval
tags:
- rag
- minirag
- faiss
- diskann
- reproducibility
size_categories:
- 100K<n<1M
---

# xrag prebuilt artifacts

Eight audited artifact bundles for LihuaWorld, ConcurrentQA, and
ConcurrentQA-Extended. Download tooling and documentation:
[lar0129/xrag-artifacts](https://github.com/lar0129/xrag-artifacts).

`manifest.json` pins an immutable commit containing the archives under
`releases/v1/` and records per-archive/per-file SHA-256 checksums, install paths,
corpus fingerprints, validation results, and known provenance limits.

| Dataset | Bundles |
|---|---|
| LihuaWorld | 425-document TXT corpus + QA, complete historical MiniRAG graph, Flat |
| ConcurrentQA | Complete 2,743-document MiniRAG graph, Flat, matching rebuilt DiskANN |
| ConcurrentQA-Extended | Flat (991,397 vectors, 991,376 distinct texts), DiskANN (991,376 passages) |

No standalone ConcurrentQA/Extended corpus dumps, Extended graph, or LihuaWorld
DiskANN are distributed. Retrieval indexes contain the text needed to return
passages; upstream data terms still apply. Model weights are not included.
Compact DiskANN requires a matching query-time embedding service.

See [REPRODUCING.md](REPRODUCING.md) for source revisions, reconstruction rules,
version selection, data terms, and the difference between structural validation
and full end-to-end reproduction.
