# Reproduction and provenance

## Release v1 decisions

Completeness is measured against the reference corpus, not directory size or date.

- **LihuaWorld graph:** the historical `minirag_v002` worktree's
  `LiHuaWorld-MiniRAG-Graph-nothinking-qwen3.5-4B` covers exactly all 425 week
  documents. April 18/19 candidates cover only 368/407. Older 442-document graphs
  contain 17 additional documents. The selected graph has 425 chunk vectors,
  1,212 graph nodes and 507 edges. Its original build commit could not be recovered
  from the relocated worktree. Adjacent historical config is supporting evidence,
  not a guarantee of every historical runtime option.
- **ConcurrentQA graph:** `Conc-MiniRAG-Graph-nothinking-qwen3.5-4B-0430-keywordFIX`
  covers all 2,743 documents. Other existing candidates cover 2,700, 2,708, or 2,740.
- **ConcurrentQA DiskANN:** rebuilt from the 2,743 documents and existing 1,024-D
  embeddings in SQLite, without calling an embedding model. The old 18,551-document
  backup exactly matches the HF train split, shares only 1,288 texts with this test
  corpus, and is excluded.
- **Extended:** Flat contains 991,397 vectors, with exactly 21 duplicate-text
  occurrences. Its 991,376 distinct texts exactly match SQLite and DiskANN.
  The original Flat vectors, IDs and duplicates are preserved. Scores between Flat
  and DiskANN are not claimed identical: duplicate multiplicity and ANN behavior differ.

All published FAISS indexes were loaded with their document stores and passed
three self-query probes. GraphML, all vector matrix shapes/finite values, document
coverage, chunk IDs and graph source references were checked. All DiskANN passage
IDs and byte offsets, corpus coverage, required files, and PQ shapes were checked.
Both DiskANN indexes also loaded successfully through the native backend.
These checks do not establish end-to-end answer accuracy. Query-time embedding and
generation still require model services.

## Corpus reconstruction

### LihuaWorld

The corpus archive installs `data/LiHuaWorld/data/week*/*.txt` and
`qa/query_set.{json,csv}`. All 425 week files are byte-identical between the inspected
old and new checkouts. `bakery_deliver` is excluded, matching the integrator.
MiniRAG strips surrounding whitespace; Flat preserves the original text, so their
text fingerprints differ even though document coverage matches.
The supplied QA set has 637 entries. Existing placeholder evidence such as `N/A`
and `What` does not resolve to TXT files; the integrator's warnings are preserved.

### ConcurrentQA

- HF dataset: `stanfordnlp/concurrentqa`.
- Cached source revision: `2cc9db602b10a561026c905403a641422f066369`.
- Split: `test`, 1,600 questions, full range.
- Traverse each question's `sp` in dataset order. For each nonempty `title` and
  `sents`, form `title + "\n" + "\n".join(sents)`. Keep the first text for each title.
  Assign document IDs in insertion order. This gives 2,743 documents.
- The original integrator downloads via HF and does not save a standalone corpus
  file. Its QA pairs refer to these document IDs; do not reorder documents when
  reconstructing them.

Example source download:

```python
from datasets import load_dataset

dataset = load_dataset(
    "stanfordnlp/concurrentqa",
    revision="2cc9db602b10a561026c905403a641422f066369",
    split="test",
)
```

The online runner currently uses xrag's integrator and verifies the resulting
evidence-text fingerprint. If upstream changes, use the source revision above;
do not silently accept another corpus.

### ConcurrentQA-Extended

This is a constructed dataset, not a single downloadable HF corpus:

1. Start with the ConcurrentQA test documents above, preserving their IDs.
2. Append parsed CMU Enron `enron_mail_20150507` emails from
   `https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz`.
   The repository constructs text from Subject, Sender, Recipients, maildir path,
   a separator and the decoded body. Empty bodies and parse failures are skipped.
3. Stream `wikimedia/wikipedia`, config `20231101.en`, split `train`, shuffle
   seed 42, buffer size 10,000, and take 500,000 rows. Skip empty texts and repeated
   titles; rename titles colliding with base evidence using `__ext`.
   The locally cached Wikipedia revision is
   `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`; the historical integrator did not pin
   a revision, so this is observed cache provenance, not proof of its exact input.
4. Keep whole documents (`identical_chunking`) and drop texts longer than 20,000
   characters. SQLite deduplicates by text hash. The stored release corpus has
   991,376 distinct texts.

Maildir traversal used `os.walk` without sorting. Exact historical ID order cannot
be guaranteed from source downloads alone, even with the same sampling seed.
The released indexes retain their own mappings and texts. For byte-identical
reproduction use the published artifacts; for source rebuilds compare corpus
fingerprints and preserve the base QA document IDs.

## Index and runtime configuration

- Flat: FAISS `IndexFlatIP`, `normalize_l2: true`, 1,024 dimensions.
- Flat/DiskANN embeddings: `Qwen/Qwen3-Embedding-0.6B`, truncation 2,048 tokens,
  local OpenAI-compatible provider. The release manifest records the complete
  xrag embedding configuration and the observed dependency versions.
- MiniRAG: Qwen3-Embedding-0.6B, 1,024 dimensions; Qwen3.5-4B extraction. LihuaWorld
  predates later keyword-extraction fixes. Its larger coverage does not imply the
  same extraction behavior as the April 30 ConcurrentQA graph.
- DiskANN: LEANN compact/recompute, inner product (`mips`), degree 32, build
  complexity 64, 8 threads, disk PQ 128 bytes, search/build memory budgets 8/16 GiB.
  The published `leann.index` is a prefix: the compact graph is stored in
  `leann_disk_graph.index`, not a file named `leann.index`.
- Runtime DiskANN config matching directory hash `0ac1b1dfe7b9d7ca`:
  `num_nodes_to_cache: 1000`, `search_complexity: 16`, `beam_width: 4`,
  `prune_ratio: 0.5`, `pruning_strategy: global`, `batch_recompute: true`,
  `dedup_node_dis: true`, `recompute_neighbors: false`. Use the audited
  `configs/pipelines/eta/qwen/agentic_rag_leann_diskann.yml`.
- LEANN requires native DiskANN/partition support from xrag's pinned dependency
  checkout. A plain upstream `diskannpy` installation is not sufficient.
- Compact DiskANN recomputes vectors at query time; the matching embedding model
  endpoint must be running. xrag refreshes the endpoint binding from your runtime
  configuration. Published metadata only contains localhost and the `EMPTY` placeholder.

The manifest's xrag/dependency commits identify the inspected code snapshot.
They are not asserted to be the original build commits for historical artifacts.
Model repository revisions and exact historical server settings were not recorded;
the release therefore preserves artifacts rather than claiming bit-for-bit rebuilds.

## Data terms

Index packages include document text in FAISS pickle stores, MiniRAG JSON, and
LEANN passages. This release does not assign a new blanket license to those texts.
Consult the original dataset cards and sources for applicable terms and attribution:

- https://huggingface.co/datasets/stanfordnlp/concurrentqa
- https://huggingface.co/datasets/wikimedia/wikipedia
- https://www.cs.cmu.edu/~enron/

LihuaWorld is supplied from the xrag project's local corpus. Use the artifacts for
the intended research reproduction and retain upstream attribution. Metadata,
sample text, logs, and credentials beyond the required retrieval payload are not
added to the release.
