"""Audit trusted local xrag artifacts and create a publication manifest."""

import argparse
import base64
import hashlib
import importlib.metadata
import json
import pickle
import sqlite3
import struct
import subprocess
from pathlib import Path

from artifactctl import document_id_fingerprint, sha256


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(hashes):
    return hashlib.sha256("\n".join(sorted(set(hashes))).encode("ascii")).hexdigest()


def corpus_hashes(connection, corpus_id, strip=False):
    return {
        text_hash(text.strip() if strip else text)
        for (text,) in connection.execute(
            "SELECT text FROM chunks WHERE corpus_id = ?", (corpus_id,)
        )
    }


def compare_texts(actual, expected):
    missing = len(expected - actual)
    extra = len(actual - expected)
    if missing or extra:
        raise ValueError(f"Corpus mismatch: {missing} missing, {extra} extra texts")
    return {"unique_texts": len(actual), "text_set_sha256": fingerprint(actual)}


def audit_flat(source, expected):
    import faiss
    import numpy as np

    with (source / "index.pkl").open("rb") as handle:
        store, mapping = pickle.load(handle)
    index = faiss.read_index(str(source / "index.faiss"))
    if not isinstance(index, faiss.IndexFlatIP) or index.ntotal != len(mapping):
        raise ValueError("FAISS type or mapping length mismatch")
    if set(mapping) != set(range(index.ntotal)) or set(mapping.values()) != set(
        store._dict
    ):
        raise ValueError("FAISS mapping does not cover the document store")
    actual = {text_hash(document.page_content) for document in store._dict.values()}
    result = compare_texts(actual, expected)
    probes = [0, index.ntotal // 2, index.ntotal - 1]
    vectors = np.stack([index.reconstruct(position) for position in probes])
    scores, neighbors = index.search(vectors, 3)
    if not np.isfinite(scores).all() or (neighbors < 0).any():
        raise ValueError("FAISS self-query failed")
    for position, neighbor in zip(probes, neighbors[:, 0], strict=True):
        if text_hash(store._dict[mapping[position]].page_content) != text_hash(
            store._dict[mapping[int(neighbor)]].page_content
        ):
            raise ValueError("FAISS self-query returned another document")
    result.update(
        {
            "vectors": index.ntotal,
            "dimensions": index.d,
            "type": "IndexFlatIP",
            "duplicate_text_occurrences": index.ntotal - len(actual),
            "validation": "Loaded FAISS and docstore; checked mapping and three self-queries.",
        }
    )
    return result


def audit_graph(source, expected):
    import networkx as nx
    import numpy as np

    documents = json.loads((source / "kv_store_full_docs.json").read_text())
    chunks = json.loads((source / "kv_store_text_chunks.json").read_text())
    result = compare_texts(
        {text_hash(row["content"].strip()) for row in documents.values()}, expected
    )
    for row in chunks.values():
        if row["full_doc_id"] not in documents:
            raise ValueError("Graph chunk references a missing document")
    vector_counts = {}
    for path in sorted(source.glob("vdb_*.json")):
        payload = json.loads(path.read_text())
        matrix = np.frombuffer(
            base64.b64decode(payload["matrix"], validate=True), dtype=np.float32
        )
        identifiers = [row["__id__"] for row in payload["data"]]
        if (
            matrix.size != len(identifiers) * payload["embedding_dim"]
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(f"Invalid vector matrix: {path.name}")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate vector IDs: {path.name}")
        if path.name == "vdb_chunks.json" and set(identifiers) != set(chunks):
            raise ValueError("Graph chunk vectors and text IDs differ")
        vector_counts[path.name] = len(identifiers)
    graph = nx.read_graphml(source / "graph_chunk_entity_relation.graphml")
    missing_sources = set()
    for attributes in [attributes for _, attributes in graph.nodes(data=True)] + [
        attributes for _, _, attributes in graph.edges(data=True)
    ]:
        for source_id in attributes.get("source_id", "").split("<SEP>"):
            if source_id and source_id not in chunks:
                missing_sources.add(source_id)
    result.update(
        {
            "documents": len(documents),
            "chunks": len(chunks),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "vector_counts": vector_counts,
            "unresolved_graph_source_ids": len(missing_sources),
            "validation": "Parsed GraphML, matched all documents and chunk vector IDs, checked matrix shapes and finite values. No LLM retrieval run.",
        }
    )
    if missing_sources:
        raise ValueError(
            f"Graph has {len(missing_sources)} unresolved chunk references"
        )
    return result


def audit_diskann(source, expected):
    from leann_backend_diskann import _diskannpy

    meta = json.loads((source / "leann.index.meta.json").read_text())
    required = [
        "leann_disk_graph.index",
        "leann_partition.bin",
        "leann_pq_compressed.bin",
        "leann_pq_pivots.bin",
        "leann.index.ids.txt",
        "leann.index.passages.idx",
        "leann_disk.index_medoids.bin",
        "leann_disk.index_max_base_norm.bin",
    ]
    if meta["backend_name"] != "diskann" or any(
        not (source / name).is_file() for name in required
    ):
        raise ValueError("Missing DiskANN components")
    if meta.get("embedding_options", {}).get("api_key") not in (None, "", "EMPTY"):
        raise ValueError("Remove credentials from DiskANN metadata before publishing")
    with (source / "leann.index.passages.idx").open("rb") as handle:
        offsets = pickle.load(handle)
    identifiers = (source / "leann.index.ids.txt").read_text().splitlines()
    if len(set(identifiers)) != len(identifiers) or len(offsets) != len(identifiers):
        raise ValueError("DiskANN ID or offset count mismatch")
    actual = set()
    count = 0
    with (source / "leann.index.passages.jsonl").open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            row = json.loads(line)
            if str(row["id"]) != identifiers[count] or offsets[row["id"]] != offset:
                raise ValueError(
                    f"DiskANN passage ordering or byte offset mismatch: {count}"
                )
            actual.add(text_hash(row["text"]))
            count += 1
    if count != len(identifiers):
        raise ValueError("Missing DiskANN passages")
    with (source / "leann_pq_compressed.bin").open("rb") as handle:
        points, dimensions = struct.unpack("<II", handle.read(8))
    if (
        points != count
        or (source / "leann_pq_compressed.bin").stat().st_size
        != 8 + points * dimensions
    ):
        raise ValueError("DiskANN PQ header mismatch")
    result = compare_texts(actual, expected)
    prefix = str((source / "leann").resolve())
    native_index = _diskannpy.StaticDiskFloatIndex(
        _diskannpy.Metric.INNER_PRODUCT, prefix, 2, 0, 0, 0, "", prefix
    )
    del native_index
    result.update(
        {
            "passages": count,
            "dimensions": meta["dimensions"],
            "backend_kwargs": meta["backend_kwargs"],
            "embedding_model": meta["embedding_model"],
            "requires_embedding_server": True,
            "validation": "Checked all passage IDs, byte offsets, text coverage and PQ shape; native DiskANN index loaded successfully. No query-time embedding server test.",
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--sources", type=Path, default=Path(__file__).with_name("sources.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("manifest.json")
    )
    args = parser.parse_args()
    manifest = json.loads(args.sources.read_text())
    connection = sqlite3.connect(
        (args.root / "storage/corpus.db").resolve().as_uri() + "?mode=ro", uri=True
    )
    connection.execute("BEGIN")
    expected_cache = {}
    for artifact in manifest["artifacts"]:
        print(f"Auditing {artifact['id']}", flush=True)
        source = args.root / artifact["source"]
        key = (artifact["corpus_id"], artifact["kind"] in ("graph", "corpus"))
        if key not in expected_cache:
            expected_cache[key] = corpus_hashes(connection, *key)
        expected = expected_cache[key]
        if artifact["kind"] == "corpus":
            texts = {
                text_hash(path.read_text().strip())
                for path in source.glob("data/week*/*.txt")
            }
            result = compare_texts(texts, expected)
        else:
            result = {
                "flat": audit_flat,
                "graph": audit_graph,
                "diskann": audit_diskann,
            }[artifact["kind"]](source, expected)
        config = connection.execute(
            "SELECT config_data FROM corpus JOIN config_versions ON corpus_config_hash = config_hash WHERE corpus_id = ?",
            (artifact["corpus_id"],),
        ).fetchone()[0]
        artifact["corpus_config"] = json.loads(config)
        artifact["audit"] = result
        paths = sorted(
            {
                path
                for pattern in artifact["include"]
                for path in source.glob(pattern)
                if path.is_file()
            }
        )
        if not paths:
            raise ValueError(f"No files for {artifact['id']}")
        artifact["files"] = [
            {
                "path": path.relative_to(source).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ]
        if artifact["kind"] in ("flat", "diskann"):
            artifact["embedding"] = json.loads(
                connection.execute(
                    "SELECT config_data FROM config_versions WHERE config_hash = 'f14d686191f1c4e5'"
                ).fetchone()[0]
            )
        print(json.dumps(result), flush=True)
    manifest["qa_contracts"] = {}
    for dataset in ("lihuaworld", "concurrentqa"):
        rows = connection.execute(
            "SELECT text, metadata FROM chunks WHERE corpus_id = ?",
            (dataset + "_4ed329c09b8e6116",),
        )
        manifest["qa_contracts"][dataset] = document_id_fingerprint(
            (json.loads(metadata)["doc_id"], text) for text, metadata in rows
        )
    connection.close()
    manifest["provenance"] = {
        "xrag_snapshot_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.root, text=True
        ).strip(),
        "dependency_commits": {
            name: subprocess.check_output(
                ["git", "-C", str(args.root / "dep" / name), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            for name in ("minirag", "leann")
        },
        "audit_environment": {
            name: importlib.metadata.version(name)
            for name in (
                "faiss-cpu",
                "numpy",
                "langchain-community",
                "nano-vectordb",
                "networkx",
            )
        },
        "fingerprint_definition": "SHA256 of sorted distinct per-text SHA256 hex digests joined with LF; graph/corpus texts are stripped, flat/diskann texts are exact.",
        "warning": "Snapshot commits document the audit environment, not proven historical build commits. See per-artifact provenance.",
    }
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
