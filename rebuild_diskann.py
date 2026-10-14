"""Build a matching compact DiskANN index from existing corpus embeddings."""

import argparse
import json
import sqlite3
from pathlib import Path


def main():
    import numpy as np
    from leann import LeannBuilder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("storage/corpus.db"))
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--embedding-hash", required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must not already exist")
    meta = json.loads(args.template.read_text())
    connection = sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        total = connection.execute(
            "SELECT count(*) FROM chunks WHERE corpus_id = ?", (args.corpus_id,)
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT chunk_id, text, metadata, encoding_block FROM chunks "
            "JOIN dense_embeddings USING(chunk_hash) WHERE corpus_id = ? "
            "AND encoding_config_hash = ? ORDER BY chunk_id",
            (args.corpus_id, args.embedding_hash),
        ).fetchall()
    finally:
        connection.close()
    if not rows or len(rows) != total:
        raise ValueError(f"Embeddings missing: {len(rows)} / {total}")
    builder = LeannBuilder(
        backend_name="diskann",
        embedding_model=meta["embedding_model"],
        embedding_mode=meta["embedding_mode"],
        dimensions=meta["dimensions"],
        embedding_options={
            "base_url": "http://localhost:8002/v1",
            "api_key": "EMPTY",
            "use_no_proxy": True,
        },
        **meta["backend_kwargs"],
    )
    vectors = []
    identifiers = []
    for chunk_id, text, metadata, encoding in rows:
        identifier = str(chunk_id)
        builder.add_text(
            text,
            metadata={"id": identifier, "chunk_id": chunk_id, **json.loads(metadata)},
        )
        identifiers.append(identifier)
        vectors.append(np.frombuffer(encoding, dtype="<f4"))
    matrix = np.stack(vectors)
    if matrix.shape != (total, meta["dimensions"]) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid embedding matrix: {matrix.shape}")
    args.output.mkdir(parents=True)
    builder.build_index_from_arrays(
        str(args.output / "leann.index"), identifiers, matrix
    )
    print(f"Built {total:,} passages at {args.output}")


if __name__ == "__main__":
    main()
