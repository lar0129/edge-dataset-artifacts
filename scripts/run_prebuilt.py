"""Run an xrag online pipeline against a verified published index."""

import argparse
import copy
import hashlib
import sys
from pathlib import Path

from artifactctl import (
    DEFAULT_MANIFEST,
    document_id_fingerprint,
    read_manifest,
    safe_path,
    select_artifacts,
    verify_directory,
)


def configure(raw_config, artifact):
    config = copy.deepcopy(raw_config)
    offline = config["offline"]
    offline["datasource"]["component_key"] = artifact["dataset"]
    offline["datasource"]["split"] = "test"
    offline["datasource"]["dataset_range"] = [-1, -1]
    if artifact["kind"] == "graph":
        if config["online"]["operator"]["component_key"] != "minirag_v2":
            raise ValueError("Graph artifacts require a minirag_v2 pipeline config")
        working_dir = Path(artifact["install_path"]).name
        config["online"]["operator"]["working_dir"] = working_dir
        offline["minirag"]["working_dir"] = working_dir
        offline["minirag"]["doc_limit"] = 0
        operator = config["online"]["operator"]
        if (
            operator["embedding_model_name"] != artifact["build"]["embedding_model"]
            or operator["embedding_dim"] != artifact["build"]["dimensions"]
        ):
            raise ValueError(
                "Graph embedding configuration does not match the artifact"
            )
    elif artifact["kind"] in ("flat", "diskann"):
        expected = "faiss" if artifact["kind"] == "flat" else "leann"
        if offline.get("vector_index", {}).get("component_key") != expected:
            raise ValueError(
                f"Select a pipeline whose vector_index.component_key is {expected}"
            )
        offline["corpus"] = artifact["corpus_config"]
        offline["encodings"] = [{"encoder_type": "dense", **artifact["embedding"]}]
    else:
        raise ValueError("Select an index artifact, not the raw corpus")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument(
        "--config", type=Path, required=True, help="An xrag pipeline YAML"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify files and configuration without models or dataset downloads",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Also run the configured evaluators"
    )
    args = parser.parse_args()
    root = Path.cwd()
    if not (root / "xrag").is_dir():
        parser.error("Run from the root of an installed xrag checkout")
    sys.path.insert(0, str(root))
    manifest = read_manifest(args.manifest)
    artifact = select_artifacts(manifest, identifiers=[args.artifact])[0]
    verify_directory(safe_path(root, artifact["install_path"]), artifact)

    from omegaconf import OmegaConf

    from xrag.config.workflow_schema import ConfigValidator

    raw = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    config = ConfigValidator.validate_pipeline_config(configure(raw, artifact))
    if artifact["kind"] in ("flat", "diskann"):
        embedding = config.offline.encodings[0]
        directory = f"{embedding.get_hash_string()}_{config.offline.vector_index.get_hash_string()}"
        if directory != Path(artifact["install_path"]).name:
            raise ValueError(
                f"Index configuration hash {directory} does not match {Path(artifact['install_path']).name}. "
                "Use the published build/search configuration described in REPRODUCING.md."
            )
    print(f"Verified files and configuration for {artifact['id']}", flush=True)
    if args.check_only:
        return

    from xrag.config.config_schema import DataSourceConfig
    from xrag.data_source.data_integrator import datasource_registry
    from xrag.workflow.workflow_executor import WorkflowExecutor, _generate_report

    dataset = artifact["dataset"]
    qa_dataset = "concurrentqa" if dataset == "concurrentqa-extended" else dataset
    datasource = datasource_registry.create(
        DataSourceConfig(component_key=qa_dataset, split="test")
    ).integrate()
    reference = next(
        entry
        for entry in manifest["artifacts"]
        if entry["dataset"] == qa_dataset and entry["kind"] == "graph"
    )
    hashes = {
        hashlib.sha256(document.text.strip().encode()).hexdigest()
        for document in datasource.documents
    }
    fingerprint = hashlib.sha256("\n".join(sorted(hashes)).encode("ascii")).hexdigest()
    if fingerprint != reference["audit"]["text_set_sha256"]:
        raise ValueError(
            "Downloaded QA evidence corpus differs from this artifact release"
        )
    id_fingerprint = document_id_fingerprint(
        (document.id, document.text) for document in datasource.documents
    )
    if id_fingerprint != manifest["qa_contracts"][qa_dataset]:
        raise ValueError("QA evidence document IDs differ from this artifact release")
    datasource.name = dataset
    executor = WorkflowExecutor(config)
    executor.datasource = datasource
    results = executor.execute_online_pipeline()
    report = _generate_report(results)
    if args.evaluate:
        evaluated = executor.evaluate_report(report)
        _generate_report(evaluated, report)
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
