"""List, download, and verify versioned xrag artifacts."""

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "manifests" / "manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id_fingerprint(documents):
    pairs = sorted(
        (identifier, hashlib.sha256(text.strip().encode("utf-8")).hexdigest())
        for identifier, text in documents
    )
    return hashlib.sha256(
        json.dumps(pairs, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def safe_path(root, relative):
    path = PurePosixPath(relative)
    if (
        not relative
        or path.is_absolute()
        or any(
            part in ("", ".", "..") or ":" in part or "\\" in part
            for part in relative.split("/")
        )
    ):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes destination: {relative!r}")
    return target


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    seen = set()
    for artifact in manifest["artifacts"]:
        if artifact["id"] in seen:
            raise ValueError(f"Duplicate artifact: {artifact['id']}")
        seen.add(artifact["id"])
        if artifact["kind"] not in ("corpus", "graph", "flat", "diskann"):
            raise ValueError(f"Unsupported artifact kind: {artifact['kind']}")
        safe_path(Path.cwd(), artifact["install_path"])
        archive = artifact.get("archive")
        if archive is not None:
            safe_path(Path.cwd(), archive["path"])
            if archive["size"] < 0 or not re.fullmatch(
                r"[0-9a-f]{64}", archive["sha256"]
            ):
                raise ValueError("Invalid archive integrity metadata")
        filenames = set()
        for entry in artifact["files"]:
            safe_path(Path.cwd(), entry["path"])
            if entry["path"] in filenames:
                raise ValueError("Duplicate file in manifest")
            filenames.add(entry["path"])
            if entry["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                raise ValueError("Invalid file integrity metadata")
    return manifest


def verify_directory(directory, artifact):
    for entry in artifact["files"]:
        path = safe_path(directory, entry["path"])
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or linked artifact file: {path}")
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")


def install_archive(archive, root, artifact):
    archive = Path(archive)
    if archive.stat().st_size != artifact["archive"]["size"]:
        raise ValueError("Archive size mismatch")
    if sha256(archive) != artifact["archive"]["sha256"]:
        raise ValueError("Archive checksum mismatch")
    root = Path(root).resolve()
    destination = safe_path(root, artifact["install_path"])
    if destination.exists():
        verify_directory(destination, artifact)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = {entry["path"]: entry for entry in artifact["files"]}
    with tempfile.TemporaryDirectory(
        prefix=".xrag-install-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "payload"
        staging.mkdir()
        seen = set()
        with tarfile.open(archive, "r|gz") as handle:
            for member in handle:
                if (
                    not member.isfile()
                    or member.name not in expected
                    or member.name in seen
                ):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                if member.size != expected[member.name]["size"]:
                    raise ValueError(f"Archive member size mismatch: {member.name}")
                target = safe_path(staging, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                seen.add(member.name)
        if seen != expected.keys():
            raise ValueError("Archive is missing manifest files")
        verify_directory(staging, artifact)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(staging, destination)
    return destination


def select_artifacts(manifest, identifiers=None, datasets=None, kinds=None):
    available = manifest["artifacts"]
    for values, key in [(identifiers, "id"), (datasets, "dataset"), (kinds, "kind")]:
        unknown = set(values or []) - {artifact[key] for artifact in available}
        if unknown:
            raise ValueError(f"Unknown {key}: {', '.join(sorted(unknown))}")
    selected = [
        artifact
        for artifact in available
        if (not identifiers or artifact["id"] in identifiers)
        and (not datasets or artifact["dataset"] in datasets)
        and (not kinds or artifact["kind"] in kinds)
    ]
    if not selected:
        raise ValueError("No artifacts match the selection")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "download", "verify"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact", action="append")
    parser.add_argument("--dataset", action="append")
    parser.add_argument(
        "--kind", action="append", choices=("corpus", "graph", "flat", "diskann")
    )
    parser.add_argument(
        "--all", action="store_true", help="Explicitly select every artifact"
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="Target xrag checkout"
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Optional Hugging Face cache directory"
    )
    args = parser.parse_args()
    if args.command != "list" and not (
        args.all or args.artifact or args.dataset or args.kind
    ):
        parser.error("Select --artifact, --dataset, --kind, or --all")
    manifest = read_manifest(args.manifest)
    selected = select_artifacts(manifest, args.artifact, args.dataset, args.kind)
    for artifact in selected:
        if args.command == "list":
            size = artifact.get("archive", {}).get("size", 0) / 1024**2
            print(
                f"{artifact['id']:32} {size:10.1f} MiB  -> {artifact['install_path']}"
            )
        elif args.command == "verify":
            verify_directory(
                safe_path(args.root.resolve(), artifact["install_path"]), artifact
            )
            print(f"Verified {artifact['id']}")
        else:
            destination = safe_path(args.root.resolve(), artifact["install_path"])
            if destination.exists():
                verify_directory(destination, artifact)
                print(f"Already verified: {artifact['id']}")
                continue
            from huggingface_hub import hf_hub_download

            hub = manifest["hub"]
            if not re.fullmatch(r"[0-9a-f]{40}", hub["revision"]):
                raise ValueError(
                    "Downloads require an immutable Hugging Face commit revision"
                )
            archive = hf_hub_download(
                repo_id=hub["repo_id"],
                repo_type="dataset",
                revision=hub["revision"],
                filename=artifact["archive"]["path"],
                cache_dir=args.cache_dir,
            )
            installed = install_archive(archive, args.root, artifact)
            print(f"Installed {artifact['id']} -> {installed}")


if __name__ == "__main__":
    main()
