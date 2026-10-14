"""Package audited artifacts and publish version-pinned release metadata."""

import argparse
import copy
import gzip
import json
import subprocess
import tarfile
from pathlib import Path

from artifactctl import (
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    read_manifest,
    sha256,
    verify_directory,
)


def write_manifest(path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def pack(manifest, manifest_path, root, staging):
    for artifact in manifest["artifacts"]:
        archive_path = f"releases/{manifest['release']}/{artifact['id']}.tar.gz"
        target = staging / archive_path
        if (
            "archive" in artifact
            and target.is_file()
            and target.stat().st_size == artifact["archive"]["size"]
            and sha256(target) == artifact["archive"]["sha256"]
        ):
            print(f"Already packed: {artifact['id']}", flush=True)
            continue
        source = root / artifact["source"]
        verify_directory(source, artifact)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        print(f"Packing {artifact['id']}", flush=True)
        with (
            partial.open("wb") as output,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=output, compresslevel=1, mtime=0
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w|") as archive,
        ):
            for entry in artifact["files"]:
                info = tarfile.TarInfo(entry["path"])
                info.size = entry["size"]
                info.mode = 0o644
                with (source / entry["path"]).open("rb") as handle:
                    archive.addfile(info, handle)
        partial.replace(target)
        artifact["archive"] = {
            "path": archive_path,
            "size": target.stat().st_size,
            "sha256": sha256(target),
        }
        write_manifest(manifest_path, manifest)


def public_manifest(manifest):
    result = copy.deepcopy(manifest)
    for artifact in result["artifacts"]:
        artifact.pop("source", None)
        artifact.pop("include", None)
    return result


def upload(manifest, manifest_path, staging):
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    repo_id = manifest["hub"]["repo_id"]
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    if api.repo_info(repo_id, repo_type="dataset").private:
        raise ValueError(
            "Existing HF repository is private; expected a public repository"
        )
    if manifest["release"] in {
        tag.name for tag in api.list_repo_refs(repo_id, repo_type="dataset").tags
    }:
        raise ValueError("This release tag is already published; choose a new release")
    operations = []
    for artifact in manifest["artifacts"]:
        archive = artifact["archive"]
        path = staging / archive["path"]
        if path.stat().st_size != archive["size"] or sha256(path) != archive["sha256"]:
            raise ValueError(f"Package changed: {path}")
        operations.append(
            CommitOperationAdd(path_in_repo=archive["path"], path_or_fileobj=str(path))
        )
    commit = api.create_commit(
        repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Publish audited xrag artifact archives {manifest['release']}",
    )
    manifest["hub"]["revision"] = commit.oid
    write_manifest(manifest_path, manifest)
    remote_files = {
        entry.rfilename: entry
        for entry in api.repo_info(
            repo_id, repo_type="dataset", revision=commit.oid, files_metadata=True
        ).siblings
    }
    for artifact in manifest["artifacts"]:
        archive = artifact["archive"]
        remote = remote_files[archive["path"]]
        if (
            remote.size != archive["size"]
            or not remote.lfs
            or remote.lfs.sha256 != archive["sha256"]
        ):
            raise ValueError(f"Published archive integrity mismatch: {archive['path']}")
    public = public_manifest(manifest)
    card = (PROJECT_ROOT / "docs" / "DATASET_CARD.md").read_bytes()
    documentation_commit = api.create_commit(
        repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card),
            CommitOperationAdd(
                path_in_repo="manifest.json",
                path_or_fileobj=(json.dumps(public, indent=2) + "\n").encode(),
            ),
            CommitOperationAdd(
                path_in_repo="REPRODUCING.md",
                path_or_fileobj=(PROJECT_ROOT / "docs" / "REPRODUCING.md").read_bytes(),
            ),
        ],
        commit_message=f"Document xrag artifacts {manifest['release']}",
    )
    api.create_tag(
        repo_id,
        tag=manifest["release"],
        revision=documentation_commit.oid,
        repo_type="dataset",
        exist_ok=False,
    )
    print(f"Published https://huggingface.co/datasets/{repo_id} at {commit.oid}")


def gh_api(endpoint, payload=None):
    command = ["gh", "api", endpoint]
    if payload is not None:
        command += ["--method", "POST", "--input", "-"]
    completed = subprocess.run(
        command,
        input=None if payload is None else json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def publish_github(manifest, create_tag=False):
    if not manifest["hub"].get("revision"):
        raise ValueError("Publish Hugging Face archives first")
    repo = manifest["github"]
    details = gh_api(f"repos/{repo}")
    if details["private"]:
        raise ValueError("GitHub repository must be public")
    branch = details["default_branch"]
    head = gh_api(f"repos/{repo}/git/ref/heads/{branch}")["object"]["sha"]
    commit = gh_api(f"repos/{repo}/git/commits/{head}")
    directory = PROJECT_ROOT
    names = [
        ".gitignore",
        "README.md",
        "README.en.md",
        "docs/REPRODUCING.md",
        "requirements.txt",
        "scripts/artifactctl.py",
        "scripts/audit.py",
        "scripts/publish.py",
        "scripts/rebuild_diskann.py",
        "scripts/run_prebuilt.py",
        "config/sources.json",
        "tests/test_artifactctl.py",
        "docs/DATASET_CARD.md",
    ]
    tree = [
        {
            "path": name,
            "mode": "100644",
            "type": "blob",
            "content": (directory / name).read_text(encoding="utf-8"),
        }
        for name in names
    ]
    tree.append(
        {
            "path": "manifests/manifest.json",
            "mode": "100644",
            "type": "blob",
            "content": json.dumps(public_manifest(manifest), indent=2) + "\n",
        }
    )
    legacy_paths = {
        "artifactctl.py",
        "audit.py",
        "publish.py",
        "rebuild_diskann.py",
        "run_prebuilt.py",
        "sources.json",
        "test_artifactctl.py",
        "DATASET_CARD.md",
        "REPRODUCING.md",
        "manifest.json",
    }
    current_paths = {
        entry["path"]
        for entry in gh_api(f"repos/{repo}/git/trees/{commit['tree']['sha']}")["tree"]
    }
    tree.extend(
        {"path": path, "mode": "100644", "type": "blob", "sha": None}
        for path in sorted(legacy_paths & current_paths)
    )
    tree_result = gh_api(
        f"repos/{repo}/git/trees", {"base_tree": commit["tree"]["sha"], "tree": tree}
    )
    new_commit = gh_api(
        f"repos/{repo}/git/commits",
        {
            "message": f"Update Edge Dataset Artifacts tooling for {manifest['release']}",
            "tree": tree_result["sha"],
            "parents": [head],
        },
    )
    subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo}/git/refs/heads/{branch}",
            "--method",
            "PATCH",
            "--input",
            "-",
        ],
        input=json.dumps({"sha": new_commit["sha"], "force": False}),
        text=True,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if create_tag:
        gh_api(
            f"repos/{repo}/git/refs",
            {"ref": f"refs/tags/{manifest['release']}", "sha": new_commit["sha"]},
        )
    print(f"Published https://github.com/{repo} at {new_commit['sha']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "upload", "github"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--tag", action="store_true", help="Create a new GitHub release tag"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--staging", type=Path, default=Path("storage/artifact-release")
    )
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    if args.command == "pack":
        pack(manifest, args.manifest, args.root, args.staging)
    elif args.command == "upload":
        upload(manifest, args.manifest, args.staging)
    else:
        publish_github(manifest, create_tag=args.tag)


if __name__ == "__main__":
    main()
