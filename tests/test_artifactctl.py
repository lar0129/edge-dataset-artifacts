"""Offline tests for release installation and selection."""

import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from artifactctl import (
    document_id_fingerprint,
    install_archive,
    safe_path,
    select_artifacts,
    sha256,
    verify_directory,
)
from run_prebuilt import configure


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.content = b"example corpus\n"
        self.source = self.root / "source.txt"
        self.source.write_bytes(self.content)
        self.artifact = {
            "id": "example-flat",
            "dataset": "example",
            "kind": "flat",
            "install_path": "storage/example/version",
            "files": [
                {
                    "path": "nested/index.txt",
                    "size": len(self.content),
                    "sha256": sha256(self.source),
                }
            ],
        }

    def make_archive(self, name="nested/index.txt", link=False, content=None):
        archive = self.root / "payload.tar.gz"
        payload = self.content if content is None else content
        with tarfile.open(archive, "w:gz") as handle:
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                handle.addfile(member)
            else:
                member.size = len(payload)
                handle.addfile(member, io.BytesIO(payload))
        self.artifact["archive"] = {
            "size": archive.stat().st_size,
            "sha256": sha256(archive),
        }
        return archive

    def test_install_and_verify_are_idempotent(self):
        archive = self.make_archive()
        target = install_archive(archive, self.root, self.artifact)
        self.assertEqual((target / "nested/index.txt").read_bytes(), self.content)
        self.assertEqual(install_archive(archive, self.root, self.artifact), target)
        verify_directory(target, self.artifact)

    def test_existing_modified_file_is_preserved(self):
        archive = self.make_archive()
        target = install_archive(archive, self.root, self.artifact)
        (target / "nested/index.txt").write_text("local changes")
        with self.assertRaises(ValueError):
            install_archive(archive, self.root, self.artifact)
        self.assertEqual((target / "nested/index.txt").read_text(), "local changes")

    def test_archive_corruption_is_rejected(self):
        archive = self.make_archive()
        archive.write_bytes(archive.read_bytes() + b"corruption")
        with self.assertRaises(ValueError):
            install_archive(archive, self.root, self.artifact)
        self.assertFalse((self.root / self.artifact["install_path"]).exists())

    def test_missing_member_is_rejected(self):
        archive = self.root / "empty.tar.gz"
        with tarfile.open(archive, "w:gz"):
            pass
        self.artifact["archive"] = {
            "size": archive.stat().st_size,
            "sha256": sha256(archive),
        }
        with self.assertRaises(ValueError):
            install_archive(archive, self.root, self.artifact)
        self.assertFalse((self.root / self.artifact["install_path"]).exists())

    def test_unexpected_and_link_members_are_rejected(self):
        for name, link in [
            ("../escape", False),
            ("nested/index.txt", True),
            ("extra", False),
        ]:
            with self.subTest(name=name, link=link):
                archive = self.make_archive(name, link)
                with self.assertRaises(ValueError):
                    install_archive(archive, self.root, self.artifact)
                self.assertFalse((self.root / self.artifact["install_path"]).exists())

    def test_wrong_file_checksum_cleans_staging(self):
        archive = self.make_archive(content=b"X" * len(self.content))
        with self.assertRaises(ValueError):
            install_archive(archive, self.root, self.artifact)
        self.assertFalse(list(self.root.glob("storage/example/.xrag-install-*")))
        self.assertFalse((self.root / self.artifact["install_path"]).exists())

    def test_path_traversal_is_rejected(self):
        for path in ["../outside", "/absolute", "C:/absolute", "a/../b", "a\\b", ""]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_path(self.root, path)

    def test_selection_and_typos(self):
        manifest = {"artifacts": [self.artifact]}
        self.assertEqual(select_artifacts(manifest, kinds=["flat"]), [self.artifact])
        with self.assertRaises(ValueError):
            select_artifacts(manifest, datasets=["typo"])

    def test_document_ids_are_part_of_qa_contract(self):
        original = [(0, "first"), (1, "second")]
        self.assertEqual(
            document_id_fingerprint(original),
            document_id_fingerprint(list(reversed(original))),
        )
        self.assertNotEqual(
            document_id_fingerprint(original),
            document_id_fingerprint([(1, "first"), (0, "second")]),
        )

    def test_prebuilt_config_does_not_mutate_input(self):
        raw = {
            "offline": {
                "datasource": {"component_key": "concurrentqa"},
                "vector_index": {"component_key": "faiss"},
            }
        }
        original = json.dumps(raw)
        artifact = {
            **self.artifact,
            "corpus_config": {"max_doc_chars": 20000},
            "embedding": {"model_name": "example"},
        }
        configured = configure(raw, artifact)
        self.assertEqual(json.dumps(raw), original)
        self.assertEqual(
            configured["offline"]["datasource"]["component_key"], "example"
        )
        self.assertEqual(configured["offline"]["encodings"][0]["encoder_type"], "dense")


if __name__ == "__main__":
    unittest.main()
