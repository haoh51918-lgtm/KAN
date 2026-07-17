from __future__ import annotations

import json

import pytest

from mirage_kan.artifacts.library import publish_library, verify_library
from mirage_kan.dsl import AstNode


def test_publication_is_no_replace_and_recomputes(tmp_path, tiny_panel) -> None:
    programs = {
        "range_close": AstNode("SafeDiv", (AstNode("Sub", (AstNode("High"), AstNode("Low"))), AstNode("Close"))),
        "return_2": AstNode("Return", (AstNode("Close"),), {"window": 2}),
    }
    destination = tmp_path / "seed"
    publish_library(
        destination,
        programs,
        tiny_panel,
        identities={"proposal_sha256": "a" * 64, "cache_sha256": "b" * 64},
        library_role="wiring_control",
        kan_mined=False,
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["library_role"] == "wiring_control"
    assert manifest["kan_mined"] is False
    assert verify_library(destination, tiny_panel)["verified"] is True
    with pytest.raises(FileExistsError):
        publish_library(destination, programs, tiny_panel, identities={}, library_role="wiring_control", kan_mined=False)


def test_library_verifier_rejects_manifest_traversal_and_extra_files(
    tmp_path, tiny_panel
) -> None:
    destination = tmp_path / "seed"
    publish_library(
        destination,
        {"return_2": AstNode("Return", (AstNode("Close"),), {"window": 2})},
        tiny_panel,
        identities={},
        library_role="wiring_control",
        kan_mined=False,
    )
    manifest_path = destination / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["../escape"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="safe filename"):
        verify_library(destination, tiny_panel)

    manifest["files"].pop("../escape")
    manifest_path.write_text(json.dumps(manifest))
    (destination / "unlisted.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="exact file set"):
        verify_library(destination, tiny_panel)
