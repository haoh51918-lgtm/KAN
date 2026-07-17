from __future__ import annotations

import os
import json

import pytest

from mirage_kan.identities import (
    regular_file_tree_identity,
    regular_file_tree_stat_identity,
)
from mirage_kan.data.pit import sha256_file
from mirage_kan.identities import source_tree_identity
from mirage_kan.mining.s2a import _contained_path, _verify_implementation_lock


def test_regular_file_tree_identity_is_deterministic_for_nested_tree(tmp_path) -> None:
    (tmp_path / "z").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "z" / "second.bin").write_bytes(b"second")
    (tmp_path / "a" / "first.bin").write_bytes(b"first")

    first = regular_file_tree_identity(tmp_path)
    second = regular_file_tree_identity(tmp_path)
    stat = regular_file_tree_stat_identity(tmp_path)

    assert first == second
    assert first["file_count"] == 2
    assert first["total_bytes"] == 11
    assert stat["stat_inventory_sha256"] == first["stat_inventory_sha256"]


def test_regular_file_tree_identity_rejects_symlinks(tmp_path) -> None:
    target = tmp_path / "target"
    target.write_text("data")
    os.symlink(target, tmp_path / "alias")
    with pytest.raises(ValueError, match="symlink"):
        regular_file_tree_identity(tmp_path)


def test_s2_implementation_lock_binds_source_files_and_provider(tmp_path) -> None:
    source = tmp_path / "src" / "mirage_kan"
    source.mkdir(parents=True)
    source_file = source / "module.py"
    source_file.write_text("VALUE = 1\n")
    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "data.bin").write_bytes(b"locked")
    (tmp_path / "prereg").mkdir()
    payload = {
        "schema_version": "mirage_s2_implementation_lock_v1",
        "protocol_id": "s2_plan_c_vertical_v1",
        "files": {"src/mirage_kan/module.py": sha256_file(source_file)},
        "source_tree": source_tree_identity(source),
        "qlib_provider": regular_file_tree_identity(provider),
    }
    lock = tmp_path / "prereg" / "s2_plan_c_vertical_v1_implementation.lock.json"
    lock.write_text(json.dumps(payload))

    verified, lock_hash = _verify_implementation_lock(tmp_path)

    assert verified == payload
    assert lock_hash == sha256_file(lock)
    source_file.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="implementation file hash mismatch"):
        _verify_implementation_lock(tmp_path)


def test_s2_authority_path_cannot_escape_workspace(tmp_path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.write_text("authority")
    try:
        with pytest.raises(ValueError, match="escapes the workspace"):
            _contained_path(tmp_path, f"../{outside.name}", label="proposal")
    finally:
        outside.unlink()
