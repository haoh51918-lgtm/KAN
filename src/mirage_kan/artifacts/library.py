"""Atomic, no-replace publication of independently executable factor libraries."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from mirage_kan.data.pit import PitPanel, sha256_file
from mirage_kan.dsl import AstNode, evaluate

SCHEMA_VERSION = "mirage_factor_library_v1"
PUBLICATION_MODE = "GPFS_EXCLUSIVE_MKDIR_OEXCL_MANIFEST_LAST_FSYNC_V1"


def claim_artifact_directory(destination: Path | str) -> Path:
    """Durably consume one no-replace artifact path before risky work begins."""
    raw = Path(destination)
    final = raw.parent.resolve(strict=True) / raw.name
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to replace artifact: {final}")
    os.mkdir(final, 0o700)
    _fsync_directory(final.parent)
    _copy_file_bytes(b"claimed publication in progress\n", final / ".INCOMPLETE", mode=0o600)
    _fsync_directory(final)
    return final


def finalize_claimed_directory(
    staging_path: Path | str,
    destination: Path | str,
    *,
    required_manifest: str,
) -> None:
    """Copy a flat staging set manifest-last into a previously claimed directory."""
    staging = Path(staging_path).resolve(strict=True)
    final = Path(destination).resolve(strict=True)
    marker = final / ".INCOMPLETE"
    if not marker.is_file() or set(entry.name for entry in final.iterdir()) != {
        ".INCOMPLETE"
    }:
        raise ValueError("claimed artifact directory is not empty and incomplete")
    manifest = staging / _safe_flat_filename(required_manifest)
    if not manifest.is_file():
        raise ValueError(f"staging lacks required manifest: {required_manifest}")
    entries = sorted(staging.iterdir(), key=lambda path: path.name)
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("artifact staging must contain flat regular files")
    for source in entries:
        if source != manifest:
            _copy_file_exclusive(source, final / source.name)
    _copy_file_exclusive(manifest, final / manifest.name)
    _fsync_directory(final)
    marker.unlink()
    _fsync_directory(final)
    _fsync_directory(final.parent)
    shutil.rmtree(staging)


def terminalize_claimed_directory(
    destination: Path | str,
    payload: Mapping[str, object],
    *,
    invalidate_published: bool = False,
) -> dict[str, object]:
    """Convert any partially copied claimed directory into a terminal artifact."""
    final = Path(destination).resolve(strict=True)
    marker = final / ".INCOMPLETE"
    entries = [entry for entry in final.iterdir() if entry != marker]
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("claimed terminal artifact contains a non-regular entry")
    terminal_path = final / "terminal_failure.json"
    if terminal_path.is_file() and not terminal_path.is_symlink():
        existing = json.loads(terminal_path.read_text(encoding="utf-8"))
        if existing.get("publication_state") != "terminal_failure":
            raise ValueError("existing terminal record is not terminal")
        marker.unlink()
        _fsync_directory(final)
        _fsync_directory(final.parent)
        return {
            "path": str(final),
            "state": "terminal_failure",
            "terminal_sha256": sha256_file(terminal_path),
        }
    if not marker.is_file() and not invalidate_published:
        return {"path": str(final), "state": "published"}
    if terminal_path.exists() or terminal_path.is_symlink():
        raise FileExistsError(f"invalid terminal failure path: {terminal_path}")
    record = dict(payload)
    record["publication_state"] = "terminal_failure"
    record["partial_files"] = {
        entry.name: sha256_file(entry) for entry in sorted(entries)
    }
    _copy_file_bytes(
        _canonical_json_bytes(record), terminal_path, mode=0o444
    )
    _fsync_directory(final)
    if marker.is_file():
        marker.unlink()
    _fsync_directory(final)
    _fsync_directory(final.parent)
    return {
        "path": str(final),
        "state": "terminal_failure",
        "terminal_sha256": sha256_file(terminal_path),
    }


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _series_hash(series: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(series, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _safe_flat_filename(filename: object) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError("artifact manifest requires a safe filename segment")
    path = Path(filename)
    if path.name != filename or path.is_absolute() or filename in {".", ".."}:
        raise ValueError(f"artifact manifest has unsafe filename: {filename!r}")
    return filename


def publish_staging_directory(
    staging_path: Path | str,
    destination: Path | str,
    *,
    required_manifest: str,
) -> None:
    """Atomically publish a verified staging directory without replacement."""
    staging = Path(staging_path).resolve(strict=True)
    final_raw = Path(destination)
    final = final_raw.parent.resolve(strict=True) / final_raw.name
    if staging.parent != final.parent:
        raise ValueError("staging and destination must share a parent")
    if not (staging / required_manifest).is_file():
        raise ValueError(f"staging lacks required manifest: {required_manifest}")
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to replace artifact: {final}")
    entries = sorted(staging.iterdir(), key=lambda path: path.name)
    manifest = staging / required_manifest
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("artifact staging must contain flat regular files")
    try:
        os.mkdir(final, 0o700)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace artifact: {final}") from error
    _fsync_directory(final.parent)

    marker = final / ".INCOMPLETE"
    _copy_file_bytes(
        b"manifest-last publication in progress\n", marker, mode=0o600
    )
    _fsync_directory(final)
    try:
        for source in entries:
            if source != manifest:
                _copy_file_exclusive(source, final / source.name)
        _copy_file_exclusive(manifest, final / manifest.name)
        _fsync_directory(final)
        marker.unlink()
        _fsync_directory(final)
        _fsync_directory(final.parent)
    except BaseException:
        _fsync_directory(final)
        raise
    shutil.rmtree(staging)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_file_bytes(data: bytes, destination: Path, *, mode: int) -> None:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_descriptor: int | None = None
    try:
        state = os.fstat(source_descriptor)
        if not stat.S_ISREG(state.st_mode):
            raise ValueError(f"publication source is not a regular file: {source}")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            stat.S_IMODE(state.st_mode) or 0o600,
        )
        while block := os.read(source_descriptor, 1024 * 1024):
            view = memoryview(block)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    raise OSError("exclusive artifact copy made no progress")
                view = view[written:]
        os.fsync(destination_descriptor)
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)


def _evaluate_library(
    programs: Mapping[str, AstNode], panel: PitPanel
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    factor_values: dict[str, pd.Series] = {}
    expression_support: dict[str, pd.Series] = {}
    records: dict[str, dict[str, object]] = {}
    for factor_id in sorted(programs):
        program = programs[factor_id]
        contract = program.validate()
        result = evaluate(program, panel)
        effective_support = result.support & panel.membership
        values = result.values.where(effective_support)
        if not np.isfinite(values[effective_support].to_numpy(dtype=float)).all():
            raise ValueError(f"factor {factor_id} has non-finite values on effective support")
        factor_values[factor_id] = values
        expression_support[factor_id] = result.support
        records[factor_id] = {
            "ast": program.to_dict(),
            "canonical_ast": program.canonical_json(),
            "canonical_hash": program.identity,
            "output_type": contract.output_type.value,
            "required_lookback": contract.lookback,
            "causal": contract.causal,
            "expression_support_hash": _series_hash(result.support),
            "expression_support_rows": int(result.support.sum()),
            "effective_membership_support_rows": int(effective_support.sum()),
        }
    values_frame = pd.DataFrame(factor_values, index=panel.raw.index)
    support_frame = pd.DataFrame(expression_support, index=panel.raw.index).astype(bool)
    return values_frame, support_frame, records


def publish_library(
    destination: Path | str,
    programs: Mapping[str, AstNode],
    panel: PitPanel,
    *,
    identities: Mapping[str, object],
    library_role: str,
    kan_mined: bool,
    preclaimed: bool = False,
) -> dict[str, object]:
    """Publish a verified library directory atomically without replacing any path."""
    if not programs:
        raise ValueError("factor library must contain at least one program")
    final = Path(destination)
    parent = final.parent.resolve(strict=True)
    final = parent / final.name
    if preclaimed:
        if not (final / ".INCOMPLETE").is_file() or set(
            entry.name for entry in final.iterdir()
        ) != {".INCOMPLETE"}:
            raise ValueError("preclaimed factor library is not empty and incomplete")
    elif final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to replace factor library: {final}")

    values, support, factors = _evaluate_library(programs, panel)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".staging", dir=parent))
    try:
        values_path = staging / "factor_panel.parquet"
        support_path = staging / "expression_support.parquet"
        values.to_parquet(values_path)
        support.to_parquet(support_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "publication_mode": PUBLICATION_MODE,
            "library_role": library_role,
            "kan_mined": bool(kan_mined),
            "scientific_result": False,
            "identities": dict(sorted(identities.items())),
            "panel_rows": len(values),
            "factor_count": len(factors),
            "index_names": list(values.index.names),
            "mask_semantics": {
                "raw_observed": "per-field finite raw observation; never imputed",
                "expression_support": "operator-contract support before universe filtering",
                "membership": "dynamic-universe membership applied to published factor values",
                "finite_output": "required on expression support and membership",
                "tradability": "separate from factor-value publication and not applied here",
            },
            "files": {
                "factor_panel.parquet": sha256_file(values_path),
                "expression_support.parquet": sha256_file(support_path),
            },
            "factors": factors,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        verify_library(staging, panel)
        for path in staging.iterdir():
            path.chmod(0o444)
        if preclaimed:
            finalize_claimed_directory(
                staging, final, required_manifest="manifest.json"
            )
        else:
            publish_staging_directory(
                staging, final, required_manifest="manifest.json"
            )
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def verify_library(path: Path | str, panel: PitPanel) -> dict[str, object]:
    """Verify file identities and independently recompute every factor from its AST."""
    library = Path(path).resolve(strict=True)
    if (library / ".INCOMPLETE").exists():
        raise ValueError("factor library publication is incomplete")
    manifest = json.loads((library / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported factor-library schema")
    filenames = {_safe_flat_filename(name) for name in manifest["files"]}
    expected_files = filenames | {"manifest.json"}
    actual_files = {
        entry.name
        for entry in library.iterdir()
        if entry.is_file() and not entry.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError("factor library does not have the manifest's exact file set")
    if any(entry.is_symlink() or not entry.is_file() for entry in library.iterdir()):
        raise ValueError("factor library must contain only flat regular files")
    for filename, expected in manifest["files"].items():
        filename = _safe_flat_filename(filename)
        actual = sha256_file(library / filename)
        if actual != expected:
            raise ValueError(f"published file hash mismatch: {filename}")

    programs = {
        factor_id: AstNode.from_dict(record["ast"])
        for factor_id, record in manifest["factors"].items()
    }
    expected_values, expected_support, expected_records = _evaluate_library(programs, panel)
    stored_values = pd.read_parquet(library / "factor_panel.parquet")
    stored_support = pd.read_parquet(library / "expression_support.parquet").astype(bool)
    pd.testing.assert_frame_equal(stored_values, expected_values, check_exact=True)
    pd.testing.assert_frame_equal(stored_support, expected_support, check_exact=True)
    for factor_id, expected_record in expected_records.items():
        recorded = manifest["factors"][factor_id]
        for key in (
            "canonical_ast",
            "canonical_hash",
            "required_lookback",
            "causal",
            "expression_support_hash",
            "expression_support_rows",
            "effective_membership_support_rows",
        ):
            if recorded[key] != expected_record[key]:
                raise ValueError(f"factor record mismatch for {factor_id}: {key}")
    return {
        "verified": True,
        "library_path": str(library),
        "factor_count": len(programs),
        "panel_rows": len(stored_values),
    }
