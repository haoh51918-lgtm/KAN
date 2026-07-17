"""Verify the exact v5 matched-control lineage that v6 must accept."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from mirage_kan.evaluation.v2_runner import _blackbox_pairing_matches_lineage


TOP_MANIFEST = Path("artifacts/s2a_kan_e3_mining_v5/manifest.json")
BLACKBOX_MANIFEST = Path("controls/s2a_matched_blackbox_v5/manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_manifest(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular manifest is missing: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be a mapping: {relative}")
    return value


def build_receipt(workspace: Path | str) -> dict[str, object]:
    root = Path(workspace).resolve(strict=True)
    top = _read_manifest(root, TOP_MANIFEST)
    blackbox = _read_manifest(root, BLACKBOX_MANIFEST)
    lineage = top.get("kan_selected_lineage")
    factor_ids = blackbox.get("selected_kan_factor_ids")
    global_indices = blackbox.get("paired_kan_global_attempt_indices")
    if not isinstance(lineage, dict):
        raise ValueError("v5 top manifest lacks selected KAN lineage")
    if not isinstance(factor_ids, list) or not isinstance(global_indices, list):
        raise ValueError("v5 blackbox manifest lacks KAN pairing lists")
    if factor_ids == list(lineage):
        raise ValueError("smoke fixture does not exercise differing order")
    if not _blackbox_pairing_matches_lineage(factor_ids, global_indices, lineage):
        raise ValueError("valid v5 blackbox mapping was rejected")
    crossed = list(global_indices)
    crossed[0], crossed[1] = crossed[1], crossed[0]
    if _blackbox_pairing_matches_lineage(factor_ids, crossed, lineage):
        raise ValueError("crossed KAN mapping was accepted")
    return {
        "blackbox_manifest_sha256": _sha256(root / BLACKBOX_MANIFEST),
        "factor_count": len(factor_ids),
        "mapping_exact": True,
        "ordered_lists_differ": True,
        "crossed_mapping_rejected": True,
        "passed": True,
        "real_scientific_label_access": False,
        "schema_version": "s2a_v6_lineage_compatibility_smoke_v1",
        "top_manifest_sha256": _sha256(root / TOP_MANIFEST),
    }


def _write_exclusive(path: Path, payload: dict[str, object]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.workspace).resolve(strict=True)
    payload = build_receipt(root)
    output = root / args.output
    _write_exclusive(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
