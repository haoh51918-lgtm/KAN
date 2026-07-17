"""Build the v8 base lock with v7 terminal custody and v6 mining custody."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import yaml


OUTPUT = Path("prereg/s2a_kan_e3_vertical_v8.lock.json")
CONFIG = Path("configs/experiments/s2a_kan_e3_vertical_v8.yaml")
PREREGISTRATION = Path("prereg/s2a_kan_e3_vertical_v8.md")
INCIDENT = Path(
    "governance/incidents/2026-07-17_s2a_v8_authority_hash_corrective_successor.md"
)
V7_IDENTITY_FILES = (
    Path("configs/experiments/s2a_kan_e3_vertical_v7.yaml"),
    Path("prereg/s2a_kan_e3_vertical_v7.md"),
    Path("prereg/s2a_kan_e3_vertical_v7.lock.json"),
    Path("prereg/s2a_kan_e3_vertical_v7_implementation.lock.json"),
    Path("governance/incidents/2026-07-17_s2a_v7_preopening_rebind_successor.md"),
)
RUNTIME_FILES = (
    Path("runtime/s2a_v8_eval/tools/build_v8_config.py"),
    Path("runtime/s2a_v8_eval/tools/build_v8_base_lock.py"),
    Path("evaluations/runtime/s2a_v8_tracking/README.md"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(root: Path, relative: Path) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing: {relative}")
    return path


def _v7_builder(root: Path):
    path = root / "runtime/s2a_v7_eval/tools/build_v7_base_lock.py"
    spec = importlib.util.spec_from_file_location("v7_base_builder", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the v7 base-lock builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_base_lock(root)


def _scientific_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(config)
    payload.pop("protocol_id", None)
    payload.pop("artifact_paths", None)
    payload.pop("mining_source", None)
    payload["claim_boundary"].pop("graph_unlock_allowed", None)
    return payload


def build_base_lock(workspace: Path | str) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    lock = copy.deepcopy(_v7_builder(root))
    v7_config = yaml.safe_load(
        _file(root, Path("configs/experiments/s2a_kan_e3_vertical_v7.yaml")).read_text(
            encoding="utf-8"
        )
    )
    v8_config = yaml.safe_load(_file(root, CONFIG).read_text(encoding="utf-8"))
    if _scientific_payload(v7_config) != _scientific_payload(v8_config):
        raise ValueError("v8 scientific configuration differs from v7")
    custody = lock["predecessor_custody"]
    custody["files"].update(
        {
            relative.as_posix(): _sha256(_file(root, relative))
            for relative in V7_IDENTITY_FILES
        }
    )
    custody.update(
        {
            "protocol_id": "s2a_kan_e3_vertical_v7",
            "scientific_observation": "terminal_preopening_rebind_software_failure",
            "candidate_membership_reuse_allowed": True,
            "reuse_scope": "v6_exact_immutable_whole_topology_via_v7_custody",
        }
    )
    governance = lock["governance"]
    governance["supersession_incident_path"] = INCIDENT.as_posix()
    governance["supersession_incident_sha256"] = _sha256(_file(root, INCIDENT))
    governance["v7_candidate_membership_reuse_allowed"] = True
    governance["v7_disposition"] = "terminal_preopening_rebind_software_failure"
    lock["preregistration"] = {
        "path": PREREGISTRATION.as_posix(),
        "sha256": _sha256(_file(root, PREREGISTRATION)),
    }
    lock["protocol"] = {
        "path": CONFIG.as_posix(),
        "protocol_id": "s2a_kan_e3_vertical_v8",
        "sha256": _sha256(_file(root, CONFIG)),
    }
    lock["runtime"] = {
        "files": {
            relative.as_posix(): _sha256(_file(root, relative))
            for relative in RUNTIME_FILES
        }
    }
    return lock


def _body(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("digest", "write", "verify"))
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()
    root = Path(args.workspace).resolve(strict=True)
    payload = build_base_lock(root)
    body = _body(payload)
    output = root / OUTPUT
    if args.mode == "write":
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            output.unlink(missing_ok=True)
            raise
    elif args.mode == "verify" and output.read_bytes() != body:
        raise ValueError("v8 base lock differs from its deterministic source build")
    print(
        json.dumps(
            {
                "custody_files": len(payload["predecessor_custody"]["files"]),
                "sha256": hashlib.sha256(body).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
