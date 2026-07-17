"""Generate v8 from the scientific-equivalent v7 config."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml


SOURCE = Path("configs/experiments/s2a_kan_e3_vertical_v7.yaml")
OUTPUT = Path("configs/experiments/s2a_kan_e3_vertical_v8.yaml")


def build_config(workspace: Path | str) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    source = yaml.safe_load((root / SOURCE).read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("protocol_id") != (
        "s2a_kan_e3_vertical_v7"
    ):
        raise ValueError("v7 source config is invalid")
    target = copy.deepcopy(source)
    target["protocol_id"] = "s2a_kan_e3_vertical_v8"

    def rewrite(value: object) -> object:
        if isinstance(value, dict):
            return {key: rewrite(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        if isinstance(value, str):
            return value.replace("_v7", "_v8")
        return value

    target["artifact_paths"] = rewrite(target["artifact_paths"])
    mining_source = target["mining_source"]
    mining_source["rebind_receipt"] = (
        "governance/openings/s2a_kan_e3_vertical_v8_mining_rebind.json"
    )
    absence = mining_source["absence_contract"]
    absence["target_mining_preclaim"] = (
        "governance/openings/s2a_kan_e3_vertical_v8_mining_preclaim.json"
    )
    absence["target_mining_entitlement"] = (
        "governance/openings/s2a_kan_e3_vertical_v8_mining.json"
    )
    return target


def _body(payload: dict[str, Any]) -> bytes:
    return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "verify"))
    parser.add_argument("--workspace", default=".")
    args = parser.parse_args()
    root = Path(args.workspace).resolve(strict=True)
    body = _body(build_config(root))
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
    elif output.read_bytes() != body:
        raise ValueError("v8 config differs from its deterministic source build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
