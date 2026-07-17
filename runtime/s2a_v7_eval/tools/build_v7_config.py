"""Generate the v7 config as an exact v6 scientific successor."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml


SOURCE = Path("configs/experiments/s2a_kan_e3_vertical_v6.yaml")
OUTPUT = Path("configs/experiments/s2a_kan_e3_vertical_v7.yaml")
SOURCE_PROTOCOL = "s2a_kan_e3_vertical_v6"
TARGET_PROTOCOL = "s2a_kan_e3_vertical_v7"


def build_config(workspace: Path | str) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    source = yaml.safe_load((root / SOURCE).read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("protocol_id") != SOURCE_PROTOCOL:
        raise ValueError("v6 source config is invalid")
    target = copy.deepcopy(source)
    target["protocol_id"] = TARGET_PROTOCOL
    target["claim_boundary"]["graph_unlock_allowed"] = True
    source_artifacts = {
        key: source["artifact_paths"][key]
        for key in (
            "mining_run",
            "kan_library",
            "gp_control_library",
            "permutation_control_library",
            "blackbox_control",
            "mechanism_cards",
            "blind_review_package",
        )
    }
    target["mining_source"] = {
        "mode": "verified_cross_protocol_rebind",
        "source_protocol_id": SOURCE_PROTOCOL,
        "source_base_lock": "prereg/s2a_kan_e3_vertical_v6.lock.json",
        "source_implementation_lock": (
            "prereg/s2a_kan_e3_vertical_v6_implementation.lock.json"
        ),
        "source_mining_entitlement": (
            "governance/openings/s2a_kan_e3_vertical_v6_mining.json"
        ),
        "source_mining_preclaim": (
            "governance/openings/s2a_kan_e3_vertical_v6_mining_preclaim.json"
        ),
        "source_artifact_paths": source_artifacts,
        "rebind_receipt": (
            "governance/openings/s2a_kan_e3_vertical_v7_mining_rebind.json"
        ),
        "absence_contract": {
            "source_development_preclaim": (
                "governance/openings/s2a_kan_e3_vertical_v6_development_preclaim.json"
            ),
            "source_development_opening": (
                "governance/openings/s2a_kan_e3_vertical_v6_development.json"
            ),
            "target_mining_preclaim": (
                "governance/openings/s2a_kan_e3_vertical_v7_mining_preclaim.json"
            ),
            "target_mining_entitlement": (
                "governance/openings/s2a_kan_e3_vertical_v7_mining.json"
            ),
        },
    }
    target["artifact_paths"] = {
        "implementation_lock": (
            "prereg/s2a_kan_e3_vertical_v7_implementation.lock.json"
        ),
        "development_preclaim": (
            "governance/openings/s2a_kan_e3_vertical_v7_development_preclaim.json"
        ),
        "development_opening": (
            "governance/openings/s2a_kan_e3_vertical_v7_development.json"
        ),
        "development_recovery_receipt": (
            "governance/recoveries/s2a_kan_e3_vertical_v7_development.json"
        ),
        "evaluations": {
            "alpha158_replay": "evaluations/s2a_v7_alpha158_replay",
            "kan_e3_selected": "evaluations/s2a_v7_kan_e3_selected",
            "typed_gp_sr_control": "evaluations/s2a_v7_typed_gp_sr_control",
            "matched_blackbox_control": (
                "evaluations/s2a_v7_matched_blackbox_control"
            ),
            "kan_e3_permutation_control": (
                "evaluations/s2a_v7_kan_e3_permutation_control"
            ),
        },
        "decision_artifact": "evaluations/s2a_kan_e3_vertical_v7_decision",
        "report": "reports/s2a_kan_e3_vertical_v7_report.md",
    }
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
        output.parent.mkdir(parents=True, exist_ok=True)
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
        raise ValueError("v7 config differs from its deterministic source build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
