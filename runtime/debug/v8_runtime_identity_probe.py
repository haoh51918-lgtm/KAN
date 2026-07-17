"""Diagnose which Quanta lifecycle step mutates the v8 runtime identity."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
LOCK = WORKSPACE / "prereg/s2a_kan_e3_vertical_v8_implementation.lock.json"
TRACKING = WORKSPACE / "evaluations/runtime/s2a_v8_tracking"


def _diff(left: object, right: object, prefix: str = "runtime") -> list[dict[str, object]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[dict[str, object]] = []
        for key in sorted(set(left) | set(right)):
            differences.extend(
                _diff(left.get(key), right.get(key), f"{prefix}.{key}")
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return []
        return [{"path": prefix, "locked": left, "live": right}]
    if left == right:
        return []
    return [{"path": prefix, "locked": left, "live": right}]


def _configure_torch() -> None:
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _mutate(stage: str, locked: Mapping[str, object]) -> None:
    os.chdir(TRACKING)
    if stage == "baseline":
        return
    import qlib

    if stage == "qlib-import":
        return
    qlib.init(provider_uri=os.environ["QLIB_DATA_DIR"], region="cn")
    if stage == "qlib-init":
        return
    from mirage_kan.evaluation.quanta import QuantaAdapter

    quanta = locked["quanta"]
    QuantaAdapter(
        quanta["repository"],
        expected_commit=quanta["commit"],
        expected_config_sha256=quanta["config"]["sha256"],
        expected_runner_sha256=quanta["runner"]["sha256"],
        expected_provider_identity=locked["qlib_provider"],
    )
    if stage == "quanta-init":
        return
    from qlib.contrib.model.gbdt import LGBModel  # noqa: F401

    if stage == "lgb-import":
        return
    from qlib.backtest import backtest  # noqa: F401
    from qlib.contrib.evaluate import risk_analysis  # noqa: F401
    from qlib.workflow import R  # noqa: F401
    from qlib.workflow.record_temp import SignalRecord, SigAnaRecord  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "baseline",
            "qlib-import",
            "qlib-init",
            "quanta-init",
            "lgb-import",
            "workflow-import",
        ),
    )
    args = parser.parse_args()
    _configure_torch()
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    _mutate(args.stage, locked)
    from mirage_kan.governance.implementation_lock import _runtime_identity

    live = _runtime_identity()
    differences = _diff(locked["runtime"], live)
    print(
        json.dumps(
            {"stage": args.stage, "passed": not differences, "differences": differences},
            indent=2,
            sort_keys=True,
        )
    )
    return int(bool(differences))


if __name__ == "__main__":
    raise SystemExit(main())
