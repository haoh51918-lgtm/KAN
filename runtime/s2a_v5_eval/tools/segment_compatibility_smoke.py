"""Exercise the actual pinned Quanta precomputed-data segmentation on synthetic rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

from mirage_kan.evaluation.quanta import QuantaAdapter


QUANTA_REPOSITORY = Path("/zju_0012/htq/aaai26_alpha/QuantaAlpha")
QUANTA_COMMIT = "b7ceb27b1001261d7a95b209a963664ae1f8ab23"
QUANTA_CONFIG_SHA256 = (
    "4e095512025a44dcca279e3d3c4d02fc83367caf044032b6c9f6eeb94405a832"
)
QUANTA_RUNNER_SHA256 = (
    "a18ec5bfbe57b452dbacb3cdd15249f99c2b53e7c0761c178e9fbb89db7d34d8"
)
EXPECTED_SEGMENTS = {
    "train": ["2016-01-01", "2020-12-31"],
    "valid": ["2021-01-01", "2021-12-31"],
    "test": ["2022-01-01", "2025-12-26"],
}
SYNTHETIC_DATES = {
    "train": pd.Timestamp("2019-01-02"),
    "valid": pd.Timestamp("2021-06-01"),
    "test": pd.Timestamp("2023-01-03"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    workspace = Path(args.workspace).resolve(strict=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = workspace / output
    output.parent.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace smoke evidence: {output}")

    qlib_loaded_before = "qlib" in sys.modules
    adapter = QuantaAdapter(
        QUANTA_REPOSITORY,
        expected_commit=QUANTA_COMMIT,
        expected_config_sha256=QUANTA_CONFIG_SHA256,
        expected_runner_sha256=QUANTA_RUNNER_SHA256,
    )
    original_segments = deepcopy(adapter.runner.config["dataset"]["segments"])
    if original_segments != EXPECTED_SEGMENTS:
        raise ValueError("pinned Quanta segment values differ from the smoke contract")

    index = pd.MultiIndex.from_product(
        [list(SYNTHETIC_DATES.values()), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    panel = pd.DataFrame(
        {"synthetic_factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, index=index
    )
    labels = pd.DataFrame(
        {"LABEL0": [0.01, -0.01, 0.02, -0.02, 0.03, -0.03]}, index=index
    )
    observed: dict[str, list[str]] = {}

    adapter.runner._init_qlib = lambda: None
    adapter._verify_effective_provider = lambda: None
    adapter.runner._compute_label = lambda expression: labels.copy(deep=True)

    def inspect_dataset(dataset, experiment, recorder, output_name=None):
        for name, expected_date in SYNTHETIC_DATES.items():
            prepared = dataset.prepare(name, col_set="feature")
            dates = pd.DatetimeIndex(
                prepared.index.get_level_values("datetime")
            ).unique()
            if not dates.equals(pd.DatetimeIndex([expected_date])):
                raise AssertionError(f"pinned Quanta {name} segment leaked dates")
            observed[name] = [timestamp.isoformat() for timestamp in dates]
        if adapter.runner.config["dataset"]["segments"] != EXPECTED_SEGMENTS:
            raise AssertionError(
                "runner config was not restored after dataset construction"
            )
        return {"synthetic_segment_smoke": True}

    adapter.runner._train_and_backtest = inspect_dataset
    metrics = adapter.evaluate_panel(
        panel,
        experiment_name="synthetic_segment_smoke",
        recorder_name="synthetic_segment_smoke",
        output_name="synthetic_segment_smoke",
    )

    import qlib
    from qlib.config import C

    payload = {
        "schema_version": "s2a_v5_segment_compatibility_smoke_v1",
        "passed": (
            metrics == {"synthetic_segment_smoke": True}
            and observed
            == {name: [date.isoformat()] for name, date in SYNTHETIC_DATES.items()}
            and adapter.identity.get("computed_factor_segments") == EXPECTED_SEGMENTS
            and adapter.runner.config["dataset"]["segments"] == EXPECTED_SEGMENTS
            and not C.registered
        ),
        "synthetic_only": True,
        "real_label_access": False,
        "qlib_loaded_before_smoke": qlib_loaded_before,
        "qlib_provider_initialized": bool(C.registered),
        "qlib_version": qlib.__version__,
        "quanta": {
            "commit": QUANTA_COMMIT,
            "config_sha256": QUANTA_CONFIG_SHA256,
            "runner_sha256": QUANTA_RUNNER_SHA256,
        },
        "original_segments": original_segments,
        "adapter_identity_segments": adapter.identity["computed_factor_segments"],
        "prepared_dates": observed,
        "config_restored": adapter.runner.config["dataset"]["segments"]
        == original_segments,
    }
    if not payload["passed"]:
        raise RuntimeError("synthetic segment compatibility smoke failed")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o444,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
