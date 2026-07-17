"""Run the frozen train-only E3/MLP compatibility gate on one CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mirage_kan.data import PitPanel
from mirage_kan.dsl import evaluate
from mirage_kan.mining.e3 import evaluate_hard_ast_from_atoms
from mirage_kan.mining.e3_runner import materialize_atom_panel, run_e3_profile
from mirage_kan.mining.mlp_control import (
    MLPControlPairing,
    replay_control_on_atom_panel,
    run_matched_blackbox_controls,
)


REPO = Path(__file__).resolve().parents[3]
PROFILE = "short_price"
TRAINING_STEPS = 300
PAIRING_COUNT = 6
SYNTHETIC_START = "2020-01-02"
SYNTHETIC_DAYS = 90
SYNTHETIC_INSTRUMENTS = 6
MLP_REPLAY_RTOL = 0.0
MLP_REPLAY_ATOL = 0.0
SOURCE_FILES = (
    "src/mirage_kan/data/pit.py",
    "src/mirage_kan/dsl/core.py",
    "src/mirage_kan/mining/e3.py",
    "src/mirage_kan/mining/e3_runner.py",
    "src/mirage_kan/mining/mlp_control.py",
    "src/mirage_kan/artifacts/v2_bundle.py",
    "src/mirage_kan/evaluation/v2_runner.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(values.dtype), "shape": list(values.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + values.numpy().tobytes()).hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def synthetic_inputs() -> tuple[PitPanel, pd.Series]:
    dates = pd.bdate_range(SYNTHETIC_START, periods=SYNTHETIC_DAYS)
    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        for instrument_index in range(SYNTHETIC_INSTRUMENTS):
            base = 20.0 + 1.7 * instrument_index + 0.09 * date_index
            cycle = math.sin((date_index + 2 * instrument_index) / 5.0)
            close = base + cycle
            rows.append(
                {
                    "datetime": date,
                    "instrument": f"S{instrument_index}",
                    "open": close * (0.996 + 0.0005 * instrument_index),
                    "high": close * 1.012,
                    "low": close * 0.988,
                    "close": close,
                    "volume": 1000.0 + 13.0 * date_index + 41.0 * instrument_index,
                    "in_universe": not (date_index == 72 and instrument_index == 5),
                }
            )
    panel = PitPanel.from_frame(pd.DataFrame(rows))
    close = panel.field("Close")
    target = (
        close.groupby(level="instrument", sort=False).shift(-1) / close - 1.0
    ).rename("synthetic_forward_return")
    return panel, target


def configure_determinism(device: str) -> dict[str, object]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED must be 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("the compatibility gate requires two visible CUDA devices")
    if device not in {"cuda:0", "cuda:1"}:
        raise ValueError("device must be cuda:0 or cuda:1")
    device_index = int(device.rsplit(":", 1)[1])
    torch.cuda.set_device(device_index)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    np.random.seed(0)
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": device,
        "device_capability": list(torch.cuda.get_device_capability(device_index)),
        "device_name": properties.name,
        "device_total_memory": properties.total_memory,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "visible_device_count": torch.cuda.device_count(),
    }


def ast_replay_evidence(
    panel: PitPanel, target: pd.Series, device: str
) -> tuple[dict[str, object], tuple[MLPControlPairing, ...]]:
    run = run_e3_profile(panel, target, PROFILE, device=device)
    if len(run.miners) != 64:
        raise AssertionError("production E3 profile did not return exactly 64 miners")
    if any(len(miner.trajectory) != TRAINING_STEPS for miner in run.miners):
        raise AssertionError("production E3 miner did not complete 300 updates")

    atom_panel = materialize_atom_panel(panel, PROFILE)
    finite_atoms = torch.from_numpy(
        np.array(
            np.where(atom_panel.support, atom_panel.values, 0.0),
            dtype=np.float64,
            copy=True,
        )
    )
    replay_hashes = []
    support_hashes = []
    candidate_hashes = []
    for miner in run.miners:
        hard_replay = evaluate_hard_ast_from_atoms(
            miner.hardening.ast, atom_panel.atom_manifest, finite_atoms
        ).numpy()
        public = evaluate(miner.hardening.ast, panel)
        public_support = public.support.reindex(
            atom_panel.index, fill_value=False
        ).to_numpy(dtype=bool)
        expected_support = (
            atom_panel.support[..., miner.hardening.positive.atom_index]
            & atom_panel.support[..., miner.hardening.negative.atom_index]
        ).reshape(-1)
        np.testing.assert_array_equal(public_support, expected_support)
        public_values = public.values.reindex(atom_panel.index).to_numpy(
            dtype=np.float64
        )
        flat_hard = hard_replay.reshape(-1)
        np.testing.assert_array_equal(
            flat_hard[public_support], public_values[public_support]
        )
        replay_hashes.append(
            tensor_sha256(torch.from_numpy(flat_hard[public_support].copy()))
        )
        support_hashes.append(tensor_sha256(torch.from_numpy(public_support.copy())))
        candidate_hashes.append(miner.candidate_ast.identity)

    pairings = tuple(
        MLPControlPairing(
            miner.profile,
            miner.global_attempt_index,
            miner.bootstrap,
        )
        for miner in run.miners[:PAIRING_COUNT]
    )
    final_logits = torch.stack(tuple(miner.final_logits for miner in run.miners))
    trajectory = torch.stack(
        tuple(
            torch.stack(tuple(step.gate_logits for step in miner.trajectory))
            for miner in run.miners
        )
    )
    return (
        {
            "atom_count": len(atom_panel.atom_manifest),
            "candidate_ast_identities_sha256": json_sha256(candidate_hashes),
            "completed_updates_per_miner": TRAINING_STEPS,
            "exact_public_ast_replay_count": len(replay_hashes),
            "exact_public_ast_replay_sha256": json_sha256(replay_hashes),
            "exact_public_support_replay_count": len(support_hashes),
            "exact_public_support_replay_sha256": json_sha256(support_hashes),
            "final_logits_sha256": tensor_sha256(final_logits),
            "hardening_receipt_count": len(run.miners),
            "miner_count": len(run.miners),
            "profile": PROFILE,
            "scheduled_updates_per_miner": TRAINING_STEPS,
            "trajectory_sha256": tensor_sha256(trajectory),
        },
        pairings,
    )


def mlp_replay_evidence(
    panel: PitPanel,
    target: pd.Series,
    pairings: tuple[MLPControlPairing, ...],
    device: str,
) -> dict[str, object]:
    result = run_matched_blackbox_controls(
        panel,
        target,
        pairings,
        minimum_library_size=PAIRING_COUNT,
        library_cap=16,
        device=device,
    )
    if len(result.controls) != PAIRING_COUNT:
        raise AssertionError("production MLP control count differs from pairings")
    if any(
        control.completed_updates != TRAINING_STEPS
        or len(control.trajectory) != TRAINING_STEPS
        for control in result.controls
    ):
        raise AssertionError("production MLP control did not complete 300 updates")

    atom_panel = materialize_atom_panel(panel, PROFILE)
    aligned_target = (
        target.reindex(atom_panel.index)
        .to_numpy(dtype=np.float64)
        .reshape(len(atom_panel.dates), len(atom_panel.instruments))
    )
    valid_mask = (
        atom_panel.joint_support & atom_panel.membership & np.isfinite(aligned_target)
    )
    objective_dates = np.any(valid_mask, axis=1)
    replay_hashes = []
    exact_compared_counts = []
    diagnostic_max_abs_errors = []
    diagnostic_max_rel_errors = []
    diagnostic_mismatch_counts = []
    diagnostic_compared_counts = []
    masks_equal = []
    expected_prediction_mask = (atom_panel.joint_support & atom_panel.membership)[
        objective_dates
    ]
    for control in result.controls:
        canonical_publication = replay_control_on_atom_panel(
            control, atom_panel
        ).to_numpy()
        independent_replay = replay_control_on_atom_panel(
            control, atom_panel
        ).to_numpy()
        canonical_mask = np.isfinite(canonical_publication)
        independent_mask = np.isfinite(independent_replay)
        expected_public_mask = (
            atom_panel.joint_support & atom_panel.membership
        ).reshape(-1)
        np.testing.assert_array_equal(canonical_mask, independent_mask)
        np.testing.assert_array_equal(canonical_mask, expected_public_mask)
        np.testing.assert_allclose(
            canonical_publication,
            independent_replay,
            rtol=MLP_REPLAY_RTOL,
            atol=MLP_REPLAY_ATOL,
            equal_nan=True,
        )
        masks_equal.append(True)
        exact_compared_counts.append(int(canonical_mask.sum()))
        replay_hashes.append(
            tensor_sha256(torch.from_numpy(independent_replay[independent_mask].copy()))
        )

        publication_on_objective_dates = canonical_publication.reshape(
            len(atom_panel.dates), len(atom_panel.instruments)
        )[objective_dates]
        prediction = control.prediction.numpy()
        prediction_mask = control.prediction_mask.numpy()
        np.testing.assert_array_equal(prediction_mask, expected_prediction_mask)
        replay_values = publication_on_objective_dates[prediction_mask]
        prediction_values = prediction[prediction_mask]
        absolute_error = np.abs(replay_values - prediction_values)
        relative_error = np.divide(
            absolute_error,
            np.abs(prediction_values),
            out=np.where(absolute_error == 0.0, 0.0, np.inf),
            where=np.abs(prediction_values) > 0.0,
        )
        diagnostic_max_abs_errors.append(float(absolute_error.max(initial=0.0)))
        diagnostic_max_rel_errors.append(float(relative_error.max(initial=0.0)))
        diagnostic_mismatch_counts.append(
            int(np.count_nonzero(replay_values != prediction_values))
        )
        diagnostic_compared_counts.append(int(replay_values.size))

    final_parameters = torch.stack(
        tuple(control.final_parameters for control in result.controls)
    )
    trajectory = torch.stack(
        tuple(
            torch.stack(tuple(step.parameters for step in control.trajectory))
            for control in result.controls
        )
    )
    return {
        "completed_updates_per_control": TRAINING_STEPS,
        "control_count": len(result.controls),
        "exact_public_replay": {
            "compared_count": sum(exact_compared_counts),
            "masks_equal": all(masks_equal),
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mismatch_count": 0,
            "replay_atol": MLP_REPLAY_ATOL,
            "replay_rtol": MLP_REPLAY_RTOL,
            "replay_sha256": json_sha256(replay_hashes),
            "seam": "canonical CPU publication replay to independent CPU checkpoint replay",
        },
        "factor_library_publication_allowed": result.factor_library_publication_allowed,
        "final_parameters_sha256": tensor_sha256(final_parameters),
        "gpu_training_vs_cpu_publication_diagnostic": {
            "compared_count": sum(diagnostic_compared_counts),
            "gating": False,
            "max_abs_error": max(diagnostic_max_abs_errors, default=0.0),
            "max_rel_error": max(diagnostic_max_rel_errors, default=0.0),
            "max_rel_error_definition": "abs(cpu_publication-gpu_training)/abs(gpu_training), with 0 for exact zero pairs and inf for nonzero error over gpu_training zero",
            "mismatch_count": sum(diagnostic_mismatch_counts),
        },
        "pairing_global_attempt_indices": [
            control.kan_global_attempt_index for control in result.controls
        ],
        "promotion_eligible": result.promotion_eligible,
        "replay_count": len(replay_hashes),
        "role": result.role,
        "scheduled_updates_per_control": TRAINING_STEPS,
        "trajectory_sha256": tensor_sha256(trajectory),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, choices=("cuda:0", "cuda:1"))
    parser.add_argument("--output-label", required=True)
    args = parser.parse_args()
    determinism = configure_determinism(args.device)
    panel, target = synthetic_inputs()
    date_values = panel.raw.index.get_level_values("datetime")
    if date_values.max() > pd.Timestamp("2020-12-31"):
        raise AssertionError("synthetic compatibility data crosses the train boundary")
    ast, pairings = ast_replay_evidence(panel, target, args.device)
    mlp = mlp_replay_evidence(panel, target, pairings, args.device)
    torch.cuda.synchronize()

    payload = {
        "ast_hardening_replay": ast,
        "environment": {
            **determinism,
            "cudnn_runtime": torch.backends.cudnn.version(),
            "cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "mlp_control_replay": mlp,
        "output_label": args.output_label,
        "passed": True,
        "role": "compatibility_gate_not_scientific_evidence",
        "schema_version": "s2a_v4_train_only_ast_mlp_compatibility_v1",
        "source_sha256": {
            path: file_sha256(REPO / path)
            for path in (
                *SOURCE_FILES,
                "runtime/s2a_v4_eval/tools/compatibility_smoke.py",
            )
        },
        "synthetic_data": {
            **panel.audit(),
            "qlib_provider_initialized": False,
            "real_quanta_label_access": False,
            "scope": "synthetic_train_only_within_train_validation_boundary",
            "synthetic_target_used": True,
            "test_period_values_used": False,
        },
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
