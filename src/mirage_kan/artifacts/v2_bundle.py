"""Deterministic flat staging builders for the frozen S2a v2 artifact topology.

These builders deliberately stop before publication.  A caller must bind the
resulting manifest to a :class:`TopologyTransaction` and let that transaction
perform the no-replace publication.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import stat
import zlib
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from mirage_kan.artifacts.library import _evaluate_library
from mirage_kan.dsl import AstNode
from mirage_kan.artifacts.mechanism import MECHANISM_CARD_FIELDS
from mirage_kan.data.pit import PitPanel, sha256_file
from mirage_kan.mining.e3 import (
    PROFILE_SPECS,
    CategoricalE3KAN,
    atom_manifest_sha256,
    build_profile_atom_bank,
    forward_mode_at_step,
    harden_checkpoint,
    temperature_at_step,
)
from mirage_kan.mining.e3_runner import (
    BOOTSTRAP_SEED_BASE,
    MINER_SEED_BASE,
    MINERS_PER_PROFILE,
    TRAINING_STEPS,
    ProfileRun,
    draw_training_bootstrap,
    materialize_atom_panel,
)
from mirage_kan.mining.gp_control import (
    ATTEMPTS_PER_PROFILE,
    GP_GENERATION_SEED,
    GpGenerationResult,
)
from mirage_kan.mining.mlp_control import (
    LEARNING_RATE,
    MLP_SEED_BASE,
    PARAMETER_GAP_MAXIMUM,
    MatchedBlackboxControlPanel,
    replay_control_on_atom_panel,
)
from mirage_kan.mining.v2_scoring import HardAstScoringRun, HardAstSelection

SCHEMA_VERSION = "mirage_s2a_v2_staging_bundle_v1"
_IDENTITY_KEYS = (
    "protocol_sha256",
    "authority_sha256",
    "implementation_sha256",
)
_TOP_CHILDREN = frozenset(
    {
        "kan_library",
        "gp_control_library",
        "permutation_control_library",
        "blackbox_control",
        "mechanism_cards",
        "blind_review_package",
    }
)
_TOP_BUDGET_KEYS = frozenset(
    {
        "kan_attempts",
        "kan_updates",
        "gp_attempts",
        "permutation_attempts",
        "mlp_controls",
        "mlp_updates",
    }
)
_BLIND_ITEM_FIELDS = frozenset(
    {
        "identity_and_canonical_ast",
        "raw_variables_windows_and_complexity",
        "edge_gate_and_shape_summary",
        "variable_and_lag_interventions",
        "response_direction_questions",
        "one_sentence_mechanism",
        "applicability_and_failure_conditions",
    }
)
_BLIND_PACKAGE_FIELDS = frozenset(
    {
        "review_status",
        "reviewers_minimum",
        "mechanism_restatement_required",
        "response_direction_accuracy_minimum",
        "inter_reviewer_agreement_reported",
        "items",
    }
)
_BLIND_HIDES = ("method_name", "pnl", "return_metrics")
_CHILD_EXPECTATIONS: dict[str, dict[str, object]] = {
    "kan_library": {
        "schema_version": "mirage_factor_library_v1",
        "library_role": "kan_e3_selected",
        "role": "kan_e3_selected",
        "output_kind": "factor_library",
        "kan_mined": True,
        "promotion_eligible": True,
        "count_field": "factor_count",
    },
    "gp_control_library": {
        "schema_version": "mirage_factor_library_v1",
        "library_role": "typed_gp_sr_control",
        "role": "typed_gp_sr_control",
        "output_kind": "factor_library",
        "kan_mined": False,
        "promotion_eligible": False,
        "count_field": "factor_count",
    },
    "permutation_control_library": {
        "schema_version": "mirage_factor_library_v1",
        "library_role": "kan_e3_permutation_control",
        "role": "kan_e3_permutation_control",
        "output_kind": "factor_library",
        "kan_mined": False,
        "promotion_eligible": False,
        "count_field": "factor_count",
    },
    "blackbox_control": {
        "schema_version": "mirage_matched_blackbox_control_v2",
        "role": "falsification_control_never_production",
        "output_kind": "control_panel_not_factor_library",
        "kan_mined": False,
        "promotion_eligible": False,
        "factor_library_publication_allowed": False,
        "count_field": "control_count",
    },
    "mechanism_cards": {
        "schema_version": SCHEMA_VERSION,
        "role": "kan_mechanism_evidence_pending_human_review",
        "output_kind": "mechanism_cards",
        "kan_mined": True,
        "promotion_eligible": False,
        "count_field": "card_count",
    },
    "blind_review_package": {
        "schema_version": SCHEMA_VERSION,
        "role": "human_blind_review_input",
        "output_kind": "blind_review_package",
        "kan_mined": False,
        "promotion_eligible": False,
        "count_field": "blind_item_count",
    },
}


@dataclass(frozen=True)
class StagedBundle:
    """A validated staging directory which is not yet a published artifact."""

    path: Path
    manifest: dict[str, object]


def _require_library_size(
    count: int,
    *,
    minimum_library_size: int,
    library_cap: int,
    label: str,
) -> None:
    if (
        type(count) is not int
        or type(minimum_library_size) is not int
        or type(library_cap) is not int
        or not 1 <= minimum_library_size <= count <= library_cap
    ):
        raise ValueError(f"{label} violates frozen admission bounds")


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_json_line(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_identities(identities: Mapping[str, object]) -> dict[str, str]:
    if set(identities) != set(_IDENTITY_KEYS):
        raise ValueError(f"identities must contain exactly {_IDENTITY_KEYS}")
    result: dict[str, str] = {}
    for key in _IDENTITY_KEYS:
        value = identities[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"identity {key} must be a lowercase SHA-256")
        result[key] = value
    return result


def _finite_float(value: object, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _canonicalize(value: object, label: str = "evidence") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return _finite_float(value, label)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, AstNode):
        return _canonicalize(value.to_dict(), label)
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{label} tensor must be stored in tensor evidence")
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be nonempty strings")
            result[key] = _canonicalize(value[key], f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize(item, f"{label}[{index}]") for index, item in enumerate(value)
        ]
    to_record = getattr(value, "to_record", None)
    if callable(to_record):
        return _canonicalize(to_record(), label)
    if is_dataclass(value):
        return _canonicalize(
            {field.name: getattr(value, field.name) for field in fields(value)}, label
        )
    raise TypeError(f"{label} has unsupported evidence type {type(value).__name__}")


class _TensorEvidence:
    """Canonical little-endian tensor stream with per-value replay metadata."""

    def __init__(self) -> None:
        self._payload = bytearray()

    def add(self, name: str, tensor: torch.Tensor) -> dict[str, object]:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.layout is not torch.strided or tensor.is_quantized:
            raise ValueError(f"{name} must be a dense, non-quantized tensor")
        value = tensor.detach().to(device="cpu").contiguous()
        if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} tensor values must be finite")
        dtype_map = {
            torch.bool: "|u1",
            torch.uint8: "|u1",
            torch.int8: "|i1",
            torch.int16: "<i2",
            torch.int32: "<i4",
            torch.int64: "<i8",
            torch.float32: "<f4",
            torch.float64: "<f8",
        }
        if value.dtype not in dtype_map:
            raise ValueError(f"{name} uses unsupported tensor dtype {value.dtype}")
        if value.dtype is torch.bool:
            array = value.to(dtype=torch.uint8).numpy()
        else:
            array = value.numpy().astype(dtype_map[value.dtype], copy=False)
        data = array.tobytes(order="C")
        offset = len(self._payload)
        self._payload.extend(data)
        return {
            "name": name,
            "offset": offset,
            "nbytes": len(data),
            "shape": list(value.shape),
            "dtype": dtype_map[value.dtype],
            "sha256": _sha256_bytes(data),
        }

    def compressed(self) -> bytes:
        return zlib.compress(bytes(self._payload), level=9)


def _stage(
    staging_path: Path | str,
    *,
    files: Mapping[str, bytes],
    identities: Mapping[str, object],
    role: str,
    kan_mined: bool | None,
    promotion_eligible: bool,
    budget_counts: Mapping[str, int],
    schema_version: str = SCHEMA_VERSION,
    extra_manifest: Mapping[str, object] | None = None,
) -> StagedBundle:
    identity_record = _validate_identities(identities)
    if not role:
        raise ValueError("artifact role must not be empty")
    canonical_budget = _canonicalize(budget_counts, "budget_counts")
    if not isinstance(canonical_budget, dict) or any(
        type(value) is not int or value < 0 for value in canonical_budget.values()
    ):
        raise ValueError("budget counts must be nonnegative integers")
    if "manifest.json" in files:
        raise ValueError("manifest is produced by the staging builder")
    for name, body in files.items():
        path = Path(name)
        if (
            not name
            or path.name != name
            or path.is_absolute()
            or not isinstance(body, bytes)
        ):
            raise ValueError("staging files must have safe flat names and bytes bodies")

    raw = Path(staging_path)
    if not raw.name.endswith(".staging"):
        raise ValueError("staging path name must end with .staging")
    parent = raw.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("staging parent must be a real directory")
    staging = parent / raw.name
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"refusing to replace staging path: {staging}")

    file_hashes = {name: _sha256_bytes(files[name]) for name in sorted(files)}
    manifest: dict[str, object] = {
        "schema_version": schema_version,
        "publication_state": "staged_unpublished",
        "role": role,
        "promotion_eligible": bool(promotion_eligible),
        "identities": identity_record,
        "budget_counts": canonical_budget,
        "files": file_hashes,
        "topology_binding_required_before_publication": True,
    }
    if kan_mined is not None:
        manifest["kan_mined"] = bool(kan_mined)
    if extra_manifest:
        overlap = set(manifest).intersection(extra_manifest)
        if overlap:
            raise ValueError(
                f"extra manifest cannot replace reserved fields: {overlap}"
            )
        extra = _canonicalize(extra_manifest, "extra_manifest")
        if not isinstance(extra, dict):
            raise TypeError("extra manifest must be a mapping")
        manifest.update(extra)

    os.mkdir(staging, 0o700)
    try:
        for name in sorted(files):
            destination = staging / name
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                view = memoryview(files[name])
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("staging write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        manifest_path = staging / "manifest.json"
        descriptor = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            data = _canonical_json_bytes(manifest)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("manifest write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_descriptor = os.open(
            staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if any(
        entry.is_symlink() or not stat.S_ISREG(entry.stat().st_mode)
        for entry in staging.iterdir()
    ):
        shutil.rmtree(staging)
        raise ValueError("staging did not produce only flat regular files")
    return StagedBundle(staging, manifest)


def _bootstrap_record(value: object) -> dict[str, object]:
    record = _canonicalize(value, "bootstrap")
    if not isinstance(record, dict):
        raise TypeError("bootstrap evidence must be a record")
    sampled = record.get("sampled_date_indices")
    multiplicities = record.get("multiplicities")
    date_count = record.get("date_count")
    target_draws = record.get("target_date_draws")
    if (
        not isinstance(sampled, list)
        or not isinstance(multiplicities, list)
        or len(multiplicities) != date_count
        or len(sampled) != target_draws
        or sum(multiplicities) != target_draws
    ):
        raise ValueError("bootstrap receipt has inconsistent budget counts")
    return record


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=True)
    return buffer.getvalue()


def stage_v2_factor_library(
    staging_path: Path | str,
    programs: Mapping[str, AstNode],
    panel: PitPanel,
    *,
    topology_key: str,
    identities: Mapping[str, object],
    minimum_library_size: int,
    library_cap: int,
    factor_lineage: Mapping[str, Mapping[str, object]] | None = None,
) -> StagedBundle:
    """Stage one enriched, executable v2 factor-library child."""
    definitions = {
        "kan_library": ("kan_e3_selected", True, True),
        "gp_control_library": ("typed_gp_sr_control", False, False),
        "permutation_control_library": (
            "kan_e3_permutation_control",
            False,
            False,
        ),
    }
    if topology_key not in definitions:
        raise ValueError("v2 factor library has an unknown topology key")
    _require_library_size(
        len(programs),
        minimum_library_size=minimum_library_size,
        library_cap=library_cap,
        label="v2 factor library size",
    )
    values, support, factor_records = _evaluate_library(programs, panel)
    lineage = dict(factor_lineage or {})
    role, kan_mined, promotion_eligible = definitions[topology_key]
    if kan_mined and set(lineage) != set(programs):
        raise ValueError("KAN library requires selected lineage for every factor")
    if not kan_mined and lineage:
        raise ValueError("control factor libraries cannot carry KAN selected lineage")
    global_indices: set[int] = set()
    for factor_id, record in lineage.items():
        canonical = record.get("canonical_hash")
        global_index = record.get("global_attempt_index")
        if (
            canonical != programs[factor_id].identity
            or type(global_index) is not int
            or not 0 <= global_index < 256
            or global_index in global_indices
        ):
            raise ValueError("KAN factor lineage differs from its executable AST")
        global_indices.add(global_index)
        factor_records[factor_id]["global_attempt_index"] = global_index
    files = {
        "expression_support.parquet": _parquet_bytes(support),
        "factor_panel.parquet": _parquet_bytes(values),
    }
    return _stage(
        staging_path,
        files=files,
        identities=identities,
        role=role,
        kan_mined=kan_mined,
        promotion_eligible=promotion_eligible,
        budget_counts={"factor_count": len(programs)},
        schema_version="mirage_factor_library_v1",
        extra_manifest={
            "library_role": role,
            "output_kind": "factor_library",
            "factor_library_publication_allowed": True,
            "scientific_result": False,
            "factor_count": len(programs),
            "panel_rows": len(values),
            "index_names": list(values.index.names),
            "factors": factor_records,
            "mask_semantics": {
                "raw_observed": "per-field finite raw observation; never imputed",
                "expression_support": "operator support before universe filtering",
                "membership": "dynamic-universe membership applied to factor values",
                "finite_output": "required on expression support and membership",
                "tradability": "separate from factor-value publication",
            },
        },
    )


def _append_kan_profile_run(
    run: ProfileRun, archive: _TensorEvidence, *, namespace: str
) -> list[dict[str, object]]:
    """Validate and append one complete profile to the top-bundle tensor stream."""
    if run.profile not in PROFILE_SPECS:
        raise ValueError("KAN profile run has an unknown profile")
    if len(run.miners) != MINERS_PER_PROFILE:
        raise ValueError("KAN profile run must contain exactly 64 miners")
    profile_index = tuple(PROFILE_SPECS).index(run.profile)
    atom_count = len(build_profile_atom_bank(run.profile))
    rows: list[dict[str, object]] = []
    for miner_index, miner in enumerate(run.miners):
        if (
            miner.profile != run.profile
            or miner.profile_index != profile_index
            or miner.miner_index != miner_index
            or miner.global_attempt_index
            != profile_index * MINERS_PER_PROFILE + miner_index
        ):
            raise ValueError(
                "KAN miner identity or global budget index is inconsistent"
            )
        expected_seed = MINER_SEED_BASE + profile_index * 1000 + miner_index
        if miner.miner_seed != expected_seed:
            raise ValueError("KAN miner seed differs from the frozen derivation")
        expected_initial = CategoricalE3KAN(
            run.profile, seed=expected_seed
        ).checkpoint_logits()
        if not torch.equal(miner.initial_logits, expected_initial):
            raise ValueError("KAN initial logits do not replay from the frozen seed")
        expected_bootstrap = draw_training_bootstrap(
            miner.bootstrap.date_count,
            BOOTSTRAP_SEED_BASE + miner.global_attempt_index,
        )
        if miner.bootstrap != expected_bootstrap:
            raise ValueError("KAN bootstrap differs from the frozen seed derivation")
        if len(miner.trajectory) != TRAINING_STEPS or [
            step.update_index for step in miner.trajectory
        ] != list(range(TRAINING_STEPS)):
            raise ValueError(
                "each KAN miner must contain the exact 300-update trajectory"
            )
        expected_shape = (2, atom_count)
        tensors = {
            "initial_logits": miner.initial_logits,
            "final_logits": miner.final_logits,
            "first_step_data_gradient": miner.first_step_data_gradient,
        }
        if any(tuple(value.shape) != expected_shape for value in tensors.values()):
            raise ValueError("KAN gate tensor shape differs from its frozen atom bank")
        if any(value.dtype is not torch.float64 for value in tensors.values()):
            raise ValueError("KAN gate tensors must be float64")
        prefix = f"{namespace}/{run.profile}/{miner_index:03d}"
        tensor_refs = {
            name: archive.add(f"{prefix}/{name}", tensor)
            for name, tensor in tensors.items()
        }
        trajectory: list[dict[str, object]] = []
        for step in miner.trajectory:
            if tuple(step.gate_logits.shape) != expected_shape:
                raise ValueError("KAN trajectory gate shape differs from its atom bank")
            if step.gate_logits.dtype is not torch.float64:
                raise ValueError("KAN trajectory gate tensors must be float64")
            if step.tau != temperature_at_step(step.update_index):
                raise ValueError(
                    "KAN temperature schedule differs from the frozen schedule"
                )
            if step.mode != forward_mode_at_step(step.update_index):
                raise ValueError(
                    "KAN forward mode schedule differs from the frozen schedule"
                )
            trajectory.append(
                {
                    "update_index": step.update_index,
                    "tau": _finite_float(step.tau, "KAN temperature"),
                    "mode": step.mode,
                    "total_loss": _finite_float(step.total_loss, "KAN loss"),
                    "mean_daily_ic": _finite_float(step.mean_daily_ic, "KAN IC"),
                    "entropy": _finite_float(step.entropy, "KAN entropy"),
                    "edge_overlap": _finite_float(step.edge_overlap, "KAN overlap"),
                    "gate_logits": archive.add(
                        f"{prefix}/trajectory/{step.update_index:03d}", step.gate_logits
                    ),
                }
            )
        if not torch.equal(miner.trajectory[-1].gate_logits, miner.final_logits):
            raise ValueError(
                "KAN final logits differ from the last trajectory checkpoint"
            )
        expected_hardening = harden_checkpoint(
            miner.final_logits, build_profile_atom_bank(run.profile), 0.1
        )
        if miner.hardening != expected_hardening:
            raise ValueError("KAN hardening receipt or checkpoint hash does not replay")
        hardening = _canonicalize(miner.hardening, "hardening")
        if not isinstance(hardening, dict):
            raise TypeError("KAN hardening evidence must be a record")
        if miner.candidate_ast.identity != miner.hardening.ast.identity:
            raise ValueError("KAN candidate AST differs from hardening receipt")
        fidelity = _canonicalize(miner.fidelity, "fidelity")
        if not isinstance(fidelity, dict) or not {"pearson", "nrmse"}.issubset(
            fidelity
        ):
            raise ValueError("KAN fidelity must contain finite pearson and nrmse")
        rows.append(
            {
                "profile": miner.profile,
                "profile_index": miner.profile_index,
                "miner_index": miner.miner_index,
                "global_attempt_index": miner.global_attempt_index,
                "miner_seed": miner.miner_seed,
                "bootstrap": _bootstrap_record(miner.bootstrap),
                **tensor_refs,
                "trajectory": trajectory,
                "hardening": hardening,
                "candidate_ast": miner.candidate_ast.to_dict(),
                "candidate_ast_sha256": miner.candidate_ast.identity,
                "fidelity": fidelity,
                "admission_failures": list(miner.admission_failures),
            }
        )
    return rows


def _records(value: object, label: str) -> list[dict[str, object]]:
    source = getattr(value, "candidates", value)
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise TypeError(f"{label} must be a candidate sequence or scoring run")
    result: list[dict[str, object]] = []
    for index, item in enumerate(source):
        record = _canonicalize(item, f"{label}[{index}]")
        if not isinstance(record, dict):
            raise TypeError(f"{label} rows must be records")
        result.append(record)
    return result


def _selection_record(selection: object) -> dict[str, object]:
    if isinstance(selection, Mapping):
        record = _canonicalize(selection, "selection")
    else:
        record = {
            "selected_candidate_ids": [
                candidate.candidate_id for candidate in selection.selected
            ],
            "dispositions": _canonicalize(selection.dispositions, "dispositions"),
            "minimum_size_met": bool(selection.minimum_size_met),
            "exact_size_met": bool(selection.exact_size_met),
            "profile_quota_met": bool(selection.profile_quota_met),
            "target_size": selection.target_size,
        }
    if not isinstance(record, dict):
        raise TypeError("selection must be a record")
    return record


def _serialize_gp_ledgers(
    generation: GpGenerationResult,
    scoring: HardAstScoringRun,
    selection: HardAstSelection,
) -> tuple[dict[str, bytes], dict[str, object]]:
    """Validate and serialize all GP budget rows for embedding in mining top."""
    if not isinstance(scoring, HardAstScoringRun) or not isinstance(
        selection, HardAstSelection
    ):
        raise TypeError("GP scoring and selection must use frozen strong types")
    expected = ATTEMPTS_PER_PROFILE * len(PROFILE_SPECS)
    attempts = tuple(generation.attempts)
    if generation.seed != GP_GENERATION_SEED:
        raise ValueError("GP generation seed differs from the frozen protocol")
    if (
        generation.attempts_per_profile != ATTEMPTS_PER_PROFILE
        or len(attempts) != expected
    ):
        raise ValueError("GP generation must consume exactly 64 attempts per profile")
    if [attempt.global_attempt_index for attempt in attempts] != list(range(expected)):
        raise ValueError("GP global attempt budget indices must be contiguous")
    for profile_index, profile in enumerate(PROFILE_SPECS):
        section = attempts[
            profile_index * ATTEMPTS_PER_PROFILE : (profile_index + 1)
            * ATTEMPTS_PER_PROFILE
        ]
        if [attempt.profile for attempt in section] != [
            profile
        ] * ATTEMPTS_PER_PROFILE or [
            attempt.attempt_index for attempt in section
        ] != list(range(ATTEMPTS_PER_PROFILE)):
            raise ValueError("GP profile attempt budget is incomplete or reordered")
    generation_rows = [
        _canonicalize(attempt.to_record(), "GP attempt") for attempt in attempts
    ]
    for attempt, row in zip(attempts, generation_rows, strict=True):
        if not isinstance(row, dict):
            raise TypeError("GP attempt record must be a mapping")
        if attempt.ast is None:
            if attempt.canonical_hash is not None:
                raise ValueError("invalid GP attempt cannot retain a canonical hash")
        elif (
            attempt.canonical_hash != attempt.ast.identity
            or row.get("canonical_hash") != attempt.ast.identity
        ):
            raise ValueError("GP attempt AST and canonical hash differ")
    scoring_rows = _records(scoring, "GP scoring rows")
    candidate_ids = [attempt.candidate_id for attempt in attempts]
    if (
        len(scoring_rows) != expected
        or [row.get("candidate_id") for row in scoring_rows] != candidate_ids
    ):
        raise ValueError(
            "GP scoring rows must preserve every generation attempt in order"
        )
    for attempt, row in zip(attempts, scoring_rows, strict=True):
        expected_fields = {
            "profile": attempt.profile,
            "attempt_index": attempt.global_attempt_index,
            "method": "typed_gp_sr",
        }
        if any(row.get(field) != value for field, value in expected_fields.items()):
            raise ValueError(
                "GP scoring row identity differs from its generation attempt"
            )
        if not {"production_disposition", "null_disposition"}.issubset(row):
            raise ValueError("GP scoring rows require both terminal dispositions")
    metadata = {
        "label_mode": getattr(scoring, "label_mode", None),
        "label_sha256": getattr(scoring, "label_sha256", None),
        "scoring_run_id": getattr(scoring, "scoring_run_id", None),
        "train": getattr(scoring, "train", None),
        "validation": getattr(scoring, "validation", None),
    }
    if (
        not isinstance(metadata["label_mode"], str)
        or not isinstance(metadata["scoring_run_id"], str)
        or not isinstance(metadata["label_sha256"], str)
        or len(metadata["label_sha256"]) != 64
        or not all(metadata[key] for key in metadata)
    ):
        raise ValueError("GP scoring metadata is incomplete")
    canonical_metadata = _canonicalize(metadata, "GP scoring metadata")
    selection_record = _selection_record(selection)
    dispositions = selection_record.get("dispositions")
    if not isinstance(dispositions, dict) or set(dispositions) != set(candidate_ids):
        raise ValueError(
            "GP selection dispositions must cover every attempted candidate"
        )
    selected = selection_record.get("selected_candidate_ids")
    if not isinstance(selected, list) or not set(selected).issubset(candidate_ids):
        raise ValueError("GP selected candidate IDs must be a subset of attempted IDs")
    disposition_counts: dict[str, int] = {}
    for row in generation_rows:
        disposition = str(row["disposition"])
        disposition_counts[disposition] = disposition_counts.get(disposition, 0) + 1
    return (
        {
            "gp_generation.jsonl": b"".join(
                _canonical_json_line(row) for row in generation_rows
            ),
            "gp_scoring.jsonl": b"".join(
                _canonical_json_line(row) for row in scoring_rows
            ),
            "gp_selection.json": _canonical_json_bytes(selection_record),
            "gp_scoring_metadata.json": _canonical_json_bytes(canonical_metadata),
        },
        {
            "generation_attempts": expected,
            "scoring_rows": expected,
            "selection_dispositions": expected,
            "generation_seed": generation.seed,
            "generation_disposition_counts": disposition_counts,
            "invalid_and_duplicate_attempts_retained": True,
        },
    )


def stage_mlp_control_panel(
    staging_path: Path | str,
    panel: MatchedBlackboxControlPanel,
    full_panel: PitPanel,
    *,
    selected_kan_factor_ids: Sequence[str],
    identities: Mapping[str, object],
    minimum_library_size: int,
    library_cap: int,
) -> StagedBundle:
    """Stage receipts plus a full locked-PIT replay from frozen final parameters."""
    controls = tuple(panel.controls)
    _require_library_size(
        len(controls),
        minimum_library_size=minimum_library_size,
        library_cap=library_cap,
        label="MLP control panel size",
    )
    indices = [control.kan_global_attempt_index for control in controls]
    if len(set(indices)) != len(indices):
        raise ValueError("MLP controls must pair to unique KAN attempts")
    selected_ids = tuple(selected_kan_factor_ids)
    if len(selected_ids) != len(controls) or len(set(selected_ids)) != len(controls):
        raise ValueError("MLP selected KAN factor IDs must be unique and size matched")
    profile_panels = {
        profile: materialize_atom_panel(full_panel, profile)
        for profile in {control.profile for control in controls}
    }
    replayed_predictions = [
        replay_control_on_atom_panel(control, profile_panels[control.profile]).rename(
            f"mlp_for_kan_{control.kan_global_attempt_index:03d}"
        )
        for control in controls
    ]
    prediction_panel = pd.concat(replayed_predictions, axis=1).reindex(
        full_panel.raw.index
    )
    if (
        not prediction_panel.index.equals(full_panel.raw.index)
        or prediction_panel.columns.has_duplicates
        or np.isinf(prediction_panel.to_numpy(dtype=float)).any()
    ):
        raise ValueError("MLP replay did not produce a valid complete locked PIT panel")
    if (
        panel.promotion_eligible
        or panel.factor_library_publication_allowed
        or panel.role != "falsification_control_never_production"
    ):
        raise ValueError("MLP panel must retain its never-production control role")
    archive = _TensorEvidence()
    receipt_rows: list[dict[str, object]] = []
    trajectory_lines: list[bytes] = []
    for control_index, (control, factor_id) in enumerate(
        zip(controls, selected_ids, strict=True)
    ):
        if control.profile not in PROFILE_SPECS:
            raise ValueError("MLP control has an unknown paired profile")
        if type(control.kan_global_attempt_index) is not int or not (
            0
            <= control.kan_global_attempt_index
            < len(PROFILE_SPECS) * MINERS_PER_PROFILE
        ):
            raise ValueError(
                "MLP paired KAN attempt index is outside the frozen budget"
            )
        expected_profile_index = control.kan_global_attempt_index // MINERS_PER_PROFILE
        if tuple(PROFILE_SPECS)[expected_profile_index] != control.profile:
            raise ValueError("MLP profile differs from its paired KAN attempt")
        if control.seed != MLP_SEED_BASE + control.kan_global_attempt_index:
            raise ValueError("MLP seed differs from the frozen paired derivation")
        expected_bootstrap = draw_training_bootstrap(
            control.bootstrap.date_count,
            BOOTSTRAP_SEED_BASE + control.kan_global_attempt_index,
        )
        if control.bootstrap != expected_bootstrap:
            raise ValueError("MLP bootstrap differs from its paired KAN bootstrap")
        if (
            control.scheduled_updates != TRAINING_STEPS
            or control.completed_updates != TRAINING_STEPS
            or len(control.trajectory) != TRAINING_STEPS
            or [step.update_index for step in control.trajectory]
            != list(range(TRAINING_STEPS))
        ):
            raise ValueError(
                "each MLP control must contain the exact 300-update budget"
            )
        if (
            control.kan_mined
            or control.promotion_eligible
            or control.factor_library_publication_allowed
        ):
            raise ValueError("MLP receipt cannot be marked mined or promotion eligible")
        atom_count = len(build_profile_atom_bank(control.profile))
        expected_kan_parameters = 2 * atom_count
        expected_mlp_parameters = 2 * atom_count + 5
        expected_gap = abs(expected_mlp_parameters - expected_kan_parameters) / float(
            expected_kan_parameters
        )
        if (
            control.optimizer != "Adam"
            or control.learning_rate != LEARNING_RATE
            or control.input_atom_count != atom_count
            or control.kan_parameter_count != expected_kan_parameters
            or control.mlp_parameter_count != expected_mlp_parameters
            or control.parameter_relative_gap != expected_gap
            or expected_gap > PARAMETER_GAP_MAXIMUM
        ):
            raise ValueError(
                "MLP parameter gap or frozen capacity budget is inconsistent"
            )
        if control.atom_manifest_sha256 != atom_manifest_sha256(
            build_profile_atom_bank(control.profile)
        ):
            raise ValueError("MLP atom-manifest identity differs from its paired KAN")
        prefix = f"mlp/{control_index:03d}"
        parameters = {
            "initial_parameters": control.initial_parameters,
            "final_parameters": control.final_parameters,
            "first_step_data_gradient": control.first_step_data_gradient,
        }
        if len({tuple(tensor.shape) for tensor in parameters.values()}) != 1:
            raise ValueError("MLP parameter evidence shapes are inconsistent")
        if tuple(control.final_parameters.shape) != (expected_mlp_parameters,):
            raise ValueError(
                "MLP parameter tensor length differs from its model capacity"
            )
        parameter_refs = {
            name: archive.add(f"{prefix}/{name}", tensor)
            for name, tensor in parameters.items()
        }
        trajectory: list[dict[str, object]] = []
        for step in control.trajectory:
            if tuple(step.parameters.shape) != tuple(control.final_parameters.shape):
                raise ValueError("MLP trajectory parameter shape is inconsistent")
            trajectory.append(
                {
                    "update_index": step.update_index,
                    "total_loss": _finite_float(step.total_loss, "MLP loss"),
                    "mean_daily_ic": _finite_float(step.mean_daily_ic, "MLP IC"),
                    "parameters": archive.add(
                        f"{prefix}/trajectory/{step.update_index:03d}", step.parameters
                    ),
                }
            )
        if (
            control.prediction.shape != control.prediction_mask.shape
            or control.prediction_mask.dtype is not torch.bool
            or control.prediction.ndim != 2
            or control.prediction.shape[0] != control.bootstrap.date_count
        ):
            raise ValueError(
                "MLP prediction and mask shapes or dtypes are inconsistent"
            )
        training_prediction_ref = archive.add(
            f"{prefix}/training_prediction", control.prediction
        )
        training_mask_ref = archive.add(
            f"{prefix}/training_prediction_mask", control.prediction_mask
        )
        replayed = prediction_panel.iloc[:, control_index].to_numpy(dtype=float)
        replay_mask = np.isfinite(replayed)
        prediction_ref = archive.add(
            f"{prefix}/prediction",
            torch.from_numpy(np.nan_to_num(replayed, nan=0.0)),
        )
        mask_ref = archive.add(
            f"{prefix}/prediction_mask", torch.from_numpy(replay_mask)
        )
        if not torch.equal(control.trajectory[-1].parameters, control.final_parameters):
            raise ValueError(
                "MLP final parameters differ from the last trajectory checkpoint"
            )
        trajectory_line = _canonical_json_line(
            {
                "receipt_index": control_index,
                "control_id": str(prediction_panel.columns[control_index]),
                "trajectory": trajectory,
            }
        )
        trajectory_lines.append(trajectory_line)
        receipt_rows.append(
            {
                "receipt_index": control_index,
                "profile": control.profile,
                "control_id": str(prediction_panel.columns[control_index]),
                "kan_factor_id": factor_id,
                "kan_global_attempt_index": control.kan_global_attempt_index,
                "seed": control.seed,
                "bootstrap": _bootstrap_record(control.bootstrap),
                "optimizer": control.optimizer,
                "learning_rate": _finite_float(control.learning_rate, "learning rate"),
                "scheduled_updates": control.scheduled_updates,
                "completed_updates": control.completed_updates,
                "input_atom_count": control.input_atom_count,
                "kan_parameter_count": control.kan_parameter_count,
                "mlp_parameter_count": control.mlp_parameter_count,
                "parameter_relative_gap": _finite_float(
                    control.parameter_relative_gap, "parameter relative gap"
                ),
                "atom_manifest_sha256": control.atom_manifest_sha256,
                "valid_support_sha256": control.valid_support_sha256,
                **parameter_refs,
                "trajectory": {
                    "file": "control_trajectories.jsonl",
                    "line": control_index,
                    "sha256": _sha256_bytes(trajectory_line),
                },
                "prediction": prediction_ref,
                "prediction_mask": mask_ref,
                "training_prediction": training_prediction_ref,
                "training_prediction_mask": training_mask_ref,
                "kan_mined": False,
                "promotion_eligible": False,
                "factor_library_publication_allowed": False,
            }
        )
    return _stage(
        staging_path,
        files={
            "control_receipts.jsonl": b"".join(
                _canonical_json_line(row) for row in receipt_rows
            ),
            "control_trajectories.jsonl": b"".join(trajectory_lines),
            "prediction_panel.parquet": _parquet_bytes(prediction_panel),
            "tensor_evidence.zlib": archive.compressed(),
        },
        identities=identities,
        role=panel.role,
        kan_mined=False,
        promotion_eligible=False,
        budget_counts={
            "controls": len(controls),
            "scheduled_updates_per_control": TRAINING_STEPS,
            "completed_updates": len(controls) * TRAINING_STEPS,
        },
        schema_version="mirage_matched_blackbox_control_v2",
        extra_manifest={
            "arm": "matched_blackbox_control",
            "output_kind": panel.output_kind,
            "factor_library_publication_allowed": False,
            "control_count": len(controls),
            "selected_kan_factor_ids": list(selected_ids),
            "paired_kan_global_attempt_indices": indices,
            "tensor_evidence": {
                "encoding": "zlib_level_9_of_concatenated_little_endian_tensors",
                "references": [
                    "control_receipts.jsonl",
                    "control_trajectories.jsonl",
                ],
            },
        },
    )


def stage_mechanism_cards(
    staging_path: Path | str,
    cards: Mapping[str, Mapping[str, object]],
    *,
    selected_factor_ids: Sequence[str],
    identities: Mapping[str, object],
) -> StagedBundle:
    """Stage sorted mechanism cards as evidence, not a human-reviewed claim."""
    if not cards:
        raise ValueError("mechanism cards must not be empty")
    selected = tuple(selected_factor_ids)
    if len(set(selected)) != len(selected) or set(cards) != set(selected):
        raise ValueError(
            "mechanism cards must correspond one-to-one with selected factors"
        )
    rows: list[dict[str, object]] = []
    for factor_id in sorted(cards):
        if not factor_id:
            raise ValueError("mechanism-card factor IDs must not be empty")
        card = _canonicalize(cards[factor_id], f"card.{factor_id}")
        if not isinstance(card, dict):
            raise TypeError("mechanism cards must be records")
        if set(card) != set(MECHANISM_CARD_FIELDS):
            raise ValueError("mechanism card must contain the exact frozen nine fields")
        identity = card.get("identity_and_canonical_ast")
        if not isinstance(identity, dict) or identity.get("factor_id") != factor_id:
            raise ValueError("mechanism card identity differs from its selected factor")
        rows.append({"factor_id": factor_id, "card": card})
    anonymous_mapping = {
        f"B{index:03d}": factor_id
        for index, factor_id in enumerate(sorted(cards), start=1)
    }
    return _stage(
        staging_path,
        files={
            "mechanism_cards.jsonl": b"".join(
                _canonical_json_line(row) for row in rows
            ),
            "blind_anonymous_mapping.json": _canonical_json_bytes(anonymous_mapping),
        },
        identities=identities,
        role="kan_mechanism_evidence_pending_human_review",
        kan_mined=True,
        promotion_eligible=False,
        budget_counts={"mechanism_cards": len(rows)},
        extra_manifest={
            "human_review_complete": False,
            "output_kind": "mechanism_cards",
            "card_count": len(rows),
            "selected_factor_ids": sorted(selected),
            "anonymous_mapping_sha256": _sha256_bytes(
                _canonical_json_bytes(anonymous_mapping)
            ),
        },
    )


def stage_blind_review_package(
    staging_path: Path | str,
    package: Mapping[str, object],
    *,
    selected_factor_ids: Sequence[str],
    hides: Sequence[str],
    identities: Mapping[str, object],
) -> StagedBundle:
    """Stage the method-blind package without asserting a human outcome."""
    record = _canonicalize(package, "blind_review_package")
    if not isinstance(record, dict) or set(record) != _BLIND_PACKAGE_FIELDS:
        raise ValueError("blind package fields differ from the frozen schema")
    if record.get("review_status") != "pending_human_review":
        raise ValueError("blind package must be pending human review")
    if tuple(hides) != _BLIND_HIDES:
        raise ValueError("blind package hiding policy differs from the frozen protocol")
    selected = tuple(selected_factor_ids)
    if len(set(selected)) != len(selected) or not selected:
        raise ValueError(
            "blind package selected factor IDs must be unique and nonempty"
        )
    if (
        type(record.get("reviewers_minimum")) is not int
        or record["reviewers_minimum"] < 2
        or record.get("mechanism_restatement_required") is not True
        or record.get("inter_reviewer_agreement_reported") is not True
    ):
        raise ValueError("blind package review protocol fields are invalid")
    accuracy = _finite_float(
        record.get("response_direction_accuracy_minimum"),
        "blind response-direction accuracy",
    )
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("blind response-direction accuracy must be in [0, 1]")
    items = record.get("items")
    expected_ids = {f"B{index:03d}" for index in range(1, len(selected) + 1)}
    if not isinstance(items, dict) or set(items) != expected_ids:
        raise ValueError("blind item count or anonymous identifiers are incomplete")
    for blind_id, item in items.items():
        if not isinstance(item, dict) or set(item) != _BLIND_ITEM_FIELDS:
            raise ValueError(f"blind item {blind_id} differs from the frozen schema")
        serialized = json.dumps(item, sort_keys=True).lower()
        if any(term in serialized for term in ("kan_e3", "pnl", "return_metrics")):
            raise ValueError("blind package exposes a hidden method or return field")
    anonymous_mapping = {
        f"B{index:03d}": factor_id
        for index, factor_id in enumerate(sorted(selected), start=1)
    }
    anonymous_mapping_sha256 = _sha256_bytes(_canonical_json_bytes(anonymous_mapping))
    return _stage(
        staging_path,
        files={"blind_review_package.json": _canonical_json_bytes(record)},
        identities=identities,
        role="human_blind_review_input",
        kan_mined=False,
        promotion_eligible=False,
        budget_counts={"blind_items": len(items)},
        extra_manifest={
            "human_review_complete": False,
            "output_kind": "blind_review_package",
            "blind_item_count": len(items),
            "hides": list(_BLIND_HIDES),
            "anonymous_mapping_sha256": anonymous_mapping_sha256,
        },
    )


def _serialize_kan_scoring_and_selection(
    scoring: HardAstScoringRun,
    selection: HardAstSelection,
    *,
    prefix: str,
    minimum_library_size: int,
    library_cap: int,
) -> tuple[dict[str, bytes], dict[str, object], list[str]]:
    if not isinstance(scoring, HardAstScoringRun) or not isinstance(
        selection, HardAstSelection
    ):
        raise TypeError("KAN scoring and selection must use frozen strong types")
    expected_label_mode = "real" if prefix == "kan_real" else "within_date_permutation"
    if (
        scoring.label_mode != expected_label_mode
        or not isinstance(scoring.label_sha256, str)
        or len(scoring.label_sha256) != 64
    ):
        raise ValueError(f"{prefix} scoring label identity is invalid")
    candidates = tuple(scoring.candidates)
    if len(candidates) != 256:
        raise ValueError(f"{prefix} scoring must contain all 256 KAN attempts")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if (
        len(set(candidate_ids)) != 256
        or [candidate.attempt_index for candidate in candidates] != list(range(256))
        or any(candidate.method != "kan_e3" for candidate in candidates)
    ):
        raise ValueError(f"{prefix} scoring candidate identity or order is invalid")
    if set(selection.dispositions) != set(candidate_ids):
        raise ValueError(f"{prefix} selection must disposition every candidate")
    selected_ids = [candidate.candidate_id for candidate in selection.selected]
    _require_library_size(
        len(selected_ids),
        minimum_library_size=minimum_library_size,
        library_cap=library_cap,
        label=f"{prefix} selection size",
    )
    expected_target_size = None if prefix == "kan_real" else len(selected_ids)
    if (
        not selection.complete
        or selection.target_size != expected_target_size
        or len(set(selected_ids)) != len(selected_ids)
        or not set(selected_ids).issubset(candidate_ids)
    ):
        raise ValueError(f"{prefix} selection is not an exact complete library")
    rows = [
        _canonicalize(candidate.to_record(), f"{prefix} score")
        for candidate in candidates
    ]
    metadata = _canonicalize(
        {
            "label_mode": scoring.label_mode,
            "label_sha256": scoring.label_sha256,
            "scoring_run_id": scoring.scoring_run_id,
            "train": scoring.train,
            "validation": scoring.validation,
        },
        f"{prefix} metadata",
    )
    selection_record = _selection_record(selection)
    files = {
        f"{prefix}_scoring.jsonl": b"".join(_canonical_json_line(row) for row in rows),
        f"{prefix}_scoring_metadata.json": _canonical_json_bytes(metadata),
        f"{prefix}_selection.json": _canonical_json_bytes(selection_record),
    }
    return (
        files,
        {
            "scoring_rows": len(rows),
            "selection_dispositions": len(selection.dispositions),
            "selected_count": len(selected_ids),
            "label_mode": scoring.label_mode,
            "label_sha256": scoring.label_sha256,
            "scoring_run_id": scoring.scoring_run_id,
        },
        selected_ids,
    )


def stage_mining_top_bundle(
    staging_path: Path | str,
    published_children: Mapping[str, Path | str],
    *,
    kan_profile_runs: Sequence[ProfileRun],
    kan_scoring: HardAstScoringRun,
    kan_selection: HardAstSelection,
    gp_generation: GpGenerationResult,
    gp_scoring: HardAstScoringRun,
    gp_selection: HardAstSelection,
    permutation_profile_runs: Sequence[ProfileRun],
    permutation_scoring: HardAstScoringRun,
    permutation_null_selection: HardAstSelection,
    identities: Mapping[str, object],
    budget_counts: Mapping[str, int],
    minimum_library_size: int,
    library_cap: int,
) -> StagedBundle:
    """Stage top evidence from four KAN runs and already-published frozen children."""
    if set(budget_counts) != _TOP_BUDGET_KEYS:
        raise ValueError("top mining budget counts are incomplete")
    _require_library_size(
        budget_counts["mlp_controls"],
        minimum_library_size=minimum_library_size,
        library_cap=library_cap,
        label="top mining MLP control count",
    )
    if (
        budget_counts["kan_attempts"] != 256
        or budget_counts["kan_updates"] != 256 * TRAINING_STEPS
        or budget_counts["gp_attempts"] != 256
        or budget_counts["permutation_attempts"] != 256
        or budget_counts["mlp_updates"]
        != budget_counts["mlp_controls"] * TRAINING_STEPS
    ):
        raise ValueError("top mining budget counts differ from the frozen protocol")
    runs = tuple(kan_profile_runs)
    if len(runs) != len(PROFILE_SPECS) or [run.profile for run in runs] != list(
        PROFILE_SPECS
    ):
        raise ValueError("top mining bundle requires all four KAN profiles in order")
    permutation_runs = tuple(permutation_profile_runs)
    if len(permutation_runs) != len(PROFILE_SPECS) or [
        run.profile for run in permutation_runs
    ] != list(PROFILE_SPECS):
        raise ValueError("top mining bundle requires all four permutation KAN profiles")
    if any(
        real is permuted for real, permuted in zip(runs, permutation_runs, strict=True)
    ):
        raise ValueError("permutation KAN runs cannot alias the real-label run objects")
    if set(published_children) != _TOP_CHILDREN:
        raise ValueError("top mining bundle requires every frozen published child")
    canonical_identities = _validate_identities(identities)
    child_paths: dict[str, Path] = {}
    for key, raw_path in published_children.items():
        raw = Path(raw_path)
        path = raw.resolve(strict=True)
        if not path.is_dir() or raw.is_symlink():
            raise ValueError(f"published child {key} must be a real directory")
        child_paths[key] = path
    if len(set(child_paths.values())) != len(child_paths):
        raise ValueError("published child topology contains a path alias")

    child_hashes: dict[str, str] = {}
    child_rows: list[dict[str, object]] = []
    topology_sha256: str | None = None
    selected_count: int | None = None
    kan_selected_lineage: dict[str, dict[str, object]] | None = None
    gp_library_factors: dict[str, object] | None = None
    permutation_library_factors: dict[str, object] | None = None
    mechanism_selected_ids: list[str] | None = None
    mechanism_mapping_sha256: str | None = None
    blind_mapping_sha256: str | None = None
    blackbox_selected_ids: list[str] | None = None
    blackbox_global_indices: list[int] | None = None
    for role in sorted(child_paths):
        child_path = child_paths[role]
        manifest_path = child_path / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"published child {role} lacks a regular final manifest")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"published child {role} manifest is invalid") from error
        if not isinstance(manifest, dict):
            raise ValueError(f"published child {role} manifest must be a record")
        if manifest.get("identities") != canonical_identities:
            raise ValueError("child manifest identities differ from the top bundle")
        if manifest.get("topology_key") != role:
            raise ValueError(f"published child {role} has the wrong topology key")
        observed_topology = manifest.get("topology_sha256")
        if not isinstance(observed_topology, str) or len(observed_topology) != 64:
            raise ValueError(f"published child {role} lacks a topology binding")
        if topology_sha256 is None:
            topology_sha256 = observed_topology
        elif observed_topology != topology_sha256:
            raise ValueError("published children belong to different topology bindings")
        expectation = _CHILD_EXPECTATIONS[role]
        count_field = str(expectation["count_field"])
        for field, expected_value in expectation.items():
            if field == "count_field":
                continue
            if manifest.get(field) != expected_value:
                raise ValueError(f"published child {role} has invalid {field}")
        count = manifest.get(count_field)
        _require_library_size(
            count,
            minimum_library_size=minimum_library_size,
            library_cap=library_cap,
            label=f"published child {role} {count_field}",
        )
        if selected_count is None:
            selected_count = count
        elif count != selected_count:
            raise ValueError("published child output counts are not size matched")
        if role in {
            "kan_library",
            "gp_control_library",
            "permutation_control_library",
        }:
            factors = manifest.get("factors")
            if not isinstance(factors, dict) or len(factors) != count:
                raise ValueError(
                    f"published child {role} factor inventory is incomplete"
                )
            if role == "kan_library":
                kan_selected_lineage = {}
                for factor_id in sorted(factors):
                    factor = factors[factor_id]
                    if not isinstance(factor, dict):
                        raise ValueError("KAN factor lineage must be a record")
                    canonical_hash = factor.get("canonical_hash")
                    global_index = factor.get("global_attempt_index")
                    if (
                        not isinstance(canonical_hash, str)
                        or len(canonical_hash) != 64
                        or type(global_index) is not int
                        or not 0 <= global_index < 256
                    ):
                        raise ValueError("KAN factor selected lineage is incomplete")
                    kan_selected_lineage[factor_id] = {
                        "canonical_hash": canonical_hash,
                        "global_attempt_index": global_index,
                    }
                if (
                    len(
                        {
                            row["global_attempt_index"]
                            for row in kan_selected_lineage.values()
                        }
                    )
                    != count
                ):
                    raise ValueError("KAN selected lineage reuses a global attempt")
            elif role == "gp_control_library":
                gp_library_factors = factors
            else:
                permutation_library_factors = factors
        elif role == "blackbox_control":
            blackbox_selected_ids = manifest.get("selected_kan_factor_ids")
            blackbox_global_indices = manifest.get("paired_kan_global_attempt_indices")
            if (
                not isinstance(blackbox_selected_ids, list)
                or not isinstance(blackbox_global_indices, list)
                or len(blackbox_selected_ids) != count
                or len(blackbox_global_indices) != count
                or len(set(blackbox_selected_ids)) != count
                or len(set(blackbox_global_indices)) != count
            ):
                raise ValueError("blackbox control KAN pairing inventory is incomplete")
        elif role == "mechanism_cards":
            mechanism_selected_ids = manifest.get("selected_factor_ids")
            mechanism_mapping_sha256 = manifest.get("anonymous_mapping_sha256")
            if (
                not isinstance(mechanism_selected_ids, list)
                or len(mechanism_selected_ids) != count
                or len(set(mechanism_selected_ids)) != count
            ):
                raise ValueError("mechanism cards selected factor IDs are incomplete")
        elif role == "blind_review_package":
            blind_mapping_sha256 = manifest.get("anonymous_mapping_sha256")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError(f"published child {role} has no file inventory")
        expected_names = set(files) | {"manifest.json"}
        entries = tuple(child_path.iterdir())
        if any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError(f"published child {role} is not flat regular files")
        if {entry.name for entry in entries} != expected_names:
            raise ValueError(f"published child {role} file inventory is incomplete")
        for filename, expected_hash in files.items():
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or sha256_file(child_path / filename) != expected_hash
            ):
                raise ValueError(f"published child {role} file hash is invalid")
        manifest_hash = sha256_file(manifest_path)
        child_hashes[role] = manifest_hash
        child_rows.append(
            {
                "topology_role": role,
                "manifest_sha256": manifest_hash,
                "artifact_role": manifest["role"],
                "kan_mined": manifest.get("kan_mined", False),
                "promotion_eligible": manifest["promotion_eligible"],
                "budget_counts": manifest.get("budget_counts"),
                "output_count": count,
            }
        )
    if selected_count != budget_counts["mlp_controls"]:
        raise ValueError("published child count differs from the frozen top budget")

    archive = _TensorEvidence()
    kan_rows: list[dict[str, object]] = []
    for run in runs:
        kan_rows.extend(_append_kan_profile_run(run, archive, namespace="kan_real"))
    if len(kan_rows) != 256:
        raise RuntimeError("top bundle KAN evidence did not consume all 256 attempts")
    permutation_rows: list[dict[str, object]] = []
    for run in permutation_runs:
        permutation_rows.extend(
            _append_kan_profile_run(run, archive, namespace="kan_permutation")
        )
    if len(permutation_rows) != 256:
        raise RuntimeError(
            "top bundle permutation evidence did not consume 256 attempts"
        )
    real_files, real_ledger, real_selected_ids = _serialize_kan_scoring_and_selection(
        kan_scoring,
        kan_selection,
        prefix="kan_real",
        minimum_library_size=minimum_library_size,
        library_cap=library_cap,
    )
    permutation_files, permutation_ledger, permutation_selected_ids = (
        _serialize_kan_scoring_and_selection(
            permutation_scoring,
            permutation_null_selection,
            prefix="kan_permutation",
            minimum_library_size=minimum_library_size,
            library_cap=library_cap,
        )
    )
    for candidate, run_row in zip(kan_scoring.candidates, kan_rows, strict=True):
        if (
            candidate.profile != run_row["profile"]
            or candidate.canonical_hash != run_row["candidate_ast_sha256"]
        ):
            raise ValueError("KAN real scoring differs from its hardened run evidence")
    for candidate, run_row in zip(
        permutation_scoring.candidates, permutation_rows, strict=True
    ):
        if (
            candidate.profile != run_row["profile"]
            or candidate.canonical_hash != run_row["candidate_ast_sha256"]
        ):
            raise ValueError(
                "permutation scoring differs from its hardened rerun evidence"
            )
    gp_files, gp_ledger = _serialize_gp_ledgers(gp_generation, gp_scoring, gp_selection)
    false_positive_rows = [
        {
            "candidate_id": candidate.candidate_id,
            "global_attempt_index": candidate.attempt_index,
            "production_eligible": candidate.production_eligible,
            "production_disposition": candidate.production_disposition,
        }
        for candidate in permutation_scoring.candidates
    ]
    false_positive_count = sum(
        bool(row["production_eligible"]) for row in false_positive_rows
    )
    permutation_files["kan_permutation_false_positive_ledger.jsonl"] = b"".join(
        _canonical_json_line(row) for row in false_positive_rows
    )
    permutation_ledger["real_threshold_false_positive_count"] = false_positive_count
    permutation_ledger["real_threshold_false_positive_rows"] = len(false_positive_rows)
    if kan_selected_lineage is None:
        raise RuntimeError("top bundle lacks KAN selected lineage")
    selected_ids = sorted(kan_selected_lineage)
    if sorted(real_selected_ids) != selected_ids:
        raise ValueError("KAN production selection differs from the published library")
    if sorted(gp_library_factors or {}) != sorted(
        candidate.candidate_id for candidate in gp_selection.selected
    ):
        raise ValueError("GP selection differs from the published control library")
    if sorted(permutation_library_factors or {}) != sorted(permutation_selected_ids):
        raise ValueError(
            "permutation null selection differs from the published control library"
        )
    for candidate in gp_selection.selected:
        factor = (gp_library_factors or {}).get(candidate.candidate_id)
        if (
            not isinstance(factor, dict)
            or factor.get("canonical_hash") != candidate.canonical_hash
        ):
            raise ValueError("GP published AST differs from its selected scoring row")
    permutation_by_id = {
        candidate.candidate_id: candidate
        for candidate in permutation_scoring.candidates
    }
    for candidate_id in permutation_selected_ids:
        factor = (permutation_library_factors or {}).get(candidate_id)
        if (
            not isinstance(factor, dict)
            or factor.get("canonical_hash")
            != permutation_by_id[candidate_id].canonical_hash
        ):
            raise ValueError(
                "permutation published AST differs from its null selection row"
            )
    selected_global_indices = [
        int(kan_selected_lineage[factor_id]["global_attempt_index"])
        for factor_id in selected_ids
    ]
    if sorted(mechanism_selected_ids or []) != selected_ids:
        raise ValueError("mechanism cards do not match the selected KAN factors")
    if (
        not isinstance(mechanism_mapping_sha256, str)
        or len(mechanism_mapping_sha256) != 64
        or blind_mapping_sha256 != mechanism_mapping_sha256
    ):
        raise ValueError("blind package anonymous mapping differs from mechanism cards")
    blackbox_pairing = dict(
        zip(
            blackbox_selected_ids or [],
            blackbox_global_indices or [],
            strict=True,
        )
    )
    selected_pairing = dict(zip(selected_ids, selected_global_indices, strict=True))
    if blackbox_pairing != selected_pairing:
        raise ValueError("blackbox controls do not match the selected KAN lineage")
    candidate_hash_by_global = {
        int(row["global_attempt_index"]): row["candidate_ast_sha256"]
        for row in kan_rows
    }
    for factor_id, lineage in kan_selected_lineage.items():
        scoring_candidate = next(
            candidate
            for candidate in kan_selection.selected
            if candidate.candidate_id == factor_id
        )
        if (
            candidate_hash_by_global[int(lineage["global_attempt_index"])]
            != lineage["canonical_hash"]
            or scoring_candidate.attempt_index != lineage["global_attempt_index"]
            or scoring_candidate.canonical_hash != lineage["canonical_hash"]
        ):
            raise ValueError(
                f"selected KAN factor {factor_id} differs from mining lineage"
            )
    return _stage(
        staging_path,
        files={
            "children.json": _canonical_json_bytes(child_rows),
            "kan_profile_runs.jsonl": b"".join(
                _canonical_json_line(row) for row in kan_rows
            ),
            "kan_permutation_profile_runs.jsonl": b"".join(
                _canonical_json_line(row) for row in permutation_rows
            ),
            "kan_tensor_evidence.zlib": archive.compressed(),
            **real_files,
            **gp_files,
            **permutation_files,
        },
        identities=identities,
        role="mining_top_bundle",
        kan_mined=True,
        promotion_eligible=False,
        budget_counts=budget_counts,
        extra_manifest={
            "child_manifest_sha256": child_hashes,
            "child_manifests": child_hashes,
            "top_bundle_must_publish_after_all_children": True,
            "published_child_topology_sha256": topology_sha256,
            "published_child_paths": {
                key: str(child_paths[key]) for key in sorted(child_paths)
            },
            "kan_selected_lineage": kan_selected_lineage,
            "mechanism_cards_manifest_sha256": child_hashes["mechanism_cards"],
            "kan_profile_evidence": {
                "profiles": len(runs),
                "miners": len(kan_rows),
                "completed_updates": len(kan_rows) * TRAINING_STEPS,
            },
            "permutation_profile_evidence": {
                "profiles": len(permutation_runs),
                "miners": len(permutation_rows),
                "completed_updates": len(permutation_rows) * TRAINING_STEPS,
            },
            "kan_real_ledger": real_ledger,
            "gp_ledger": gp_ledger,
            "permutation_ledger": permutation_ledger,
            "tensor_evidence": {
                "encoding": "zlib_level_9_of_concatenated_little_endian_tensors",
                "references": [
                    "kan_profile_runs.jsonl",
                    "kan_permutation_profile_runs.jsonl",
                ],
            },
        },
    )


__all__ = [
    "StagedBundle",
    "stage_blind_review_package",
    "stage_mechanism_cards",
    "stage_mining_top_bundle",
    "stage_mlp_control_panel",
    "stage_v2_factor_library",
]
