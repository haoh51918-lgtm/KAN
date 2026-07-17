"""No-replace orchestration and provenance for the frozen S2a mining screen."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from mirage_kan.artifacts.library import (
    claim_artifact_directory,
    finalize_claimed_directory,
    publish_library,
    terminalize_claimed_directory,
)
from mirage_kan.data import PitPanel
from mirage_kan.data.pit import RAW_FIELDS, sha256_file
from mirage_kan.identities import (
    regular_file_tree_identity,
    source_tree_identity,
)
from mirage_kan.mining.core import (
    CandidateScore,
    MiningAttempt,
    ScoringRun,
    SelectionResult,
    generate_attempts,
    greedy_select,
    permute_labels_within_date,
    score_attempts,
    select_random_control,
)

LIBRARY_ROLES = (
    "heterogeneous_selected",
    "random_typed",
    "label_permutation_selected",
)
IMPLEMENTATION_LOCK = "prereg/s2_plan_c_vertical_v1_implementation.lock.json"
MINING_ENTITLEMENT = "governance/openings/s2_plan_c_vertical_v1_mining.json"
MINING_PRECLAIM = "governance/openings/s2_plan_c_vertical_v1_mining_preclaim.json"
PREREGISTRATION_LOCK_SHA256 = (
    "7d1345713b49a9e146a733444cb7cf92758c441b41436984f2c17afbd3496b3d"
)


def _contained_path(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"S2 {label} path must be workspace-relative")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"S2 {label} path escapes the workspace") from error
    return resolved


def _publication_targets(
    workspace: Path | str,
    destination: Path | str,
    library_destinations: Mapping[str, Path | str],
    *,
    allow_claimed_run: bool,
) -> tuple[Path, dict[str, Path]]:
    if set(library_destinations) != set(LIBRARY_ROLES):
        raise ValueError(f"library destinations must be exactly {LIBRARY_ROLES}")
    root = Path(workspace).resolve(strict=True)
    artifacts_root = (root / "artifacts").resolve(strict=True)
    libraries_root = (root / "factor_libraries").resolve(strict=True)
    run_raw = Path(destination)
    run_path = run_raw.parent.resolve(strict=True) / run_raw.name
    libraries = {
        role: Path(library_destinations[role]).parent.resolve(strict=True)
        / Path(library_destinations[role]).name
        for role in LIBRARY_ROLES
    }
    if run_path.parent != artifacts_root:
        raise ValueError("S2a mining run must be a direct child of artifacts/")
    if any(path.parent != libraries_root for path in libraries.values()):
        raise ValueError("S2a libraries must be direct children of factor_libraries/")
    targets = [run_path, *libraries.values()]
    if len(set(targets)) != len(targets):
        raise ValueError("S2a publication destinations must be distinct and non-aliased")
    if allow_claimed_run:
        if not (run_path / ".INCOMPLETE").is_file():
            raise ValueError("S2a mining run was not prospectively claimed")
    elif run_path.exists() or run_path.is_symlink():
        raise FileExistsError(f"refusing to replace S2a artifact: {run_path}")
    for path in libraries.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace S2a artifact: {path}")
    return run_path, libraries


def claim_s2a_mining_run(
    workspace: Path | str,
    destination: Path | str,
    library_destinations: Mapping[str, Path | str],
) -> Path:
    """Consume the mining run path before any label or candidate access."""
    run_path, _ = _publication_targets(
        workspace,
        destination,
        library_destinations,
        allow_claimed_run=False,
    )
    return claim_artifact_directory(run_path)


def claim_s2a_mining_preclaim(
    workspace: Path | str,
    destination: Path | str,
    library_destinations: Mapping[str, Path | str],
) -> dict[str, object]:
    """Register the only recoverable run topology before claiming any directory."""
    root = Path(workspace).resolve(strict=True)
    run_path, libraries = _publication_targets(
        root,
        destination,
        library_destinations,
        allow_claimed_run=False,
    )
    path = root / MINING_PRECLAIM
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mirage_s2a_mining_preclaim_v1",
        "protocol_id": "s2_plan_c_vertical_v1",
        "state": "consumed_before_run_claim",
        "run_path": str(run_path),
        "library_destinations": {
            role: str(libraries[role]) for role in LIBRARY_ROLES
        },
        "attempt_budget": 256,
    }
    _write_exclusive(path, _canonical_bytes(payload))
    return {"path": str(path), "sha256": sha256_file(path), **payload}


def claim_s2a_mining_entitlement(
    workspace: Path | str,
    run_path: Path,
    library_destinations: Mapping[str, Path | str],
    identities: Mapping[str, object],
) -> dict[str, object]:
    """Consume the protocol-global 256-attempt entitlement exactly once."""
    root = Path(workspace).resolve(strict=True)
    path = root / MINING_ENTITLEMENT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mirage_s2a_mining_entitlement_v1",
        "protocol_id": "s2_plan_c_vertical_v1",
        "state": "consumed_before_attempt_generation",
        "run_path": str(run_path.resolve(strict=True)),
        "library_destinations": {
            role: str(Path(library_destinations[role]).resolve())
            for role in LIBRARY_ROLES
        },
        "implementation_lock_sha256": identities["implementation_lock"]["sha256"],
        "preregistration_lock_sha256": identities["preregistration_lock"]["sha256"],
        "attempt_budget": 256,
        "preclaim_sha256": identities["mining_preclaim"]["sha256"],
    }
    _write_exclusive(path, _canonical_bytes(payload))
    return {"path": str(path), "sha256": sha256_file(path), **payload}


def _finalize_premining_failure(
    run_path: Path,
    error: BaseException,
    attempts: Sequence[MiningAttempt],
    identities: Mapping[str, object],
) -> None:
    if not (run_path / ".INCOMPLETE").is_file():
        return
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{run_path.name}.", suffix=".failure", dir=run_path.parent
        )
    )
    _write_jsonl(staging / "attempts.jsonl", [attempt.to_record() for attempt in attempts])
    manifest = {
        "schema_version": "mirage_s2a_mining_run_v1",
        "stage": "S2a",
        "publication_state": "terminal_failure",
        "scientific_result": False,
        "final_claim_allowed": False,
        "kan_mined": False,
        "attempt_count": len(attempts),
        "actual_evaluated_attempt_count": 0,
        "error_type": type(error).__name__,
        "error": str(error),
        "identities": dict(identities),
        "files": {"attempts.jsonl": sha256_file(staging / "attempts.jsonl")},
    }
    _write_exclusive(staging / "manifest.json", _canonical_bytes(manifest))
    finalize_claimed_directory(staging, run_path, required_manifest="manifest.json")


def _verify_implementation_lock(
    root: Path, provider_receipt: Mapping[str, object] | None = None
) -> tuple[dict[str, object], str]:
    path = root / IMPLEMENTATION_LOCK
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mirage_s2_implementation_lock_v1":
        raise ValueError("unsupported S2 implementation-lock schema")
    if payload.get("protocol_id") != "s2_plan_c_vertical_v1":
        raise ValueError("S2 implementation lock has the wrong protocol")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("S2 implementation lock has no bound files")
    for relative, expected in files.items():
        file_path = _contained_path(root, relative, label="implementation")
        if file_path == path:
            raise ValueError("S2 implementation lock cannot hash itself")
        if sha256_file(file_path) != expected:
            raise ValueError(f"S2 implementation file hash mismatch: {relative}")
    source = source_tree_identity(root / "src" / "mirage_kan")
    if source != payload.get("source_tree"):
        raise ValueError("S2 implementation source-tree identity mismatch")
    provider = payload.get("qlib_provider")
    if not isinstance(provider, dict):
        raise ValueError("S2 implementation lock lacks qlib_provider")
    observed_provider = regular_file_tree_identity(provider.get("path"))
    if provider_receipt is not None:
        for key in (
            "path",
            "tree_sha256",
            "stat_inventory_sha256",
            "file_count",
            "total_bytes",
        ):
            if observed_provider[key] != provider_receipt.get(key):
                raise ValueError(f"S2 Qlib provider changed in-process: {key}")
    for key in (
        "path",
        "tree_sha256",
        "stat_inventory_sha256",
        "file_count",
        "total_bytes",
    ):
        if observed_provider[key] != provider.get(key):
            raise ValueError(f"S2 locked Qlib provider mismatch: {key}")
    return payload, sha256_file(path)


def _canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _normalize_yaml_dates(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _normalize_yaml_dates(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_yaml_dates(item) for item in value]
    return value


def _write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    data = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for record in records
    )
    _write_exclusive(path, data)


def _selection_provenance(
    role: str,
    candidates: Sequence[CandidateScore],
    scoring: ScoringRun | None,
    selection_config: Mapping[str, object],
) -> dict[str, object]:
    label_free = scoring is None
    return {
        "role": role,
        "label_free": label_free,
        "label_sha256": None if label_free else scoring.label_sha256,
        "label_mode": None if label_free else scoring.label_mode,
        "scoring_run_id": None if label_free else scoring.scoring_run_id,
        "candidate_ids_in_selection_order": [
            candidate.candidate_id for candidate in candidates
        ],
        "canonical_hashes_in_selection_order": [
            candidate.canonical_hash for candidate in candidates
        ],
        "selection_config": dict(selection_config),
        "random_control_semantics": (
            "unique typed causal finite-support candidates; membership coverage only; "
            "no label values or label-derived metrics"
            if label_free
            else None
        ),
    }


def _candidate_records(
    scoring: ScoringRun, selection: SelectionResult
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    selection_order = {
        candidate.candidate_id: order
        for order, candidate in enumerate(selection.selected, start=1)
    }
    for candidate in scoring.candidates:
        record = candidate.to_record()
        record["scoring_disposition"] = candidate.disposition
        record["selection_disposition"] = selection.by_candidate[candidate.candidate_id]
        record["selection_order"] = selection_order.get(candidate.candidate_id)
        record["label_mode"] = scoring.label_mode
        record["scoring_run_id"] = scoring.scoring_run_id
        records.append(record)
    return records


def verify_s2a_scoring_ledgers(
    run_path: Path | str,
    protocol: Mapping[str, Any],
    identities: Mapping[str, Any],
) -> dict[str, object]:
    """Recompute all 256 scores, selections, and the label-free control."""
    root = Path(run_path).resolve(strict=True)
    search = protocol["search"]
    admission = protocol["admission"]
    attempts = generate_attempts(
        seed=int(search["generation_seed"]),
        attempts_per_profile=int(search["attempts_per_profile"]),
        profiles=tuple(search["miner_profiles"]),
    )
    recorded_attempts = [
        json.loads(line)
        for line in (root / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(recorded_attempts) != len(attempts):
        raise ValueError("S2a attempt ledger is not independently reproducible")
    for attempt, recorded in zip(attempts, recorded_attempts, strict=True):
        expected = attempt.to_record()
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError("S2a attempt AST ledger differs from frozen generation")

    cache_path = Path(identities["data"]["cache_path"])
    panel, labels = _load_screening_data(
        cache_path,
        identities["data"]["cache_sha256"],
        protocol["data"]["validation"][1],
    )
    score_kwargs = {
        "train": tuple(protocol["data"]["train"]),
        "validation": tuple(protocol["data"]["validation"]),
        "minimum_coverage": float(admission["minimum_coverage"]),
        "minimum_absolute_rank_ic": float(
            admission["minimum_absolute_train_rank_ic"]
        ),
    }

    observed = score_attempts(
        attempts, panel, labels, label_mode="observed", **score_kwargs
    )
    observed_selection = greedy_select(
        observed.candidates,
        library_cap=int(admission["library_cap"]),
        minimum_library_size=int(admission["minimum_library_size"]),
        minimum_profiles=int(admission["minimum_miner_profiles"]),
        maximum_absolute_spearman=float(
            admission["maximum_absolute_validation_spearman"]
        ),
    )
    expected_observed = _candidate_records(observed, observed_selection)
    observed_selected = [item.candidate_id for item in observed_selection.selected]
    observed_run_id = observed.scoring_run_id
    observed_label_sha256 = observed.label_sha256
    del observed, observed_selection

    permuted = permute_labels_within_date(labels, seed=int(search["permutation_seed"]))
    permutation = score_attempts(
        attempts,
        panel,
        permuted,
        label_mode="within_date_permutation",
        **score_kwargs,
    )
    permutation_selection = greedy_select(
        permutation.candidates,
        library_cap=int(admission["library_cap"]),
        minimum_library_size=int(admission["minimum_library_size"]),
        minimum_profiles=int(admission["minimum_miner_profiles"]),
        maximum_absolute_spearman=float(
            admission["maximum_absolute_validation_spearman"]
        ),
    )
    expected_permutation = _candidate_records(permutation, permutation_selection)
    permutation_selected = [
        item.candidate_id for item in permutation_selection.selected
    ]
    permutation_run_id = permutation.scoring_run_id
    permutation_label_sha256 = permutation.label_sha256
    del permutation, permutation_selection

    random_selection = select_random_control(
        attempts,
        panel,
        seed=int(search["random_control_seed"]),
        library_cap=int(admission["library_cap"]),
        minimum_coverage=float(admission["minimum_coverage"]),
        period=(protocol["data"]["train"][0], protocol["data"]["validation"][1]),
    )
    expected_random = [candidate.to_record() for candidate in random_selection]
    expected_tables = {
        "candidate_table_observed.jsonl": expected_observed,
        "candidate_table_permutation.jsonl": expected_permutation,
        "random_control.jsonl": expected_random,
    }
    for filename, expected in expected_tables.items():
        recorded = [
            json.loads(line)
            for line in (root / filename).read_text(encoding="utf-8").splitlines()
        ]
        if recorded != expected:
            raise ValueError(f"S2a scoring ledger is not recomputable: {filename}")
    return {
        "verified": True,
        "observed_scoring_run_id": observed_run_id,
        "observed_label_sha256": observed_label_sha256,
        "permutation_scoring_run_id": permutation_run_id,
        "permutation_label_sha256": permutation_label_sha256,
        "selected_candidate_ids": {
            "heterogeneous_selected": observed_selected,
            "random_typed": [item.candidate_id for item in random_selection],
            "label_permutation_selected": permutation_selected,
        },
    }


def publish_s2a_run(
    destination: Path | str,
    library_destinations: Mapping[str, Path | str],
    publication_panel: PitPanel,
    *,
    attempts: Sequence[MiningAttempt],
    observed: ScoringRun,
    permutation: ScoringRun,
    observed_selection: SelectionResult,
    permutation_selection: SelectionResult,
    random_selection: Sequence[CandidateScore],
    identities: Mapping[str, object],
    selection_config: Mapping[str, object],
    data_access_records: Sequence[Mapping[str, object]] = (),
    workspace: Path | str,
    run_preclaimed: bool = False,
    random_control_seed: int | None = None,
    random_control_period: tuple[str, str] | None = None,
) -> dict[str, object]:
    """Transactionally publish all mining evidence, including terminal failures."""
    run_path, resolved_libraries = _publication_targets(
        workspace,
        destination,
        library_destinations,
        allow_claimed_run=run_preclaimed,
    )
    if not run_preclaimed:
        run_path = claim_artifact_directory(run_path)
    selections = {
        "heterogeneous_selected": (observed_selection.selected, observed),
        "random_typed": (tuple(random_selection), None),
        "label_permutation_selected": (
            permutation_selection.selected,
            permutation,
        ),
    }
    parent = run_path.parent.resolve(strict=True)
    staging: Path | None = None
    library_manifests: dict[str, dict[str, object]] = {}
    try:
        for role in LIBRARY_ROLES:
            claim_artifact_directory(resolved_libraries[role])
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{run_path.name}.", suffix=".staging", dir=parent
            )
        )
        attempt_records = []
        observed_by_id = {score.candidate_id: score for score in observed.candidates}
        permutation_by_id = {
            score.candidate_id: score for score in permutation.candidates
        }
        for attempt in attempts:
            record = attempt.to_record()
            observed_score = observed_by_id.get(attempt.candidate_id)
            permutation_score = permutation_by_id.get(attempt.candidate_id)
            record["observed_scoring_disposition"] = (
                observed_score.disposition if observed_score is not None else None
            )
            record["permutation_scoring_disposition"] = (
                permutation_score.disposition if permutation_score is not None else None
            )
            attempt_records.append(record)
        _write_jsonl(staging / "attempts.jsonl", attempt_records)
        _write_jsonl(
            staging / "candidate_table_observed.jsonl",
            _candidate_records(observed, observed_selection),
        )
        _write_jsonl(
            staging / "candidate_table_permutation.jsonl",
            _candidate_records(permutation, permutation_selection),
        )
        _write_jsonl(
            staging / "random_control.jsonl",
            [candidate.to_record() for candidate in random_selection],
        )
        _write_jsonl(
            staging / "data_access_ledger.jsonl",
            list(data_access_records)
            or [
                {
                    "access": "synthetic_or_injected_publication_panel",
                    "label_access": False,
                    "rows": len(publication_panel.raw),
                }
            ],
        )
        attempt_ids = [attempt.candidate_id for attempt in attempts]
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("S2a attempt IDs are not unique")
        if set(attempt_ids) != set(observed_by_id) or set(attempt_ids) != set(
            permutation_by_id
        ):
            raise ValueError("S2a attempt and candidate ledgers are not aligned")
        if int(selection_config["library_cap"]) == 16 and len(attempts) != 256:
            raise ValueError("formal S2a publication requires exactly 256 attempts")
        recomputed_observed = greedy_select(
            observed.candidates,
            library_cap=int(selection_config["library_cap"]),
            minimum_library_size=int(selection_config["minimum_library_size"]),
            minimum_profiles=int(selection_config["minimum_miner_profiles"]),
            maximum_absolute_spearman=float(
                selection_config["maximum_absolute_validation_spearman"]
            ),
        )
        recomputed_permutation = greedy_select(
            permutation.candidates,
            library_cap=int(selection_config["library_cap"]),
            minimum_library_size=int(selection_config["minimum_library_size"]),
            minimum_profiles=int(selection_config["minimum_miner_profiles"]),
            maximum_absolute_spearman=float(
                selection_config["maximum_absolute_validation_spearman"]
            ),
        )
        for recorded, recomputed, label in (
            (observed_selection, recomputed_observed, "observed"),
            (permutation_selection, recomputed_permutation, "permutation"),
        ):
            if (
                [item.candidate_id for item in recorded.selected]
                != [item.candidate_id for item in recomputed.selected]
                or recorded.by_candidate != recomputed.by_candidate
            ):
                raise ValueError(f"S2a {label} selection is not recomputable")
            if not recomputed.minimum_size_met or not recomputed.profile_quota_met:
                raise ValueError(f"S2a {label} admission is below the frozen minimum")
        if len(random_selection) != int(selection_config["library_cap"]):
            raise ValueError("S2a random control must exactly match the library cap")
        if any(
            candidate.train_rank_ic is not None
            or candidate.validation_rank_ic is not None
            or candidate.minimum_score is not None
            or candidate.disposition != "random_label_free_eligible"
            for candidate in random_selection
        ):
            raise ValueError("S2a random control is not label-free by provenance")
        if random_control_seed is not None:
            recomputed_random = select_random_control(
                attempts,
                publication_panel,
                seed=random_control_seed,
                library_cap=int(selection_config["library_cap"]),
                minimum_coverage=float(selection_config["minimum_coverage"]),
                period=random_control_period,
            )
            if [item.candidate_id for item in random_selection] != [
                item.candidate_id for item in recomputed_random
            ]:
                raise ValueError("S2a random control membership is not recomputable")
        if any(not candidates for candidates, _ in selections.values()):
            raise ValueError("S2a publication cannot create an empty factor library")

        for role in LIBRARY_ROLES:
            candidates, scoring = selections[role]
            library_identities = dict(identities)
            provenance_config = dict(selection_config)
            if role == "random_typed":
                provenance_config.update(
                    {
                        "random_control_seed": random_control_seed,
                        "random_control_period": random_control_period,
                    }
                )
            library_identities["selection_provenance"] = _selection_provenance(
                role, candidates, scoring, provenance_config
            )
            programs = {
                candidate.candidate_id: candidate.program
                for candidate in candidates
                if candidate.program is not None
            }
            if len(programs) != len(candidates):
                raise ValueError(f"S2a {role} selection contains an invalid program")
            library_manifests[role] = publish_library(
                resolved_libraries[role],
                programs,
                publication_panel,
                identities=library_identities,
                library_role=role,
                kan_mined=False,
                preclaimed=True,
            )
        file_hashes = {
            path.name: sha256_file(path) for path in sorted(staging.iterdir())
        }
        manifest = {
            "schema_version": "mirage_s2a_mining_run_v1",
            "stage": "S2a",
            "evidence_class": "prospective_development_screen",
            "scientific_result": False,
            "final_claim_allowed": False,
            "kan_mined": False,
            "publication_state": "complete",
            "identities": dict(identities),
            "attempt_count": len(attempts),
            "actual_evaluated_attempt_count": sum(
                attempt.program is not None for attempt in attempts
            ),
            "observed_scoring_run_id": observed.scoring_run_id,
            "permutation_scoring_run_id": permutation.scoring_run_id,
            "selection_config": dict(selection_config),
            "libraries": {
                role: {
                    "path": str(resolved_libraries[role].resolve()),
                    "manifest_sha256": sha256_file(
                        resolved_libraries[role] / "manifest.json"
                    ),
                    "factor_count": library_manifests[role]["factor_count"],
                    "role": role,
                    "kan_mined": False,
                }
                for role in LIBRARY_ROLES
            },
            "files": file_hashes,
        }
        _write_exclusive(staging / "manifest.json", _canonical_bytes(manifest))
        for path in staging.iterdir():
            path.chmod(0o444)
        finalize_claimed_directory(
            staging, run_path, required_manifest="manifest.json"
        )
        return manifest
    except BaseException as error:
        library_terminal_states = {}
        for role, path in resolved_libraries.items():
            if path.exists():
                try:
                    library_terminal_states[role] = terminalize_claimed_directory(
                        path,
                        {
                            "schema_version": "mirage_factor_library_terminal_v1",
                            "library_role": role,
                            "scientific_result": False,
                            "kan_mined": False,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        invalidate_published=True,
                    )
                except BaseException as terminal_error:
                    library_terminal_states[role] = {
                        "path": str(path),
                        "state": "terminalization_failed",
                        "error_type": type(terminal_error).__name__,
                        "error": str(terminal_error),
                    }
        if staging is not None and staging.exists() and (staging / "manifest.json").exists():
            terminal = {
                "schema_version": "mirage_s2a_mining_finalization_failure_v1",
                "publication_state": "terminal_failure_incomplete_copy",
                "error_type": type(error).__name__,
                "error": str(error),
                "staging_manifest_sha256": sha256_file(staging / "manifest.json"),
                "formal_promotion_allowed": False,
                "library_terminal_states": library_terminal_states,
            }
            if (run_path / ".INCOMPLETE").is_file():
                try:
                    terminalize_claimed_directory(run_path, terminal)
                except BaseException:
                    pass
        elif staging is not None and staging.exists():
            try:
                failure = {
                    "schema_version": "mirage_s2a_mining_run_v1",
                    "stage": "S2a",
                    "publication_state": "terminal_failure",
                    "scientific_result": False,
                    "final_claim_allowed": False,
                    "kan_mined": False,
                    "attempt_count": len(attempts),
                    "actual_evaluated_attempt_count": sum(
                        attempt.program is not None for attempt in attempts
                    ),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "published_libraries": {
                        role: str(Path(path).resolve())
                        for role, path in resolved_libraries.items()
                        if Path(path).exists()
                    },
                    "identities": dict(identities),
                    "library_terminal_states": library_terminal_states,
                    "files": {
                        path.name: sha256_file(path)
                        for path in sorted(staging.iterdir())
                        if path.is_file()
                    },
                }
                _write_exclusive(
                    staging / "terminal_failure.json", _canonical_bytes(failure)
                )
                failure["files"]["terminal_failure.json"] = sha256_file(
                    staging / "terminal_failure.json"
                )
                _write_exclusive(
                    staging / "manifest.json", _canonical_bytes(failure)
                )
                try:
                    finalize_claimed_directory(
                        staging, run_path, required_manifest="manifest.json"
                    )
                except BaseException:
                    if (run_path / ".INCOMPLETE").is_file():
                        terminalize_claimed_directory(run_path, failure)
            except BaseException as terminal_error:
                error.add_note(
                    "S2a terminal evidence publication also failed: "
                    f"{type(terminal_error).__name__}: {terminal_error}"
                )
        elif (run_path / ".INCOMPLETE").is_file():
            try:
                terminalize_claimed_directory(
                    run_path,
                    {
                        "schema_version": "mirage_s2a_mining_claim_failure_v1",
                        "scientific_result": False,
                        "formal_promotion_allowed": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "library_terminal_states": library_terminal_states,
                    },
                )
            except BaseException:
                pass
        raise


def verified_s2_identities(
    workspace: Path | str,
    *,
    provider_receipt: Mapping[str, object] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every prospective S2 authority before candidate labels are loaded."""
    root = Path(workspace).resolve(strict=True)
    lock_path = root / "prereg" / "s2_plan_c_vertical_v1.lock.json"
    if sha256_file(lock_path) != PREREGISTRATION_LOCK_SHA256:
        raise ValueError("S2 preregistration lock identity mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "proposal": lock["proposal"],
        "protocol": lock["protocol"],
        "preregistration": lock["preregistration"],
    }
    for name, record in checks.items():
        actual = sha256_file(_contained_path(root, record["path"], label=name))
        if actual != record["sha256"]:
            raise ValueError(f"S2 {name} authority hash mismatch")
    data_config_path = _contained_path(
        root, lock["data"]["config_path"], label="data config"
    )
    if sha256_file(data_config_path) != lock["data"]["config_sha256"]:
        raise ValueError("S2 data config hash mismatch")
    quanta_config_path = _contained_path(
        root, lock["quanta"]["pinned_config_path"], label="Quanta config"
    )
    if sha256_file(quanta_config_path) != lock["quanta"]["pinned_config_sha256"]:
        raise ValueError("S2 Quanta config hash mismatch")
    baseline = Path(lock["baseline_metric"]["path"]).resolve(strict=True)
    if sha256_file(baseline) != lock["baseline_metric"]["sha256"]:
        raise ValueError("S2 baseline metric hash mismatch")
    implementation_lock, implementation_lock_sha256 = _verify_implementation_lock(
        root, provider_receipt
    )
    protocol = _normalize_yaml_dates(
        yaml.safe_load(
            _contained_path(root, lock["protocol"]["path"], label="protocol").read_text()
        )
    )
    identities = {
        "proposal": checks["proposal"],
        "config": checks["protocol"],
        "preregistration": checks["preregistration"],
        "preregistration_lock": {
            "path": str(lock_path),
            "sha256": sha256_file(lock_path),
        },
        "data": lock["data"],
        "quanta": lock["quanta"],
        "baseline_metric": lock["baseline_metric"],
        "implementation_lock": {
            "path": str(root / IMPLEMENTATION_LOCK),
            "sha256": implementation_lock_sha256,
            "protocol_id": implementation_lock["protocol_id"],
        },
        "qlib_provider": implementation_lock["qlib_provider"],
        "code": source_tree_identity(root / "src" / "mirage_kan"),
    }
    return protocol, identities


def _load_screening_data(
    cache_path: Path, expected_sha256: str, validation_end: str
) -> tuple[PitPanel, pd.Series]:
    actual = sha256_file(cache_path)
    if actual != expected_sha256:
        raise ValueError("S2 PIT cache hash mismatch")
    frame = pd.read_parquet(
        cache_path,
        columns=["datetime", "instrument", *RAW_FIELDS, "in_universe", "fwd"],
        filters=[("datetime", "<=", pd.Timestamp(validation_end))],
    )
    labels_frame = frame.loc[:, ["datetime", "instrument", "fwd"]].copy()
    labels_frame["datetime"] = pd.to_datetime(labels_frame["datetime"])
    labels = labels_frame.set_index(["datetime", "instrument"])["fwd"].sort_index()
    panel = PitPanel.from_frame(
        frame.drop(columns="fwd"), source_path=cache_path, source_sha256=actual
    )
    return panel, labels.reindex(panel.raw.index).rename("fwd")


def _load_publication_panel(cache_path: Path, expected_sha256: str) -> PitPanel:
    actual = sha256_file(cache_path)
    if actual != expected_sha256:
        raise ValueError("S2 PIT cache hash mismatch before library execution")
    frame = pd.read_parquet(
        cache_path,
        columns=["datetime", "instrument", *RAW_FIELDS, "in_universe"],
    )
    return PitPanel.from_frame(frame, source_path=cache_path, source_sha256=actual)


def run_s2a_mining(
    workspace: Path | str,
    destination: Path | str,
    library_destinations: Mapping[str, Path | str],
) -> dict[str, object]:
    """Claim first, then execute the frozen prospective mining chain."""
    root = Path(workspace).resolve(strict=True)
    entitlement_path = root / MINING_ENTITLEMENT
    preclaim_path = root / MINING_PRECLAIM
    if (
        entitlement_path.exists()
        or entitlement_path.is_symlink()
        or preclaim_path.exists()
        or preclaim_path.is_symlink()
    ):
        raise FileExistsError("S2a protocol-global mining topology is consumed")
    preclaim = claim_s2a_mining_preclaim(root, destination, library_destinations)
    run_path = claim_s2a_mining_run(root, destination, library_destinations)
    identities: dict[str, object] = {}
    attempts: tuple[MiningAttempt, ...] = ()
    try:
        protocol, identities = verified_s2_identities(root)
        identities = dict(identities)
        identities["mining_preclaim"] = preclaim
        identities["mining_entitlement"] = claim_s2a_mining_entitlement(
            root, run_path, library_destinations, identities
        )
        search = protocol["search"]
        admission = protocol["admission"]
        if (
            tuple(search["windows"]) != (2, 3, 5, 10, 20, 40, 60)
            or search["max_ast_depth"] != 6
            or search["max_ast_nodes"] != 20
            or search["full_development_evaluations"] != search["total_attempts"]
        ):
            raise ValueError("frozen S2 typed-search contract is internally inconsistent")
        if admission["minimum_per_profile"] != 1:
            raise ValueError(
                "S2a implementation supports the frozen one-per-profile quota"
            )
        attempts = generate_attempts(
            seed=search["generation_seed"],
            attempts_per_profile=search["attempts_per_profile"],
            profiles=tuple(search["miner_profiles"]),
        )
        if len(attempts) != search["total_attempts"]:
            raise ValueError("generated attempt count differs from frozen S2 budget")
        return _run_claimed_s2a_mining(
            root,
            run_path,
            library_destinations,
            protocol,
            identities,
            attempts,
        )
    except BaseException as error:
        try:
            _finalize_premining_failure(run_path, error, attempts, identities)
        except BaseException as terminal_error:
            error.add_note(
                "S2a premining terminal publication also failed: "
                f"{type(terminal_error).__name__}: {terminal_error}"
            )
        raise


def _run_claimed_s2a_mining(
    root: Path,
    run_path: Path,
    library_destinations: Mapping[str, Path | str],
    protocol: Mapping[str, Any],
    identities: Mapping[str, Any],
    attempts: tuple[MiningAttempt, ...],
) -> dict[str, object]:
    """Run after the no-replace root and all frozen identities are established."""
    cache_path = Path(identities["data"]["cache_path"])
    cache_sha256 = identities["data"]["cache_sha256"]
    panel, labels = _load_screening_data(
        cache_path, cache_sha256, protocol["data"]["validation"][1]
    )
    search = protocol["search"]
    admission = protocol["admission"]
    score_kwargs = {
        "train": tuple(protocol["data"]["train"]),
        "validation": tuple(protocol["data"]["validation"]),
        "minimum_coverage": admission["minimum_coverage"],
        "minimum_absolute_rank_ic": admission["minimum_absolute_train_rank_ic"],
    }
    if (
        admission["minimum_absolute_train_rank_ic"]
        != admission["minimum_absolute_validation_rank_ic"]
    ):
        raise ValueError(
            "current scorer requires equal frozen train/validation IC floors"
        )
    observed = score_attempts(
        attempts, panel, labels, label_mode="observed", **score_kwargs
    )
    observed_selection = greedy_select(
        observed.candidates,
        library_cap=admission["library_cap"],
        minimum_library_size=admission["minimum_library_size"],
        minimum_profiles=admission["minimum_miner_profiles"],
        maximum_absolute_spearman=admission["maximum_absolute_validation_spearman"],
    )
    permuted_labels = permute_labels_within_date(
        labels, seed=protocol["search"]["permutation_seed"]
    )
    permutation = score_attempts(
        attempts,
        panel,
        permuted_labels,
        label_mode="within_date_permutation",
        **score_kwargs,
    )
    permutation_selection = greedy_select(
        permutation.candidates,
        library_cap=admission["library_cap"],
        minimum_library_size=admission["minimum_library_size"],
        minimum_profiles=admission["minimum_miner_profiles"],
        maximum_absolute_spearman=admission["maximum_absolute_validation_spearman"],
    )
    random_selection = select_random_control(
        attempts,
        panel,
        seed=search["random_control_seed"],
        library_cap=admission["library_cap"],
        minimum_coverage=admission["minimum_coverage"],
        period=(protocol["data"]["train"][0], protocol["data"]["validation"][1]),
    )
    publication_panel = _load_publication_panel(cache_path, cache_sha256)
    data_access_records = [
        {
            "access": "candidate_screening",
            "columns": [*RAW_FIELDS, "in_universe", "fwd"],
            "label": "fwd",
            "label_access": True,
            "maximum_datetime": protocol["data"]["validation"][1],
            "development_test_access": False,
            "cache_sha256": cache_sha256,
        },
        {
            "access": "immutable_library_execution",
            "columns": [*RAW_FIELDS, "in_universe"],
            "label": None,
            "label_access": False,
            "development_test_raw_features_access": True,
            "development_test_outcome_access": False,
            "cache_sha256": cache_sha256,
        },
    ]
    return publish_s2a_run(
        run_path,
        library_destinations,
        publication_panel,
        attempts=attempts,
        observed=observed,
        permutation=permutation,
        observed_selection=observed_selection,
        permutation_selection=permutation_selection,
        random_selection=random_selection,
        identities=identities,
        selection_config=admission,
        data_access_records=data_access_records,
        workspace=root,
        run_preclaimed=True,
        random_control_seed=int(search["random_control_seed"]),
        random_control_period=(
            protocol["data"]["train"][0],
            protocol["data"]["validation"][1],
        ),
    )
