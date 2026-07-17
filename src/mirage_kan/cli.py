"""Command-line entry points for the faithful S0 vertical slice."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mirage_kan.artifacts.library import (
    publish_library,
    publish_staging_directory,
    terminalize_claimed_directory,
    verify_library,
)
from mirage_kan.data.pit import PitPanel, sha256_file
from mirage_kan.dsl import AstNode
from mirage_kan.evaluation.quanta import QuantaAdapter
from mirage_kan.evaluation.s2a import (
    ARMS,
    chinese_report,
    decide_s2a,
    replay_anchor_checks,
)
from mirage_kan.identities import source_tree_identity
from mirage_kan.mining import ast_depth, ast_node_count
from mirage_kan.mining.s2a import (
    LIBRARY_ROLES,
    MINING_ENTITLEMENT,
    MINING_PRECLAIM,
    run_s2a_mining,
    verify_s2a_scoring_ledgers,
    verified_s2_identities,
)
from mirage_kan.seed import seed_wiring_programs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_exclusive(path: Path, payload: object) -> None:
    _write_bytes_exclusive(
        path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _write_bytes_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
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
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _write_text_exclusive(path: Path, value: str) -> None:
    _write_bytes_exclusive(path, value.encode("utf-8"))


def _write_text_exclusive_or_exact(path: Path, value: str) -> None:
    expected = value.encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"existing recovery artifact differs: {path}")
        return
    _write_bytes_exclusive(path, expected)


def _direct_child(workspace: Path, path: Path, category: str) -> Path:
    category_root = (workspace / category).resolve(strict=True)
    if path.parent.resolve(strict=True) != category_root:
        raise ValueError(f"artifact must be a direct child of {category}/")
    resolved = category_root / path.name
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"refusing to replace artifact: {resolved}")
    return resolved


def _recorded_direct_child(
    workspace: Path, raw_path: object, category: str
) -> Path:
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"recorded artifact must be an absolute direct child of {category}/")
    path = Path(raw_path)
    category_root = (workspace / category).resolve(strict=True)
    if path.parent.resolve(strict=True) != category_root:
        raise ValueError(f"recorded artifact must be a direct child of {category}/")
    resolved = category_root / path.name
    if resolved.is_symlink():
        raise ValueError(f"recorded artifact cannot be a symlink in {category}/")
    return resolved


def _recovery_child(workspace: Path, path: Path, category: str) -> Path:
    category_root = (workspace / category).resolve(strict=True)
    if path.parent.resolve(strict=True) != category_root:
        raise ValueError(f"recovery artifact must be a direct child of {category}/")
    resolved = category_root / path.name
    if resolved.is_symlink() or (resolved.exists() and not resolved.is_file()):
        raise ValueError(f"invalid existing recovery artifact: {resolved}")
    return resolved


def _safe_manifest_filename(filename: object) -> str:
    if not isinstance(filename, str) or not filename:
        raise ValueError("manifest requires a safe filename segment")
    path = Path(filename)
    if path.name != filename or path.is_absolute() or filename in {".", ".."}:
        raise ValueError(f"manifest has unsafe filename: {filename!r}")
    return filename


_S2_ORCHESTRATOR_TOKEN = object()


def _workspace(args: argparse.Namespace) -> Path:
    return Path(args.workspace).resolve(strict=True)


def _configs(workspace: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _load_json(workspace / "configs" / "data" / "pit_cache.json"),
        _load_json(workspace / "configs" / "evaluation" / "quanta_pinned.json"),
    )


def _load_panel(data_config: dict[str, Any]) -> PitPanel:
    return PitPanel.load(
        data_config["cache_path"], data_config["cache_sha256"]
    )


def _label_parity(cache_path: Path) -> dict[str, object]:
    frame = pd.read_parquet(
        cache_path,
        columns=[
            "instrument",
            "datetime",
            "close",
            "fwd",
            "fwd20",
            "in_universe",
        ],
    ).sort_values(["instrument", "datetime"])
    close = frame.groupby("instrument", sort=False)["close"]
    recomputed = {
        "fwd": close.shift(-2) / close.shift(-1) - 1.0,
        "fwd20": close.shift(-21) / close.shift(-1) - 1.0,
    }
    result: dict[str, object] = {}
    for name, expected in recomputed.items():
        recorded = frame[name]
        comparable = recorded.notna() & expected.notna()
        difference = (recorded[comparable] - expected[comparable]).abs()
        maximum = float(difference.max(skipna=True))
        result[name] = {
            "comparable_rows": int(comparable.sum()),
            "max_abs_difference": maximum,
            "recorded_finite_outside_universe_rows": int(
                (recorded.notna() & ~frame["in_universe"]).sum()
            ),
            "recorded_finite_beyond_cache_recompute_rows": int(
                (recorded.notna() & expected.isna()).sum()
            ),
            "recorded_missing_but_raw_recomputable_rows": int(
                (recorded.isna() & expected.notna()).sum()
            ),
            "missing_mask_interpretation": (
                "recorded labels are dynamic-universe scoped; finite right-boundary "
                "labels may use Qlib dates beyond the raw cache end"
            ),
            "exact": bool(
                maximum == 0.0
                and int((recorded.notna() & ~frame["in_universe"]).sum()) == 0
            ),
        }
    return result


def audit_data(workspace: Path, output: Path) -> dict[str, object]:
    """Verify the real cache, mask split, and exact label identities."""
    data_config, _ = _configs(workspace)
    panel = _load_panel(data_config)
    labels = _label_parity(Path(data_config["cache_path"]))
    payload = {"pit": panel.audit(), "label_parity": labels}
    if not all(record["exact"] for record in labels.values()):
        raise ValueError("PIT label parity failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(output, payload)
    return payload


def _publication_identities(
    workspace: Path, data_config: dict[str, Any], quanta_config: dict[str, Any]
) -> dict[str, object]:
    package_root = workspace / "src" / "mirage_kan"
    baseline = QuantaAdapter.baseline_link(quanta_config["baseline_metric"])
    if baseline["sha256"] != quanta_config["baseline_metric_sha256"]:
        raise ValueError("historical Alpha158 metric identity mismatch")
    return {
        "proposal": {
            "path": str(workspace / "KAN_Alpha_PR.md"),
            "sha256": sha256_file(workspace / "KAN_Alpha_PR.md"),
            "authority": "idea_draft",
        },
        "pit_cache": {
            "path": data_config["cache_path"],
            "sha256": data_config["cache_sha256"],
        },
        "quanta": {
            "commit": quanta_config["commit"],
            "config_sha256": quanta_config["config_sha256"],
            "runner_sha256": quanta_config["runner_sha256"],
        },
        "code": source_tree_identity(package_root),
        "baseline": baseline,
    }


def publish_seed(workspace: Path, destination: Path) -> dict[str, object]:
    """Publish the S0 seed wiring control and append accepted candidate records."""
    data_config, quanta_config = _configs(workspace)
    panel = _load_panel(data_config)
    programs = seed_wiring_programs()
    manifest = publish_library(
        destination,
        programs,
        panel,
        identities=_publication_identities(workspace, data_config, quanta_config),
        library_role="wiring_control",
        kan_mined=False,
    )
    ledger = workspace / "ledgers" / "attempts.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as stream:
        for factor_id, record in manifest["factors"].items():
            stream.write(
                json.dumps(
                    {
                        "stage": "S0",
                        "candidate_id": factor_id,
                        "canonical_hash": record["canonical_hash"],
                        "status": "accepted_wiring_control",
                        "kan_mined": False,
                        "library": str(destination.resolve()),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return manifest


def evaluate_library(
    workspace: Path, library_path: Path, destination: Path
) -> dict[str, object]:
    """Verify a library and run the real pinned Quanta evaluation into no-replace output."""
    data_config, quanta_config = _configs(workspace)
    panel = _load_panel(data_config)
    verification = verify_library(library_path, panel)
    factor_panel = pd.read_parquet(library_path / "factor_panel.parquet")
    parent = destination.parent.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace evaluation: {destination}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".staging", dir=parent
        )
    )
    try:
        adapter = QuantaAdapter(
            quanta_config["repository"],
            expected_commit=quanta_config["commit"],
            expected_config_sha256=quanta_config["config_sha256"],
            expected_runner_sha256=quanta_config["runner_sha256"],
            output_dir=staging,
        )
        metrics = adapter.evaluate_panel(
            factor_panel,
            experiment_name="mirage_kan_s0_seed_ast_v1",
            recorder_name="mirage_kan_s0_seed_ast_v1",
            output_name="seed_ast_v1",
        )
        payload = {
            "stage": "S0",
            "scientific_result": False,
            "library_role": "wiring_control",
            "kan_mined": False,
            "library_path": str(library_path.resolve()),
            "library_manifest_sha256": sha256_file(library_path / "manifest.json"),
            "library_verification": verification,
            "quanta_identity": adapter.identity,
            "metrics": metrics,
            "baseline_link": QuantaAdapter.baseline_link(
                quanta_config["baseline_metric"]
            ),
        }
        _write_json_exclusive(staging / "evaluation_manifest.json", payload)
        publish_staging_directory(
            staging,
            destination,
            required_manifest="evaluation_manifest.json",
        )
        return payload
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _publish_s2a_evaluation(
    workspace: Path,
    destination: Path,
    *,
    library_path: Path | None,
    mining_run: Path,
    authorization: object,
    opening: dict[str, object],
    provider_receipt: dict[str, object],
) -> dict[str, object]:
    if authorization is not _S2_ORCHESTRATOR_TOKEN:
        raise PermissionError("S2a arms may only run through the single orchestrator")
    destination = _direct_child(workspace, destination, "evaluations")
    parent = destination.parent.resolve(strict=True)
    protocol, current_identities = verified_s2_identities(
        workspace, provider_receipt=provider_receipt
    )
    if current_identities["implementation_lock"]["sha256"] != opening[
        "implementation_lock_sha256"
    ]:
        raise ValueError("S2a implementation identity changed after opening")
    if current_identities["qlib_provider"]["tree_sha256"] != opening[
        "qlib_provider_tree_sha256"
    ]:
        raise ValueError("S2a provider identity changed after opening")
    if current_identities["baseline_metric"]["sha256"] != opening[
        "baseline_metric_sha256"
    ]:
        raise ValueError("S2a baseline identity changed after opening")
    data_config, quanta_config = _configs(workspace)
    panel = _load_panel(data_config)
    mining_verification = _verify_s2a_mining_run(
        mining_run,
        panel,
        workspace,
        current_identities,
        deep_scoring=False,
    )
    if mining_verification["manifest_sha256"] != opening["mining_manifest_sha256"]:
        raise ValueError("S2a mining identity changed after opening")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".staging", dir=parent
        )
    )
    try:
        console_path = staging / "console.log"
        with console_path.open("x", encoding="utf-8") as console, contextlib.redirect_stdout(
            console
        ), contextlib.redirect_stderr(console):
            adapter = QuantaAdapter(
                quanta_config["repository"],
                expected_commit=quanta_config["commit"],
                expected_config_sha256=quanta_config["config_sha256"],
                expected_runner_sha256=quanta_config["runner_sha256"],
                expected_provider_identity=provider_receipt,
                output_dir=staging,
            )
            adapter.initialize_and_verify_provider()
            if library_path is None:
                arm = "alpha158_replay"
                verification = None
                library_identity = None
                metrics = adapter.evaluate_alpha158(
                    experiment_name="mirage_kan_s2a_alpha158_replay",
                    output_name="alpha158_replay",
                    capture_report=True,
                )
            else:
                verification = verify_library(library_path, panel)
                library_manifest = _load_json(library_path / "manifest.json")
                arm = library_manifest["library_role"]
                if arm not in LIBRARY_ROLES:
                    raise ValueError(f"library role is not an S2a arm: {arm}")
                if library_manifest.get("kan_mined") is not False:
                    raise ValueError("S2a custom library must record kan_mined=false")
                registered_path = mining_verification["libraries"][arm]["path"]
                if library_path.resolve() != Path(registered_path):
                    raise ValueError("custom library is not registered by the mining run")
                factor_panel = pd.read_parquet(library_path / "factor_panel.parquet")
                metrics = adapter.evaluate_panel(
                    factor_panel,
                    experiment_name=f"mirage_kan_s2a_{arm}",
                    recorder_name=f"mirage_kan_s2a_{arm}",
                    output_name=arm,
                    capture_report=True,
                )
                library_identity = {
                    "path": str(library_path.resolve()),
                    "manifest_sha256": sha256_file(library_path / "manifest.json"),
                    "identities": library_manifest.get("identities"),
                }
            diagnostic_files = adapter.write_portfolio_diagnostics(staging)
        if str(destination) != opening["evaluation_destinations"][arm]:
            raise ValueError("S2a arm destination differs from fixed opening topology")
        artifact_index = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(staging.iterdir())
            if path.is_file() and not path.is_symlink()
        }
        payload = {
            "schema_version": "mirage_s2a_quanta_evaluation_v1",
            "stage": "S2a",
            "evidence_class": "prospective_development_screen",
            "arm": arm,
            "scientific_result": False,
            "final_claim_allowed": False,
            "formal_promotion_allowed": False,
            "library": library_identity,
            "library_verification": verification,
            "mining_run_verification": mining_verification,
            "quanta_identity": adapter.identity,
            "metrics": metrics,
            "diagnostic_files": diagnostic_files,
            "artifact_index": artifact_index,
            "opening": {
                "path": opening["path"],
                "sha256": opening["sha256"],
            },
            "baseline_link": QuantaAdapter.baseline_link(
                quanta_config["baseline_metric"]
            ),
            "execution_identities": _publication_identities(
                workspace, data_config, quanta_config
            ),
            "prospective_identities": current_identities,
            "protocol_id": protocol["protocol_id"],
        }
        _write_json_exclusive(staging / "evaluation_manifest.json", payload)
        publish_staging_directory(
            staging,
            destination,
            required_manifest="evaluation_manifest.json",
        )
        return payload
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _verify_s2a_mining_run(
    mining_run: Path,
    panel: PitPanel,
    workspace: Path,
    current_identities: dict[str, object],
    *,
    deep_scoring: bool = True,
) -> dict[str, object]:
    root = mining_run.resolve(strict=True)
    if root.parent != (workspace / "artifacts").resolve(strict=True):
        raise ValueError("S2a mining run is outside artifacts/")
    if (root / ".INCOMPLETE").exists():
        raise ValueError("S2a mining run publication is incomplete")
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema_version") != "mirage_s2a_mining_run_v1":
        raise ValueError("unsupported S2a mining-run manifest")
    if manifest.get("publication_state") != "complete":
        raise ValueError("S2a mining run is not a complete publication")
    for key in ("scientific_result", "final_claim_allowed", "kan_mined"):
        if manifest.get(key) is not False:
            raise ValueError(f"S2a mining manifest has invalid {key}")
    if manifest.get("attempt_count") != 256:
        raise ValueError("S2a mining manifest must record exactly 256 attempts")
    expected_files = {_safe_manifest_filename(name) for name in manifest["files"]}
    actual_files = {
        entry.name
        for entry in root.iterdir()
        if entry.is_file() and not entry.is_symlink()
    }
    if actual_files != expected_files | {"manifest.json"}:
        raise ValueError("S2a mining run does not have the exact manifest file set")
    if any(entry.is_symlink() or not entry.is_file() for entry in root.iterdir()):
        raise ValueError("S2a mining run must contain only flat regular files")
    for filename, expected in manifest["files"].items():
        filename = _safe_manifest_filename(filename)
        if sha256_file(root / filename) != expected:
            raise ValueError(f"S2a mining-run file hash mismatch: {filename}")
    attempts = [
        json.loads(line)
        for line in (root / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(attempts) != 256 or len({row["candidate_id"] for row in attempts}) != 256:
        raise ValueError("S2a attempt ledger count or IDs are invalid")
    evaluated = 0
    for row in attempts:
        if row["ast"] is None:
            if row["generation_disposition"] != "invalid_generation":
                raise ValueError("S2a null AST lacks invalid-generation disposition")
            continue
        program = AstNode.from_dict(row["ast"])
        contract = program.validate()
        if row.get("canonical_hash") != program.identity:
            raise ValueError("S2a attempt canonical identity is not recomputable")
        if row.get("required_lookback") != contract.lookback or row.get(
            "required_lag"
        ) != contract.lookback - 1:
            raise ValueError("S2a attempt lookback provenance is not recomputable")
        evaluated += 1
    if manifest.get("actual_evaluated_attempt_count") != evaluated:
        raise ValueError("S2a actual evaluated-attempt count is inconsistent")
    with (workspace / "configs" / "experiments" / "s2_plan_c_vertical_v1.yaml").open(
        encoding="utf-8"
    ) as stream:
        protocol = yaml.safe_load(stream)
    admission = protocol["admission"]
    candidate_tables = {}
    for mode, filename, expected_run in (
        ("observed", "candidate_table_observed.jsonl", manifest["observed_scoring_run_id"]),
        (
            "within_date_permutation",
            "candidate_table_permutation.jsonl",
            manifest["permutation_scoring_run_id"],
        ),
    ):
        records = [
            json.loads(line)
            for line in (root / filename).read_text(encoding="utf-8").splitlines()
        ]
        if len(records) != 256 or {row["candidate_id"] for row in records} != {
            row["candidate_id"] for row in attempts
        }:
            raise ValueError(f"S2a {mode} candidate ledger is incomplete")
        selected_records = [row for row in records if row["selection_order"] is not None]
        selected_records.sort(key=lambda row: int(row["selection_order"]))
        if len(selected_records) < int(admission["minimum_library_size"]):
            raise ValueError(f"S2a {mode} selected library is below minimum size")
        if len(selected_records) > int(admission["library_cap"]):
            raise ValueError(f"S2a {mode} selected library exceeds cap")
        if len({row["profile"] for row in selected_records}) < int(
            admission["minimum_miner_profiles"]
        ):
            raise ValueError(f"S2a {mode} selected profile quota fails")
        for row in records:
            if row["label_mode"] != mode or row["scoring_run_id"] != expected_run:
                raise ValueError(f"S2a {mode} scoring provenance mismatch")
            if row["ast"] is not None:
                program = AstNode.from_dict(row["ast"])
                contract = program.validate()
                if (
                    row["required_lookback"] != contract.lookback
                    or row["required_lag"] != contract.lookback - 1
                    or ast_depth(program) != row["ast_depth"]
                    or ast_node_count(program) != row["ast_nodes"]
                ):
                    raise ValueError(f"S2a {mode} AST provenance mismatch")
            if row["selection_order"] is not None:
                if (
                    row["selection_disposition"] != "selected"
                    or row["scoring_disposition"] != "eligible"
                    or row["coverage"] < float(admission["minimum_coverage"])
                    or abs(row["train_rank_ic"])
                    < float(admission["minimum_absolute_train_rank_ic"])
                    or abs(row["validation_rank_ic"])
                    < float(admission["minimum_absolute_validation_rank_ic"])
                    or row["sign_agreement"] is not True
                ):
                    raise ValueError(f"S2a {mode} selected candidate is inadmissible")
        candidate_tables[mode] = selected_records
    scoring_verification = None
    if deep_scoring:
        scoring_verification = verify_s2a_scoring_ledgers(
            root, protocol, current_identities
        )
        if (
            scoring_verification["observed_scoring_run_id"]
            != manifest["observed_scoring_run_id"]
            or scoring_verification["permutation_scoring_run_id"]
            != manifest["permutation_scoring_run_id"]
        ):
            raise ValueError("S2a independently recomputed scoring identities differ")
    recorded_identities = manifest.get("identities", {})
    for key in ("proposal", "config", "preregistration", "implementation_lock"):
        if recorded_identities.get(key) != current_identities.get(key):
            raise ValueError(f"S2a mining identity is stale: {key}")
    for key in ("tree_sha256", "file_count", "total_bytes"):
        if recorded_identities.get("qlib_provider", {}).get(key) != current_identities[
            "qlib_provider"
        ][key]:
            raise ValueError(f"S2a mining provider identity is stale: {key}")
    if recorded_identities.get("baseline_metric") != current_identities.get(
        "baseline_metric"
    ):
        raise ValueError("S2a mining baseline identity is stale")
    entitlement = recorded_identities.get("mining_entitlement", {})
    entitlement_path = (workspace / MINING_ENTITLEMENT).resolve(strict=True)
    preclaim = recorded_identities.get("mining_preclaim", {})
    preclaim_path = (workspace / MINING_PRECLAIM).resolve(strict=True)
    if (
        entitlement.get("path") != str(entitlement_path)
        or entitlement.get("sha256") != sha256_file(entitlement_path)
        or entitlement.get("run_path") != str(root)
        or entitlement.get("attempt_budget") != 256
        or entitlement.get("preclaim_sha256") != sha256_file(preclaim_path)
    ):
        raise ValueError("S2a protocol-global mining entitlement is invalid")
    if (
        preclaim.get("path") != str(preclaim_path)
        or preclaim.get("sha256") != sha256_file(preclaim_path)
        or preclaim.get("run_path") != str(root)
        or preclaim.get("attempt_budget") != 256
    ):
        raise ValueError("S2a protocol-global mining preclaim is invalid")
    if set(manifest["libraries"]) != set(LIBRARY_ROLES):
        raise ValueError("S2a mining run does not register all three custom arms")
    if entitlement.get("library_destinations") != {
        role: manifest["libraries"][role]["path"] for role in LIBRARY_ROLES
    }:
        raise ValueError("S2a mining entitlement library topology is invalid")
    if preclaim.get("library_destinations") != entitlement.get(
        "library_destinations"
    ):
        raise ValueError("S2a mining preclaim library topology is invalid")
    libraries: dict[str, object] = {}
    for role in LIBRARY_ROLES:
        record = manifest["libraries"][role]
        library_path = Path(record["path"]).resolve(strict=True)
        if library_path.parent != (workspace / "factor_libraries").resolve(strict=True):
            raise ValueError(f"S2a library is outside factor_libraries/: {role}")
        if record.get("role") != role or record.get("kan_mined") is not False:
            raise ValueError(f"S2a registered library role is invalid: {role}")
        if sha256_file(library_path / "manifest.json") != record["manifest_sha256"]:
            raise ValueError(f"S2a library manifest hash mismatch: {role}")
        verification = verify_library(library_path, panel)
        library_manifest = _load_json(library_path / "manifest.json")
        if (
            library_manifest.get("library_role") != role
            or library_manifest.get("kan_mined") is not False
            or library_manifest.get("scientific_result") is not False
            or library_manifest.get("factor_count") != record.get("factor_count")
        ):
            raise ValueError(f"S2a library invariants fail: {role}")
        provenance = library_manifest.get("identities", {}).get(
            "selection_provenance", {}
        )
        if provenance.get("role") != role:
            raise ValueError(f"S2a selection provenance role mismatch: {role}")
        if deep_scoring and provenance.get("candidate_ids_in_selection_order") != (
            scoring_verification["selected_candidate_ids"][role]
        ):
            raise ValueError(f"S2a selected ordering is not reproducible: {role}")
        if role == "random_typed":
            if provenance.get("label_free") is not True or provenance.get(
                "label_sha256"
            ) is not None:
                raise ValueError("S2a random library is not label-free")
            if record.get("factor_count") != 16:
                raise ValueError("S2a random library does not match the cap of 16")
            random_config = provenance.get("selection_config", {})
            if (
                random_config.get("random_control_seed")
                != protocol["search"]["random_control_seed"]
                or random_config.get("random_control_period")
                != [
                    str(protocol["data"]["train"][0]),
                    str(protocol["data"]["validation"][1]),
                ]
            ):
                raise ValueError("S2a random-control provenance is not frozen")
        elif provenance.get("label_free") is not False:
            raise ValueError(f"S2a scored library lacks scoring provenance: {role}")
        if role != "random_typed":
            mode = (
                "observed"
                if role == "heterogeneous_selected"
                else "within_date_permutation"
            )
            selected_ids = [row["candidate_id"] for row in candidate_tables[mode]]
            if provenance.get("candidate_ids_in_selection_order") != selected_ids:
                raise ValueError(f"S2a {role} selection order is not ledger-bound")
            factor_panel = pd.read_parquet(library_path / "factor_panel.parquet")
            dates = pd.to_datetime(factor_panel.index.get_level_values("datetime"))
            validation = factor_panel.loc[
                (dates >= pd.Timestamp(protocol["data"]["validation"][0]))
                & (dates <= pd.Timestamp(protocol["data"]["validation"][1]))
            ]
            for left, right in combinations(validation.columns, 2):
                daily_correlations = []
                pair = validation[[left, right]].dropna()
                for _, daily_values in pair.groupby(level="datetime", sort=False):
                    if (
                        len(daily_values) < 2
                        or daily_values[left].nunique() < 2
                        or daily_values[right].nunique() < 2
                    ):
                        continue
                    value = daily_values[left].corr(daily_values[right], method="spearman")
                    if pd.notna(value):
                        daily_correlations.append(float(value))
                if not daily_correlations or abs(sum(daily_correlations) / len(daily_correlations)) >= float(
                    admission["maximum_absolute_validation_spearman"]
                ):
                    raise ValueError(f"S2a {role} diversity is not recomputable")
        libraries[role] = {
            "path": str(library_path),
            "manifest_sha256": record["manifest_sha256"],
            "verification": verification,
        }
    return {
        "verified": True,
        "path": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "libraries": libraries,
        "scoring_verification": scoring_verification,
    }


def _verify_s2a_evaluation(
    path: Path,
    workspace: Path,
    opening: dict[str, object],
    expected_arm: str,
) -> dict[str, object]:
    root = path.resolve(strict=True)
    if root.parent != (workspace / "evaluations").resolve(strict=True):
        raise ValueError("S2a evaluation is outside evaluations/")
    if (root / ".INCOMPLETE").exists():
        raise ValueError("S2a evaluation publication is incomplete")
    manifest = _load_json(root / "evaluation_manifest.json")
    if manifest.get("schema_version") != "mirage_s2a_quanta_evaluation_v1":
        raise ValueError("unsupported S2a evaluation schema")
    if manifest.get("arm") != expected_arm:
        raise ValueError("S2a evaluation is in the wrong fixed-topology arm slot")
    if opening["evaluation_destinations"].get(expected_arm) != str(root):
        raise ValueError("S2a opening does not bind this arm destination")
    if manifest.get("scientific_result") is not False or manifest.get(
        "formal_promotion_allowed"
    ) is not False:
        raise ValueError("S2a evaluation overstates its evidence class")
    if manifest.get("opening", {}).get("sha256") != opening["sha256"]:
        raise ValueError("S2a evaluation is not bound to the shared opening")
    index = manifest.get("artifact_index")
    if not isinstance(index, dict):
        raise ValueError("S2a evaluation lacks a complete artifact index")
    indexed = {_safe_manifest_filename(name) for name in index}
    actual = {
        entry.name
        for entry in root.iterdir()
        if entry.is_file() and not entry.is_symlink()
    }
    if actual != indexed | {"evaluation_manifest.json"}:
        raise ValueError("S2a evaluation does not have the exact indexed file set")
    required_diagnostics = {
        "console.log",
        "qlib_report.parquet",
        "portfolio_daily.parquet",
        "prediction_coverage.parquet",
    }
    if not required_diagnostics.issubset(indexed):
        raise ValueError("S2a evaluation lacks a required diagnostic artifact")
    diagnostic_files = manifest.get("diagnostic_files")
    parquet_diagnostics = required_diagnostics - {"console.log"}
    if (
        not isinstance(diagnostic_files, dict)
        or set(diagnostic_files) != parquet_diagnostics
    ):
        raise ValueError("S2a evaluation required diagnostic index is inconsistent")
    if any(
        diagnostic_files[name] != index[name].get("sha256")
        for name in parquet_diagnostics
    ):
        raise ValueError("S2a evaluation diagnostic hashes are inconsistent")
    if any(entry.is_symlink() or not entry.is_file() for entry in root.iterdir()):
        raise ValueError("S2a evaluation must contain only flat regular files")
    for filename, record in index.items():
        filename = _safe_manifest_filename(filename)
        artifact = root / filename
        if sha256_file(artifact) != record["sha256"] or artifact.stat().st_size != record[
            "bytes"
        ]:
            raise ValueError(f"S2a evaluation artifact mismatch: {filename}")
    return manifest


def run_s2a_development(
    workspace: Path,
    mining_run: Path,
    evaluation_destinations: dict[str, Path],
    decision_path: Path,
    report_path: Path,
) -> dict[str, object]:
    """Open the development test once and execute the frozen four-arm topology."""
    if set(evaluation_destinations) != set(ARMS):
        raise ValueError("S2a orchestrator requires exactly the four frozen arms")
    destinations = {
        arm: _direct_child(workspace, evaluation_destinations[arm], "evaluations")
        for arm in ARMS
    }
    if len(set(destinations.values())) != len(ARMS):
        raise ValueError("S2a evaluation destinations must be distinct")
    decision_path = _direct_child(workspace, decision_path, "governance/decisions")
    report_path = _direct_child(workspace, report_path, "reports")
    protocol, identities = verified_s2_identities(workspace)
    provider_receipt = dict(identities["qlib_provider"])
    data_config, _ = _configs(workspace)
    panel = _load_panel(data_config)
    mining = _verify_s2a_mining_run(
        mining_run, panel, workspace, identities
    )
    opening_directory = workspace / "governance" / "openings"
    opening_directory.mkdir(parents=True, exist_ok=True)
    opening_path = opening_directory / "s2_plan_c_vertical_v1.json"
    opening_payload = {
        "schema_version": "mirage_s2a_development_opening_v1",
        "protocol_id": protocol["protocol_id"],
        "state": "consumed_before_first_test_access",
        "mining_manifest_sha256": mining["manifest_sha256"],
        "mining_scoring_verification": mining["scoring_verification"],
        "implementation_lock_sha256": identities["implementation_lock"]["sha256"],
        "qlib_provider_tree_sha256": provider_receipt["tree_sha256"],
        "baseline_metric_sha256": identities["baseline_metric"]["sha256"],
        "evaluation_destinations": {
            arm: str(path) for arm, path in destinations.items()
        },
        "decision_path": str(decision_path),
        "report_path": str(report_path),
        "formal_promotion_allowed": False,
    }
    _write_json_exclusive(opening_path, opening_payload)
    opening = dict(opening_payload)
    opening.update({"path": str(opening_path), "sha256": sha256_file(opening_path)})
    evaluations: dict[str, dict[str, object]] = {}
    try:
        evaluations["alpha158_replay"] = _publish_s2a_evaluation(
            workspace,
            destinations["alpha158_replay"],
            library_path=None,
            mining_run=mining_run,
            authorization=_S2_ORCHESTRATOR_TOKEN,
            opening=opening,
            provider_receipt=provider_receipt,
        )
        evaluations["alpha158_replay"] = _verify_s2a_evaluation(
            destinations["alpha158_replay"],
            workspace,
            opening,
            "alpha158_replay",
        )
        replay_check = replay_anchor_checks(
            evaluations["alpha158_replay"]["metrics"], protocol
        )
        if not replay_check["passed"]:
            replay_metrics = evaluations["alpha158_replay"]["metrics"]
            decision = {
                "outcome": "s2a_inconclusive_infrastructure",
                "formal_promotion_allowed": False,
                "replay_anchor_checks": replay_check,
                "headline_metrics": {
                    "alpha158_replay": {
                        "information_ratio": replay_metrics["information_ratio"],
                        "max_drawdown": replay_metrics["max_drawdown"],
                        "rank_ic": replay_metrics["Rank IC"],
                    }
                },
                "criteria": {},
                "all_criteria_passed": False,
                "calendar_active_return": {},
            }
        else:
            for arm in LIBRARY_ROLES:
                library = Path(mining["libraries"][arm]["path"])
                evaluations[arm] = _publish_s2a_evaluation(
                    workspace,
                    destinations[arm],
                    library_path=library,
                    mining_run=mining_run,
                    authorization=_S2_ORCHESTRATOR_TOKEN,
                    opening=opening,
                    provider_receipt=provider_receipt,
                )
            verified_evaluations = {
                arm: _verify_s2a_evaluation(
                    destinations[arm], workspace, opening, arm
                )
                for arm in ARMS
            }
            daily = {
                arm: pd.read_parquet(destinations[arm] / "portfolio_daily.parquet")
                for arm in ARMS
            }
            test_start, test_end = protocol["data"]["development_test"]
            panel_dates = pd.DatetimeIndex(
                panel.raw.index.get_level_values("datetime").unique()
            ).sort_values()
            expected_calendar = panel_dates[
                (panel_dates >= pd.Timestamp(test_start))
                & (panel_dates <= pd.Timestamp(test_end))
            ]
            decision = decide_s2a(
                verified_evaluations, daily, protocol, expected_calendar
            )
    except Exception as error:
        decision = {
            "outcome": "s2a_inconclusive_infrastructure",
            "formal_promotion_allowed": False,
            "infrastructure_error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "replay_anchor_checks": {},
            "headline_metrics": {},
            "criteria": {},
            "all_criteria_passed": False,
            "calendar_active_return": {},
        }
    decision.update(
        {
            "schema_version": "mirage_s2a_decision_v1",
            "protocol_id": protocol["protocol_id"],
            "opening": {"path": opening["path"], "sha256": opening["sha256"]},
            "mining_manifest_sha256": mining["manifest_sha256"],
            "evaluation_manifests": {
                arm: {
                    "path": str(destinations[arm]),
                    "sha256": sha256_file(destinations[arm] / "evaluation_manifest.json"),
                }
                for arm in evaluations
            },
        }
    )
    if "infrastructure_error" in decision:
        report = (
            "# MIRAGE-KAN S2a 完整链路报告\n\n"
            "## ⚠️ 结论：s2a_inconclusive_infrastructure\n\n"
            "基础设施或证据完整性检查失败，未形成科学比较。"
            f"错误：{decision['infrastructure_error']['type']} — "
            f"{decision['infrastructure_error']['message']}\n\n"
            "formal_promotion_allowed = false。图控制模块保持锁定。\n"
        )
    else:
        report = chinese_report(decision)
    _write_text_exclusive(report_path, report)
    decision["human_report"] = {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "bytes": report_path.stat().st_size,
    }
    _write_json_exclusive(decision_path, decision)
    return decision


def recover_s2a_interruption(workspace: Path) -> dict[str, object]:
    """Finalize a consumed, interrupted opening without rerunning any arm."""
    opening_raw = (
        workspace / "governance" / "openings" / "s2_plan_c_vertical_v1.json"
    )
    if opening_raw.is_symlink():
        raise ValueError("S2a development opening cannot be a symlink")
    opening_path = opening_raw.resolve(strict=True)
    opening = _load_json(opening_path)
    normal_decision = _recorded_direct_child(
        workspace, opening["decision_path"], "governance/decisions"
    )
    normal_report = _recorded_direct_child(
        workspace, opening["report_path"], "reports"
    )
    destinations = {
        arm: _recorded_direct_child(
            workspace, opening["evaluation_destinations"][arm], "evaluations"
        )
        for arm in ARMS
    }
    if normal_decision.is_file():
        recorded = _load_json(normal_decision)
        report_record = recorded.get("human_report", {})
        if (
            normal_report.is_file()
            and report_record.get("sha256") == sha256_file(normal_report)
            and report_record.get("bytes") == normal_report.stat().st_size
        ):
            raise FileExistsError("S2a development opening already has a terminal decision")

    recovery_decision = workspace / "governance" / "decisions" / (
        "s2_plan_c_vertical_v1_interruption.json"
    )
    recovery_report = workspace / "reports" / "s2_plan_c_vertical_v1_interruption.md"
    recovery_decision = _recovery_child(
        workspace, recovery_decision, "governance/decisions"
    )
    recovery_report = _recovery_child(workspace, recovery_report, "reports")
    if recovery_decision.is_file():
        raise FileExistsError("S2a interruption recovery already has a decision")
    evaluations: dict[str, object] = {}
    for arm, path in destinations.items():
        if not path.exists():
            evaluations[arm] = {"path": str(path), "state": "missing"}
        elif (path / ".INCOMPLETE").exists():
            evaluations[arm] = {"path": str(path), "state": "incomplete"}
        elif (path / "evaluation_manifest.json").is_file():
            evaluations[arm] = {
                "path": str(path),
                "state": "published_unconsumed_by_recovery",
                "manifest_sha256": sha256_file(path / "evaluation_manifest.json"),
            }
        else:
            evaluations[arm] = {"path": str(path), "state": "invalid"}
    report = (
        "# MIRAGE-KAN S2a 中断终态报告\n\n"
        "## ⚠️ 结论：s2a_inconclusive_infrastructure\n\n"
        "一次性开发测试 opening 已被消耗，但进程未发布完整且可验证的常规终局。"
        "本恢复流程不会重跑任何实验臂，也不会读取或比较已产生的指标；"
        "它只把本协议永久终止为基础设施不确定。\n\n"
        "formal_promotion_allowed = false。图控制模块保持锁定。\n"
    )
    _write_text_exclusive_or_exact(recovery_report, report)
    decision = {
        "schema_version": "mirage_s2a_interruption_decision_v1",
        "protocol_id": "s2_plan_c_vertical_v1",
        "outcome": "s2a_inconclusive_infrastructure",
        "formal_promotion_allowed": False,
        "opening": {"path": str(opening_path), "sha256": sha256_file(opening_path)},
        "evaluations": evaluations,
        "normal_decision_state": (
            "partial_or_invalid" if normal_decision.exists() else "missing"
        ),
        "human_report": {
            "path": str(recovery_report),
            "sha256": sha256_file(recovery_report),
            "bytes": recovery_report.stat().st_size,
        },
    }
    _write_json_exclusive(recovery_decision, decision)
    return decision


def _interrupted_artifact_state(path: Path, manifest_name: str) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "state": "missing"}
    if (path / ".INCOMPLETE").is_file():
        return {"path": str(path), "state": "incomplete"}
    terminal = path / "terminal_failure.json"
    if terminal.is_file():
        payload = _load_json(terminal)
        return {
            "path": str(path),
            "state": "terminal_failure",
            "terminal_sha256": sha256_file(terminal),
            "schema_version": payload.get("schema_version"),
        }
    manifest = path / manifest_name
    if manifest.is_file():
        payload = _load_json(manifest)
        publication_state = payload.get("publication_state", "published")
        return {
            "path": str(path),
            "state": str(publication_state),
            "manifest_sha256": sha256_file(manifest),
        }
    return {"path": str(path), "state": "invalid"}


def recover_s2a_mining_interruption(workspace: Path) -> dict[str, object]:
    """Terminalize a consumed mining entitlement without generating attempts."""
    entitlement_raw = workspace / MINING_ENTITLEMENT
    preclaim_raw = workspace / MINING_PRECLAIM
    if entitlement_raw.is_symlink() or preclaim_raw.is_symlink():
        raise ValueError("S2a mining authority cannot be a symlink")
    if entitlement_raw.is_file():
        authority_path = entitlement_raw.resolve(strict=True)
        entitlement = _load_json(authority_path)
        authority_source = "entitlement"
        expected_schema = "mirage_s2a_mining_entitlement_v1"
        expected_state = "consumed_before_attempt_generation"
    else:
        authority_path = preclaim_raw.resolve(strict=True)
        entitlement = _load_json(authority_path)
        authority_source = "preclaim"
        expected_schema = "mirage_s2a_mining_preclaim_v1"
        expected_state = "consumed_before_run_claim"
    if (
        entitlement.get("schema_version") != expected_schema
        or entitlement.get("protocol_id") != "s2_plan_c_vertical_v1"
        or entitlement.get("state") != expected_state
        or entitlement.get("attempt_budget") != 256
        or set(entitlement.get("library_destinations", {})) != set(LIBRARY_ROLES)
    ):
        raise ValueError("invalid S2a mining authority")
    run_path = _recorded_direct_child(
        workspace, entitlement["run_path"], "artifacts"
    )
    libraries = {
        role: _recorded_direct_child(
            workspace,
            entitlement["library_destinations"][role],
            "factor_libraries",
        )
        for role in LIBRARY_ROLES
    }
    initial_run_state = _interrupted_artifact_state(run_path, "manifest.json")
    initial_library_states = {
        role: _interrupted_artifact_state(path, "manifest.json")
        for role, path in libraries.items()
    }
    interrupted = initial_run_state["state"] == "incomplete" or any(
        state["state"] == "incomplete"
        for state in initial_library_states.values()
    )
    recovery_owned_terminal = (
        initial_run_state.get("schema_version")
        == "mirage_s2a_mining_interruption_v1"
    )
    if not interrupted and not recovery_owned_terminal and initial_run_state["state"] in {
        "complete",
        "terminal_failure",
    }:
        raise FileExistsError("S2a mining entitlement already has a terminal run")
    for role, path in libraries.items():
        if path.exists() and interrupted:
            terminalize_claimed_directory(
                path,
                {
                    "schema_version": "mirage_factor_library_terminal_v1",
                    "library_role": role,
                    "scientific_result": False,
                    "kan_mined": False,
                    "error_type": "ProcessInterruption",
                    "error": "recovered after mining process interruption",
                },
                invalidate_published=True,
            )
    if (run_path / ".INCOMPLETE").is_file():
        terminalize_claimed_directory(
            run_path,
            {
                "schema_version": "mirage_s2a_mining_interruption_v1",
                "scientific_result": False,
                "formal_promotion_allowed": False,
                "error_type": "ProcessInterruption",
                "error": "recovered after mining process interruption",
            },
        )
    run_state = _interrupted_artifact_state(run_path, "manifest.json")
    decision_path = _recovery_child(
        workspace,
        workspace
        / "governance"
        / "decisions"
        / "s2_plan_c_vertical_v1_mining_interruption.json",
        "governance/decisions",
    )
    report_path = _recovery_child(
        workspace,
        workspace / "reports" / "s2_plan_c_vertical_v1_mining_interruption.md",
        "reports",
    )
    if decision_path.is_file():
        raise FileExistsError("S2a mining recovery already has a decision")
    library_states = {
        role: _interrupted_artifact_state(path, "manifest.json")
        for role, path in libraries.items()
    }
    report = (
        "# MIRAGE-KAN S2a 挖掘中断终态报告\n\n"
        "## ⚠️ 结论：s2a_mining_inconclusive_infrastructure\n\n"
        "协议级 256 次尝试额度已经被消耗，但挖掘运行未形成完整终局。"
        "恢复流程不生成候选、不读取标签、不恢复或重跑部分结果；"
        "本协议的挖掘与后续四臂回测永久停止。\n\n"
        "formal_promotion_allowed = false。图控制模块保持锁定。\n"
    )
    _write_text_exclusive_or_exact(report_path, report)
    decision = {
        "schema_version": "mirage_s2a_mining_interruption_decision_v1",
        "protocol_id": "s2_plan_c_vertical_v1",
        "outcome": "s2a_mining_inconclusive_infrastructure",
        "formal_promotion_allowed": False,
        "entitlement": {
            "path": str(authority_path),
            "sha256": sha256_file(authority_path),
            "attempt_budget": 256,
            "source": authority_source,
        },
        "run": run_state,
        "libraries": library_states,
        "human_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "bytes": report_path.stat().st_size,
        },
    }
    _write_json_exclusive(decision_path, decision)
    return decision


def replay_alpha158(
    workspace: Path, destination: Path, mining_run: Path
) -> dict[str, object]:
    """Reject independent test opening; retained only for a clear API error."""
    raise PermissionError(
        "use the single orchestrator run_s2a_development; independent arms are forbidden"
    )


def evaluate_s2a_library(
    workspace: Path, library_path: Path, destination: Path, mining_run: Path
) -> dict[str, object]:
    """Reject independent test opening; retained only for a clear API error."""
    raise PermissionError(
        "use the single orchestrator run_s2a_development; independent arms are forbidden"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mirage-kan")
    parser.add_argument("--workspace", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-data")
    audit.add_argument("--output", required=True)
    publish = subparsers.add_parser("publish-seed")
    publish.add_argument("--destination", required=True)
    verify = subparsers.add_parser("verify-library")
    verify.add_argument("--library", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--library", required=True)
    evaluate.add_argument("--destination", required=True)
    run = subparsers.add_parser("run-s0")
    run.add_argument("--library", required=True)
    run.add_argument("--evaluation", required=True)
    run.add_argument("--audit-output", required=True)
    mine_s2a = subparsers.add_parser("mine-s2a")
    mine_s2a.add_argument("--destination", required=True)
    mine_s2a.add_argument("--selected-library", required=True)
    mine_s2a.add_argument("--random-library", required=True)
    mine_s2a.add_argument("--permutation-library", required=True)
    replay = subparsers.add_parser("replay-alpha158")
    replay.add_argument("--destination", required=True)
    replay.add_argument("--mining-run", required=True)
    evaluate_s2a = subparsers.add_parser("evaluate-s2a")
    evaluate_s2a.add_argument("--library", required=True)
    evaluate_s2a.add_argument("--destination", required=True)
    evaluate_s2a.add_argument("--mining-run", required=True)
    development = subparsers.add_parser("run-s2a-development")
    development.add_argument("--mining-run", required=True)
    development.add_argument("--alpha158-evaluation", required=True)
    development.add_argument("--selected-evaluation", required=True)
    development.add_argument("--random-evaluation", required=True)
    development.add_argument("--permutation-evaluation", required=True)
    development.add_argument("--decision", required=True)
    development.add_argument("--report", required=True)
    subparsers.add_parser("recover-s2a-interruption")
    subparsers.add_parser("recover-s2a-mining-interruption")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one S0 command and print a machine-readable summary."""
    args = _parser().parse_args(argv)
    workspace = _workspace(args)
    if args.command == "audit-data":
        payload = audit_data(workspace, Path(args.output))
    elif args.command == "publish-seed":
        payload = publish_seed(workspace, Path(args.destination))
    elif args.command == "verify-library":
        data_config, _ = _configs(workspace)
        payload = verify_library(Path(args.library), _load_panel(data_config))
    elif args.command == "evaluate":
        payload = evaluate_library(
            workspace, Path(args.library), Path(args.destination)
        )
    elif args.command == "run-s0":
        audit_data(workspace, Path(args.audit_output))
        publish_seed(workspace, Path(args.library))
        payload = evaluate_library(
            workspace, Path(args.library), Path(args.evaluation)
        )
    elif args.command == "mine-s2a":
        payload = run_s2a_mining(
            workspace,
            Path(args.destination),
            {
                "heterogeneous_selected": Path(args.selected_library),
                "random_typed": Path(args.random_library),
                "label_permutation_selected": Path(args.permutation_library),
            },
        )
    elif args.command in {"replay-alpha158", "evaluate-s2a"}:
        raise PermissionError(
            "independent S2a arm commands are disabled; use run-s2a-development"
        )
    elif args.command == "run-s2a-development":
        payload = run_s2a_development(
            workspace,
            Path(args.mining_run),
            {
                "alpha158_replay": Path(args.alpha158_evaluation),
                "heterogeneous_selected": Path(args.selected_evaluation),
                "random_typed": Path(args.random_evaluation),
                "label_permutation_selected": Path(args.permutation_evaluation),
            },
            Path(args.decision),
            Path(args.report),
        )
    elif args.command == "recover-s2a-interruption":
        payload = recover_s2a_interruption(workspace)
    else:
        payload = recover_s2a_mining_interruption(workspace)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
