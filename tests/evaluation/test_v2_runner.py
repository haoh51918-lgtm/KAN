from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mirage_kan.data import PitPanel
from mirage_kan.governance.authority import AuthorityGuard, AuthoritySuperseded


ARMS = (
    "alpha158_replay",
    "kan_e3_selected",
    "typed_gp_sr_control",
    "matched_blackbox_control",
    "kan_e3_permutation_control",
)
BOUNDARIES = (
    "before_first_label_access",
    "before_each_scientific_or_control_arm",
    "before_each_artifact_publication",
    "before_development_opening",
    "before_final_decision_publication",
)
PROVIDER = {
    "path": "/locked/qlib/provider",
    "tree_sha256": "1" * 64,
    "stat_inventory_sha256": "2" * 64,
    "file_count": 7,
    "total_bytes": 101,
}
QUANTA = {
    "commit": "a" * 40,
    "config_sha256": "b" * 64,
    "runner_sha256": "c" * 64,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _panel(tmp_path: Path, *, sparse: bool = False) -> PitPanel:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "cache.parquet"
    dates = pd.bdate_range(end="2021-12-29", periods=60).append(
        pd.DatetimeIndex(
            pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-03", "2022-01-04"])
        )
    )
    frame = pd.DataFrame(
        [
            {
                "datetime": date,
                "instrument": instrument,
                "open": 10.0 + instrument_index,
                "high": 11.0 + instrument_index,
                "low": 9.0 + instrument_index,
                "close": 10.5 + instrument_index,
                "volume": 100.0 + instrument_index,
                "in_universe": True,
            }
            for date in dates
            for instrument_index, instrument in enumerate(("A", "B"))
        ]
    )
    if sparse:
        frame = frame.drop(frame.index[1]).reset_index(drop=True)
    frame.to_parquet(source)
    return PitPanel.from_frame(
        frame, source_path=source.resolve(), source_sha256=_sha256(source)
    )


def _input_artifact(
    workspace: Path, arm: str, panel: PitPanel, *, control_count: int = 8
) -> tuple[Path, dict[str, object]]:
    if arm == "matched_blackbox_control":
        path = workspace / "controls" / "blackbox"
        role = "falsification_control_never_production"
        kind = "computed_factor_control"
        output_kind = "control_panel_not_factor_library"
    else:
        path = workspace / "factor_libraries" / arm
        role = arm
        kind = "verified_factor_library"
        output_kind = "factor_library"
    path.mkdir(parents=True)
    values = pd.DataFrame(
        {
            f"{arm}_{index:03d}": np.arange(len(panel.raw), dtype=float) + float(index)
            for index in range(control_count)
        },
        index=panel.raw.index,
    )
    filename = (
        "prediction_panel.parquet"
        if arm == "matched_blackbox_control"
        else "factor_panel.parquet"
    )
    if arm == "matched_blackbox_control":
        from mirage_kan.mining.e3_runner import materialize_atom_panel

        atom_panel = materialize_atom_panel(panel, "short_price")
        zero_replay = pd.Series(
            np.where(
                (atom_panel.joint_support & atom_panel.membership).reshape(-1),
                0.0,
                np.nan,
            ),
            index=atom_panel.index,
        )
        values = pd.DataFrame(
            {
                f"mlp_for_kan_{index:03d}": zero_replay.reindex(panel.raw.index)
                for index in range(control_count)
            },
            index=panel.raw.index,
        )
    values.to_parquet(path / filename)
    extra_files: dict[str, str] = {}
    selected_kan_factor_ids: list[str] | None = None
    paired_indices: list[int] | None = None
    if arm == "matched_blackbox_control":
        from mirage_kan.mining.e3 import (
            atom_manifest_sha256,
            build_profile_atom_bank,
        )
        from mirage_kan.mining.e3_runner import draw_training_bootstrap

        selected_kan_factor_ids = [
            f"kan_e3_selected_{index:03d}" for index in range(control_count)
        ]
        paired_indices = list(range(control_count))
        archive = bytearray()

        def add_tensor(name: str, array: np.ndarray) -> dict[str, object]:
            value = np.ascontiguousarray(array)
            body = value.tobytes()
            offset = len(archive)
            archive.extend(body)
            return {
                "name": name,
                "offset": offset,
                "nbytes": len(body),
                "shape": list(value.shape),
                "dtype": value.dtype.str,
                "sha256": hashlib.sha256(body).hexdigest(),
            }

        parameter_reference = add_tensor("shared/parameters", np.zeros(69, dtype="<f8"))
        prediction_reference = add_tensor(
            "shared/prediction",
            np.nan_to_num(values.iloc[:, 0].to_numpy(dtype="<f8"), nan=0.0),
        )
        prediction_mask_reference = add_tensor(
            "shared/prediction_mask",
            np.isfinite(values.iloc[:, 0].to_numpy(dtype=float)).astype("|u1"),
        )
        training_prediction_reference = add_tensor(
            "shared/training_prediction", np.zeros((25, 2), dtype="<f8")
        )
        training_mask_reference = add_tensor(
            "shared/training_mask", np.ones((25, 2), dtype="|u1")
        )
        trajectory = [
            {
                "update_index": update,
                "total_loss": 0.0,
                "mean_daily_ic": 0.0,
                "parameters": parameter_reference,
            }
            for update in range(300)
        ]
        trajectory_lines = [
            (
                json.dumps(
                    {
                        "receipt_index": index,
                        "control_id": values.columns[index],
                        "trajectory": trajectory,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode()
            for index in range(control_count)
        ]
        rows = []
        for index in range(control_count):
            rows.append(
                {
                    "receipt_index": index,
                    "control_id": values.columns[index],
                    "profile": "short_price",
                    "kan_factor_id": selected_kan_factor_ids[index],
                    "kan_global_attempt_index": index,
                    "seed": 32452843 + index,
                    "bootstrap": asdict(draw_training_bootstrap(25, 49979687 + index)),
                    "optimizer": "Adam",
                    "learning_rate": 0.03,
                    "scheduled_updates": 300,
                    "completed_updates": 300,
                    "input_atom_count": 32,
                    "kan_parameter_count": 64,
                    "mlp_parameter_count": 69,
                    "parameter_relative_gap": 5 / 64,
                    "atom_manifest_sha256": atom_manifest_sha256(
                        build_profile_atom_bank("short_price")
                    ),
                    "valid_support_sha256": "7" * 64,
                    "initial_parameters": parameter_reference,
                    "final_parameters": parameter_reference,
                    "first_step_data_gradient": parameter_reference,
                    "trajectory": {
                        "file": "control_trajectories.jsonl",
                        "line": index,
                        "sha256": hashlib.sha256(trajectory_lines[index]).hexdigest(),
                    },
                    "prediction": prediction_reference,
                    "prediction_mask": prediction_mask_reference,
                    "training_prediction": training_prediction_reference,
                    "training_prediction_mask": training_mask_reference,
                    "kan_mined": False,
                    "promotion_eligible": False,
                    "factor_library_publication_allowed": False,
                }
            )
        receipt_path = path / "control_receipts.jsonl"
        receipt_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        extra_files[receipt_path.name] = _sha256(receipt_path)
        trajectories_path = path / "control_trajectories.jsonl"
        trajectories_path.write_bytes(b"".join(trajectory_lines))
        extra_files[trajectories_path.name] = _sha256(trajectories_path)
        tensor_path = path / "tensor_evidence.zlib"
        tensor_path.write_bytes(zlib.compress(bytes(archive), level=9))
        extra_files[tensor_path.name] = _sha256(tensor_path)
    manifest = {
        "schema_version": (
            "mirage_matched_blackbox_control_v2"
            if arm == "matched_blackbox_control"
            else "mirage_factor_library_v1"
        ),
        "arm": arm,
        "role": role,
        "output_kind": output_kind,
        "promotion_eligible": False,
        "kan_mined": False,
        "factor_library_publication_allowed": arm != "matched_blackbox_control",
        "control_count": control_count,
        "factor_count": control_count,
        "files": {filename: _sha256(path / filename), **extra_files},
    }
    if selected_kan_factor_ids is not None and paired_indices is not None:
        manifest["selected_kan_factor_ids"] = selected_kan_factor_ids
        manifest["paired_kan_global_attempt_indices"] = paired_indices
    if arm != "matched_blackbox_control":
        from mirage_kan.dsl import AstNode

        program = AstNode("Close")
        manifest["library_role"] = role
        manifest["kan_mined"] = arm == "kan_e3_selected"
        manifest["selected_candidate_ids"] = list(values.columns)
        manifest["factors"] = {
            factor_id: {
                "canonical_hash": program.identity,
                "ast": program.to_dict(),
                **({"global_attempt_index": index} if arm == "kan_e3_selected" else {}),
            }
            for index, factor_id in enumerate(values.columns)
        }
    _write_json(path / "manifest.json", manifest)
    return path, {
        "kind": kind,
        "path": str(path.relative_to(workspace)),
        "manifest_sha256": _sha256(path / "manifest.json"),
    }


def _workspace(
    tmp_path: Path, *, control_count: int = 8, sparse: bool = False
) -> tuple[Path, PitPanel, dict[str, object]]:
    workspace = tmp_path
    panel = _panel(workspace, sparse=sparse)
    mining_index = panel.raw.index[
        panel.raw.index.get_level_values("datetime") <= pd.Timestamp("2021-12-31")
    ]
    mining_panel = replace(
        panel,
        raw=panel.raw.loc[mining_index],
        membership=panel.membership.loc[mining_index],
        observed={
            field: values.loc[mining_index] for field, values in panel.observed.items()
        },
        tradability=(
            None if panel.tradability is None else panel.tradability.loc[mining_index]
        ),
    )
    input_paths: dict[str, Path] = {}
    for arm in ARMS[1:]:
        input_paths[arm], _ = _input_artifact(
            workspace, arm, mining_panel, control_count=control_count
        )

    authority_paths = {
        "proposal": workspace / "KAN_Alpha_PR.md",
        "preregistration": workspace / "prereg" / "protocol.md",
        "directive": workspace / "governance" / "decisions" / "directive.md",
        "incident": workspace / "governance" / "incidents" / "incident.md",
    }
    for name, path in authority_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {name}\n", encoding="utf-8")

    artifact_paths = {
        "mining_run": "artifacts/mining",
        "implementation_lock": "prereg/implementation.lock.json",
        "kan_library": "factor_libraries/kan_e3_selected",
        "gp_control_library": "factor_libraries/typed_gp_sr_control",
        "blackbox_control": "controls/blackbox",
        "permutation_control_library": ("factor_libraries/kan_e3_permutation_control"),
        "mechanism_cards": "mechanism_cards/kan",
        "blind_review_package": "reviews/blind",
        "development_preclaim": "governance/openings/development_preclaim.json",
        "development_opening": "governance/openings/development.json",
        "development_recovery_receipt": "governance/recoveries/development.json",
        "evaluations": {arm: f"evaluations/{arm}" for arm in ARMS},
        "decision_artifact": "evaluations/decision",
    }
    config = {
        "protocol_id": "s2a_kan_e3_vertical_v8",
        "admission": {
            "minimum_library_size": 6,
            "library_cap": 16,
        },
        "authority_revalidation": {"boundaries": list(BOUNDARIES)},
        "controls": {"arms": list(ARMS)},
        "data": {
            "train": ["2021-12-30", "2021-12-30"],
            "validation": ["2021-12-31", "2021-12-31"],
            "development_test": ["2022-01-01", "2022-01-04"],
            "feature_warmup": {
                "trading_dates": 60,
                "raw_only": True,
                "labels_outside_objective_split_are_null": True,
            },
        },
        "artifact_paths": artifact_paths,
    }
    config_path = workspace / "configs" / "experiments" / "protocol.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    data_config_path = workspace / "configs" / "data" / "pit_cache.json"
    _write_json(
        data_config_path,
        {"cache_path": str(panel.source_path), "cache_sha256": panel.source_sha256},
    )
    base_lock = {
        "protocol": {
            "protocol_id": config["protocol_id"],
            "path": str(config_path.relative_to(workspace)),
            "sha256": _sha256(config_path),
        },
        "proposal": {
            "authority": "sole_proposal_authority",
            "path": "KAN_Alpha_PR.md",
            "sha256": _sha256(authority_paths["proposal"]),
        },
        "preregistration": {
            "path": str(authority_paths["preregistration"].relative_to(workspace)),
            "sha256": _sha256(authority_paths["preregistration"]),
        },
        "governance": {
            "active_directive_path": str(
                authority_paths["directive"].relative_to(workspace)
            ),
            "active_directive_sha256": _sha256(authority_paths["directive"]),
            "supersession_incident_path": str(
                authority_paths["incident"].relative_to(workspace)
            ),
            "supersession_incident_sha256": _sha256(authority_paths["incident"]),
        },
        "data": {
            "cache_path": str(panel.source_path),
            "cache_sha256": panel.source_sha256,
            "config_path": str(data_config_path.relative_to(workspace)),
            "config_sha256": _sha256(data_config_path),
        },
        "quanta": QUANTA,
    }
    base_lock_path = workspace / "prereg" / "s2a_kan_e3_vertical_v8.lock.json"
    _write_json(base_lock_path, base_lock)
    implementation_path = workspace / artifact_paths["implementation_lock"]
    _write_json(
        implementation_path,
        {
            "schema_version": "mirage_s2_implementation_lock_v2",
            "protocol_id": config["protocol_id"],
            "qlib_provider": PROVIDER,
        },
    )
    topology_sha256 = "8" * 64
    child_paths = {
        "kan_library": input_paths["kan_e3_selected"],
        "gp_control_library": input_paths["typed_gp_sr_control"],
        "permutation_control_library": input_paths["kan_e3_permutation_control"],
        "blackbox_control": input_paths["matched_blackbox_control"],
    }
    kan_manifest = json.loads(
        (child_paths["kan_library"] / "manifest.json").read_text(encoding="utf-8")
    )
    selected_ids = list(kan_manifest["factors"])
    lineage = {
        factor_id: {
            "canonical_hash": kan_manifest["factors"][factor_id]["canonical_hash"],
            "global_attempt_index": index,
        }
        for index, factor_id in enumerate(selected_ids)
    }
    cards_path = workspace / artifact_paths["mechanism_cards"]
    cards_path.mkdir(parents=True)
    cards_file = cards_path / "mechanism_cards.jsonl"
    cards_file.write_text(
        "".join(
            json.dumps(
                {
                    "factor_id": factor_id,
                    "card": {
                        "identity_and_canonical_ast": {
                            "factor_id": factor_id,
                            "canonical_hash": lineage[factor_id]["canonical_hash"],
                        }
                    },
                },
                sort_keys=True,
            )
            + "\n"
            for factor_id in selected_ids
        ),
        encoding="utf-8",
    )
    anonymous = cards_path / "blind_anonymous_mapping.json"
    _write_json(
        anonymous,
        {f"B{index:03d}": factor_id for index, factor_id in enumerate(selected_ids, 1)},
    )
    cards_manifest = {
        "schema_version": "mirage_s2a_v2_staging_bundle_v1",
        "role": "kan_mechanism_evidence_pending_human_review",
        "output_kind": "mechanism_cards",
        "kan_mined": True,
        "promotion_eligible": False,
        "card_count": len(selected_ids),
        "selected_factor_ids": selected_ids,
        "topology_key": "mechanism_cards",
        "topology_sha256": topology_sha256,
        "files": {
            cards_file.name: _sha256(cards_file),
            anonymous.name: _sha256(anonymous),
        },
    }
    _write_json(cards_path / "manifest.json", cards_manifest)
    child_paths["mechanism_cards"] = cards_path

    blind_path = workspace / artifact_paths["blind_review_package"]
    blind_path.mkdir(parents=True)
    blind_file = blind_path / "blind_review_package.json"
    _write_json(blind_file, {"items": len(selected_ids)})
    _write_json(
        blind_path / "manifest.json",
        {
            "schema_version": "mirage_s2a_v2_staging_bundle_v1",
            "role": "human_blind_review_input",
            "topology_key": "blind_review_package",
            "topology_sha256": topology_sha256,
            "files": {blind_file.name: _sha256(blind_file)},
        },
    )
    child_paths["blind_review_package"] = blind_path
    for child_key, child_path in child_paths.items():
        manifest_path = child_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["topology_key"] = child_key
        manifest["topology_sha256"] = topology_sha256
        _write_json(manifest_path, manifest)

    mining_directory = workspace / artifact_paths["mining_run"]
    mining_directory.mkdir(parents=True)
    children_file = mining_directory / "children.json"
    _write_json(children_file, sorted(child_paths))
    child_hashes = {
        key: _sha256(path / "manifest.json") for key, path in child_paths.items()
    }
    mining_path = mining_directory / "manifest.json"
    _write_json(
        mining_path,
        {
            "schema_version": "mirage_s2a_v2_staging_bundle_v1",
            "role": "mining_top_bundle",
            "topology_key": "mining_run",
            "topology_sha256": topology_sha256,
            "published_child_topology_sha256": topology_sha256,
            "identities": {
                "protocol_sha256": _sha256(config_path),
                "authority_sha256": "9" * 64,
                "implementation_sha256": _sha256(implementation_path),
            },
            "child_manifest_sha256": child_hashes,
            "published_child_paths": {
                key: str(path.resolve()) for key, path in child_paths.items()
            },
            "kan_selected_lineage": lineage,
            "mechanism_cards_manifest_sha256": child_hashes["mechanism_cards"],
            "files": {children_file.name: _sha256(children_file)},
        },
    )
    for relative in (
        "evaluations",
        "governance/openings",
        "governance/recoveries",
        "mechanism_cards",
    ):
        (workspace / relative).mkdir(parents=True, exist_ok=True)
    return (
        workspace,
        panel,
        {
            "base_lock_sha256": _sha256(base_lock_path),
            "implementation_lock_sha256": _sha256(implementation_path),
            "mining_manifest_sha256": _sha256(mining_path),
            "provider_identity": PROVIDER,
        },
    )


class FakeAdapter:
    def __init__(self, output_dir: Path, provider: dict[str, object]) -> None:
        self.output_dir = output_dir
        self.calls: list[object] = []
        self.identity = {
            "verified": True,
            **QUANTA,
            "effective_qlib_provider": provider["path"],
            "qlib_provider_tree_sha256": provider["tree_sha256"],
        }

    def initialize_and_verify_provider(self) -> None:
        self.calls.append("provider")

    def evaluate_alpha158(self, **kwargs):
        self.calls.append(("alpha158", kwargs))
        return {"information_ratio": 0.22, "rank_ic": 0.03311}

    def evaluate_panel(self, panel: pd.DataFrame, **kwargs):
        self.calls.append(("panel", panel.copy(), kwargs))
        return {"information_ratio": 0.21, "rank_ic": 0.032}

    def write_portfolio_diagnostics(self, destination: Path):
        assert destination == self.output_dir
        dates = pd.date_range("2022-01-03", periods=2)
        report = pd.DataFrame(
            {
                "return": [0.01, 0.02],
                "bench": [0.0, 0.01],
                "cost": [0.001, 0.001],
                "turnover": [0.2, 0.3],
            },
            index=dates,
        )
        daily = pd.DataFrame(
            {
                "daily_excess_return": [0.009, 0.009],
                "turnover": [0.2, 0.3],
                "realized_cost": [0.001, 0.001],
            },
            index=dates,
        )
        coverage = pd.DataFrame(
            {
                "total_predictions": [2, 2],
                "finite_predictions": [2, 2],
                "prediction_coverage": [1.0, 1.0],
            },
            index=dates,
        )
        frames = {
            "qlib_report.parquet": report,
            "portfolio_daily.parquet": daily,
            "prediction_coverage.parquet": coverage,
        }
        for filename, frame in frames.items():
            frame.to_parquet(destination / filename)
        return {filename: _sha256(destination / filename) for filename in frames}


class AdapterFactory:
    def __init__(self) -> None:
        self.adapters: list[FakeAdapter] = []

    def __call__(self, output_dir: Path, provider: dict[str, object]) -> FakeAdapter:
        adapter = FakeAdapter(output_dir, provider)
        self.adapters.append(adapter)
        return adapter


def _consume_development_opening(
    workspace: Path, panel: PitPanel, pins_payload: dict[str, object]
) -> tuple[AuthorityGuard, str, dict[str, object]]:
    from mirage_kan.artifacts.topology import TopologyTransaction
    from mirage_kan.evaluation.v2_runner import _calendar_sha256

    guard = AuthorityGuard(workspace)
    receipt = guard.revalidate("before_development_opening")
    topology = TopologyTransaction.from_frozen_config(workspace, phase="development")
    config = yaml.safe_load(
        (workspace / "configs" / "experiments" / "protocol.yaml").read_text()
    )
    dates = tuple(
        pd.Timestamp(value)
        for value in panel.raw.index.get_level_values("datetime").unique()
        if pd.Timestamp(value) >= pd.Timestamp("2022-01-01")
    )
    opening = {
        "schema_version": (
            "mirage_s2a_development_opening_v3"
            if "mining_source" in config
            else "mirage_s2a_development_opening_v2"
        ),
        "protocol_id": "s2a_kan_e3_vertical_v8",
        "state": "consumed_before_first_development_access",
        "topology_sha256": topology.topology_sha256,
        "evaluation_paths": config["artifact_paths"]["evaluations"],
        "identity_pins": {
            "base_lock_sha256": pins_payload["base_lock_sha256"],
            "implementation_lock_sha256": pins_payload["implementation_lock_sha256"],
            "mining_manifest_sha256": pins_payload["mining_manifest_sha256"],
            "provider_identity": pins_payload["provider_identity"],
        },
        "development_calendar_sha256": _calendar_sha256(dates),
        "development_calendar_count": len(dates),
        "development_calendar_start": dates[0].isoformat(),
        "development_calendar_end": dates[-1].isoformat(),
        "authority_receipt": {
            "receipt_sha256": receipt.receipt_sha256,
            "sequence": receipt.sequence,
            "boundary": receipt.boundary,
            "authority_sha256": receipt.authority_sha256,
            "base_lock_sha256": receipt.base_lock_sha256,
            "capability_sha256": hashlib.sha256(
                receipt.capability.encode("utf-8")
            ).hexdigest(),
        },
    }
    mining_source = config.get("mining_source")
    if isinstance(mining_source, dict):
        rebind_path = workspace / mining_source["rebind_receipt"]
        rebind_sha256 = _sha256(rebind_path)
        opening["mining_authorization_kind"] = "verified_cross_protocol_rebind"
        opening["identity_pins"]["mining_rebind_receipt_sha256"] = rebind_sha256
        updated_rebind = rebind_sha256
    else:
        updated_rebind = None
    opening_path = workspace / config["artifact_paths"]["development_opening"]
    _write_json(opening_path, opening)
    updated = {
        **pins_payload,
        "development_opening_sha256": _sha256(opening_path),
        "development_topology_sha256": topology.topology_sha256,
    }
    if updated_rebind is not None:
        updated["mining_rebind_receipt_sha256"] = updated_rebind
    return guard, receipt.capability, updated


def _run(
    workspace: Path,
    panel: PitPanel,
    pins_payload: dict[str, object],
    arm: str,
    factory: AdapterFactory,
):
    from mirage_kan.evaluation.v2_runner import (
        EvaluationIdentityPins,
        _stage_v2_arm_for_test,
    )

    guard, development_capability, pins_payload = _consume_development_opening(
        workspace, panel, pins_payload
    )
    receipt = guard.revalidate("before_each_scientific_or_control_arm", arm=arm)
    pins = EvaluationIdentityPins(**pins_payload)
    return _stage_v2_arm_for_test(
        workspace,
        arm=arm,
        panel=panel,
        pins=pins,
        authority_guard=guard,
        development_capability=development_capability,
        capability=receipt.capability,
        adapter_factory=factory,
        staging_parent=workspace / "staging",
    )


def _rebind_mining_child(
    workspace: Path,
    pins: dict[str, object],
    child_key: str,
    child_path: Path,
) -> None:
    mining_path = workspace / "artifacts" / "mining" / "manifest.json"
    mining = json.loads(mining_path.read_text(encoding="utf-8"))
    child_hash = _sha256(child_path / "manifest.json")
    mining["child_manifest_sha256"][child_key] = child_hash
    if child_key == "mechanism_cards":
        mining["mechanism_cards_manifest_sha256"] = child_hash
    _write_json(mining_path, mining)
    pins["mining_manifest_sha256"] = _sha256(mining_path)


@pytest.mark.parametrize(
    "arm",
    ("kan_e3_selected", "typed_gp_sr_control", "kan_e3_permutation_control"),
)
def test_factor_library_arms_are_verified_and_staged_for_topology_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    verified: list[Path] = []

    def fake_verify(path: Path, observed_panel: PitPanel):
        verified.append(Path(path).resolve())
        observed_dates = observed_panel.raw.index.get_level_values("datetime")
        assert observed_dates.max() == pd.Timestamp("2021-12-31")
        assert len(observed_panel.raw) < len(panel.raw)
        return {"verified": True, "factor_count": 8}

    monkeypatch.setattr("mirage_kan.evaluation.v2_runner.verify_library", fake_verify)
    factory = AdapterFactory()

    result = _run(workspace, panel, pins, arm, factory)

    assert verified == [(workspace / "factor_libraries" / arm).resolve()]
    adapter = factory.adapters[0]
    assert adapter.calls[0] == "provider"
    assert adapter.calls[1][0] == "panel"
    replayed = adapter.calls[1][1]
    assert replayed.index.equals(panel.raw.index)
    expected_close = panel.raw["close"]
    for factor_id in replayed:
        pd.testing.assert_series_equal(
            replayed[factor_id], expected_close.rename(factor_id), check_exact=True
        )
    assert adapter.calls[1][2]["capture_report"] is True
    assert result.topology_key == f"evaluation:{arm}"
    assert result.staging_path.parent == (workspace / "staging").resolve()
    assert not (workspace / "evaluations" / arm).exists()
    assert _sha256(result.staging_path / "manifest.json") == result.manifest_sha256
    assert set(result.manifest["files"]) == {
        "qlib_report.parquet",
        "portfolio_daily.parquet",
        "prediction_coverage.parquet",
    }


def test_alpha158_uses_only_the_official_adapter_path(tmp_path: Path) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    factory = AdapterFactory()

    result = _run(workspace, panel, pins, "alpha158_replay", factory)

    call = factory.adapters[0].calls[1]
    assert call[0] == "alpha158"
    assert call[1]["experiment_name"] == (
        "mirage_kan_s2a_kan_e3_vertical_v8_alpha158_replay"
    )
    assert result.manifest["input"]["kind"] == "official_alpha158"
    assert result.manifest["input"]["path"] is None


def test_blackbox_is_control_only_but_uses_computed_factor_evaluation(
    tmp_path: Path,
) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    factory = AdapterFactory()

    result = _run(workspace, panel, pins, "matched_blackbox_control", factory)

    call = factory.adapters[0].calls[1]
    assert call[0] == "panel"
    assert call[1].index.equals(panel.raw.index)
    assert call[1].columns.tolist() == [
        f"mlp_for_kan_{index:03d}" for index in range(8)
    ]
    assert result.manifest["input"]["kind"] == "computed_factor_control"
    assert result.manifest["input"]["factor_library"] is False
    assert result.manifest["input"]["promotion_eligible"] is False


def test_blackbox_checkpoint_replay_preserves_sparse_raw_pit_index(
    tmp_path: Path,
) -> None:
    workspace, panel, pins = _workspace(tmp_path, sparse=True)
    (workspace / "staging").mkdir()
    factory = AdapterFactory()

    result = _run(workspace, panel, pins, "matched_blackbox_control", factory)

    replayed = factory.adapters[0].calls[1][1]
    assert replayed.index.equals(panel.raw.index)
    assert result.manifest["input"]["control_count"] == 8


def test_blackbox_sparse_reindex_does_not_hide_checkpoint_value_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mirage_kan.evaluation import v2_runner

    workspace, panel, pins = _workspace(tmp_path, sparse=True)
    (workspace / "staging").mkdir()
    exact_replay = v2_runner.replay_control_on_atom_panel
    published = pd.read_parquet(
        workspace / "controls" / "blackbox" / "prediction_panel.parquet"
    )
    finite_index = published.index[np.isfinite(published.iloc[:, 0])][0]

    def mismatched_replay(control, atom_panel):
        replay = exact_replay(control, atom_panel)
        replay.loc[finite_index] += 1.0
        return replay

    monkeypatch.setattr(v2_runner, "replay_control_on_atom_panel", mismatched_replay)

    with pytest.raises(ValueError, match="exact checkpoint replay"):
        _run(
            workspace,
            panel,
            pins,
            "matched_blackbox_control",
            AdapterFactory(),
        )


def test_runner_resolves_rebound_source_before_quanta_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    config_path = workspace / "configs/experiments/protocol.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_paths = {
        key: config["artifact_paths"][key]
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
    source_protocol_sha256 = _sha256(config_path)
    rebind_relative = "governance/openings/mining_rebind.json"
    config["mining_source"] = {
        "mode": "verified_cross_protocol_rebind",
        "rebind_receipt": rebind_relative,
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    rebind_path = workspace / rebind_relative
    _write_json(rebind_path, {})
    base_path = workspace / "prereg/s2a_kan_e3_vertical_v8.lock.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base["protocol"]["sha256"] = _sha256(config_path)
    _write_json(base_path, base)
    pins["base_lock_sha256"] = _sha256(base_path)
    artifacts = {
        key: {
            "path": path,
            "manifest_sha256": _sha256(workspace / path / "manifest.json"),
            "files": {},
        }
        for key, path in source_paths.items()
    }
    binding = {
        "source": {
            "protocol_id": "s2a_kan_e3_vertical_v6_source",
            "config": {"sha256": source_protocol_sha256},
            "implementation_lock": {
                "sha256": pins["implementation_lock_sha256"]
            },
            "artifacts": artifacts,
        }
    }
    monkeypatch.setattr(
        "mirage_kan.evaluation.v2_runner.verify_mining_rebind_receipt",
        lambda *args, **kwargs: binding,
    )
    monkeypatch.setattr(
        "mirage_kan.evaluation.v2_runner.verify_library",
        lambda path, observed_panel: {"verified": True, "factor_count": 8},
    )
    factory = AdapterFactory()

    result = _run(workspace, panel, pins, "kan_e3_selected", factory)

    assert len(factory.adapters) == 1
    assert result.manifest["input"]["source_protocol_id"] == (
        "s2a_kan_e3_vertical_v6_source"
    )
    assert result.manifest["input"]["mining_rebind_receipt_sha256"] == _sha256(
        rebind_path
    )
    assert result.manifest["identity_pins"]["mining_rebind_receipt_sha256"] == (
        _sha256(rebind_path)
    )


@pytest.mark.parametrize("arm", ARMS)
def test_six_factor_v5_lineage_cross_links_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    workspace, panel, pins = _workspace(tmp_path, control_count=6)
    (workspace / "staging").mkdir()
    monkeypatch.setattr(
        "mirage_kan.evaluation.v2_runner.verify_library",
        lambda path, observed_panel: {"verified": True, "factor_count": 6},
    )

    result = _run(workspace, panel, pins, arm, AdapterFactory())

    assert result.manifest["arm"] == arm
    if arm in {
        "kan_e3_selected",
        "typed_gp_sr_control",
        "kan_e3_permutation_control",
    }:
        assert result.manifest["input"]["factor_count"] == 6
    elif arm == "matched_blackbox_control":
        assert result.manifest["input"]["control_count"] == 6


@pytest.mark.parametrize("control_count", (5, 17))
def test_runner_rejects_counts_outside_frozen_admission_bounds(
    tmp_path: Path, control_count: int
) -> None:
    workspace, panel, pins = _workspace(tmp_path, control_count=control_count)
    (workspace / "staging").mkdir()

    with pytest.raises(ValueError, match="frozen admission bounds"):
        _run(workspace, panel, pins, "alpha158_replay", AdapterFactory())


@pytest.mark.parametrize(
    "admission",
    (
        None,
        {},
        {"minimum_library_size": True, "library_cap": 16},
        {"minimum_library_size": 0, "library_cap": 16},
        {"minimum_library_size": 6, "library_cap": 5},
    ),
)
def test_frozen_library_size_bounds_fail_closed(admission: object) -> None:
    from mirage_kan.evaluation.v2_runner import _library_size_bounds

    with pytest.raises(ValueError, match="invalid library-size bounds|mapping"):
        _library_size_bounds({"admission": admission})


@pytest.mark.parametrize(
    ("arm", "child_key", "relative"),
    (
        ("kan_e3_selected", "kan_library", "factor_libraries/kan_e3_selected"),
        (
            "matched_blackbox_control",
            "blackbox_control",
            "controls/blackbox",
        ),
    ),
)
def test_mining_artifact_rejects_any_prediction_after_validation_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    child_key: str,
    relative: str,
) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    artifact = workspace / relative
    filename = (
        "prediction_panel.parquet"
        if arm == "matched_blackbox_control"
        else "factor_panel.parquet"
    )
    values = pd.read_parquet(artifact / filename)
    development_index = panel.raw.index[
        panel.raw.index.get_level_values("datetime") > pd.Timestamp("2021-12-31")
    ]
    leaked = pd.DataFrame(0.0, index=development_index, columns=values.columns)
    pd.concat([values, leaked]).sort_index().to_parquet(artifact / filename)
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename] = _sha256(artifact / filename)
    _write_json(manifest_path, manifest)
    _rebind_mining_child(workspace, pins, child_key, artifact)
    if arm != "matched_blackbox_control":
        monkeypatch.setattr(
            "mirage_kan.evaluation.v2_runner.verify_library",
            lambda path, observed_panel: {"verified": True, "factor_count": 8},
        )

    with pytest.raises(ValueError, match="after validation end"):
        _run(workspace, panel, pins, arm, AdapterFactory())


def test_expected_mining_index_keeps_exact_last_60_pretrain_dates(
    tmp_path: Path,
) -> None:
    from mirage_kan.evaluation.v2_runner import (
        _ExecutionContext,
        _mining_panel,
        _verify_mining_artifact_index,
    )

    pretrain_dates = pd.date_range("2020-09-01", periods=65, freq="B")
    train_and_later = pd.to_datetime(
        ["2021-01-04", "2021-01-05", "2021-02-01", "2022-01-03"]
    )
    dates = pretrain_dates.append(pd.DatetimeIndex(train_and_later))
    frame = pd.DataFrame(
        {
            "datetime": dates,
            "instrument": "A",
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
            "in_universe": True,
        }
    )
    panel = PitPanel.from_frame(frame)
    placeholder = tmp_path / "placeholder.json"
    _write_json(placeholder, {})
    context = _ExecutionContext(
        protocol_id="s2a_kan_e3_vertical_v8",
        config={
            "data": {
                "train": ["2021-01-01", "2021-01-31"],
                "validation": ["2021-02-01", "2021-12-31"],
                "feature_warmup": {
                    "trading_dates": 60,
                    "raw_only": True,
                    "labels_outside_objective_split_are_null": True,
                },
            }
        },
        base_lock={},
        base_lock_path=placeholder,
        implementation_lock_path=placeholder,
        mining_manifest_path=placeholder,
        mining_manifest={},
        development_opening_path=placeholder,
        development_opening={},
        development_dates=(pd.Timestamp("2022-01-03"),),
        provider_identity={},
        mining_child_paths={},
        mining_source_protocol_id="s2a_kan_e3_vertical_v8",
        mining_rebind_receipt_sha256=None,
    )

    mining, validation_end = _mining_panel(panel, context)
    observed_dates = pd.DatetimeIndex(
        mining.raw.index.get_level_values("datetime").unique()
    )
    expected_dates = pretrain_dates[-60:].append(pd.DatetimeIndex(train_and_later[:3]))
    assert observed_dates.equals(expected_dates)
    _verify_mining_artifact_index(
        pd.DataFrame({"factor": 0.0}, index=mining.raw.index),
        mining,
        validation_end,
        label="test artifact",
    )
    too_early = panel.raw.loc[
        panel.raw.index.get_level_values("datetime") <= validation_end,
        ["close"],
    ].rename(columns={"close": "factor"})
    with pytest.raises(ValueError, match="exactly cover"):
        _verify_mining_artifact_index(
            too_early,
            mining,
            validation_end,
            label="test artifact",
        )


def test_library_count_selected_ids_and_ast_lineage_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    library = workspace / "factor_libraries" / "kan_e3_selected"
    manifest_path = library / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["factor_count"] = 7
    _write_json(manifest_path, manifest)
    _rebind_mining_child(workspace, pins, "kan_library", library)
    monkeypatch.setattr(
        "mirage_kan.evaluation.v2_runner.verify_library",
        lambda path, observed_panel: {"verified": True, "factor_count": 8},
    )

    with pytest.raises(ValueError, match="factor count or selected IDs"):
        _run(workspace, panel, pins, "kan_e3_selected", AdapterFactory())


def test_blackbox_pairing_seed_and_bootstrap_receipts_fail_closed(
    tmp_path: Path,
) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    control = workspace / "controls" / "blackbox"
    receipts_path = control / "control_receipts.jsonl"
    rows = receipts_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["seed"] += 1
    rows[0] = json.dumps(first, sort_keys=True)
    receipts_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest_path = control / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][receipts_path.name] = _sha256(receipts_path)
    _write_json(manifest_path, manifest)
    _rebind_mining_child(workspace, pins, "blackbox_control", control)

    with pytest.raises(ValueError, match="pairing or seed"):
        _run(
            workspace,
            panel,
            pins,
            "matched_blackbox_control",
            AdapterFactory(),
        )


def test_blackbox_pairing_accepts_valid_mapping_in_selection_order() -> None:
    from mirage_kan.evaluation.v2_runner import _blackbox_pairing_matches_lineage

    lineage = {
        "kan_a": {"global_attempt_index": 20},
        "kan_b": {"global_attempt_index": 215},
        "kan_c": {"global_attempt_index": 188},
    }

    assert _blackbox_pairing_matches_lineage(
        ["kan_c", "kan_a", "kan_b"],
        [188, 20, 215],
        lineage,
    )


def test_blackbox_pairing_rejects_crossed_global_attempt_indices() -> None:
    from mirage_kan.evaluation.v2_runner import _blackbox_pairing_matches_lineage

    lineage = {
        "kan_a": {"global_attempt_index": 20},
        "kan_b": {"global_attempt_index": 215},
        "kan_c": {"global_attempt_index": 188},
    }

    assert not _blackbox_pairing_matches_lineage(
        ["kan_c", "kan_a", "kan_b"],
        [188, 215, 20],
        lineage,
    )


def test_mechanism_cards_must_match_selected_kan_lineage(tmp_path: Path) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    cards = workspace / "mechanism_cards" / "kan"
    manifest_path = cards / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_factor_ids"] = list(reversed(manifest["selected_factor_ids"]))
    _write_json(manifest_path, manifest)
    _rebind_mining_child(workspace, pins, "mechanism_cards", cards)

    with pytest.raises(ValueError, match="mechanism cards"):
        _run(workspace, panel, pins, "alpha158_replay", AdapterFactory())


def test_capability_is_verified_before_adapter_creation(tmp_path: Path) -> None:
    from mirage_kan.evaluation.v2_runner import (
        EvaluationIdentityPins,
        _stage_v2_arm_for_test,
    )

    workspace, panel, payload = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    guard, development_capability, payload = _consume_development_opening(
        workspace, panel, payload
    )
    receipt = guard.revalidate(
        "before_each_scientific_or_control_arm", arm="kan_e3_selected"
    )
    factory = AdapterFactory()

    with pytest.raises(ValueError, match="boundary or arm"):
        _stage_v2_arm_for_test(
            workspace,
            arm="typed_gp_sr_control",
            panel=panel,
            pins=EvaluationIdentityPins(**payload),
            authority_guard=guard,
            development_capability=development_capability,
            capability=receipt.capability,
            adapter_factory=factory,
            staging_parent=workspace / "staging",
        )

    assert factory.adapters == []


def test_public_production_entrypoint_has_no_adapter_injection_seam() -> None:
    import inspect

    from mirage_kan.evaluation.v2_runner import stage_v2_arm

    assert "adapter_factory" not in inspect.signature(stage_v2_arm).parameters


def test_opening_hash_and_development_capability_are_both_required(
    tmp_path: Path,
) -> None:
    from mirage_kan.evaluation.v2_runner import (
        EvaluationIdentityPins,
        _stage_v2_arm_for_test,
    )

    workspace, panel, payload = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    guard, development_capability, payload = _consume_development_opening(
        workspace, panel, payload
    )
    arm_receipt = guard.revalidate(
        "before_each_scientific_or_control_arm", arm="alpha158_replay"
    )
    factory = AdapterFactory()
    with pytest.raises(ValueError, match="boundary or arm"):
        _stage_v2_arm_for_test(
            workspace,
            arm="alpha158_replay",
            panel=panel,
            pins=EvaluationIdentityPins(**payload),
            authority_guard=guard,
            development_capability=arm_receipt.capability,
            capability=arm_receipt.capability,
            adapter_factory=factory,
            staging_parent=workspace / "staging",
        )
    assert factory.adapters == []

    opening = workspace / "governance" / "openings" / "development.json"
    opening.write_bytes(opening.read_bytes() + b"x")
    with pytest.raises(ValueError, match="development opening identity"):
        _stage_v2_arm_for_test(
            workspace,
            arm="alpha158_replay",
            panel=panel,
            pins=EvaluationIdentityPins(**payload),
            authority_guard=guard,
            development_capability=development_capability,
            capability=arm_receipt.capability,
            adapter_factory=factory,
            staging_parent=workspace / "staging",
        )
    assert factory.adapters == []


def test_arm_capability_is_atomically_consumed_once_even_when_execution_fails(
    tmp_path: Path,
) -> None:
    from mirage_kan.evaluation.v2_runner import (
        EvaluationIdentityPins,
        _stage_v2_arm_for_test,
    )

    workspace, panel, payload = _workspace(tmp_path)
    staging_parent = workspace / "staging"
    staging_parent.mkdir()
    guard, development_capability, payload = _consume_development_opening(
        workspace, panel, payload
    )
    receipt = guard.revalidate(
        "before_each_scientific_or_control_arm", arm="alpha158_replay"
    )
    pins = EvaluationIdentityPins(**payload)
    factory = AdapterFactory()
    _stage_v2_arm_for_test(
        workspace,
        arm="alpha158_replay",
        panel=panel,
        pins=pins,
        authority_guard=guard,
        development_capability=development_capability,
        capability=receipt.capability,
        adapter_factory=factory,
        staging_parent=staging_parent,
    )
    with pytest.raises(PermissionError, match="already consumed"):
        _stage_v2_arm_for_test(
            workspace,
            arm="alpha158_replay",
            panel=panel,
            pins=pins,
            authority_guard=guard,
            development_capability=development_capability,
            capability=receipt.capability,
            adapter_factory=factory,
            staging_parent=staging_parent,
        )
    assert len(factory.adapters) == 1

    second, second_panel, second_payload = _workspace(tmp_path / "failed")
    (second / "staging").mkdir()
    second_guard, second_development, second_payload = _consume_development_opening(
        second, second_panel, second_payload
    )
    second_receipt = second_guard.revalidate(
        "before_each_scientific_or_control_arm", arm="alpha158_replay"
    )

    def failing_factory(output_dir, provider):
        raise RuntimeError("adapter construction failed")

    arguments = {
        "arm": "alpha158_replay",
        "panel": second_panel,
        "pins": EvaluationIdentityPins(**second_payload),
        "authority_guard": second_guard,
        "development_capability": second_development,
        "capability": second_receipt.capability,
        "adapter_factory": failing_factory,
        "staging_parent": second / "staging",
    }
    with pytest.raises(RuntimeError, match="adapter construction failed"):
        _stage_v2_arm_for_test(second, **arguments)
    with pytest.raises(PermissionError, match="already consumed"):
        _stage_v2_arm_for_test(second, **arguments)


def test_live_cache_and_exact_panel_replay_cannot_be_forged_from_source_attrs(
    tmp_path: Path,
) -> None:
    workspace, panel, payload = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    forged_raw = panel.raw.copy()
    forged_raw.iloc[0, 0] += 1.0
    forged = replace(panel, raw=forged_raw)

    with pytest.raises(ValueError, match="panel content"):
        _run(workspace, forged, payload, "alpha158_replay", AdapterFactory())

    workspace, panel, payload = _workspace(tmp_path / "live-drift")
    (workspace / "staging").mkdir()
    panel.source_path.write_bytes(panel.source_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="live PIT cache hash"):
        _run(workspace, panel, payload, "alpha158_replay", AdapterFactory())


def test_diagnostic_calendar_must_exactly_equal_consumed_development_calendar(
    tmp_path: Path,
) -> None:
    workspace, panel, payload = _workspace(tmp_path)
    staging_parent = workspace / "staging"
    staging_parent.mkdir()

    class ShortCalendarAdapter(FakeAdapter):
        def write_portfolio_diagnostics(self, destination):
            files = super().write_portfolio_diagnostics(destination)
            daily_path = destination / "portfolio_daily.parquet"
            pd.read_parquet(daily_path).iloc[:1].to_parquet(daily_path)
            files["portfolio_daily.parquet"] = _sha256(daily_path)
            return files

    class ShortCalendarFactory(AdapterFactory):
        def __call__(self, output_dir, provider):
            adapter = ShortCalendarAdapter(output_dir, provider)
            self.adapters.append(adapter)
            return adapter

    with pytest.raises(ValueError, match="exact development calendar"):
        _run(
            workspace,
            panel,
            payload,
            "alpha158_replay",
            ShortCalendarFactory(),
        )
    assert list(staging_parent.iterdir()) == []


def test_nonfinite_quanta_diagnostic_fails_closed(tmp_path: Path) -> None:
    workspace, panel, payload = _workspace(tmp_path)
    staging_parent = workspace / "staging"
    staging_parent.mkdir()

    class NonfiniteAdapter(FakeAdapter):
        def write_portfolio_diagnostics(self, destination):
            files = super().write_portfolio_diagnostics(destination)
            report_path = destination / "qlib_report.parquet"
            report = pd.read_parquet(report_path)
            report.iloc[0, report.columns.get_loc("turnover")] = np.nan
            report.to_parquet(report_path)
            files["qlib_report.parquet"] = _sha256(report_path)
            return files

    class NonfiniteFactory(AdapterFactory):
        def __call__(self, output_dir, provider):
            adapter = NonfiniteAdapter(output_dir, provider)
            self.adapters.append(adapter)
            return adapter

    with pytest.raises(ValueError, match="non-finite diagnostics"):
        _run(workspace, panel, payload, "alpha158_replay", NonfiniteFactory())
    assert list(staging_parent.iterdir()) == []


@pytest.mark.parametrize(
    ("identity", "match"),
    (
        ("base", "base.?lock"),
        ("implementation", "implementation lock"),
        ("mining", "mining manifest"),
    ),
)
def test_identity_drift_fails_before_quanta(
    tmp_path: Path, identity: str, match: str
) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    paths = {
        "base": workspace / "prereg" / "s2a_kan_e3_vertical_v8.lock.json",
        "implementation": workspace / "prereg" / "implementation.lock.json",
        "mining": workspace / "artifacts" / "mining" / "manifest.json",
    }
    paths[identity].write_bytes(paths[identity].read_bytes() + b"x")
    factory = AdapterFactory()

    with pytest.raises((ValueError, AuthoritySuperseded), match=match):
        _run(workspace, panel, pins, "alpha158_replay", factory)

    assert factory.adapters == []


def test_provider_and_quanta_identity_mismatches_fail_closed(tmp_path: Path) -> None:
    workspace, panel, payload = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    factory = AdapterFactory()
    bad_provider = dict(payload)
    bad_provider["provider_identity"] = {**PROVIDER, "tree_sha256": "9" * 64}

    with pytest.raises(ValueError, match="provider identity"):
        _run(workspace, panel, bad_provider, "alpha158_replay", factory)
    assert factory.adapters == []

    class BadQuantaFactory(AdapterFactory):
        def __call__(self, output_dir, provider):
            adapter = super().__call__(output_dir, provider)
            adapter.identity["runner_sha256"] = "0" * 64
            return adapter

    workspace, panel, payload = _workspace(tmp_path / "bad_quanta")
    (workspace / "staging").mkdir()
    bad_factory = BadQuantaFactory()
    with pytest.raises(ValueError, match="Quanta runner"):
        _run(workspace, panel, payload, "alpha158_replay", bad_factory)


def test_unregistered_input_and_control_masquerade_fail_closed(tmp_path: Path) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    (workspace / "staging").mkdir()
    mining_path = workspace / "artifacts" / "mining" / "manifest.json"
    mining = json.loads(mining_path.read_text(encoding="utf-8"))
    mining["child_manifest_sha256"]["blackbox_control"] = "0" * 64
    _write_json(mining_path, mining)
    pins["mining_manifest_sha256"] = _sha256(mining_path)

    with pytest.raises(ValueError, match="child manifest identity"):
        _run(
            workspace,
            panel,
            pins,
            "matched_blackbox_control",
            AdapterFactory(),
        )

    _, second_panel, pins = _workspace(tmp_path / "second")
    second = tmp_path / "second"
    (second / "staging").mkdir()
    manifest_path = second / "controls" / "blackbox" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["factor_library_publication_allowed"] = True
    _write_json(manifest_path, manifest)
    mining_path = second / "artifacts" / "mining" / "manifest.json"
    mining = json.loads(mining_path.read_text(encoding="utf-8"))
    mining["child_manifest_sha256"]["blackbox_control"] = _sha256(manifest_path)
    _write_json(mining_path, mining)
    pins["mining_manifest_sha256"] = _sha256(mining_path)
    with pytest.raises(ValueError, match="factor library"):
        _run(
            second,
            second_panel,
            pins,
            "matched_blackbox_control",
            AdapterFactory(),
        )


def test_failed_or_nonflat_quanta_run_leaves_no_staging_artifact(
    tmp_path: Path,
) -> None:
    workspace, panel, pins = _workspace(tmp_path)
    staging_parent = workspace / "staging"
    staging_parent.mkdir()

    class NonFlatAdapter(FakeAdapter):
        def evaluate_alpha158(self, **kwargs):
            result = super().evaluate_alpha158(**kwargs)
            (self.output_dir / "nested").mkdir()
            return result

    class NonFlatFactory(AdapterFactory):
        def __call__(self, output_dir, provider):
            adapter = NonFlatAdapter(output_dir, provider)
            self.adapters.append(adapter)
            return adapter

    with pytest.raises(ValueError, match="flat regular files"):
        _run(workspace, panel, pins, "alpha158_replay", NonFlatFactory())

    assert list(staging_parent.iterdir()) == []
