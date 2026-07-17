from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


ARMS = (
    "alpha158_replay",
    "kan_e3_selected",
    "typed_gp_sr_control",
    "matched_blackbox_control",
    "kan_e3_permutation_control",
)


@pytest.fixture(autouse=True)
def _verified_implementation_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mirage_kan.evaluation.v2_decision_assembler.verify_implementation_lock",
        lambda workspace: {"protocol_id": "s2a_kan_e3_vertical_v8"},
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _calendar_hash(calendar: pd.DatetimeIndex) -> str:
    return hashlib.sha256(
        json.dumps(
            [value.isoformat() for value in calendar], separators=(",", ":")
        ).encode()
    ).hexdigest()


def _ir(values: np.ndarray) -> float:
    return float(np.sqrt(252.0) * values.mean() / values.std(ddof=1))


def _publish_flat(
    path: Path, manifest: dict[str, object], files: dict[str, bytes]
) -> None:
    path.mkdir(parents=True)
    hashes: dict[str, str] = {}
    for name, body in files.items():
        (path / name).write_bytes(body)
        hashes[name] = _sha256(path / name)
    manifest["files"] = hashes
    _write_json(path / "manifest.json", manifest)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    import io

    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return buffer.getvalue()


def _protocol(
    *, minimum_library_size: int = 8, integrity_minimum: int | None = None
) -> dict[str, object]:
    if integrity_minimum is None:
        integrity_minimum = minimum_library_size
    return {
        "claim_boundary": {"graph_unlock_allowed": True},
        "interpretability": {
            "human_blind_review": {
                "reviewers_minimum": 2,
                "mechanism_restatement_required": True,
                "response_direction_accuracy_minimum": 0.8,
                "inter_reviewer_agreement_reported": True,
            }
        },
        "evaluation": {
            "historical_anchor": {},
            "replay_tolerances": {
                "information_ratio_absolute": 1e-12,
                "max_drawdown_absolute": 1e-12,
                "rank_ic_absolute": 1e-12,
            },
            "bootstrap": {
                "seed": 17,
                "block_length_trading_days": 2,
                "replicates": 20,
                "confidence_level": 0.95,
            },
        },
        "admission": {
            "minimum_library_size": minimum_library_size,
            "library_cap": 16,
        },
        "s2a_decision": {
            "performance_noninferiority": {
                "kan_minus_alpha158_delta_ir_lcb_minimum": -100.0,
                "kan_minus_alpha158_rank_ic_minimum": -1.0,
                "kan_minus_alpha158_max_drawdown_minimum": -1.0,
                "kan_to_alpha158_mean_turnover_ratio_maximum": 2.0,
                "kan_to_alpha158_mean_cost_ratio_maximum": 2.0,
                "calendar_years_not_worse_than_alpha158_minimum": 0,
            },
            "integrity": {
                "production_library_size_minimum": integrity_minimum,
                "production_library_profiles_minimum": 3,
                "production_factors_strict_fraction": 1.0,
                "production_factors_with_mechanism_card_fraction": 1.0,
                "permutation_false_positive_count_maximum": 0,
                "alpha158_replay_must_match_anchor": True,
            },
            "method_falsification": {
                "numerical_tolerance": {
                    "delta_ir_lcb": 1e-12,
                    "effective_rank": 1e-12,
                }
            },
        },
    }


def _workspace(
    tmp_path: Path,
    *,
    factor_count: int = 8,
    minimum_library_size: int | None = None,
    integrity_minimum: int | None = None,
) -> tuple[Path, pd.DatetimeIndex]:
    if minimum_library_size is None:
        minimum_library_size = factor_count
    workspace = tmp_path / "workspace"
    for parent in (
        "configs/experiments",
        "prereg",
        "artifacts",
        "factor_libraries",
        "controls",
        "mechanism_cards",
        "reviews",
        "evaluations",
        "governance/openings",
        "governance/recoveries",
    ):
        (workspace / parent).mkdir(parents=True, exist_ok=True)

    evaluation_paths = {arm: f"evaluations/{arm}" for arm in ARMS}
    artifact_paths = {
        "mining_run": "artifacts/mining",
        "kan_library": "factor_libraries/kan",
        "gp_control_library": "factor_libraries/gp",
        "permutation_control_library": "factor_libraries/permutation",
        "blackbox_control": "controls/blackbox",
        "mechanism_cards": "mechanism_cards/cards",
        "blind_review_package": "reviews/blind",
        "implementation_lock": "prereg/implementation.json",
        "development_preclaim": "governance/openings/development_preclaim.json",
        "development_opening": "governance/openings/development.json",
        "mining_preclaim": "governance/openings/mining_preclaim.json",
        "mining_entitlement": "governance/openings/mining.json",
        "mining_recovery_receipt": "governance/recoveries/mining.json",
        "development_recovery_receipt": "governance/recoveries/development.json",
        "evaluations": evaluation_paths,
        "decision_artifact": "evaluations/decision",
    }
    config = {
        "protocol_id": "s2a_kan_e3_vertical_v8",
        "data": {
            "validation": ["2021-01-01", "2021-12-31"],
            "development_test": ["2022-01-01", "2025-12-31"],
        },
        "diversity_metrics": {
            "selected_library_effective_rank": {"minimum_joint_rows": 2}
        },
        "artifact_paths": artifact_paths,
        **_protocol(
            minimum_library_size=minimum_library_size,
            integrity_minimum=integrity_minimum,
        ),
    }
    config_path = workspace / "configs/experiments/protocol.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    implementation_path = workspace / "prereg/implementation.json"
    _write_json(implementation_path, {"schema_version": "implementation"})
    base_lock = {
        "protocol": {
            "path": "configs/experiments/protocol.yaml",
            "protocol_id": config["protocol_id"],
            "sha256": _sha256(config_path),
        },
        "data": {"cache_sha256": "1" * 64},
        "baseline_metric": {"anchor": "frozen"},
    }
    lock_path = workspace / "prereg/s2a_kan_e3_vertical_v8.lock.json"
    _write_json(lock_path, base_lock)

    from mirage_kan.artifacts.topology import TopologyTransaction

    topology = TopologyTransaction.from_frozen_config(workspace, phase="development")
    topology.preclaim()
    topology.claim_all()

    identities = {
        "protocol_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(implementation_path),
        "authority_sha256": "a" * 64,
    }
    from mirage_kan.dsl import AstNode
    from mirage_kan.mining.e3 import build_profile_atom_bank

    factor_ids = [f"factor_{index:02d}" for index in range(factor_count)]
    programs = {}
    for index_value, factor_id in enumerate(factor_ids):
        profile = ("short_price", "long_price", "reversal", "price_volume")[
            index_value % 4
        ]
        bank = build_profile_atom_bank(profile)
        left, right = (
            (bank[0].ast, bank[1].ast)
            if index_value < 4
            else (bank[1].ast, bank[0].ast)
        )
        programs[factor_id] = AstNode(
            "Sub",
            (left, right),
        )
    canonical = {factor_id: program.identity for factor_id, program in programs.items()}
    validation_dates = pd.date_range("2021-01-04", periods=4, freq="B")
    index = pd.MultiIndex.from_product(
        [validation_dates, ["A", "B"]], names=["datetime", "instrument"]
    )
    panel = pd.DataFrame(
        {
            factor_id: np.tile([-(index_value + 1.0), index_value + 1.0], 4)
            for index_value, factor_id in enumerate(factor_ids)
        },
        index=index,
    )

    child_paths = {
        "kan_library": workspace / artifact_paths["kan_library"],
        "gp_control_library": workspace / artifact_paths["gp_control_library"],
        "permutation_control_library": workspace
        / artifact_paths["permutation_control_library"],
        "blackbox_control": workspace / artifact_paths["blackbox_control"],
        "mechanism_cards": workspace / artifact_paths["mechanism_cards"],
        "blind_review_package": workspace / artifact_paths["blind_review_package"],
    }
    mining_topology = "b" * 64
    for key in ("kan_library", "gp_control_library", "permutation_control_library"):
        role = {
            "kan_library": "kan_e3_selected",
            "gp_control_library": "typed_gp_sr_control",
            "permutation_control_library": "kan_e3_permutation_control",
        }[key]
        factors = {
            factor_id: {
                "canonical_hash": canonical[factor_id],
                "ast": programs[factor_id].to_dict(),
                **(
                    {"global_attempt_index": index_value}
                    if key == "kan_library"
                    else {}
                ),
            }
            for index_value, factor_id in enumerate(factor_ids)
        }
        _publish_flat(
            child_paths[key],
            {
                "schema_version": "mirage_factor_library_v1",
                "role": role,
                "library_role": role,
                "output_kind": "factor_library",
                "kan_mined": key == "kan_library",
                "promotion_eligible": key == "kan_library",
                "factor_count": factor_count,
                "selected_candidate_ids": factor_ids,
                "factors": factors,
                "identities": identities,
                "topology_sha256": mining_topology,
                "topology_key": key,
            },
            {"factor_panel.parquet": _parquet_bytes(panel)},
        )
    _publish_flat(
        child_paths["blackbox_control"],
        {
            "schema_version": "mirage_matched_blackbox_control_v2",
            "role": "falsification_control_never_production",
            "output_kind": "control_panel_not_factor_library",
            "kan_mined": False,
            "promotion_eligible": False,
            "factor_library_publication_allowed": False,
            "control_count": factor_count,
            "selected_kan_factor_ids": factor_ids,
            "paired_kan_global_attempt_indices": list(range(factor_count)),
            "identities": identities,
            "topology_sha256": mining_topology,
            "topology_key": "blackbox_control",
        },
        {"prediction_panel.parquet": _parquet_bytes(panel)},
    )
    mapping = {
        f"B{index + 1:03d}": factor_id for index, factor_id in enumerate(factor_ids)
    }
    card_rows = [
        {
            "factor_id": factor_id,
            "card": {
                "identity_and_canonical_ast": {
                    "factor_id": factor_id,
                    "canonical_hash": canonical[factor_id],
                }
            },
        }
        for factor_id in factor_ids
    ]
    mapping_bytes = (json.dumps(mapping, indent=2, sort_keys=True) + "\n").encode()
    mapping_hash = hashlib.sha256(mapping_bytes).hexdigest()
    _publish_flat(
        child_paths["mechanism_cards"],
        {
            "schema_version": "mirage_s2a_v2_staging_bundle_v1",
            "role": "kan_mechanism_evidence_pending_human_review",
            "output_kind": "mechanism_cards",
            "kan_mined": True,
            "promotion_eligible": False,
            "card_count": factor_count,
            "selected_factor_ids": factor_ids,
            "anonymous_mapping_sha256": mapping_hash,
            "identities": identities,
            "topology_sha256": mining_topology,
            "topology_key": "mechanism_cards",
        },
        {
            "mechanism_cards.jsonl": b"".join(
                (json.dumps(row, sort_keys=True) + "\n").encode() for row in card_rows
            ),
            "blind_anonymous_mapping.json": mapping_bytes,
        },
    )
    blind_package = {
        "review_status": "pending_human_review",
        "items": {blind_id: {"question": "direction?"} for blind_id in mapping},
    }
    _publish_flat(
        child_paths["blind_review_package"],
        {
            "schema_version": "mirage_s2a_v2_staging_bundle_v1",
            "role": "human_blind_review_input",
            "output_kind": "blind_review_package",
            "kan_mined": False,
            "promotion_eligible": False,
            "blind_item_count": factor_count,
            "anonymous_mapping_sha256": mapping_hash,
            "identities": identities,
            "topology_sha256": mining_topology,
            "topology_key": "blind_review_package",
        },
        {"blind_review_package.json": (json.dumps(blind_package) + "\n").encode()},
    )

    def scoring_rows(
        method: str, *, eligible_not_selected: bool = False
    ) -> list[dict[str, object]]:
        rows = []
        for index_value in range(256):
            selected = index_value < factor_count
            eligible = selected or (
                eligible_not_selected and index_value == factor_count
            )
            factor_id = (
                factor_ids[index_value] if selected else f"{method}_{index_value:03d}"
            )
            rows.append(
                {
                    "candidate_id": factor_id,
                    "profile": (
                        "short_price",
                        "long_price",
                        "reversal",
                        "price_volume",
                    )[index_value % 4],
                    "attempt_index": index_value,
                    "method": method,
                    "canonical_hash": canonical[factor_id]
                    if selected
                    else hashlib.sha256(factor_id.encode()).hexdigest(),
                    "ast": programs[factor_id].to_dict()
                    if selected
                    else programs[factor_ids[index_value % factor_count]].to_dict(),
                    "ast_depth": 3,
                    "ast_nodes": 7,
                    "output_type": "dimensionless_ts",
                    "causal": True,
                    "unique": True,
                    "fidelity_gate_met": method != "typed_gp_sr" or selected,
                    "lineage_gate_met": True,
                    "production_eligible": eligible,
                    "production_disposition": "production_eligible"
                    if eligible
                    else "below_threshold",
                }
            )
        return rows

    kan_rows = scoring_rows("kan_e3", eligible_not_selected=True)
    gp_rows = scoring_rows("typed_gp_sr")
    kan_selection = {
        "selected_candidate_ids": factor_ids,
        "dispositions": {
            row["candidate_id"]: (
                "selected"
                if index_value < factor_count
                else (
                    "eligible_not_selected_cap"
                    if index_value == factor_count
                    else row["production_disposition"]
                )
            )
            for index_value, row in enumerate(kan_rows)
        },
        "minimum_size_met": True,
        "exact_size_met": True,
        "profile_quota_met": True,
        "target_size": None,
    }
    gp_selection = {
        "selected_candidate_ids": factor_ids,
        "dispositions": {
            row["candidate_id"]: (
                "selected"
                if index_value < factor_count
                else row["production_disposition"]
            )
            for index_value, row in enumerate(gp_rows)
        },
        "minimum_size_met": True,
        "exact_size_met": True,
        "profile_quota_met": True,
        "target_size": factor_count,
    }
    profile_rows = [
        {
            "global_attempt_index": index_value,
            "profile": row["profile"],
            "candidate_ast_sha256": row["canonical_hash"],
            "admission_failures": [],
        }
        for index_value, row in enumerate(kan_rows)
    ]
    permutation_rows = [
        {"global_attempt_index": index_value, "production_eligible": False}
        for index_value in range(256)
    ]
    mining_files = {
        "kan_real_scoring.jsonl": b"".join(
            (json.dumps(row) + "\n").encode() for row in kan_rows
        ),
        "gp_scoring.jsonl": b"".join(
            (json.dumps(row) + "\n").encode() for row in gp_rows
        ),
        "kan_real_selection.json": (
            json.dumps(kan_selection, sort_keys=True) + "\n"
        ).encode(),
        "gp_selection.json": (json.dumps(gp_selection, sort_keys=True) + "\n").encode(),
        "kan_profile_runs.jsonl": b"".join(
            (json.dumps(row) + "\n").encode() for row in profile_rows
        ),
        "kan_permutation_false_positive_ledger.jsonl": b"".join(
            (json.dumps(row) + "\n").encode() for row in permutation_rows
        ),
    }
    lineage = {
        factor_id: {
            "canonical_hash": canonical[factor_id],
            "global_attempt_index": index_value,
        }
        for index_value, factor_id in enumerate(factor_ids)
    }
    mining_path = workspace / artifact_paths["mining_run"]
    _publish_flat(
        mining_path,
        {
            "schema_version": "mirage_s2a_v2_staging_bundle_v1",
            "role": "mining_top_bundle",
            "promotion_eligible": False,
            "kan_mined": True,
            "identities": identities,
            "topology_sha256": mining_topology,
            "topology_key": "mining_run",
            "published_child_topology_sha256": mining_topology,
            "published_child_paths": {
                key: str(path) for key, path in child_paths.items()
            },
            "child_manifest_sha256": {
                key: _sha256(path / "manifest.json")
                for key, path in child_paths.items()
            },
            "mechanism_cards_manifest_sha256": _sha256(
                child_paths["mechanism_cards"] / "manifest.json"
            ),
            "kan_selected_lineage": lineage,
            "kan_real_ledger": {
                "scoring_rows": 256,
                "selected_count": factor_count,
            },
            "gp_ledger": {"scoring_rows": 256, "selected_count": factor_count},
            "permutation_ledger": {
                "real_threshold_false_positive_count": 0,
                "real_threshold_false_positive_rows": 256,
            },
        },
        mining_files,
    )

    calendar = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{year}-01-03") + pd.offsets.BDay(offset)
            for year in range(2022, 2026)
            for offset in range(2)
        ]
    )
    wave = np.array([-0.001, 0.002] * 4)
    means = dict(zip(ARMS, (0.0002, 0.0004, 0.00025, 0.0003, 0.00005), strict=True))
    metrics_by_arm: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        values = wave + means[arm]
        daily = pd.DataFrame(
            {"daily_excess_return": values, "turnover": 0.1, "realized_cost": 0.001},
            index=calendar,
        )
        metrics = {
            "information_ratio": _ir(values),
            "max_drawdown": -0.1,
            "Rank IC": 0.03,
        }
        metrics_by_arm[arm] = metrics
        path = topology.targets[f"evaluation:{arm}"]
        (path / ".INCOMPLETE").unlink()
        portfolio = _parquet_bytes(daily)
        (path / "portfolio_daily.parquet").write_bytes(portfolio)
        file_hash = _sha256(path / "portfolio_daily.parquet")
        manifest = {
            "schema_version": "mirage_s2a_quanta_evaluation_v2",
            "protocol_id": config["protocol_id"],
            "arm": arm,
            "topology_sha256": topology.topology_sha256,
            "topology_key": f"evaluation:{arm}",
            "metrics": metrics,
            "diagnostic_files": {"portfolio_daily.parquet": file_hash},
            "files": {
                "portfolio_daily.parquet": {
                    "sha256": file_hash,
                    "bytes": len(portfolio),
                }
            },
            "identity_pins": {
                "base_lock_sha256": _sha256(lock_path),
                "implementation_lock_sha256": _sha256(implementation_path),
                "mining_manifest_sha256": _sha256(mining_path / "manifest.json"),
                "development_opening_sha256": "0" * 64,
                "development_topology_sha256": topology.topology_sha256,
                "provider_identity": {"path": "provider"},
            },
        }
        _write_json(path / "manifest.json", manifest)
    replay_metrics = metrics_by_arm["alpha158_replay"]
    config["evaluation"]["historical_anchor"] = {
        "information_ratio": replay_metrics["information_ratio"],
        "max_drawdown": replay_metrics["max_drawdown"],
        "rank_ic": replay_metrics["Rank IC"],
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    base_lock["protocol"]["sha256"] = _sha256(config_path)
    _write_json(lock_path, base_lock)
    final_identities = {**identities, "protocol_sha256": _sha256(config_path)}
    for child_path in child_paths.values():
        manifest_path = child_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identities"] = final_identities
        _write_json(manifest_path, manifest)
    mining_manifest_path = mining_path / "manifest.json"
    mining_manifest = json.loads(mining_manifest_path.read_text())
    mining_manifest["identities"] = final_identities
    mining_manifest["child_manifest_sha256"] = {
        key: _sha256(path / "manifest.json") for key, path in child_paths.items()
    }
    mining_manifest["mechanism_cards_manifest_sha256"] = _sha256(
        child_paths["mechanism_cards"] / "manifest.json"
    )
    _write_json(mining_manifest_path, mining_manifest)
    # Rebuild because the topology hash binds the final config hash.
    topology = TopologyTransaction.from_frozen_config(workspace, phase="development")
    for arm in ARMS:
        path = workspace / evaluation_paths[arm]
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["topology_sha256"] = topology.topology_sha256
        manifest["identity_pins"]["base_lock_sha256"] = _sha256(lock_path)
        manifest["identity_pins"]["development_topology_sha256"] = (
            topology.topology_sha256
        )
        manifest["identity_pins"]["mining_manifest_sha256"] = _sha256(
            mining_path / "manifest.json"
        )
        _write_json(path / "manifest.json", manifest)
    preclaim = {
        "schema_version": "mirage_topology_preclaim_v2",
        "topology_sha256": topology.topology_sha256,
    }
    _write_json(topology.preclaim_path, preclaim)
    decision = topology.targets["decision_artifact"]
    _write_json(
        decision / ".INCOMPLETE",
        {
            "topology_sha256": topology.topology_sha256,
            "topology_key": "decision_artifact",
        },
    )

    opening_path = workspace / artifact_paths["development_opening"]
    opening = {
        "schema_version": "mirage_s2a_development_opening_v2",
        "protocol_id": config["protocol_id"],
        "state": "consumed_before_first_development_access",
        "topology_sha256": topology.topology_sha256,
        "evaluation_paths": evaluation_paths,
        "identity_pins": {
            "base_lock_sha256": _sha256(lock_path),
            "implementation_lock_sha256": _sha256(implementation_path),
            "mining_manifest_sha256": _sha256(mining_path / "manifest.json"),
            "provider_identity": {"path": "provider"},
        },
        "development_calendar_sha256": _calendar_hash(calendar),
        "development_calendar_count": len(calendar),
        "development_calendar_start": calendar[0].isoformat(),
        "development_calendar_end": calendar[-1].isoformat(),
    }
    _write_json(opening_path, opening)
    opening_sha = _sha256(opening_path)
    for arm in ARMS:
        path = workspace / evaluation_paths[arm] / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["identity_pins"]["development_opening_sha256"] = opening_sha
        _write_json(path, manifest)
    return workspace, calendar


def test_stages_decision_from_published_files_without_caller_evidence(
    tmp_path: Path,
) -> None:
    from mirage_kan.evaluation.v2_decision_assembler import stage_v2_decision_artifact

    workspace, _ = _workspace(tmp_path)
    result = stage_v2_decision_artifact(
        workspace, workspace / "evaluations/result.staging"
    )

    assert result.decision["outcome"] == "advance_s2_formal_pending_human_blind_review"
    assert result.decision["graph_unlock_allowed"] is True
    assert result.manifest["graph_unlock_allowed"] is True
    assert result.evidence["production_library_size"] == 8
    assert result.evidence["production_profile_count"] == 4
    assert result.evidence["permutation_false_positive_count"] == 0
    assert result.evidence["human_blind_review"]["status"] == "pending"
    assert result.evidence["kan_selected_library_effective_rank"] > 0.0
    assert result.evidence["kan_unique_admitted_count"] == 9
    assert result.evidence["gp_unique_admitted_count"] == 8
    assert result.manifest["final_decision_authority_consumed"] is False
    assert set(result.manifest["evaluation_manifest_sha256"]) == set(ARMS)


def test_eligible_but_not_selected_candidate_is_not_treated_as_library_member(
    tmp_path: Path,
) -> None:
    from mirage_kan.evaluation.v2_decision_assembler import stage_v2_decision_artifact

    workspace, _ = _workspace(tmp_path)
    result = stage_v2_decision_artifact(
        workspace, workspace / "evaluations/result.staging"
    )

    assert result.evidence["production_library_size"] == 8
    assert result.evidence["kan_unique_admitted_count"] == 9


def test_v5_decision_assembler_accepts_complete_six_factor_evidence(
    tmp_path: Path,
) -> None:
    from mirage_kan.evaluation.v2_decision_assembler import stage_v2_decision_artifact

    workspace, _ = _workspace(tmp_path, factor_count=6)
    result = stage_v2_decision_artifact(
        workspace, workspace / "evaluations/result.staging"
    )

    assert result.evidence["production_library_size"] == 6
    assert result.evidence["kan_unique_admitted_count"] == 7
    assert result.evidence["gp_unique_admitted_count"] == 6


@pytest.mark.parametrize("factor_count", (5, 17))
def test_v5_decision_assembler_rejects_counts_outside_frozen_bounds(
    tmp_path: Path, factor_count: int
) -> None:
    from mirage_kan.evaluation.v2_decision_assembler import stage_v2_decision_artifact

    workspace, _ = _workspace(
        tmp_path,
        factor_count=factor_count,
        minimum_library_size=6,
        integrity_minimum=6,
    )
    with pytest.raises(ValueError, match="frozen admission bounds"):
        stage_v2_decision_artifact(workspace, workspace / "evaluations/result.staging")


def test_v5_decision_assembler_rejects_admission_integrity_floor_drift(
    tmp_path: Path,
) -> None:
    from mirage_kan.evaluation.v2_decision_assembler import stage_v2_decision_artifact

    workspace, _ = _workspace(
        tmp_path,
        factor_count=6,
        minimum_library_size=6,
        integrity_minimum=8,
    )
    with pytest.raises(ValueError, match="integrity.*admission"):
        stage_v2_decision_artifact(workspace, workspace / "evaluations/result.staging")


def test_rejects_live_implementation_drift_before_decision_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mirage_kan.evaluation.v2_decision_assembler as assembler

    workspace, _ = _workspace(tmp_path)

    def reject_drift(workspace: Path) -> dict[str, object]:
        del workspace
        raise ValueError("implementation lock source hash mismatch")

    monkeypatch.setattr(assembler, "verify_implementation_lock", reject_drift)
    staging = workspace / "evaluations/result.staging"
    with pytest.raises(ValueError, match="implementation lock source hash mismatch"):
        assembler.stage_v2_decision_artifact(workspace, staging)
    assert not staging.exists()


def test_rejects_headline_ir_that_does_not_recompute_from_daily(tmp_path: Path) -> None:
    from mirage_kan.evaluation.v2_decision_assembler import stage_v2_decision_artifact

    workspace, _ = _workspace(tmp_path)
    manifest_path = workspace / "evaluations/kan_e3_selected/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metrics"]["information_ratio"] += 0.01
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="information ratio"):
        stage_v2_decision_artifact(workspace, workspace / "evaluations/result.staging")


def test_rejects_rehashed_mechanism_tampering_against_frozen_opening(
    tmp_path: Path,
) -> None:
    from mirage_kan.evaluation.v2_decision_assembler import stage_v2_decision_artifact

    workspace, _ = _workspace(tmp_path)
    cards_path = workspace / "mechanism_cards/cards/mechanism_cards.jsonl"
    rows = cards_path.read_text().splitlines()
    row = json.loads(rows[0])
    row["card"]["identity_and_canonical_ast"]["canonical_hash"] = "f" * 64
    rows[0] = json.dumps(row)
    cards_path.write_text("\n".join(rows) + "\n")
    manifest_path = cards_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][cards_path.name] = _sha256(cards_path)
    _write_json(manifest_path, manifest)
    top_path = workspace / "artifacts/mining/manifest.json"
    top = json.loads(top_path.read_text())
    top["child_manifest_sha256"]["mechanism_cards"] = _sha256(manifest_path)
    top["mechanism_cards_manifest_sha256"] = _sha256(manifest_path)
    _write_json(top_path, top)

    with pytest.raises(ValueError, match="development opening"):
        stage_v2_decision_artifact(workspace, workspace / "evaluations/result.staging")
