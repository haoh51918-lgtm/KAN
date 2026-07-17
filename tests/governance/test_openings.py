from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
import yaml
import pandas as pd

from mirage_kan.artifacts.topology import TopologyTransaction
from mirage_kan.governance.authority import AuthorityGuard


@pytest.fixture(autouse=True)
def _verified_test_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mirage_kan.governance.implementation_lock.verify_implementation_lock",
        lambda workspace: {"protocol_id": "s2a_kan_e3_vertical_v8"},
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace(tmp_path: Path) -> Path:
    for parent in (
        "artifacts",
        "factor_libraries",
        "controls",
        "mechanism_cards",
        "reviews",
        "evaluations",
        "reports",
        "governance/openings",
        "governance/recoveries",
        "governance/decisions",
        "governance/incidents",
        "prereg",
        "configs/experiments",
    ):
        (tmp_path / parent).mkdir(parents=True, exist_ok=True)
    proposal = tmp_path / "KAN_Alpha_PR.md"
    preregistration = tmp_path / "prereg/protocol.md"
    directive = tmp_path / "governance/decisions/directive.md"
    incident = tmp_path / "governance/incidents/incident.md"
    for path in (proposal, preregistration, directive, incident):
        path.write_text(f"frozen {path.name}\n", encoding="utf-8")
    artifact_paths = {
        "mining_run": "artifacts/mining",
        "mining_preclaim": "governance/openings/mining_preclaim.json",
        "mining_entitlement": "governance/openings/mining_entitlement.json",
        "kan_library": "factor_libraries/kan",
        "gp_control_library": "factor_libraries/gp",
        "permutation_control_library": "factor_libraries/permutation",
        "blackbox_control": "controls/blackbox",
        "mechanism_cards": "mechanism_cards/kan",
        "blind_review_package": "reviews/blind",
        "implementation_lock": "prereg/implementation.lock.json",
        "development_preclaim": "governance/openings/development_preclaim.json",
        "development_opening": "governance/openings/development.json",
        "mining_recovery_receipt": "governance/recoveries/mining.json",
        "development_recovery_receipt": "governance/recoveries/development.json",
        "evaluations": {
            arm: f"evaluations/{arm}"
            for arm in (
                "alpha158_replay",
                "kan_e3_selected",
                "typed_gp_sr_control",
                "matched_blackbox_control",
                "kan_e3_permutation_control",
            )
        },
        "decision_artifact": "evaluations/decision",
        "report": "reports/report.md",
    }
    config = {
        "protocol_id": "s2a_kan_e3_vertical_v8",
        "authority_revalidation": {
            "boundaries": [
                "before_first_label_access",
                "before_each_scientific_or_control_arm",
                "before_each_artifact_publication",
                "before_development_opening",
                "before_final_decision_publication",
            ]
        },
        "controls": {"arms": list(artifact_paths["evaluations"])},
        "data": {
            "train": ["2016-01-01", "2020-12-31"],
            "validation": ["2021-01-01", "2021-12-31"],
            "development_test": ["2022-01-01", "2025-12-26"],
        },
        "kan_e3": {"total_miner_attempts": 256},
        "artifact_paths": artifact_paths,
    }
    config_path = tmp_path / "configs/experiments/protocol.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    implementation = tmp_path / artifact_paths["implementation_lock"]
    implementation.write_text(
        json.dumps(
            {
                "schema_version": "mirage_s2_implementation_lock_v2",
                "protocol_id": "s2a_kan_e3_vertical_v8",
                "qlib_provider": {
                    "path": "/provider",
                    "tree_sha256": "1" * 64,
                    "stat_inventory_sha256": "2" * 64,
                    "file_count": 10,
                    "total_bytes": 100,
                },
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache.parquet"
    pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2021-12-31", "2022-01-03", "2023-01-03", "2025-12-26"]
            ),
            "instrument": ["A"] * 4,
            "open": [1.0] * 4,
            "high": [1.0] * 4,
            "low": [1.0] * 4,
            "close": [1.0] * 4,
            "volume": [1.0] * 4,
            "in_universe": [True] * 4,
        }
    ).to_parquet(cache, index=False)
    lock = {
        "schema_version": "mirage_s2_prereg_lock_v2",
        "protocol": {
            "protocol_id": config["protocol_id"],
            "path": str(config_path.relative_to(tmp_path)),
            "sha256": _sha(config_path),
        },
        "proposal": {
            "authority": "sole_proposal_authority",
            "path": "KAN_Alpha_PR.md",
            "sha256": _sha(proposal),
        },
        "preregistration": {
            "path": str(preregistration.relative_to(tmp_path)),
            "sha256": _sha(preregistration),
        },
        "governance": {
            "active_directive_path": str(directive.relative_to(tmp_path)),
            "active_directive_sha256": _sha(directive),
            "supersession_incident_path": str(incident.relative_to(tmp_path)),
            "supersession_incident_sha256": _sha(incident),
        },
        "data": {"cache_path": str(cache), "cache_sha256": _sha(cache)},
        "baseline_metric": {"path": "/data/baseline", "sha256": "b" * 64},
    }
    (tmp_path / "prereg/s2a_kan_e3_vertical_v8.lock.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    return tmp_path


def _staging(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f".{name}.staging"
    path.mkdir()
    (path / "payload.json").write_text("{}\n", encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": "test"}), encoding="utf-8"
    )
    return path


def _use_yaml_date_scalars(workspace: Path) -> None:
    config_path = workspace / "configs/experiments/protocol.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"] = {
        "train": [date(2016, 1, 1), date(2020, 12, 31)],
        "validation": [date(2021, 1, 1), date(2021, 12, 31)],
        "development_test": [date(2022, 1, 1), date(2025, 12, 26)],
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    lock_path = workspace / "prereg/s2a_kan_e3_vertical_v8.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["protocol"]["sha256"] = _sha(config_path)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")


def _replace_period_scalar(
    workspace: Path, period: str, index: int, value: object
) -> None:
    config_path = workspace / "configs/experiments/protocol.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["data"][period][index] = value
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    lock_path = workspace / "prereg/s2a_kan_e3_vertical_v8.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["protocol"]["sha256"] = _sha(config_path)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")


def _publish_mining(workspace: Path, guard: AuthorityGuard) -> TopologyTransaction:
    from mirage_kan.governance.openings import consume_mining_entitlement

    topology = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    topology.preclaim()
    topology.claim_all()
    label_capability = guard.revalidate("before_first_label_access").capability
    consume_mining_entitlement(workspace, topology, guard, label_capability)
    for key in topology.child_keys:
        capability = guard.revalidate(
            "before_each_artifact_publication", arm=key
        ).capability
        topology.publish_child(
            key,
            _staging(workspace, key),
            authority_guard=guard,
            authority_capability=capability,
        )
    capability = guard.revalidate(
        "before_each_artifact_publication", arm=topology.top_key
    ).capability
    top_staging = _staging(workspace, "mining_top")
    (top_staging / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "test_mining_top",
                "child_manifests": {
                    key: _sha(topology.targets[key] / "manifest.json")
                    for key in topology.child_keys
                },
            }
        ),
        encoding="utf-8",
    )
    topology.publish_top_bundle(
        top_staging,
        authority_guard=guard,
        authority_capability=capability,
    )
    return topology


def test_mining_entitlement_is_fixed_single_use_and_authority_bound(
    tmp_path: Path,
) -> None:
    from mirage_kan.governance.openings import consume_mining_entitlement

    workspace = _workspace(tmp_path)
    guard = AuthorityGuard(workspace)
    topology = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    topology.preclaim()
    topology.claim_all()
    capability = guard.revalidate("before_first_label_access").capability

    receipt = consume_mining_entitlement(
        workspace,
        topology,
        guard,
        capability,
    )

    assert receipt["state"] == "consumed_before_first_label_access"
    assert receipt["topology_sha256"] == topology.topology_sha256
    assert receipt["attempt_budget"] == 256
    assert receipt["authority_receipt_sha256"]
    with pytest.raises(FileExistsError):
        consume_mining_entitlement(workspace, topology, guard, capability)


def test_mining_entitlement_normalizes_yaml_date_scalars_to_iso_strings(
    tmp_path: Path,
) -> None:
    from mirage_kan.governance.openings import consume_mining_entitlement

    workspace = _workspace(tmp_path)
    _use_yaml_date_scalars(workspace)
    guard = AuthorityGuard(workspace)
    topology = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    topology.preclaim()
    topology.claim_all()
    capability = guard.revalidate("before_first_label_access").capability

    receipt = consume_mining_entitlement(
        workspace,
        topology,
        guard,
        capability,
    )

    assert receipt["train"] == ["2016-01-01", "2020-12-31"]
    assert receipt["validation"] == ["2021-01-01", "2021-12-31"]
    stored = json.loads(
        (workspace / "governance/openings/mining_entitlement.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["train"] == receipt["train"]
    assert stored["validation"] == receipt["validation"]


def test_mining_entitlement_rejects_non_date_period_scalar(tmp_path: Path) -> None:
    from mirage_kan.governance.openings import consume_mining_entitlement

    workspace = _workspace(tmp_path)
    _replace_period_scalar(workspace, "train", 0, 20160101)
    guard = AuthorityGuard(workspace)
    topology = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    topology.preclaim()
    topology.claim_all()
    capability = guard.revalidate("before_first_label_access").capability

    with pytest.raises(ValueError, match="date or ISO date string"):
        consume_mining_entitlement(
            workspace,
            topology,
            guard,
            capability,
        )

    assert not (workspace / "governance/openings/mining_entitlement.json").exists()


def test_development_opening_requires_all_immutable_mining_children_and_is_replayable(
    tmp_path: Path,
) -> None:
    from mirage_kan.governance.openings import (
        consume_development_opening,
        verify_development_opening,
    )

    workspace = _workspace(tmp_path)
    guard = AuthorityGuard(workspace)
    mining = _publish_mining(workspace, guard)
    development = TopologyTransaction.from_frozen_config(workspace, phase="development")
    development.preclaim()
    capability = guard.revalidate("before_development_opening").capability

    receipt = consume_development_opening(
        workspace,
        mining,
        development,
        guard,
        capability,
    )

    assert receipt["state"] == "consumed_before_first_development_access"
    assert receipt["mining_topology_sha256"] == mining.topology_sha256
    assert receipt["topology_sha256"] == development.topology_sha256
    assert receipt["mining_entitlement_sha256"]
    assert set(receipt["mining_manifests"]) == set(mining.targets)
    assert receipt["development_calendar_count"] == 3
    assert receipt["development_calendar_start"] == "2022-01-03T00:00:00"
    assert receipt["identity_pins"]["mining_manifest_sha256"]
    assert set(receipt["evaluation_paths"]) == set(
        configured
        for configured in (
            "alpha158_replay",
            "kan_e3_selected",
            "typed_gp_sr_control",
            "matched_blackbox_control",
            "kan_e3_permutation_control",
        )
    )
    assert verify_development_opening(workspace) == receipt


def test_development_opening_normalizes_yaml_date_scalars_to_iso_strings(
    tmp_path: Path,
) -> None:
    from mirage_kan.governance.openings import consume_development_opening

    workspace = _workspace(tmp_path)
    _use_yaml_date_scalars(workspace)
    guard = AuthorityGuard(workspace)
    mining = _publish_mining(workspace, guard)
    development = TopologyTransaction.from_frozen_config(workspace, phase="development")
    development.preclaim()
    capability = guard.revalidate("before_development_opening").capability

    receipt = consume_development_opening(
        workspace,
        mining,
        development,
        guard,
        capability,
    )

    assert receipt["development_period"] == ["2022-01-01", "2025-12-26"]
    stored = json.loads(
        (workspace / "governance/openings/development.json").read_text(
            encoding="utf-8"
        )
    )
    assert stored["development_period"] == receipt["development_period"]


def test_development_opening_rejects_non_date_period_scalar(tmp_path: Path) -> None:
    from mirage_kan.governance.openings import consume_development_opening

    workspace = _workspace(tmp_path)
    _replace_period_scalar(workspace, "development_test", 0, 20220101)
    guard = AuthorityGuard(workspace)
    mining = _publish_mining(workspace, guard)
    development = TopologyTransaction.from_frozen_config(workspace, phase="development")
    development.preclaim()
    capability = guard.revalidate("before_development_opening").capability

    with pytest.raises(ValueError, match="date or ISO date string"):
        consume_development_opening(
            workspace,
            mining,
            development,
            guard,
            capability,
        )

    assert not (workspace / "governance/openings/development.json").exists()


def test_exclusive_opening_write_does_not_create_path_when_serialization_fails(
    tmp_path: Path,
) -> None:
    from mirage_kan.governance.openings import _write_exclusive

    destination = tmp_path / "receipt.json"

    with pytest.raises(TypeError):
        _write_exclusive(destination, {"unsupported": object()})

    assert not destination.exists()


def test_development_opening_rejects_missing_or_mutated_mining_evidence(
    tmp_path: Path,
) -> None:
    from mirage_kan.governance.openings import consume_development_opening

    workspace = _workspace(tmp_path)
    guard = AuthorityGuard(workspace)
    mining = _publish_mining(workspace, guard)
    development = TopologyTransaction.from_frozen_config(workspace, phase="development")
    development.preclaim()
    target = mining.targets["kan_library"] / "manifest.json"
    target.write_bytes(target.read_bytes() + b" ")
    capability = guard.revalidate("before_development_opening").capability

    with pytest.raises(ValueError, match="manifest"):
        consume_development_opening(
            workspace,
            mining,
            development,
            guard,
            capability,
        )


def test_development_opening_rejects_mutated_mining_entitlement(
    tmp_path: Path,
) -> None:
    from mirage_kan.governance.openings import consume_development_opening

    workspace = _workspace(tmp_path)
    guard = AuthorityGuard(workspace)
    mining = _publish_mining(workspace, guard)
    entitlement = workspace / "governance/openings/mining_entitlement.json"
    record = json.loads(entitlement.read_text(encoding="utf-8"))
    record["state"] = "tampered"
    entitlement.chmod(0o644)
    entitlement.write_text(json.dumps(record), encoding="utf-8")
    development = TopologyTransaction.from_frozen_config(workspace, phase="development")
    development.preclaim()
    capability = guard.revalidate("before_development_opening").capability

    with pytest.raises(ValueError, match="mining entitlement identity"):
        consume_development_opening(
            workspace,
            mining,
            development,
            guard,
            capability,
        )


def test_rebound_development_opening_binds_source_receipt_without_new_entitlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mirage_kan.governance.openings import (
        consume_rebound_development_opening,
        verify_development_opening,
    )

    workspace = _workspace(tmp_path)
    config_path = workspace / "configs/experiments/protocol.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["mining_source"] = {
        "rebind_receipt": "governance/openings/mining_rebind.json"
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    lock_path = workspace / "prereg/s2a_kan_e3_vertical_v8.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["protocol"]["sha256"] = _sha(config_path)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    rebind_path = workspace / "governance/openings/mining_rebind.json"
    rebind_path.write_text("{}", encoding="utf-8")

    guard = AuthorityGuard(workspace)
    mining = _publish_mining(workspace, guard)
    artifacts = {
        key: {
            "manifest_sha256": _sha(path / "manifest.json"),
            "path": str(path.relative_to(workspace)),
            "files": {},
        }
        for key, path in mining.targets.items()
    }
    entitlement = workspace / "governance/openings/mining_entitlement.json"
    binding = {
        "source": {
            "protocol_id": "source_v6",
            "topology_sha256": mining.topology_sha256,
            "entitlement": {"sha256": _sha(entitlement)},
            "artifacts": artifacts,
        }
    }
    monkeypatch.setattr(
        "mirage_kan.governance.openings.verify_mining_rebind_receipt",
        lambda *args, **kwargs: binding,
    )
    development = TopologyTransaction.from_frozen_config(workspace, phase="development")
    development.preclaim()
    capability = guard.revalidate("before_development_opening").capability

    receipt = consume_rebound_development_opening(
        workspace,
        development,
        guard,
        capability,
    )

    assert receipt["schema_version"] == "mirage_s2a_development_opening_v3"
    assert receipt["mining_authorization_kind"] == "verified_cross_protocol_rebind"
    assert receipt["identity_pins"]["mining_rebind_receipt_sha256"] == _sha(
        rebind_path
    )
    assert receipt["source_mining_entitlement_sha256"] == _sha(entitlement)
    assert verify_development_opening(workspace) == receipt
