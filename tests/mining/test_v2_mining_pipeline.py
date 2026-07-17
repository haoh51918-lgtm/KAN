from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

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


def _workspace(
    tmp_path: Path, *, warmup_date_count: int = 60
) -> tuple[Path, pd.DataFrame]:
    for parent in (
        "artifacts",
        "factor_libraries",
        "controls",
        "mechanism_cards",
        "reviews",
        "evaluations",
        "configs/data",
        "configs/experiments",
        "prereg",
        "governance/openings",
        "governance/recoveries",
        "governance/decisions",
        "governance/incidents",
    ):
        (tmp_path / parent).mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(end="2015-12-31", periods=warmup_date_count).append(
        pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2016-01-04",
                    "2016-01-05",
                    "2020-12-30",
                    "2020-12-31",
                    "2021-01-04",
                    "2021-01-05",
                    "2021-01-06",
                    "2022-01-03",
                ]
            )
        )
    )
    rows: list[dict[str, object]] = []
    for instrument_index, instrument in enumerate(("A", "B", "C")):
        closes = [
            10.0 + instrument_index + date_index for date_index in range(len(dates))
        ]
        for date_index, (date, close) in enumerate(zip(dates, closes, strict=True)):
            rows.append(
                {
                    "datetime": date,
                    "instrument": instrument,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1000.0 + 10 * date_index + instrument_index,
                    "in_universe": True,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["datetime", "instrument"])
    indexed = frame.set_index(["datetime", "instrument"])
    indexed["fwd"] = (
        indexed["close"].groupby(level="instrument", sort=False).shift(-2)
        / indexed["close"].groupby(level="instrument", sort=False).shift(-1)
        - 1.0
    )
    frame = indexed.reset_index()
    cache = tmp_path / "cache.parquet"
    frame.to_parquet(cache, index=False)
    data_config = {
        "cache_path": str(cache),
        "cache_sha256": _sha(cache),
        "raw_fields": ["open", "high", "low", "close", "volume"],
        "label_1d": "Ref($close, -2) / Ref($close, -1) - 1",
    }
    data_path = tmp_path / "configs/data/pit_cache.json"
    data_path.write_text(json.dumps(data_config), encoding="utf-8")
    proposal = tmp_path / "KAN_Alpha_PR.md"
    preregistration = tmp_path / "prereg/protocol.md"
    directive = tmp_path / "governance/decisions/directive.md"
    incident = tmp_path / "governance/incidents/incident.md"
    for path in (proposal, preregistration, directive, incident):
        path.write_text(f"frozen {path.name}\n", encoding="utf-8")
    arms = [
        "alpha158_replay",
        "kan_e3_selected",
        "typed_gp_sr_control",
        "matched_blackbox_control",
        "kan_e3_permutation_control",
    ]
    artifact_paths = {
        "implementation_lock": "prereg/implementation.lock.json",
        "mining_entitlement": "governance/openings/mining.json",
        "mining_run": "artifacts/mining",
        "mining_preclaim": "governance/openings/mining_preclaim.json",
        "kan_library": "factor_libraries/kan",
        "gp_control_library": "factor_libraries/gp",
        "permutation_control_library": "factor_libraries/permutation",
        "blackbox_control": "controls/blackbox",
        "mechanism_cards": "mechanism_cards/kan",
        "blind_review_package": "reviews/blind",
        "mining_recovery_receipt": "governance/recoveries/mining.json",
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
        "data": {
            "train": ["2016-01-01", "2020-12-31"],
            "validation": ["2021-01-01", "2021-12-31"],
            "development_test": ["2022-01-01", "2025-12-26"],
            "labels": ["fwd"],
            "raw_miner_inputs": ["open", "high", "low", "close", "volume"],
            "label_horizon_purge": {
                "trading_dates": 2,
                "boundaries": ["train_end", "validation_end"],
                "cross_split_labels_forbidden": True,
            },
            "feature_warmup": {
                "trading_dates": 60,
                "raw_only": True,
                "labels_outside_objective_split_are_null": True,
            },
        },
        "kan_e3": {"total_miner_attempts": 256},
        "controls": {"arms": arms},
        "artifact_paths": artifact_paths,
    }
    config_path = tmp_path / "configs/experiments/protocol.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    implementation = tmp_path / "prereg/implementation.lock.json"
    implementation.write_text("{}\n", encoding="utf-8")
    base_lock = {
        "schema_version": "mirage_s2_prereg_lock_v2",
        "protocol": {
            "protocol_id": config["protocol_id"],
            "path": "configs/experiments/protocol.yaml",
            "sha256": _sha(config_path),
        },
        "data": {
            "cache_path": str(cache),
            "cache_sha256": _sha(cache),
            "config_path": "configs/data/pit_cache.json",
            "config_sha256": _sha(data_path),
        },
        "proposal": {
            "authority": "sole_proposal_authority",
            "path": "KAN_Alpha_PR.md",
            "sha256": _sha(proposal),
        },
        "preregistration": {
            "path": "prereg/protocol.md",
            "sha256": _sha(preregistration),
        },
        "governance": {
            "active_directive_path": "governance/decisions/directive.md",
            "active_directive_sha256": _sha(directive),
            "supersession_incident_path": "governance/incidents/incident.md",
            "supersession_incident_sha256": _sha(incident),
        },
    }
    lock_path = tmp_path / "prereg/s2a_kan_e3_vertical_v8.lock.json"
    lock_path.write_text(json.dumps(base_lock), encoding="utf-8")
    return tmp_path, frame


def _consume_live_capability(workspace: Path):
    from mirage_kan.governance.openings import consume_mining_entitlement
    from mirage_kan.mining.v2_pipeline import _MiningDataCapability

    topology = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    topology.preclaim()
    topology.claim_all()
    guard = AuthorityGuard(workspace)
    receipt = guard.revalidate("before_first_label_access")
    record = consume_mining_entitlement(workspace, topology, guard, receipt.capability)
    return _MiningDataCapability(topology, guard, receipt.capability, record)


def test_locked_loader_physically_filters_development_and_verifies_exact_fwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mirage_kan.mining.v2_pipeline import _load_locked_mining_inputs

    workspace, _ = _workspace(tmp_path)
    original = pd.read_parquet
    observed: list[dict[str, object]] = []

    def recording_read_parquet(*args, **kwargs):
        observed.append(
            {"columns": kwargs.get("columns"), "filters": kwargs.get("filters")}
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read_parquet)
    capability = _consume_live_capability(workspace)
    inputs = _load_locked_mining_inputs(workspace, capability)

    assert observed == [
        {
            "columns": [
                "datetime",
                "instrument",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "in_universe",
            ],
            "filters": [("datetime", "<=", pd.Timestamp("2021-12-31"))],
        },
        {
            "columns": ["datetime", "instrument", "fwd"],
            "filters": [
                [
                    ("datetime", ">=", pd.Timestamp("2016-01-01")),
                    ("datetime", "<=", pd.Timestamp("2016-01-05")),
                ],
                [
                    ("datetime", ">=", pd.Timestamp("2021-01-01")),
                    ("datetime", "<=", pd.Timestamp("2021-01-04")),
                ],
            ],
        },
    ]
    dates = inputs.panel.raw.index.get_level_values("datetime")
    assert len(dates[dates < pd.Timestamp("2016-01-01")].unique()) == 60
    assert dates.max() <= pd.Timestamp("2021-12-31")
    assert inputs.labels.name == "fwd"
    assert inputs.train == ("2016-01-01", "2020-12-31")
    assert inputs.validation == ("2021-01-01", "2021-12-31")
    dates = inputs.labels.index.get_level_values("datetime")
    assert inputs.labels[dates == pd.Timestamp("2020-12-31")].isna().all()
    assert inputs.labels[dates == pd.Timestamp("2021-01-05")].isna().all()
    assert inputs.labels[dates == pd.Timestamp("2021-01-06")].isna().all()


def test_locked_loader_rejects_label_or_entitlement_drift(tmp_path: Path) -> None:
    from mirage_kan.mining.v2_pipeline import _load_locked_mining_inputs

    workspace, frame = _workspace(tmp_path)
    cache = workspace / "cache.parquet"
    frame.loc[frame["datetime"] == pd.Timestamp("2016-01-04"), "fwd"] = 0.123
    frame.to_parquet(cache, index=False)
    data_path = workspace / "configs/data/pit_cache.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    data["cache_sha256"] = _sha(cache)
    data_path.write_text(json.dumps(data), encoding="utf-8")
    lock_path = workspace / "prereg/s2a_kan_e3_vertical_v8.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["data"]["cache_sha256"] = _sha(cache)
    lock["data"]["config_sha256"] = _sha(data_path)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    capability = _consume_live_capability(workspace)
    with pytest.raises(ValueError, match="exact fwd"):
        _load_locked_mining_inputs(workspace, capability)

    entitlement_path = workspace / "governance/openings/mining.json"
    entitlement = json.loads(entitlement_path.read_text(encoding="utf-8"))
    entitlement["topology_sha256"] = "x" * 64
    entitlement_path.write_text(json.dumps(entitlement), encoding="utf-8")
    with pytest.raises(ValueError, match="entitlement"):
        _load_locked_mining_inputs(workspace, capability)


def test_handwritten_entitlement_cannot_open_labels(tmp_path: Path) -> None:
    from mirage_kan.mining.v2_pipeline import _load_locked_mining_inputs

    workspace, _ = _workspace(tmp_path)
    (workspace / "governance/openings/mining.json").write_text(
        json.dumps({"state": "consumed_before_first_label_access"}),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="live in-process"):
        _load_locked_mining_inputs(workspace, object())


def test_locked_loader_keeps_exactly_last_sixty_pretrain_raw_dates(
    tmp_path: Path,
) -> None:
    from mirage_kan.mining.v2_pipeline import _load_locked_mining_inputs

    workspace, _ = _workspace(tmp_path, warmup_date_count=65)
    pretrain_dates = pd.bdate_range(end="2015-12-31", periods=65)

    inputs = _load_locked_mining_inputs(workspace, _consume_live_capability(workspace))

    dates = pd.DatetimeIndex(
        inputs.panel.raw.index.get_level_values("datetime").unique()
    )
    warmup_dates = dates[dates < pd.Timestamp("2016-01-01")]
    assert len(warmup_dates) == 60
    assert warmup_dates[0] == pretrain_dates[-60]
    assert (
        inputs.labels.reindex(inputs.panel.raw.index)[
            inputs.panel.raw.index.get_level_values("datetime")
            < pd.Timestamp("2016-01-01")
        ]
        .isna()
        .all()
    )


def test_production_entrypoint_has_no_data_or_scientific_injection_seam() -> None:
    from mirage_kan.mining.v2_pipeline import run_s2a_v2_mining

    assert tuple(inspect.signature(run_s2a_v2_mining).parameters) == (
        "workspace",
        "devices",
    )
    source = inspect.getsource(run_s2a_v2_mining)
    assert source.index("consume_mining_entitlement") < source.index(
        "_load_locked_mining_inputs"
    )
    assert source.index("try:") < source.index("guard = AuthorityGuard")
    assert "panel" not in inspect.signature(run_s2a_v2_mining).parameters
    assert "labels" not in inspect.signature(run_s2a_v2_mining).parameters


def test_rebind_protocol_refuses_mining_before_preclaim_or_label_access(
    tmp_path: Path,
) -> None:
    from mirage_kan.mining.v2_pipeline import run_s2a_v2_mining

    workspace, _ = _workspace(tmp_path)
    config_path = workspace / "configs/experiments/protocol.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["mining_source"] = {"mode": "verified_cross_protocol_rebind"}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    lock_path = workspace / "prereg/s2a_kan_e3_vertical_v8.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["protocol"]["sha256"] = _sha(config_path)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(RuntimeError, match="forbids a new mining"):
        run_s2a_v2_mining(workspace)

    assert not (workspace / "governance/openings/mining_preclaim.json").exists()
    assert not (workspace / "governance/openings/mining_entitlement.json").exists()
    assert not (workspace / "artifacts/mining").exists()


def test_authority_drift_during_guard_construction_terminalizes_every_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mirage_kan.mining.v2_pipeline as pipeline
    from mirage_kan.governance.authority import AuthoritySuperseded

    workspace, _ = _workspace(tmp_path)

    def superseded(_workspace: Path) -> None:
        raise AuthoritySuperseded("proposal drift")

    monkeypatch.setattr(pipeline, "AuthorityGuard", superseded)
    with pytest.raises(AuthoritySuperseded, match="proposal drift"):
        pipeline.run_s2a_v2_mining(workspace, devices=("cuda:0", "cuda:1"))

    topology = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    for target in topology.targets.values():
        assert not (target / ".INCOMPLETE").exists()
        terminal = json.loads(
            (target / "terminal_failure.json").read_text(encoding="utf-8")
        )
        assert terminal["failure_class"] == "superseded_authority"
        assert terminal["topology_sha256"] == topology.topology_sha256
