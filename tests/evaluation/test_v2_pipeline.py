from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ARMS = (
    "alpha158_replay",
    "kan_e3_selected",
    "typed_gp_sr_control",
    "matched_blackbox_control",
    "kan_e3_permutation_control",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_panel_loader_reads_all_rows_but_no_label_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mirage_kan.evaluation.v2_pipeline import (
        _FrozenDevelopmentContext,
        _load_raw_pit_panel,
    )

    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2016-01-04", "2025-12-26"]),
            "instrument": ["A", "A"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [10.0, 20.0],
            "in_universe": [True, True],
            "tradable": [True, False],
            "fwd": [0.1, 0.2],
        }
    )
    cache = tmp_path / "pit.parquet"
    frame.to_parquet(cache, index=False)
    placeholder = tmp_path / "placeholder"
    placeholder.write_text("x", encoding="utf-8")
    context = _FrozenDevelopmentContext(
        root=tmp_path,
        base_lock_path=placeholder,
        base_lock={},
        config_path=placeholder,
        config={},
        implementation_lock_path=placeholder,
        development_opening_path=placeholder,
        cache_path=cache,
        cache_sha256=_sha(cache),
    )
    original = pd.read_parquet
    observed: dict[str, object] = {}

    def recording_read(*args, **kwargs):
        observed["columns"] = kwargs.get("columns")
        observed["filters"] = kwargs.get("filters")
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read)
    panel = _load_raw_pit_panel(context)

    assert "fwd" not in observed["columns"]
    assert observed["filters"] is None
    assert len(panel.raw) == len(frame)
    assert panel.tradability is not None
    assert panel.source_sha256 == _sha(cache)


class _FakeTopology:
    def __init__(self, root: Path, phase: str, events: list[str]) -> None:
        self.workspace = root
        self.phase = phase
        self.topology_sha256 = ("a" if phase == "mining" else "d") * 64
        self.top_key = "mining_run" if phase == "mining" else "decision_artifact"
        self.child_keys = (
            tuple(f"evaluation:{arm}" for arm in ARMS)
            if phase == "development"
            else ("kan_library",)
        )
        self.targets = {
            key: root / "evaluations" / key.replace(":", "_")
            for key in (self.top_key, *self.child_keys)
        }
        self.preclaim_path = root / f"{phase}_preclaim.json"
        self.events = events
        self.published: list[str] = []
        self.terminalized = False
        self.terminal_payloads: list[dict[str, object]] = []

    def preclaim(self) -> None:
        self.events.append("preclaim")

    def claim_all(self) -> None:
        self.events.append("claim")
        for path in self.targets.values():
            path.mkdir(parents=True)
            (path / ".INCOMPLETE").write_text("{}", encoding="utf-8")

    def publish_child(self, key, staging, **kwargs) -> None:
        self.events.append(f"publish:{key}")
        assert len([event for event in self.events if event.startswith("staged:")]) == 5
        target = self.targets[key]
        (target / ".INCOMPLETE").unlink()
        shutil.copy2(Path(staging) / "manifest.json", target / "manifest.json")
        shutil.rmtree(staging)
        self.published.append(key)

    def publish_top_bundle(self, *args, **kwargs) -> None:
        raise AssertionError("development pipeline must not publish an empty decision")

    def terminalize(self, payload) -> None:
        self.terminalized = True
        self.terminal_payloads.append(dict(payload))
        for path in self.targets.values():
            if path.exists():
                (path / "terminal_failure.json").write_text("{}", encoding="utf-8")
                marker = path / ".INCOMPLETE"
                if marker.exists():
                    marker.unlink()


class _FakeGuard:
    def __init__(self, root: Path, events: list[str]) -> None:
        self.workspace = root
        self.base_lock_sha256 = _sha(root / "base.json")
        self.events = events
        self.count = 0

    def revalidate(self, boundary: str, arm: str | None = None):
        self.count += 1
        capability = f"capability-{self.count}-{boundary}-{arm}"
        self.events.append(f"authority:{boundary}:{arm}")
        return SimpleNamespace(
            capability=capability,
            authority_sha256="c" * 64,
        )


class _DriftGuard(_FakeGuard):
    def __init__(self, root: Path, events: list[str], drift_boundary: str) -> None:
        super().__init__(root, events)
        self.drift_boundary = drift_boundary

    def revalidate(self, boundary: str, arm: str | None = None):
        from mirage_kan.governance.authority import AuthoritySuperseded

        if boundary == self.drift_boundary:
            raise AuthoritySuperseded(f"drift at {boundary}")
        return super().revalidate(boundary, arm)


def _patch_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failing_arm: str | None = None,
):
    import mirage_kan.evaluation.v2_pipeline as pipeline
    from mirage_kan.data import PitPanel
    from mirage_kan.evaluation.v2_runner import StagedArmEvaluation

    events: list[str] = []
    mining = _FakeTopology(tmp_path, "mining", events)
    development = _FakeTopology(tmp_path, "development", events)
    topologies = {"mining": mining, "development": development}
    opening_path = tmp_path / "development_opening.json"
    implementation = tmp_path / "implementation.json"
    cache = tmp_path / "cache.parquet"
    base = tmp_path / "base.json"
    config = tmp_path / "config.yaml"
    base.write_text("{}\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    implementation.write_text("{}\n", encoding="utf-8")
    cache.write_text("cache\n", encoding="utf-8")
    context = pipeline._FrozenDevelopmentContext(
        root=tmp_path,
        base_lock_path=base,
        base_lock={},
        config_path=config,
        config={},
        implementation_lock_path=implementation,
        development_opening_path=opening_path,
        cache_path=cache,
        cache_sha256=_sha(cache),
    )
    panel_frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2022-01-03"]),
            "instrument": ["A"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
            "in_universe": [True],
        }
    )
    panel = PitPanel.from_frame(
        panel_frame, source_path=cache, source_sha256=_sha(cache)
    )
    opening = {
        "identity_pins": {
            "base_lock_sha256": _sha(base),
            "implementation_lock_sha256": _sha(implementation),
            "mining_manifest_sha256": "e" * 64,
            "provider_identity": {
                "path": "/provider",
                "tree_sha256": "1" * 64,
                "stat_inventory_sha256": "2" * 64,
                "file_count": 1,
                "total_bytes": 1,
            },
        }
    }
    guards: list[_FakeGuard] = []

    monkeypatch.setattr(pipeline, "_load_frozen_context", lambda root: context)
    monkeypatch.setattr(
        pipeline, "verify_implementation_lock", lambda root: {"protocol_id": "v2"}
    )
    monkeypatch.setattr(
        pipeline.TopologyTransaction,
        "from_frozen_config",
        lambda root, phase: topologies[phase],
    )
    monkeypatch.setattr(
        pipeline, "_verify_published_mining_topology", lambda *args: "e" * 64
    )

    def make_guard(root):
        guard = _FakeGuard(root, events)
        guards.append(guard)
        return guard

    monkeypatch.setattr(pipeline, "AuthorityGuard", make_guard)

    def consume(*args):
        events.append("opening")
        opening_path.write_text(json.dumps(opening), encoding="utf-8")
        return opening

    monkeypatch.setattr(pipeline, "consume_development_opening", consume)
    monkeypatch.setattr(pipeline, "verify_development_opening", lambda root: opening)

    def load_raw(_context):
        assert "opening" in events
        events.append("raw_panel")
        return panel

    monkeypatch.setattr(pipeline, "_load_raw_pit_panel", load_raw)

    def stage(workspace, **kwargs):
        assert workspace == tmp_path
        arm = kwargs["arm"]
        assert kwargs["capability"] != kwargs["development_capability"]
        events.append(f"start:{arm}")
        if arm == failing_arm:
            raise RuntimeError("arm failed")
        staging = tmp_path / f".{arm}.staging"
        staging.mkdir()
        manifest = {"arm": arm}
        (staging / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        events.append(f"staged:{arm}")
        return StagedArmEvaluation(
            arm=arm,
            topology_key=f"evaluation:{arm}",
            staging_path=staging,
            manifest=manifest,
            manifest_sha256=_sha(staging / "manifest.json"),
        )

    monkeypatch.setattr(pipeline, "stage_v2_arm", stage)
    return events, development, guards


def test_public_pipeline_opens_before_raw_data_and_publishes_five_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mirage_kan.evaluation.v2_pipeline import run_s2a_v2_development

    events, development, guards = _patch_orchestration(tmp_path, monkeypatch)
    pending = run_s2a_v2_development(tmp_path)

    assert events.index("opening") < events.index("raw_panel")
    assert development.published == [f"evaluation:{arm}" for arm in ARMS]
    assert (development.targets[development.top_key] / ".INCOMPLETE").is_file()
    assert not (development.targets[development.top_key] / "manifest.json").exists()
    arm_authorities = [
        event
        for event in events
        if event.startswith("authority:before_each_scientific_or_control_arm")
    ]
    assert len(arm_authorities) == 5
    assert len(set(arm_authorities)) == 5
    assert pending.arm_manifest_sha256.keys() == dict.fromkeys(ARMS).keys()
    assert len(guards) == 1


def test_five_arms_finish_in_frozen_order_without_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mirage_kan.evaluation.v2_pipeline import run_s2a_v2_development

    events, _, _ = _patch_orchestration(tmp_path, monkeypatch)
    run_s2a_v2_development(tmp_path)

    arm_events = [
        event
        for event in events
        if event.startswith("authority:before_each_scientific_or_control_arm")
        or event.startswith("start:")
    ]
    assert arm_events == [
        event
        for arm in ARMS
        for event in (
            f"authority:before_each_scientific_or_control_arm:{arm}",
            f"start:{arm}",
        )
    ]
    for previous, current in zip(ARMS, ARMS[1:], strict=False):
        assert events.index(f"staged:{previous}") < events.index(f"start:{current}")


def test_arm_failure_terminalizes_topology_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mirage_kan.evaluation.v2_pipeline import run_s2a_v2_development

    _, development, _ = _patch_orchestration(
        tmp_path, monkeypatch, failing_arm="typed_gp_sr_control"
    )

    with pytest.raises(RuntimeError, match="arm failed"):
        run_s2a_v2_development(tmp_path)

    assert development.terminalized
    assert not list(tmp_path.glob(".*.staging"))
    assert [event for event in development.events if event.startswith("start:")] == [
        f"start:{arm}" for arm in ARMS[:3]
    ]
    assert [
        event
        for event in development.events
        if event.startswith("authority:before_each_scientific_or_control_arm")
    ] == [f"authority:before_each_scientific_or_control_arm:{arm}" for arm in ARMS[:3]]
    assert all(
        (path / "terminal_failure.json").is_file()
        for path in development.targets.values()
    )


def test_cleanup_failure_is_not_allowed_to_replace_the_arm_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mirage_kan.evaluation.v2_pipeline as pipeline

    _patch_orchestration(tmp_path, monkeypatch, failing_arm="typed_gp_sr_control")
    original_remove = pipeline._remove_staging
    cleanup_attempts: list[Path] = []

    def flaky_remove(root: Path, path: Path | str) -> None:
        cleanup_attempts.append(Path(path))
        if len(cleanup_attempts) == 1:
            raise OSError("cleanup exploded")
        original_remove(root, path)

    monkeypatch.setattr(pipeline, "_remove_staging", flaky_remove)
    with pytest.raises(RuntimeError, match="arm failed") as captured:
        pipeline.run_s2a_v2_development(tmp_path)

    assert [path.name for path in cleanup_attempts] == [
        f".{arm}.staging" for arm in ARMS[:2]
    ]
    assert any("cleanup exploded" in note for note in captured.value.__notes__)
    assert cleanup_attempts[0].is_dir()
    assert not cleanup_attempts[1].exists()


def test_development_authority_drift_terminalizes_as_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mirage_kan.evaluation.v2_pipeline as pipeline
    from mirage_kan.governance.authority import AuthoritySuperseded

    events, development, _ = _patch_orchestration(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pipeline,
        "AuthorityGuard",
        lambda root: _DriftGuard(root, events, "before_development_opening"),
    )

    with pytest.raises(AuthoritySuperseded, match="before_development_opening"):
        pipeline.run_s2a_v2_development(tmp_path)

    assert development.terminal_payloads[-1] == {
        "failure_class": "superseded_authority",
        "error": "drift at before_development_opening",
    }


def test_production_entrypoint_has_no_adapter_or_data_injection_seam() -> None:
    from mirage_kan.evaluation.v2_pipeline import run_s2a_v2_development

    assert tuple(inspect.signature(run_s2a_v2_development).parameters) == ("workspace",)
    source = inspect.getsource(run_s2a_v2_development)
    assert "adapter" not in source
    assert source.index("consume_development_opening") < source.index(
        "_load_raw_pit_panel"
    )


def test_decision_assembly_authority_drift_terminalizes_as_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mirage_kan.evaluation.v2_decision_assembler as assembler
    import mirage_kan.evaluation.v2_pipeline as pipeline
    from mirage_kan.governance.authority import AuthoritySuperseded

    events: list[str] = []
    topology = _FakeTopology(tmp_path, "development", events)
    pending = pipeline.PendingDevelopmentDecision(
        workspace=tmp_path,
        topology=topology,
        authority_guard=SimpleNamespace(),
        pins=SimpleNamespace(),
        arm_manifest_sha256={},
    )

    def reject_drift(*args, **kwargs):
        raise AuthoritySuperseded("drift during decision assembly")

    monkeypatch.setattr(assembler, "stage_v2_decision_artifact", reject_drift)

    with pytest.raises(AuthoritySuperseded, match="decision assembly"):
        pending.stage_decision(tmp_path / ".decision.staging")

    assert topology.terminal_payloads[-1] == {
        "failure_class": "superseded_authority",
        "error": "drift during decision assembly",
    }


@pytest.mark.parametrize("mutation", ["modify", "add", "delete"])
def test_decision_publication_reverifies_implementation_and_exact_staging_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import mirage_kan.evaluation.v2_pipeline as pipeline
    from mirage_kan.evaluation.v2_decision_assembler import StagedDecisionArtifact
    from mirage_kan.evaluation.v2_runner import EvaluationIdentityPins

    events: list[str] = []
    topology = _FakeTopology(tmp_path, "development", events)
    topology.claim_all()
    (tmp_path / "base.json").write_text("{}\n", encoding="utf-8")
    implementation_path = tmp_path / "implementation.json"
    implementation_path.write_text("{}\n", encoding="utf-8")
    guard = _FakeGuard(tmp_path, events)
    staging = tmp_path / ".decision.staging"
    staging.mkdir()
    decision_path = staging / "decision.json"
    decision_path.write_text('{"outcome":"screen"}\n', encoding="utf-8")
    manifest = {
        "development_topology_sha256": topology.topology_sha256,
        "evaluation_manifest_sha256": {arm: arm[0] * 64 for arm in ARMS},
        "files": {"decision.json": _sha(decision_path)},
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staged = StagedDecisionArtifact(
        path=staging,
        manifest=manifest,
        decision={"outcome": "screen"},
        evidence={},
        manifest_sha256=_sha(manifest_path),
    )
    pending = pipeline.PendingDevelopmentDecision(
        workspace=tmp_path,
        topology=topology,
        authority_guard=guard,
        pins=EvaluationIdentityPins(
            base_lock_sha256=guard.base_lock_sha256,
            implementation_lock_sha256=_sha(implementation_path),
            mining_manifest_sha256="2" * 64,
            development_opening_sha256="3" * 64,
            development_topology_sha256=topology.topology_sha256,
            provider_identity={
                "path": "/provider",
                "tree_sha256": "4" * 64,
                "stat_inventory_sha256": "5" * 64,
                "file_count": 1,
                "total_bytes": 1,
            },
        ),
        arm_manifest_sha256=manifest["evaluation_manifest_sha256"],
    )
    monkeypatch.setattr(
        pipeline,
        "verify_implementation_lock",
        lambda root: events.append("implementation_verified"),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_frozen_context",
        lambda root: SimpleNamespace(implementation_lock_path=implementation_path),
    )

    if mutation == "modify":
        decision_path.write_text('{"outcome":"forged"}\n', encoding="utf-8")
    elif mutation == "add":
        (staging / "unregistered.txt").write_text("forged\n", encoding="utf-8")
    else:
        decision_path.unlink()

    with pytest.raises(ValueError, match="decision staging"):
        pending.publish_decision(staged)

    assert "implementation_verified" in events
    assert not any(event.startswith("authority:") for event in events)
    assert topology.terminalized
    assert not staging.exists()


def test_decision_publication_authority_drift_terminalizes_as_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mirage_kan.evaluation.v2_pipeline as pipeline
    from mirage_kan.evaluation.v2_decision_assembler import StagedDecisionArtifact
    from mirage_kan.evaluation.v2_runner import EvaluationIdentityPins
    from mirage_kan.governance.authority import AuthoritySuperseded

    events: list[str] = []
    topology = _FakeTopology(tmp_path, "development", events)
    topology.claim_all()
    (tmp_path / "base.json").write_text("{}\n", encoding="utf-8")
    implementation_path = tmp_path / "implementation.json"
    implementation_path.write_text("{}\n", encoding="utf-8")
    guard = _DriftGuard(
        tmp_path, events, drift_boundary="before_final_decision_publication"
    )
    staging = tmp_path / ".decision.staging"
    staging.mkdir()
    decision_path = staging / "decision.json"
    decision_path.write_text('{"outcome":"screen"}\n', encoding="utf-8")
    evaluation_hashes = {arm: arm[0] * 64 for arm in ARMS}
    manifest = {
        "development_topology_sha256": topology.topology_sha256,
        "evaluation_manifest_sha256": evaluation_hashes,
        "files": {"decision.json": _sha(decision_path)},
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staged = StagedDecisionArtifact(
        path=staging,
        manifest=manifest,
        decision={"outcome": "screen"},
        evidence={},
        manifest_sha256=_sha(manifest_path),
    )
    pending = pipeline.PendingDevelopmentDecision(
        workspace=tmp_path,
        topology=topology,
        authority_guard=guard,
        pins=EvaluationIdentityPins(
            base_lock_sha256=guard.base_lock_sha256,
            implementation_lock_sha256=_sha(implementation_path),
            mining_manifest_sha256="2" * 64,
            development_opening_sha256="3" * 64,
            development_topology_sha256=topology.topology_sha256,
            provider_identity={
                "path": "/provider",
                "tree_sha256": "4" * 64,
                "stat_inventory_sha256": "5" * 64,
                "file_count": 1,
                "total_bytes": 1,
            },
        ),
        arm_manifest_sha256=evaluation_hashes,
    )
    monkeypatch.setattr(
        pipeline,
        "verify_implementation_lock",
        lambda root: events.append("implementation_verified"),
    )
    monkeypatch.setattr(
        pipeline,
        "_load_frozen_context",
        lambda root: SimpleNamespace(implementation_lock_path=implementation_path),
    )

    with pytest.raises(AuthoritySuperseded, match="final_decision_publication"):
        pending.publish_decision(staged)

    assert "implementation_verified" in events
    assert topology.terminal_payloads[-1] == {
        "failure_class": "superseded_authority",
        "error": "drift at before_final_decision_publication",
    }
    assert not staging.exists()
