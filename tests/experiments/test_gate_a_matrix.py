from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
import sys

import pytest


def test_matrix_modes_reject_scientific_overrides_and_frozen_smoke_seeds(
    tmp_path, monkeypatch
) -> None:
    import mirage_kan.experiments.gate_a.matrix as matrix_module

    from mirage_kan.experiments.gate_a.matrix import (
        run_gate_a_matrix,
        scientific_matrix_settings,
        smoke_matrix_settings,
    )

    lock = json.loads(Path("prereg/s1_gate_a_v1.lock.json").read_text())
    # The v1 lock pins the proposal version in force when Gate A v1 ran. That
    # version was superseded by the 2026-07-17 principal directive (proposal
    # Section 25); the byte-exact archive keeps the pinned hash resolvable.
    archived_proposal = Path(
        "governance/archives/KAN_Alpha_PR_v2026-07-16_sha1880ccf1.md"
    )
    assert (
        hashlib.sha256(archived_proposal.read_bytes()).hexdigest()
        == lock["proposal_sha256"]
    )
    assert (
        hashlib.sha256(Path(lock["proposal_authority"]).read_bytes()).hexdigest()
        != lock["proposal_sha256"]
    )
    for hash_key, path_key in (
        ("protocol_sha256", "protocol_path"),
        ("config_sha256", "config_path"),
    ):
        assert hashlib.sha256(Path(lock[path_key]).read_bytes()).hexdigest() == lock[hash_key]
    assert lock["scientific_results_observed"] is False
    assert lock["non_scientific_smoke_observed"] is True
    implementation_lock = json.loads(
        Path("prereg/s1_gate_a_v1_implementation.lock.json").read_text()
    )
    from mirage_kan.experiments.gate_a.matrix import _implementation_snapshot

    historical_files = implementation_lock["snapshot"]["files"]
    historical_aggregate = hashlib.sha256(
        json.dumps(
            historical_files, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert historical_aggregate == implementation_lock["snapshot"][
        "aggregate_sha256"
    ]
    assert implementation_lock["snapshot"] != _implementation_snapshot()
    assert "s1_gate_a_v1_implementation.lock.json" not in implementation_lock[
        "snapshot"
    ]["files"]
    with pytest.raises(ValueError, match="implementation differs"):
        scientific_matrix_settings("current_source_must_not_pose_as_s1")

    monkeypatch.setattr(
        matrix_module,
        "_verify_implementation_lock",
        lambda: implementation_lock["snapshot"],
    )
    scientific = scientific_matrix_settings("scientific_literal")
    assert scientific.mode == "scientific"
    assert scientific.seeds == (1729, 2718, 31415)
    assert scientific.config["mechanism"] == {
        "source": "Return(Close,5)",
        "input_scale": 0.03,
        "input_clip": [-4.0, 4.0],
        "negative_amplitude": -0.6,
        "negative_rate": 2.5,
        "positive_amplitude": 1.8,
        "positive_rate": 0.25,
        "noise_fraction_of_train_clean_std": 0.10,
    }
    with pytest.raises(ValueError, match="does not accept overrides"):
        scientific_matrix_settings("bad", overrides={"training.max_steps": 1})
    with pytest.raises(ValueError, match="does not accept overrides"):
        scientific_matrix_settings("bad", seeds=(999,))
    scientific.config["training"]["max_steps"] = 1
    with pytest.raises(ValueError, match="differs from the sealed config"):
        run_gate_a_matrix(scientific, artifact_base=tmp_path, device="cpu")
    assert not (tmp_path / "s1_gate_a_scientific").exists()
    # Scientific mode rejects any non-default artifact base before it can
    # reach the seal, so the superseded-proposal refusal ("proposal authority
    # hash mismatch") is exercised by the smoke-mode tests below instead.

    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("scientific artifact guard ran too late")

    monkeypatch.setattr(matrix_module, "_seal_snapshot", forbidden)
    monkeypatch.setattr(matrix_module, "_prepare_root", forbidden)
    monkeypatch.setattr(matrix_module, "generate_gate_a_replication", forbidden)
    fixed_root = scientific_matrix_settings("fixed_project_artifacts")
    with pytest.raises(ValueError, match="fixed project artifacts"):
        run_gate_a_matrix(fixed_root, artifact_base=tmp_path, device="cpu")
    assert called == []
    assert not (tmp_path / "s1_gate_a_scientific").exists()
    with pytest.raises(ValueError, match="frozen scientific seed"):
        smoke_matrix_settings(
            "bad_smoke",
            seeds=(1729, 8675309),
            assets=2,
            burn_in_dates=25,
            train_dates=40,
            validation_dates=40,
            test_dates=40,
            max_steps=1,
            batch_size=16,
            e5_candidate_budget=2,
            bootstrap_replicates=2,
        )


@pytest.mark.parametrize(
    "run_id",
    ("", ".", "..", "../escape", "nested/run", r"nested\run", "/tmp/escape"),
)
def test_matrix_run_id_is_one_safe_filename_segment(run_id) -> None:
    from mirage_kan.experiments.gate_a.matrix import smoke_matrix_settings

    with pytest.raises(ValueError, match="run ID"):
        smoke_matrix_settings(
            run_id,
            seeds=(8675801, 8675802),
            assets=2,
            burn_in_dates=25,
            train_dates=40,
            validation_dates=40,
            test_dates=40,
            max_steps=1,
            batch_size=16,
            e5_candidate_budget=2,
            bootstrap_replicates=2,
        )


def test_tiny_fresh_seed_matrix_refuses_superseded_proposal_authority(
    tmp_path, monkeypatch
) -> None:
    """The consumed v1 machinery must fail closed under the revised proposal.

    Gate A v1 is consumed and its preregistration lock pins the archived
    proposal version, so every run mode now refuses with a proposal authority
    hash mismatch and publishes nothing. Behavioral smoke coverage of a full
    matrix run returns with the S1b protocol machinery (proposal Section
    25.7), which binds to its own preregistration lock.
    """
    import mirage_kan.experiments.gate_a.matrix as matrix_module

    from mirage_kan.experiments.gate_a.matrix import (
        _implementation_snapshot,
        run_gate_a_matrix,
        smoke_matrix_settings,
    )

    monkeypatch.setattr(
        matrix_module, "_verify_implementation_lock", _implementation_snapshot
    )

    settings = smoke_matrix_settings(
        "fresh_matrix",
        seeds=(8675309, 8675310),
        assets=2,
        burn_in_dates=25,
        train_dates=40,
        validation_dates=40,
        test_dates=40,
        max_steps=1,
        batch_size=16,
        e5_candidate_budget=2,
        bootstrap_replicates=2,
    )
    with pytest.raises(ValueError, match="proposal authority hash mismatch"):
        run_gate_a_matrix(settings, artifact_base=tmp_path, device="cpu")
    root = tmp_path / "s1_gate_a_matrix_smoke" / settings.run_id
    assert not (root / "manifests" / "matrix.json").exists()


def test_artifact_index_verification_detects_tampering(tmp_path) -> None:
    """Index closure and tamper detection stay covered without the v1 runner."""
    from mirage_kan.experiments.gate_a.matrix import (
        _build_artifact_index,
        verify_artifact_index,
    )

    root = tmp_path / "run"
    (root / "manifests").mkdir(parents=True)
    (root / "ledgers").mkdir()
    (root / "ledgers" / "console.log").write_text("alpha\n", encoding="utf-8")
    (root / "metrics.json").write_text("{}\n", encoding="utf-8")
    index = _build_artifact_index(root)
    assert set(index["files"]) == {"ledgers/console.log", "metrics.json"}
    manifest_path = root / "manifests" / "matrix.json"
    manifest_path.write_text(
        json.dumps({"artifact_index": index}), encoding="utf-8"
    )
    verify_artifact_index(root, manifest_path)
    with (root / "ledgers" / "console.log").open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")
    with pytest.raises(
        ValueError, match=r"artifact index (?:hash|byte-size) mismatch"
    ):
        verify_artifact_index(root, manifest_path)
    (root / "extra.bin").write_bytes(b"\x00")
    with pytest.raises(ValueError, match="artifact index file set mismatch"):
        verify_artifact_index(root, manifest_path)


def test_scientific_cli_has_no_artifact_root_override(monkeypatch, capsys) -> None:
    from mirage_kan.experiments.gate_a.matrix import main

    monkeypatch.setattr(sys, "argv", ["gate-a-matrix", "--help"])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--artifact-base" not in help_text
    assert "--matrix-id" in help_text


def test_implementation_lock_timestamp_is_truthful() -> None:
    lock = json.loads(
        Path("prereg/s1_gate_a_v1_implementation.lock.json").read_text()
    )
    actual_max_ns = max(
        Path(path).stat().st_mtime_ns for path in lock["snapshot"]["files"]
    )
    recorded_max = datetime.fromisoformat(lock["max_locked_file_mtime_utc"])
    created = datetime.fromisoformat(lock["created_at_utc"])
    recorded_ns = (
        int(recorded_max.timestamp()) * 1_000_000_000
        + recorded_max.microsecond * 1_000
    )
    created_ns = (
        int(created.timestamp()) * 1_000_000_000 + created.microsecond * 1_000
    )
    locked_max_ns = lock["max_locked_file_mtime_ns"]
    assert recorded_ns <= locked_max_ns
    assert locked_max_ns - recorded_ns < 1_000
    assert created_ns >= locked_max_ns
    assert actual_max_ns >= locked_max_ns
    incidents = lock["known_preformal_incidents"]
    assert incidents == [
        {
            "report_path": "governance/incidents/2026-07-16_frozen_seed_red_test_incident.md",
            "report_sha256": hashlib.sha256(
                Path(
                    "governance/incidents/2026-07-16_frozen_seed_red_test_incident.md"
                ).read_bytes()
            ).hexdigest(),
            "classification": "pre-test partial scientific attempt, invalidated",
            "test_opened": False,
            "eligible_for_clean_rerun": True,
        }
    ]
    addendum = Path(
        "governance/incidents/2026-07-16_frozen_seed_red_test_cleanup_addendum.md"
    )
    assert lock["governance_addenda"] == [
        {
            "path": str(addendum),
            "sha256": hashlib.sha256(addendum.read_bytes()).hexdigest(),
            "relation": "custody update to known_preformal_incidents[0]",
        }
    ]


def test_post_open_hook_is_unreachable_under_superseded_proposal_authority(
    tmp_path, monkeypatch
) -> None:
    """Refusal must precede any test opening or terminal-failure publication.

    Terminal-failure publication behavior remains exercised only by the
    sealed v1 evidence; live coverage returns with the S1b machinery.
    """
    import mirage_kan.experiments.gate_a.matrix as matrix_module

    from mirage_kan.experiments.gate_a.matrix import (
        _implementation_snapshot,
        run_gate_a_matrix,
        smoke_matrix_settings,
    )

    monkeypatch.setattr(
        matrix_module, "_verify_implementation_lock", _implementation_snapshot
    )

    settings = smoke_matrix_settings(
        "post_open_failure",
        seeds=(8675321, 8675322),
        assets=2,
        burn_in_dates=25,
        train_dates=40,
        validation_dates=40,
        test_dates=40,
        max_steps=1,
        batch_size=16,
        e5_candidate_budget=2,
        bootstrap_replicates=2,
    )

    def fail_after_open(seed: int, opening: Path) -> None:
        raise AssertionError(
            "post-open hook must be unreachable under superseded authority"
        )

    with pytest.raises(ValueError, match="proposal authority hash mismatch"):
        run_gate_a_matrix(
            settings,
            artifact_base=tmp_path,
            device="cpu",
            post_open_hook=fail_after_open,
        )
    root = tmp_path / "s1_gate_a_matrix_smoke" / settings.run_id
    assert not (root / "manifests" / "matrix.json").exists()
    assert not (root / "manifests" / "terminal_failure.json").exists()
