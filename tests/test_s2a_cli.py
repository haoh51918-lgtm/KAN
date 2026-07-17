from __future__ import annotations

import json
from pathlib import Path

import pytest

import mirage_kan.cli as cli


def _protocol() -> dict[str, object]:
    return {
        "protocol_id": "s2_plan_c_vertical_v1",
        "evaluation": {
            "historical_anchor": {
                "information_ratio": 0.22,
                "max_drawdown": -0.1376,
                "rank_ic": 0.03311,
            },
            "replay_tolerances": {
                "information_ratio_absolute": 0.03,
                "max_drawdown_absolute": 0.02,
                "rank_ic_absolute": 0.003,
            },
        },
    }


def test_new_publication_identity_labels_proposal_as_idea_draft(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "src" / "mirage_kan").mkdir(parents=True)
    (tmp_path / "KAN_Alpha_PR.md").write_text("idea draft\n", encoding="utf-8")
    monkeypatch.setattr(
        cli.QuantaAdapter,
        "baseline_link",
        lambda metric: {"sha256": "b" * 64},
    )
    monkeypatch.setattr(
        cli,
        "source_tree_identity",
        lambda package_root: {"tree_sha256": "c" * 64},
    )

    identities = cli._publication_identities(
        tmp_path,
        {"cache_path": "cache.pkl", "cache_sha256": "d" * 64},
        {
            "baseline_metric": "baseline.json",
            "baseline_metric_sha256": "b" * 64,
            "commit": "e" * 40,
            "config_sha256": "f" * 64,
            "runner_sha256": "0" * 64,
        },
    )

    assert identities["proposal"]["authority"] == "idea_draft"


def test_independent_s2a_arm_entry_points_are_forbidden(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="single orchestrator"):
        cli.replay_alpha158(tmp_path, tmp_path / "eval", tmp_path / "mining")
    with pytest.raises(PermissionError, match="single orchestrator"):
        cli.evaluate_s2a_library(
            tmp_path, tmp_path / "library", tmp_path / "eval", tmp_path / "mining"
        )


def test_single_opening_stops_after_invalid_replay_and_binds_report(
    tmp_path: Path, monkeypatch
) -> None:
    for relative in (
        "artifacts",
        "factor_libraries",
        "evaluations",
        "governance/decisions",
        "reports",
    ):
        (tmp_path / relative).mkdir(parents=True)
    mining_run = tmp_path / "artifacts" / "mining"
    mining_run.mkdir()
    calls = []
    identities = {
        "implementation_lock": {"sha256": "i" * 64},
        "qlib_provider": {"tree_sha256": "q" * 64},
        "baseline_metric": {"sha256": "b" * 64},
    }
    monkeypatch.setattr(cli, "verified_s2_identities", lambda workspace: (_protocol(), identities))
    monkeypatch.setattr(cli, "_configs", lambda workspace: ({}, {}))
    monkeypatch.setattr(cli, "_load_panel", lambda config: object())
    monkeypatch.setattr(
        cli,
        "_verify_s2a_mining_run",
        lambda *args: {
            "manifest_sha256": "m" * 64,
            "scoring_verification": {"verified": True},
            "libraries": {
                role: {"path": str(tmp_path / "factor_libraries" / role)}
                for role in cli.LIBRARY_ROLES
            },
        },
    )

    replay_payload = {
        "arm": "alpha158_replay",
        "metrics": {
            "information_ratio": -1.0,
            "max_drawdown": -0.1376,
            "Rank IC": 0.03311,
        },
    }

    def fake_publish(workspace, destination, **kwargs):
        calls.append(destination.name)
        destination.mkdir()
        (destination / "evaluation_manifest.json").write_text("{}")
        return replay_payload

    monkeypatch.setattr(cli, "_publish_s2a_evaluation", fake_publish)
    monkeypatch.setattr(cli, "_verify_s2a_evaluation", lambda *args: replay_payload)
    destinations = {
        arm: tmp_path / "evaluations" / arm for arm in cli.ARMS
    }
    decision_path = tmp_path / "governance" / "decisions" / "decision.json"
    report_path = tmp_path / "reports" / "report.md"

    result = cli.run_s2a_development(
        tmp_path, mining_run, destinations, decision_path, report_path
    )

    assert calls == ["alpha158_replay"]
    assert result["outcome"] == "s2a_inconclusive_infrastructure"
    assert result["formal_promotion_allowed"] is False
    assert result["human_report"]["sha256"] == cli.sha256_file(report_path)
    opening = tmp_path / "governance" / "openings" / "s2_plan_c_vertical_v1.json"
    assert opening.is_file()
    assert json.loads(opening.read_text())["state"] == "consumed_before_first_test_access"
    with pytest.raises((FileExistsError, PermissionError)):
        cli.run_s2a_development(
            tmp_path, mining_run, destinations, decision_path, report_path
        )


def test_cli_exposes_only_one_result_bearing_s2a_orchestrator() -> None:
    parser = cli._parser()
    choices = next(
        action.choices
        for action in parser._actions
        if getattr(action, "choices", None)
    )
    assert "run-s2a-development" in choices
    assert list(choices).count("replay-alpha158") == 1


def test_interrupted_opening_recovers_to_terminal_infrastructure_without_rerun(
    tmp_path: Path,
) -> None:
    for relative in (
        "evaluations",
        "governance/openings",
        "governance/decisions",
        "reports",
    ):
        (tmp_path / relative).mkdir(parents=True)
    destinations = {
        arm: tmp_path / "evaluations" / arm for arm in cli.ARMS
    }
    destinations["alpha158_replay"].mkdir()
    (destinations["alpha158_replay"] / ".INCOMPLETE").write_text("claimed")
    opening = {
        "protocol_id": "s2_plan_c_vertical_v1",
        "evaluation_destinations": {
            arm: str(path) for arm, path in destinations.items()
        },
        "decision_path": str(
            tmp_path / "governance" / "decisions" / "normal.json"
        ),
        "report_path": str(tmp_path / "reports" / "normal.md"),
    }
    opening_path = (
        tmp_path / "governance" / "openings" / "s2_plan_c_vertical_v1.json"
    )
    opening_path.write_text(json.dumps(opening))

    decision = cli.recover_s2a_interruption(tmp_path)

    assert decision["outcome"] == "s2a_inconclusive_infrastructure"
    assert decision["formal_promotion_allowed"] is False
    assert decision["evaluations"]["alpha158_replay"]["state"] == "incomplete"
    assert all(
        state["state"] == "missing"
        for arm, state in decision["evaluations"].items()
        if arm != "alpha158_replay"
    )
    with pytest.raises(FileExistsError):
        cli.recover_s2a_interruption(tmp_path)


def test_interruption_recovery_rejects_recorded_paths_outside_workspace(
    tmp_path: Path,
) -> None:
    for relative in (
        "evaluations",
        "governance/openings",
        "governance/decisions",
        "reports",
    ):
        (tmp_path / relative).mkdir(parents=True)
    outside = tmp_path.parent / "outside-evaluation"
    opening = {
        "protocol_id": "s2_plan_c_vertical_v1",
        "evaluation_destinations": {
            arm: str(outside if arm == "alpha158_replay" else tmp_path / "evaluations" / arm)
            for arm in cli.ARMS
        },
        "decision_path": str(tmp_path / "governance" / "decisions" / "normal.json"),
        "report_path": str(tmp_path / "reports" / "normal.md"),
    }
    opening_path = (
        tmp_path / "governance" / "openings" / "s2_plan_c_vertical_v1.json"
    )
    opening_path.write_text(json.dumps(opening))

    with pytest.raises(ValueError, match="direct child of evaluations"):
        cli.recover_s2a_interruption(tmp_path)

    assert not (
        tmp_path
        / "governance"
        / "decisions"
        / "s2_plan_c_vertical_v1_interruption.json"
    ).exists()


def test_development_recovery_resumes_after_report_only_crash(
    tmp_path: Path, monkeypatch
) -> None:
    for relative in (
        "evaluations",
        "governance/openings",
        "governance/decisions",
        "reports",
    ):
        (tmp_path / relative).mkdir(parents=True)
    opening = {
        "protocol_id": "s2_plan_c_vertical_v1",
        "evaluation_destinations": {
            arm: str(tmp_path / "evaluations" / arm) for arm in cli.ARMS
        },
        "decision_path": str(tmp_path / "governance" / "decisions" / "normal.json"),
        "report_path": str(tmp_path / "reports" / "normal.md"),
    }
    opening_path = (
        tmp_path / "governance" / "openings" / "s2_plan_c_vertical_v1.json"
    )
    opening_path.write_text(json.dumps(opening))
    original_write = cli._write_json_exclusive
    monkeypatch.setattr(
        cli,
        "_write_json_exclusive",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decision crash")),
    )
    with pytest.raises(RuntimeError, match="decision crash"):
        cli.recover_s2a_interruption(tmp_path)
    report = tmp_path / "reports" / "s2_plan_c_vertical_v1_interruption.md"
    assert report.is_file()
    monkeypatch.setattr(cli, "_write_json_exclusive", original_write)

    decision = cli.recover_s2a_interruption(tmp_path)
    assert decision["outcome"] == "s2a_inconclusive_infrastructure"


def test_mining_interruption_recovery_terminalizes_consumed_entitlement(
    tmp_path: Path,
) -> None:
    for relative in (
        "artifacts",
        "factor_libraries",
        "governance/openings",
        "governance/decisions",
        "reports",
    ):
        (tmp_path / relative).mkdir(parents=True)
    run = tmp_path / "artifacts" / "mining"
    run.mkdir()
    (run / ".INCOMPLETE").write_text("claimed")
    libraries = {
        role: tmp_path / "factor_libraries" / role for role in cli.LIBRARY_ROLES
    }
    libraries["heterogeneous_selected"].mkdir()
    (libraries["heterogeneous_selected"] / ".INCOMPLETE").write_text("claimed")
    entitlement = {
        "schema_version": "mirage_s2a_mining_entitlement_v1",
        "protocol_id": "s2_plan_c_vertical_v1",
        "state": "consumed_before_attempt_generation",
        "run_path": str(run),
        "library_destinations": {
            role: str(path) for role, path in libraries.items()
        },
        "attempt_budget": 256,
    }
    entitlement_path = tmp_path / cli.MINING_ENTITLEMENT
    entitlement_path.write_text(json.dumps(entitlement))

    decision = cli.recover_s2a_mining_interruption(tmp_path)

    assert decision["outcome"] == "s2a_mining_inconclusive_infrastructure"
    assert decision["formal_promotion_allowed"] is False
    assert decision["run"]["state"] == "terminal_failure"
    assert decision["libraries"]["heterogeneous_selected"]["state"] == "terminal_failure"
    assert decision["libraries"]["random_typed"]["state"] == "missing"
    with pytest.raises(FileExistsError):
        cli.recover_s2a_mining_interruption(tmp_path)


def test_mining_recovery_resumes_after_report_only_crash(
    tmp_path: Path, monkeypatch
) -> None:
    for relative in (
        "artifacts",
        "factor_libraries",
        "governance/openings",
        "governance/decisions",
        "reports",
    ):
        (tmp_path / relative).mkdir(parents=True)
    run = tmp_path / "artifacts" / "mining"
    run.mkdir()
    (run / ".INCOMPLETE").write_text("claimed")
    libraries = {
        role: tmp_path / "factor_libraries" / role for role in cli.LIBRARY_ROLES
    }
    entitlement = {
        "schema_version": "mirage_s2a_mining_entitlement_v1",
        "protocol_id": "s2_plan_c_vertical_v1",
        "state": "consumed_before_attempt_generation",
        "run_path": str(run),
        "library_destinations": {
            role: str(path) for role, path in libraries.items()
        },
        "attempt_budget": 256,
    }
    (tmp_path / cli.MINING_ENTITLEMENT).write_text(json.dumps(entitlement))
    original_write = cli._write_json_exclusive
    monkeypatch.setattr(
        cli,
        "_write_json_exclusive",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("decision crash")),
    )
    with pytest.raises(RuntimeError, match="decision crash"):
        cli.recover_s2a_mining_interruption(tmp_path)
    monkeypatch.setattr(cli, "_write_json_exclusive", original_write)

    decision = cli.recover_s2a_mining_interruption(tmp_path)
    assert decision["outcome"] == "s2a_mining_inconclusive_infrastructure"


def test_mining_recovery_accepts_preclaim_before_entitlement(
    tmp_path: Path,
) -> None:
    for relative in (
        "artifacts",
        "factor_libraries",
        "governance/openings",
        "governance/decisions",
        "reports",
    ):
        (tmp_path / relative).mkdir(parents=True)
    run = tmp_path / "artifacts" / "mining"
    run.mkdir()
    (run / ".INCOMPLETE").write_text("claimed")
    libraries = {
        role: tmp_path / "factor_libraries" / role for role in cli.LIBRARY_ROLES
    }
    preclaim = {
        "schema_version": "mirage_s2a_mining_preclaim_v1",
        "protocol_id": "s2_plan_c_vertical_v1",
        "state": "consumed_before_run_claim",
        "run_path": str(run),
        "library_destinations": {
            role: str(path) for role, path in libraries.items()
        },
        "attempt_budget": 256,
    }
    (tmp_path / cli.MINING_PRECLAIM).write_text(json.dumps(preclaim))

    decision = cli.recover_s2a_mining_interruption(tmp_path)

    assert decision["run"]["state"] == "terminal_failure"
    assert decision["entitlement"]["source"] == "preclaim"


def test_evaluation_verifier_rejects_wrong_arm_and_unindexed_file(tmp_path: Path) -> None:
    (tmp_path / "evaluations").mkdir()
    destination = tmp_path / "evaluations" / "selected"
    destination.mkdir()
    artifact = destination / "console.log"
    artifact.write_text("log")
    opening = {
        "sha256": "o" * 64,
        "evaluation_destinations": {"heterogeneous_selected": str(destination)},
    }
    manifest = {
        "schema_version": "mirage_s2a_quanta_evaluation_v1",
        "arm": "random_typed",
        "scientific_result": False,
        "formal_promotion_allowed": False,
        "opening": {"sha256": "o" * 64},
        "artifact_index": {
            "console.log": {
                "sha256": cli.sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            }
        },
    }
    (destination / "evaluation_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="wrong fixed-topology"):
        cli._verify_s2a_evaluation(
            destination, tmp_path, opening, "heterogeneous_selected"
        )

    manifest["arm"] = "heterogeneous_selected"
    (destination / "evaluation_manifest.json").write_text(json.dumps(manifest))
    (destination / "extra.csv").write_text("unindexed")
    with pytest.raises(ValueError, match="exact indexed file set"):
        cli._verify_s2a_evaluation(
            destination, tmp_path, opening, "heterogeneous_selected"
        )

    (destination / "extra.csv").unlink()
    with pytest.raises(ValueError, match="required diagnostic"):
        cli._verify_s2a_evaluation(
            destination, tmp_path, opening, "heterogeneous_selected"
        )
