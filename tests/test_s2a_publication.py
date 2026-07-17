from __future__ import annotations

import json

import pandas as pd
import pytest

from mirage_kan.dsl import AstNode
from mirage_kan.artifacts.library import verify_library
from mirage_kan.mining import CandidateScore, MiningAttempt, ScoringRun, SelectionResult
from mirage_kan.mining.s2a import publish_s2a_run
import mirage_kan.mining.s2a as s2a


def _candidate(candidate_id: str, profile: str) -> CandidateScore:
    program = AstNode("Return", (AstNode("Close"),), {"window": 2})
    return CandidateScore(
        candidate_id=candidate_id,
        profile=profile,
        attempt_index=0,
        program=program,
        canonical_hash=program.identity,
        ast_depth=2,
        ast_nodes=2,
        output_type="dimensionless_ts",
        causal=True,
        unique=True,
        support_rows=1,
        eligible_rows=1,
        coverage=1.0,
        train_rank_ic=0.02,
        validation_rank_ic=0.01,
        sign_agreement=True,
        minimum_score=0.01,
        eligible=True,
        disposition="eligible",
        values=pd.Series(dtype=float),
    )


def test_s2a_publication_is_no_replace_and_records_exact_provenance(
    tmp_path, tiny_panel, monkeypatch
) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "factor_libraries").mkdir()
    candidate = _candidate("trend_000", "trend")
    attempt = MiningAttempt("trend", 0, "trend_000", candidate.program, "generated")
    observed = ScoringRun(
        (candidate,),
        "observed",
        "a" * 64,
        "b" * 64,
        ("2016-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
    )
    permutation_candidate = CandidateScore(
        **{
            **candidate.__dict__,
            "train_rank_ic": 0.01,
            "validation_rank_ic": 0.02,
        }
    )
    permutation = ScoringRun(
        (permutation_candidate,),
        "within_date_permutation",
        "c" * 64,
        "d" * 64,
        observed.train,
        observed.validation,
    )
    observed_selection = SelectionResult(
        (candidate,), {candidate.candidate_id: "selected"}, True, True
    )
    permutation_selection = SelectionResult(
        (permutation_candidate,),
        {permutation_candidate.candidate_id: "selected"},
        True,
        True,
    )
    random_candidate = CandidateScore(
        **{
            **candidate.__dict__,
            "train_rank_ic": None,
            "validation_rank_ic": None,
            "minimum_score": None,
            "disposition": "random_label_free_eligible",
        }
    )
    libraries = {
        "heterogeneous_selected": tmp_path / "factor_libraries" / "selected",
        "random_typed": tmp_path / "factor_libraries" / "random",
        "label_permutation_selected": tmp_path / "factor_libraries" / "permutation",
    }
    identities = {
        "proposal": {"sha256": "1" * 64, "authority": "sole_proposal_authority"},
        "config": {"sha256": "2" * 64},
        "preregistration": {"sha256": "3" * 64},
        "data": {"sha256": "4" * 64},
        "code": {"tree_sha256": "5" * 64},
    }
    run_path = tmp_path / "artifacts" / "run"
    selection_config = {
        "library_cap": 1,
        "minimum_library_size": 1,
        "minimum_miner_profiles": 1,
        "minimum_coverage": 0.0,
        "maximum_absolute_validation_spearman": 0.80,
        "padding_forbidden": True,
    }

    manifest = publish_s2a_run(
        run_path,
        libraries,
        tiny_panel,
        attempts=(attempt,),
        observed=observed,
        permutation=permutation,
        observed_selection=observed_selection,
        permutation_selection=permutation_selection,
        random_selection=(random_candidate,),
        identities=identities,
        selection_config=selection_config,
        workspace=tmp_path,
        random_control_seed=8675309,
        random_control_period=None,
    )

    assert manifest["stage"] == "S2a"
    assert manifest["scientific_result"] is False
    assert manifest["final_claim_allowed"] is False
    assert (run_path / "attempts.jsonl").is_file()
    assert (run_path / "candidate_table_observed.jsonl").is_file()
    assert (run_path / "candidate_table_permutation.jsonl").is_file()
    assert (run_path / "data_access_ledger.jsonl").is_file()
    for role, path in libraries.items():
        library_manifest = json.loads((path / "manifest.json").read_text())
        assert library_manifest["library_role"] == role
        assert library_manifest["kan_mined"] is False
        assert library_manifest["scientific_result"] is False
        provenance = library_manifest["identities"]["selection_provenance"]
        if role == "random_typed":
            assert provenance["label_free"] is True
            assert provenance["label_sha256"] is None
            assert provenance["selection_config"]["random_control_seed"] == 8675309
        else:
            assert provenance["scoring_run_id"] in {"b" * 64, "d" * 64}

    before = {
        role: (path / "manifest.json").read_bytes() for role, path in libraries.items()
    }
    with pytest.raises(FileExistsError, match="refusing to replace"):
        publish_s2a_run(
            run_path,
            libraries,
            tiny_panel,
            attempts=(attempt,),
            observed=observed,
            permutation=permutation,
            observed_selection=observed_selection,
            permutation_selection=permutation_selection,
            random_selection=(random_candidate,),
            identities=identities,
            selection_config=selection_config,
            workspace=tmp_path,
            random_control_seed=8675309,
            random_control_period=None,
        )
    assert before == {
        role: (path / "manifest.json").read_bytes() for role, path in libraries.items()
    }

    failure_run = tmp_path / "artifacts" / "finalize_failure"
    failure_libraries = {
        role: tmp_path / "factor_libraries" / f"failure_{role}"
        for role in s2a.LIBRARY_ROLES
    }

    def fail_finalize(*args, **kwargs):
        raise RuntimeError("finalize boom")

    monkeypatch.setattr(s2a, "finalize_claimed_directory", fail_finalize)
    with pytest.raises(RuntimeError, match="finalize boom"):
        publish_s2a_run(
            failure_run,
            failure_libraries,
            tiny_panel,
            attempts=(attempt,),
            observed=observed,
            permutation=permutation,
            observed_selection=observed_selection,
            permutation_selection=permutation_selection,
            random_selection=(random_candidate,),
            identities=identities,
            selection_config=selection_config,
            workspace=tmp_path,
            random_control_seed=8675309,
            random_control_period=None,
        )
    terminal = json.loads((failure_run / "terminal_failure.json").read_text())
    assert terminal["error"] == "finalize boom"
    assert not (failure_run / ".INCOMPLETE").exists()
    for path in failure_libraries.values():
        assert not (path / ".INCOMPLETE").exists()
        assert (path / "terminal_failure.json").is_file()
        with pytest.raises(ValueError, match="exact file set"):
            verify_library(path, tiny_panel)


def test_mining_run_is_claimed_before_any_identity_or_label_access(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "factor_libraries").mkdir()
    destinations = {
        role: tmp_path / "factor_libraries" / role for role in s2a.LIBRARY_ROLES
    }
    run_path = tmp_path / "artifacts" / "claimed_failure"
    calls = []

    def fail_identity(workspace):
        calls.append("identity")
        raise ValueError("future implementation lock missing")

    monkeypatch.setattr(s2a, "verified_s2_identities", fail_identity)
    monkeypatch.setattr(
        s2a,
        "_load_screening_data",
        lambda *args: pytest.fail("labels must not be accessed after identity failure"),
    )

    with pytest.raises(ValueError, match="implementation lock"):
        s2a.run_s2a_mining(tmp_path, run_path, destinations)

    assert calls == ["identity"]
    manifest = json.loads((run_path / "manifest.json").read_text())
    assert manifest["publication_state"] == "terminal_failure"
    assert (run_path / "attempts.jsonl").is_file()
    assert not (run_path / ".INCOMPLETE").exists()
    with pytest.raises(FileExistsError, match="protocol-global mining topology"):
        s2a.run_s2a_mining(tmp_path, run_path, destinations)


def test_protocol_global_mining_entitlement_cannot_move_to_new_destinations(
    tmp_path,
) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "factor_libraries").mkdir()
    first_run = tmp_path / "artifacts" / "first"
    second_run = tmp_path / "artifacts" / "second"
    first_run.mkdir()
    second_run.mkdir()
    first_libraries = {
        role: tmp_path / "factor_libraries" / f"first_{role}"
        for role in s2a.LIBRARY_ROLES
    }
    second_libraries = {
        role: tmp_path / "factor_libraries" / f"second_{role}"
        for role in s2a.LIBRARY_ROLES
    }
    identities = {
        "implementation_lock": {"sha256": "i" * 64},
        "preregistration_lock": {"sha256": "p" * 64},
        "mining_preclaim": {"sha256": "c" * 64},
    }

    receipt = s2a.claim_s2a_mining_entitlement(
        tmp_path, first_run, first_libraries, identities
    )

    assert receipt["attempt_budget"] == 256
    with pytest.raises(FileExistsError):
        s2a.claim_s2a_mining_entitlement(
            tmp_path, second_run, second_libraries, identities
        )


def test_all_library_destinations_are_claimed_before_first_publication(
    tmp_path, tiny_panel, monkeypatch
) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "factor_libraries").mkdir()
    candidate = _candidate("trend_000", "trend")
    attempt = MiningAttempt("trend", 0, "trend_000", candidate.program, "generated")
    observed = ScoringRun(
        (candidate,),
        "observed",
        "a" * 64,
        "b" * 64,
        ("2016-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
    )
    permutation = ScoringRun(
        (candidate,),
        "within_date_permutation",
        "c" * 64,
        "d" * 64,
        observed.train,
        observed.validation,
    )
    selection = SelectionResult(
        (candidate,), {candidate.candidate_id: "selected"}, True, True
    )
    random_candidate = CandidateScore(
        **{
            **candidate.__dict__,
            "train_rank_ic": None,
            "validation_rank_ic": None,
            "minimum_score": None,
            "disposition": "random_label_free_eligible",
        }
    )
    libraries = {
        role: tmp_path / "factor_libraries" / role for role in s2a.LIBRARY_ROLES
    }

    def inspect_claims(*args, **kwargs):
        assert kwargs["preclaimed"] is True
        assert all((path / ".INCOMPLETE").is_file() for path in libraries.values())
        raise RuntimeError("publication stopped after topology claim")

    monkeypatch.setattr(s2a, "publish_library", inspect_claims)
    with pytest.raises(RuntimeError, match="topology claim"):
        publish_s2a_run(
            tmp_path / "artifacts" / "run",
            libraries,
            tiny_panel,
            attempts=(attempt,),
            observed=observed,
            permutation=permutation,
            observed_selection=selection,
            permutation_selection=selection,
            random_selection=(random_candidate,),
            identities={},
            selection_config={
                "library_cap": 1,
                "minimum_library_size": 1,
                "minimum_miner_profiles": 1,
                "minimum_coverage": 0.0,
                "maximum_absolute_validation_spearman": 0.80,
            },
            workspace=tmp_path,
        )
    assert not (tmp_path / "artifacts" / "run" / ".INCOMPLETE").exists()
    assert (tmp_path / "artifacts" / "run" / "terminal_failure.json").is_file()
    for path in libraries.values():
        assert not (path / ".INCOMPLETE").exists()
        terminal = json.loads((path / "terminal_failure.json").read_text())
        assert terminal["publication_state"] == "terminal_failure"


def test_library_claim_failure_terminalizes_all_prior_claims(
    tmp_path, tiny_panel, monkeypatch
) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "factor_libraries").mkdir()
    libraries = {
        role: tmp_path / "factor_libraries" / role for role in s2a.LIBRARY_ROLES
    }
    candidate = _candidate("trend_000", "trend")
    attempt = MiningAttempt("trend", 0, "trend_000", candidate.program, "generated")
    scoring = ScoringRun(
        (candidate,), "observed", "a" * 64, "b" * 64,
        ("2016-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31")
    )
    selection = SelectionResult(
        (candidate,), {candidate.candidate_id: "selected"}, True, True
    )
    random_candidate = CandidateScore(
        **{
            **candidate.__dict__,
            "train_rank_ic": None,
            "validation_rank_ic": None,
            "minimum_score": None,
            "disposition": "random_label_free_eligible",
        }
    )
    original_claim = s2a.claim_artifact_directory
    calls = 0

    def fail_second_library_claim(path):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("claim race")
        return original_claim(path)

    monkeypatch.setattr(s2a, "claim_artifact_directory", fail_second_library_claim)
    with pytest.raises(RuntimeError, match="claim race"):
        publish_s2a_run(
            tmp_path / "artifacts" / "run",
            libraries,
            tiny_panel,
            attempts=(attempt,),
            observed=scoring,
            permutation=scoring,
            observed_selection=selection,
            permutation_selection=selection,
            random_selection=(random_candidate,),
            identities={},
            selection_config={
                "library_cap": 1,
                "minimum_library_size": 1,
                "minimum_miner_profiles": 1,
                "minimum_coverage": 0.0,
                "maximum_absolute_validation_spearman": 0.80,
            },
            workspace=tmp_path,
        )
    assert not (tmp_path / "artifacts" / "run" / ".INCOMPLETE").exists()
    first = libraries[s2a.LIBRARY_ROLES[0]]
    assert not (first / ".INCOMPLETE").exists()
    assert (first / "terminal_failure.json").is_file()


def test_second_library_failure_invalidates_already_published_first_library(
    tmp_path, tiny_panel, monkeypatch
) -> None:
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "factor_libraries").mkdir()
    libraries = {
        role: tmp_path / "factor_libraries" / role for role in s2a.LIBRARY_ROLES
    }
    candidate = _candidate("trend_000", "trend")
    attempt = MiningAttempt("trend", 0, "trend_000", candidate.program, "generated")
    scoring = ScoringRun(
        (candidate,), "observed", "a" * 64, "b" * 64,
        ("2016-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31")
    )
    selection = SelectionResult(
        (candidate,), {candidate.candidate_id: "selected"}, True, True
    )
    random_candidate = CandidateScore(
        **{
            **candidate.__dict__,
            "train_rank_ic": None,
            "validation_rank_ic": None,
            "minimum_score": None,
            "disposition": "random_label_free_eligible",
        }
    )
    original_publish = s2a.publish_library
    calls = 0

    def fail_second_publish(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second publish failed")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(s2a, "publish_library", fail_second_publish)
    with pytest.raises(RuntimeError, match="second publish failed"):
        publish_s2a_run(
            tmp_path / "artifacts" / "run",
            libraries,
            tiny_panel,
            attempts=(attempt,),
            observed=scoring,
            permutation=scoring,
            observed_selection=selection,
            permutation_selection=selection,
            random_selection=(random_candidate,),
            identities={},
            selection_config={
                "library_cap": 1,
                "minimum_library_size": 1,
                "minimum_miner_profiles": 1,
                "minimum_coverage": 0.0,
                "maximum_absolute_validation_spearman": 0.80,
            },
            workspace=tmp_path,
        )
    for path in libraries.values():
        assert not (path / ".INCOMPLETE").exists()
        assert (path / "terminal_failure.json").is_file()
