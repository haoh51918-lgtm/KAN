from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest
import torch


IDENTITIES = {
    "protocol_sha256": "1" * 64,
    "authority_sha256": "2" * 64,
    "implementation_sha256": "3" * 64,
}
TOP_BUDGET = {
    "kan_attempts": 256,
    "kan_updates": 256 * 300,
    "gp_attempts": 256,
    "permutation_attempts": 256,
    "mlp_controls": 8,
    "mlp_updates": 8 * 300,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_run(profile: str):
    from mirage_kan.mining.e3 import (
        CategoricalE3KAN,
        PROFILE_SPECS,
        build_profile_atom_bank,
        forward_mode_at_step,
        harden_checkpoint,
        temperature_at_step,
    )
    from mirage_kan.mining.e3_runner import (
        MINER_SEED_BASE,
        MinerReceipt,
        ProfileRun,
        TrainingStepReceipt,
        draw_training_bootstrap,
    )

    profile_index = tuple(PROFILE_SPECS).index(profile)
    bank = build_profile_atom_bank(profile)
    miners = []
    for miner_index in range(64):
        seed = MINER_SEED_BASE + profile_index * 1000 + miner_index
        logits = CategoricalE3KAN(profile, seed=seed).checkpoint_logits()
        hardening = harden_checkpoint(logits, bank, 0.1)
        trajectory = tuple(
            TrainingStepReceipt(
                update_index=index,
                tau=temperature_at_step(index),
                mode=forward_mode_at_step(index),
                total_loss=-0.01,
                mean_daily_ic=0.01,
                entropy=0.2,
                edge_overlap=0.01,
                gate_logits=logits,
            )
            for index in range(300)
        )
        global_index = profile_index * 64 + miner_index
        miners.append(
            MinerReceipt(
                profile=profile,
                profile_index=profile_index,
                miner_index=miner_index,
                global_attempt_index=global_index,
                miner_seed=seed,
                bootstrap=draw_training_bootstrap(20, 49979687 + global_index),
                initial_logits=logits,
                final_logits=logits,
                first_step_data_gradient=torch.zeros_like(logits),
                trajectory=trajectory,
                hardening=hardening,
                candidate_ast=hardening.ast,
                fidelity={"pearson": 0.99, "nrmse": 0.01},
                admission_failures=(),
            )
        )
    return ProfileRun(profile=profile, device="cpu", miners=tuple(miners))


@lru_cache(maxsize=1)
def _profile_runs():
    from mirage_kan.mining.e3 import PROFILE_SPECS

    return tuple(_profile_run(profile) for profile in PROFILE_SPECS)


def _score_from_miner(miner, *, label_mode: str, factor_count: int = 8):
    from mirage_kan.mining.v2_scoring import HardAstScore

    production = label_mode == "real" and miner.global_attempt_index < factor_count
    return HardAstScore(
        candidate_id=f"kan_{miner.profile}_{miner.miner_index:03d}",
        profile=miner.profile,
        attempt_index=miner.global_attempt_index,
        method="kan_e3",
        ast=miner.candidate_ast,
        canonical_hash=miner.candidate_ast.identity,
        ast_depth=2,
        ast_nodes=3,
        output_type="dimensionless_ts",
        causal=True,
        unique=True,
        support_rows=100,
        eligible_rows=100,
        coverage=1.0,
        train_rank_ic=0.01,
        validation_rank_ic=0.01,
        sign_agreement=True,
        minimum_score=0.01,
        fidelity_pearson=0.99,
        fidelity_nrmse=0.01,
        gate_probability_margin=0.1,
        fidelity_gate_met=True,
        lineage_gate_met=True,
        production_eligible=production,
        null_eligible=True,
        production_disposition="admitted" if production else "below_threshold",
        null_disposition="eligible",
        validation_values=pd.Series(dtype=float),
    )


def _selection(candidates, selected, *, target_size="exact"):
    from mirage_kan.mining.v2_scoring import HardAstSelection

    selected_ids = {candidate.candidate_id for candidate in selected}
    return HardAstSelection(
        selected=tuple(selected),
        dispositions={
            candidate.candidate_id: (
                "selected" if candidate.candidate_id in selected_ids else "not_selected"
            )
            for candidate in candidates
        },
        minimum_size_met=True,
        exact_size_met=True,
        profile_quota_met=True,
        target_size=len(selected) if target_size == "exact" else None,
    )


@lru_cache(maxsize=None)
def _experiment_evidence(factor_count: int = 8):
    from mirage_kan.mining.gp_control import generate_gp_attempts
    from mirage_kan.mining.v2_scoring import HardAstScore, HardAstScoringRun

    runs = _profile_runs()
    permutation_runs = tuple(
        replace(run, miners=tuple(replace(miner) for miner in run.miners))
        for run in runs
    )
    real_scores = tuple(
        _score_from_miner(miner, label_mode="real", factor_count=factor_count)
        for run in runs
        for miner in run.miners
    )
    permutation_scores = tuple(
        _score_from_miner(
            miner,
            label_mode="within_date_permutation",
            factor_count=factor_count,
        )
        for run in permutation_runs
        for miner in run.miners
    )
    real_scoring = HardAstScoringRun(
        real_scores,
        "real",
        "6" * 64,
        "kan-real-score",
        ("2016-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
    )
    permutation_scoring = HardAstScoringRun(
        permutation_scores,
        "within_date_permutation",
        "7" * 64,
        "kan-permutation-score",
        ("2016-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
    )
    real_selection = _selection(
        real_scores, real_scores[:factor_count], target_size=None
    )
    permutation_selection = _selection(
        permutation_scores, permutation_scores[:factor_count]
    )

    generation = generate_gp_attempts(lambda ast, context: 0.01)
    gp_scores = tuple(
        HardAstScore(
            candidate_id=attempt.candidate_id,
            profile=attempt.profile,
            attempt_index=attempt.global_attempt_index,
            method="typed_gp_sr",
            ast=attempt.ast,
            canonical_hash=attempt.canonical_hash,
            ast_depth=2 if attempt.ast is not None else None,
            ast_nodes=attempt.ast_nodes,
            output_type="dimensionless_ts" if attempt.ast is not None else None,
            causal=attempt.ast is not None,
            unique=attempt.disposition == "generated",
            support_rows=100 if attempt.ast is not None else 0,
            eligible_rows=100,
            coverage=1.0 if attempt.ast is not None else 0.0,
            train_rank_ic=attempt.train_rank_ic,
            validation_rank_ic=0.01 if attempt.ast is not None else None,
            sign_agreement=attempt.ast is not None,
            minimum_score=0.01 if attempt.ast is not None else None,
            fidelity_pearson=None,
            fidelity_nrmse=None,
            gate_probability_margin=None,
            fidelity_gate_met=True,
            lineage_gate_met=True,
            production_eligible=attempt.disposition == "generated",
            null_eligible=attempt.disposition == "generated",
            production_disposition=attempt.disposition,
            null_disposition=attempt.disposition,
            validation_values=pd.Series(dtype=float),
        )
        for attempt in generation.attempts
    )
    gp_scoring = HardAstScoringRun(
        gp_scores,
        "real",
        "8" * 64,
        "gp-score",
        ("2016-01-01", "2020-12-31"),
        ("2021-01-01", "2021-12-31"),
    )
    gp_selected = tuple(score for score in gp_scores if score.production_eligible)[
        :factor_count
    ]
    gp_selection = _selection(gp_scores, gp_selected)
    return {
        "kan_profile_runs": runs,
        "kan_scoring": real_scoring,
        "kan_selection": real_selection,
        "gp_generation": generation,
        "gp_scoring": gp_scoring,
        "gp_selection": gp_selection,
        "permutation_profile_runs": permutation_runs,
        "permutation_scoring": permutation_scoring,
        "permutation_null_selection": permutation_selection,
    }


def _published_children(tmp_path: Path, *, factor_count: int = 8) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    topology_sha = "a" * 64
    evidence = _experiment_evidence(factor_count)
    selected_ids = [
        candidate.candidate_id for candidate in evidence["kan_selection"].selected
    ]
    selected_miners = _profile_runs()[0].miners[:factor_count]
    selected_factors = {
        factor_id: {
            "canonical_hash": miner.candidate_ast.identity,
            "global_attempt_index": miner.global_attempt_index,
        }
        for factor_id, miner in zip(selected_ids, selected_miners, strict=True)
    }
    anonymous_mapping = {
        f"B{index:03d}": factor_id
        for index, factor_id in enumerate(sorted(selected_ids), start=1)
    }
    mapping_sha256 = hashlib.sha256(
        (json.dumps(anonymous_mapping, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    gp_control_factors = {
        candidate.candidate_id: {"canonical_hash": candidate.canonical_hash}
        for candidate in evidence["gp_selection"].selected
    }
    permutation_control_factors = {
        candidate.candidate_id: {"canonical_hash": candidate.canonical_hash}
        for candidate in evidence["permutation_null_selection"].selected
    }
    definitions = {
        "kan_library": {
            "schema_version": "mirage_factor_library_v1",
            "library_role": "kan_e3_selected",
            "role": "kan_e3_selected",
            "output_kind": "factor_library",
            "kan_mined": True,
            "promotion_eligible": True,
            "factor_count": factor_count,
            "factors": selected_factors,
        },
        "gp_control_library": {
            "schema_version": "mirage_factor_library_v1",
            "library_role": "typed_gp_sr_control",
            "role": "typed_gp_sr_control",
            "output_kind": "factor_library",
            "kan_mined": False,
            "promotion_eligible": False,
            "factor_count": factor_count,
            "factors": gp_control_factors,
        },
        "permutation_control_library": {
            "schema_version": "mirage_factor_library_v1",
            "library_role": "kan_e3_permutation_control",
            "role": "kan_e3_permutation_control",
            "output_kind": "factor_library",
            "kan_mined": False,
            "promotion_eligible": False,
            "factor_count": factor_count,
            "factors": permutation_control_factors,
        },
        "blackbox_control": {
            "schema_version": "mirage_matched_blackbox_control_v2",
            "role": "falsification_control_never_production",
            "output_kind": "control_panel_not_factor_library",
            "kan_mined": False,
            "promotion_eligible": False,
            "factor_library_publication_allowed": False,
            "control_count": factor_count,
            "selected_kan_factor_ids": selected_ids,
            "paired_kan_global_attempt_indices": list(range(factor_count)),
        },
        "mechanism_cards": {
            "schema_version": "mirage_s2a_v2_staging_bundle_v1",
            "role": "kan_mechanism_evidence_pending_human_review",
            "output_kind": "mechanism_cards",
            "kan_mined": True,
            "promotion_eligible": False,
            "card_count": factor_count,
            "selected_factor_ids": selected_ids,
            "anonymous_mapping_sha256": mapping_sha256,
        },
        "blind_review_package": {
            "schema_version": "mirage_s2a_v2_staging_bundle_v1",
            "role": "human_blind_review_input",
            "output_kind": "blind_review_package",
            "kan_mined": False,
            "promotion_eligible": False,
            "blind_item_count": factor_count,
            "anonymous_mapping_sha256": mapping_sha256,
        },
    }
    result: dict[str, Path] = {}
    for key, fields in definitions.items():
        path = tmp_path / f"published_{key}"
        path.mkdir()
        payload = path / "payload.bin"
        payload.write_bytes(key.encode())
        manifest = {
            **fields,
            "identities": IDENTITIES,
            "files": {"payload.bin": _sha256(payload)},
            "topology_sha256": topology_sha,
            "topology_key": key,
        }
        (path / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result[key] = path
    return result


def _card(factor_id: str) -> dict[str, object]:
    return {
        "identity_and_canonical_ast": {
            "factor_id": factor_id,
            "canonical_hash": "4" * 64,
        },
        "raw_variables_windows_and_complexity": {"windows": [5]},
        "edge_gate_and_shape_summary": {"shape": "sub"},
        "soft_hard_fidelity": {"pearson": 0.99, "nrmse": 0.01},
        "variable_and_lag_interventions": {"variable": {"Close": 0.1}},
        "local_counterfactual_response": {"up": 0.1},
        "one_sentence_mechanism": "price response",
        "applicability_and_failure_conditions": {"failure_conditions": "missing"},
        "complete_kan_lineage_and_hardening_receipt": {"miner": "kan_e3"},
    }


def _blind_item() -> dict[str, object]:
    card = _card("hidden")
    item = {
        key: value
        for key, value in card.items()
        if key
        not in {
            "soft_hard_fidelity",
            "complete_kan_lineage_and_hardening_receipt",
            "local_counterfactual_response",
        }
    }
    item["response_direction_questions"] = [
        {"question_id": "positive", "prompt": "increase or decrease?"},
        {"question_id": "negative", "prompt": "increase or decrease?"},
    ]
    return item


def test_v2_factor_library_staging_is_executable_and_enriched(
    tmp_path: Path, tiny_panel
) -> None:
    from mirage_kan.artifacts.library import verify_library
    from mirage_kan.artifacts.v2_bundle import stage_v2_factor_library
    from mirage_kan.mining.e3 import build_profile_atom_bank

    programs = {
        f"f{index:03d}": atom.ast
        for index, atom in enumerate(build_profile_atom_bank("short_price")[:8])
    }
    lineage = {
        factor_id: {
            "canonical_hash": ast.identity,
            "global_attempt_index": index,
        }
        for index, (factor_id, ast) in enumerate(programs.items())
    }
    result = stage_v2_factor_library(
        tmp_path / "library.staging",
        programs,
        tiny_panel,
        topology_key="kan_library",
        identities=IDENTITIES,
        minimum_library_size=8,
        library_cap=16,
        factor_lineage=lineage,
    )

    assert verify_library(result.path, tiny_panel)["verified"] is True
    assert result.manifest["library_role"] == "kan_e3_selected"
    assert result.manifest["output_kind"] == "factor_library"
    assert result.manifest["promotion_eligible"] is True
    assert all(
        "global_attempt_index" in record
        for record in result.manifest["factors"].values()
    )


def test_top_bundle_contains_all_four_profile_runs_and_hashes_published_children(
    tmp_path: Path,
) -> None:
    from mirage_kan.artifacts.v2_bundle import stage_mining_top_bundle

    evidence = _experiment_evidence()
    children = _published_children(tmp_path)
    first = stage_mining_top_bundle(
        tmp_path / "first.staging",
        children,
        **evidence,
        identities=IDENTITIES,
        budget_counts=TOP_BUDGET,
        minimum_library_size=8,
        library_cap=16,
    )
    second = stage_mining_top_bundle(
        tmp_path / "second.staging",
        children,
        **evidence,
        identities=IDENTITIES,
        budget_counts=TOP_BUDGET,
        minimum_library_size=8,
        library_cap=16,
    )

    assert first.manifest["files"] == second.manifest["files"]
    assert first.manifest["budget_counts"] == TOP_BUDGET
    assert first.manifest["kan_profile_evidence"] == {
        "profiles": 4,
        "miners": 256,
        "completed_updates": 256 * 300,
    }
    rows = (first.path / "kan_profile_runs.jsonl").read_text().splitlines()
    assert len(rows) == 256
    permutation_rows = (
        (first.path / "kan_permutation_profile_runs.jsonl").read_text().splitlines()
    )
    assert len(permutation_rows) == 256
    assert len(json.loads(rows[0])["trajectory"]) == 300
    assert len((first.path / "gp_generation.jsonl").read_text().splitlines()) == 256
    assert len((first.path / "kan_real_scoring.jsonl").read_text().splitlines()) == 256
    assert (
        len(
            (first.path / "kan_permutation_false_positive_ledger.jsonl")
            .read_text()
            .splitlines()
        )
        == 256
    )
    assert (
        first.manifest["permutation_ledger"]["real_threshold_false_positive_count"] == 0
    )
    assert (first.path / "kan_tensor_evidence.zlib").stat().st_size > 0
    assert list(first.path.iterdir())[-1].name == "manifest.json"
    for key, path in children.items():
        assert first.manifest["child_manifest_sha256"][key] == _sha256(
            path / "manifest.json"
        )
        assert (
            first.manifest["child_manifests"][key]
            == first.manifest["child_manifest_sha256"][key]
        )


def test_top_bundle_rejects_crossed_blackbox_kan_pairing(tmp_path: Path) -> None:
    from mirage_kan.artifacts.v2_bundle import stage_mining_top_bundle

    evidence = _experiment_evidence()
    children = _published_children(tmp_path)
    manifest_path = children["blackbox_control"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paired = manifest["paired_kan_global_attempt_indices"]
    paired[0], paired[1] = paired[1], paired[0]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="blackbox controls do not match"):
        stage_mining_top_bundle(
            tmp_path / "crossed-pairing.staging",
            children,
            **evidence,
            identities=IDENTITIES,
            budget_counts=TOP_BUDGET,
            minimum_library_size=8,
            library_cap=16,
        )


def test_top_bundle_rejects_orphaned_profile_budget_spoofed_child_and_alias(
    tmp_path: Path,
) -> None:
    from mirage_kan.artifacts.v2_bundle import stage_mining_top_bundle

    evidence = _experiment_evidence()
    runs = evidence["kan_profile_runs"]
    children = _published_children(tmp_path)
    with pytest.raises(ValueError, match="four KAN profiles"):
        stage_mining_top_bundle(
            tmp_path / "missing-profile.staging",
            children,
            **{**evidence, "kan_profile_runs": runs[:-1]},
            identities=IDENTITIES,
            budget_counts=TOP_BUDGET,
            minimum_library_size=8,
            library_cap=16,
        )
    bad_manifest = children["gp_control_library"] / "manifest.json"
    bad = json.loads(bad_manifest.read_text())
    bad["kan_mined"] = True
    bad_manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="gp_control_library.*kan_mined"):
        stage_mining_top_bundle(
            tmp_path / "spoof.staging",
            children,
            **evidence,
            identities=IDENTITIES,
            budget_counts=TOP_BUDGET,
            minimum_library_size=8,
            library_cap=16,
        )
    children = _published_children(tmp_path / "fresh")
    children["gp_control_library"] = children["kan_library"]
    with pytest.raises(ValueError, match="alias"):
        stage_mining_top_bundle(
            tmp_path / "alias.staging",
            children,
            **evidence,
            identities=IDENTITIES,
            budget_counts=TOP_BUDGET,
            minimum_library_size=8,
            library_cap=16,
        )


def test_kan_lineage_seed_bootstrap_schedule_and_last_checkpoint_fail_closed(
    tmp_path: Path,
) -> None:
    from mirage_kan.artifacts.v2_bundle import stage_mining_top_bundle

    evidence = _experiment_evidence()
    runs = list(evidence["kan_profile_runs"])
    children = _published_children(tmp_path)
    bad_miner = replace(runs[0].miners[0], miner_seed=7)
    runs[0] = replace(runs[0], miners=(bad_miner, *runs[0].miners[1:]))
    with pytest.raises(ValueError, match="seed"):
        stage_mining_top_bundle(
            tmp_path / "seed.staging",
            children,
            **{**evidence, "kan_profile_runs": runs},
            identities=IDENTITIES,
            budget_counts=TOP_BUDGET,
            minimum_library_size=8,
            library_cap=16,
        )
    runs = list(evidence["kan_profile_runs"])
    last = replace(
        runs[0].miners[0].trajectory[-1],
        gate_logits=torch.ones_like(runs[0].miners[0].final_logits),
    )
    bad_miner = replace(
        runs[0].miners[0], trajectory=(*runs[0].miners[0].trajectory[:-1], last)
    )
    runs[0] = replace(runs[0], miners=(bad_miner, *runs[0].miners[1:]))
    with pytest.raises(ValueError, match="final logits"):
        stage_mining_top_bundle(
            tmp_path / "last.staging",
            children,
            **{**evidence, "kan_profile_runs": runs},
            identities=IDENTITIES,
            budget_counts=TOP_BUDGET,
            minimum_library_size=8,
            library_cap=16,
        )


def _mlp_panel(control_count: int = 8):
    from mirage_kan.mining.e3 import atom_manifest_sha256, build_profile_atom_bank
    from mirage_kan.mining.e3_runner import draw_training_bootstrap
    from mirage_kan.mining.mlp_control import (
        MLPControlReceipt,
        MLPTrainingStepReceipt,
        MatchedBlackboxControlPanel,
    )

    parameters = torch.arange(69, dtype=torch.float64)
    trajectory = tuple(
        MLPTrainingStepReceipt(index, -0.01, 0.01, parameters) for index in range(300)
    )
    controls = tuple(
        MLPControlReceipt(
            profile="short_price",
            kan_global_attempt_index=index,
            seed=32452843 + index,
            bootstrap=draw_training_bootstrap(20, 49979687 + index),
            optimizer="Adam",
            learning_rate=0.03,
            scheduled_updates=300,
            completed_updates=300,
            input_atom_count=32,
            kan_parameter_count=64,
            mlp_parameter_count=69,
            parameter_relative_gap=5 / 64,
            atom_manifest_sha256=atom_manifest_sha256(
                build_profile_atom_bank("short_price")
            ),
            valid_support_sha256="5" * 64,
            initial_parameters=parameters,
            final_parameters=parameters,
            first_step_data_gradient=parameters,
            trajectory=trajectory,
            prediction=torch.ones((20, 2), dtype=torch.float64),
            prediction_mask=torch.ones((20, 2), dtype=torch.bool),
        )
        for index in range(control_count)
    )
    return MatchedBlackboxControlPanel(controls)


def test_mlp_control_validates_pairing_lineage_and_prediction_evidence(
    tmp_path: Path, tiny_panel
) -> None:
    from mirage_kan.artifacts.v2_bundle import stage_mlp_control_panel

    panel = _mlp_panel()
    factor_ids = tuple(f"f{index:03d}" for index in range(8))
    result = stage_mlp_control_panel(
        tmp_path / "mlp.staging",
        panel,
        tiny_panel,
        selected_kan_factor_ids=factor_ids,
        identities=IDENTITIES,
        minimum_library_size=8,
        library_cap=16,
    )
    assert len((result.path / "control_receipts.jsonl").read_text().splitlines()) == 8
    assert result.manifest["budget_counts"]["completed_updates"] == 2400
    assert result.manifest["promotion_eligible"] is False
    assert result.manifest["schema_version"] == "mirage_matched_blackbox_control_v2"
    assert (result.path / "prediction_panel.parquet").is_file()
    bad = replace(panel.controls[0], seed=1)
    with pytest.raises(ValueError, match="seed"):
        stage_mlp_control_panel(
            tmp_path / "bad-mlp.staging",
            replace(panel, controls=(bad, *panel.controls[1:])),
            tiny_panel,
            selected_kan_factor_ids=factor_ids,
            identities=IDENTITIES,
            minimum_library_size=8,
            library_cap=16,
        )
    bad = replace(panel.controls[0], parameter_relative_gap=0.2)
    with pytest.raises(ValueError, match="parameter gap"):
        stage_mlp_control_panel(
            tmp_path / "gap-mlp.staging",
            replace(panel, controls=(bad, *panel.controls[1:])),
            tiny_panel,
            selected_kan_factor_ids=factor_ids,
            identities=IDENTITIES,
            minimum_library_size=8,
            library_cap=16,
        )


def test_v5_artifact_bounds_accept_six_and_reject_five_or_seventeen(
    tmp_path: Path, tiny_panel
) -> None:
    from mirage_kan.artifacts.v2_bundle import (
        stage_mining_top_bundle,
        stage_mlp_control_panel,
        stage_v2_factor_library,
    )
    from mirage_kan.mining.e3 import build_profile_atom_bank

    atoms = build_profile_atom_bank("short_price")
    programs = {f"v4_{index:03d}": atom.ast for index, atom in enumerate(atoms[:6])}
    library = stage_v2_factor_library(
        tmp_path / "v4-library.staging",
        programs,
        tiny_panel,
        topology_key="gp_control_library",
        identities=IDENTITIES,
        minimum_library_size=6,
        library_cap=16,
    )
    assert library.manifest["factor_count"] == 6

    factor_ids = tuple(f"v4_{index:03d}" for index in range(6))
    mlp = stage_mlp_control_panel(
        tmp_path / "v4-mlp.staging",
        _mlp_panel(6),
        tiny_panel,
        selected_kan_factor_ids=factor_ids,
        identities=IDENTITIES,
        minimum_library_size=6,
        library_cap=16,
    )
    assert mlp.manifest["control_count"] == 6

    evidence = _experiment_evidence(6)
    children = _published_children(tmp_path / "v4-children", factor_count=6)
    budget = {**TOP_BUDGET, "mlp_controls": 6, "mlp_updates": 6 * 300}
    top = stage_mining_top_bundle(
        tmp_path / "v4-top.staging",
        children,
        **evidence,
        identities=IDENTITIES,
        budget_counts=budget,
        minimum_library_size=6,
        library_cap=16,
    )
    assert top.manifest["budget_counts"]["mlp_controls"] == 6

    for invalid in (5, 17):
        invalid_programs = {
            f"invalid_{invalid}_{index:03d}": atom.ast
            for index, atom in enumerate(atoms[:invalid])
        }
        with pytest.raises(ValueError, match="frozen admission bounds"):
            stage_v2_factor_library(
                tmp_path / f"invalid-library-{invalid}.staging",
                invalid_programs,
                tiny_panel,
                topology_key="gp_control_library",
                identities=IDENTITIES,
                minimum_library_size=6,
                library_cap=16,
            )
        with pytest.raises(ValueError, match="frozen admission bounds"):
            stage_mlp_control_panel(
                tmp_path / f"invalid-mlp-{invalid}.staging",
                _mlp_panel(invalid),
                tiny_panel,
                selected_kan_factor_ids=tuple(
                    f"invalid_{invalid}_{index:03d}" for index in range(invalid)
                ),
                identities=IDENTITIES,
                minimum_library_size=6,
                library_cap=16,
            )
        invalid_evidence = _experiment_evidence(invalid)
        invalid_children = _published_children(
            tmp_path / f"invalid-children-{invalid}", factor_count=invalid
        )
        invalid_budget = {
            **TOP_BUDGET,
            "mlp_controls": invalid,
            "mlp_updates": invalid * 300,
        }
        with pytest.raises(ValueError, match="frozen admission bounds"):
            stage_mining_top_bundle(
                tmp_path / f"invalid-top-{invalid}.staging",
                invalid_children,
                **invalid_evidence,
                identities=IDENTITIES,
                budget_counts=invalid_budget,
                minimum_library_size=6,
                library_cap=16,
            )


def test_cards_and_blind_package_require_exact_schema_ids_mapping_and_hiding(
    tmp_path: Path,
) -> None:
    from mirage_kan.artifacts.v2_bundle import (
        stage_blind_review_package,
        stage_mechanism_cards,
    )

    ids = ("f2", "f1")
    cards = stage_mechanism_cards(
        tmp_path / "cards.staging",
        {factor_id: _card(factor_id) for factor_id in ids},
        selected_factor_ids=ids,
        identities=IDENTITIES,
    )
    package = {
        "review_status": "pending_human_review",
        "reviewers_minimum": 2,
        "mechanism_restatement_required": True,
        "response_direction_accuracy_minimum": 0.8,
        "inter_reviewer_agreement_reported": True,
        "items": {"B001": _blind_item(), "B002": _blind_item()},
    }
    blind = stage_blind_review_package(
        tmp_path / "blind.staging",
        package,
        selected_factor_ids=ids,
        hides=("method_name", "pnl", "return_metrics"),
        identities=IDENTITIES,
    )

    mapping = json.loads((cards.path / "blind_anonymous_mapping.json").read_text())
    assert mapping == {"B001": "f1", "B002": "f2"}
    assert cards.manifest["card_count"] == 2
    assert blind.manifest["blind_item_count"] == 2
    assert blind.manifest["hides"] == ["method_name", "pnl", "return_metrics"]
    with pytest.raises(ValueError, match="nine fields"):
        stage_mechanism_cards(
            tmp_path / "bad-card.staging",
            {"f1": {"identity_and_canonical_ast": {"factor_id": "f1"}}},
            selected_factor_ids=("f1",),
            identities=IDENTITIES,
        )
    with pytest.raises(ValueError, match="hiding policy"):
        stage_blind_review_package(
            tmp_path / "bad-blind.staging",
            package,
            selected_factor_ids=ids,
            hides=("pnl",),
            identities=IDENTITIES,
        )


def test_nonfinite_and_existing_staging_fail_before_publication(tmp_path: Path) -> None:
    from mirage_kan.artifacts.v2_bundle import stage_mechanism_cards

    existing = tmp_path / "existing.staging"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        stage_mechanism_cards(
            existing,
            {"f": _card("f")},
            selected_factor_ids=("f",),
            identities=IDENTITIES,
        )
    card = _card("f")
    card["soft_hard_fidelity"] = {"pearson": float("nan"), "nrmse": 0.1}
    with pytest.raises(ValueError, match="finite"):
        stage_mechanism_cards(
            tmp_path / "nonfinite.staging",
            {"f": card},
            selected_factor_ids=("f",),
            identities=IDENTITIES,
        )
