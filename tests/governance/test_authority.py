from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from mirage_kan.governance.authority import (
    AuthorityGuard,
    AuthorityReceipt,
    AuthoritySuperseded,
)


BOUNDARIES = (
    "before_first_label_access",
    "before_each_scientific_or_control_arm",
    "before_each_artifact_publication",
    "before_development_opening",
    "before_final_decision_publication",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority_workspace(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    paths = {
        "proposal": tmp_path / "KAN_Alpha_PR.md",
        "config": tmp_path / "configs" / "experiments" / "protocol.yaml",
        "preregistration": tmp_path / "prereg" / "protocol.md",
        "directive": tmp_path / "governance" / "decisions" / "directive.md",
        "incident": tmp_path / "governance" / "incidents" / "incident.md",
        "predecessor": (tmp_path / "governance" / "incidents" / "predecessor.json"),
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {name}\n", encoding="utf-8")
    config = {
        "protocol_id": "s2a_kan_e3_vertical_v8",
        "authority_revalidation": {"boundaries": list(BOUNDARIES)},
        "controls": {
            "arms": [
                "alpha158_replay",
                "kan_e3_selected",
                "typed_gp_sr_control",
                "matched_blackbox_control",
                "kan_e3_permutation_control",
            ]
        },
    }
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    lock = {
        "schema_version": "mirage_s2_prereg_lock_v2",
        "protocol": {
            "protocol_id": "s2a_kan_e3_vertical_v8",
            "path": str(paths["config"].relative_to(tmp_path)),
            "sha256": _sha256(paths["config"]),
        },
        "proposal": {
            "authority": "sole_proposal_authority",
            "path": "KAN_Alpha_PR.md",
            "sha256": _sha256(paths["proposal"]),
        },
        "preregistration": {
            "path": str(paths["preregistration"].relative_to(tmp_path)),
            "sha256": _sha256(paths["preregistration"]),
        },
        "governance": {
            "active_directive_path": str(paths["directive"].relative_to(tmp_path)),
            "active_directive_sha256": _sha256(paths["directive"]),
            "supersession_incident_path": str(paths["incident"].relative_to(tmp_path)),
            "supersession_incident_sha256": _sha256(paths["incident"]),
        },
        "predecessor_custody": {
            "protocol_id": "terminal_predecessor",
            "scientific_observation": "none",
            "files": {
                str(paths["predecessor"].relative_to(tmp_path)): _sha256(
                    paths["predecessor"]
                )
            },
        },
    }
    lock_path = tmp_path / "prereg" / "s2a_kan_e3_vertical_v8.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    paths["base_lock"] = lock_path
    return tmp_path, paths


def test_guard_records_every_frozen_boundary_with_monotonic_receipts(
    tmp_path: Path,
) -> None:
    workspace, _ = _authority_workspace(tmp_path)
    guard = AuthorityGuard(workspace)
    labels = {
        "before_each_scientific_or_control_arm": "kan_e3_selected",
        "before_each_artifact_publication": "kan_library",
    }

    receipts = [
        guard.revalidate(boundary, arm=labels.get(boundary)) for boundary in BOUNDARIES
    ]

    assert all(isinstance(receipt, AuthorityReceipt) for receipt in receipts)
    assert [receipt.sequence for receipt in receipts] == [1, 2, 3, 4, 5]
    assert all(len(receipt.authority_sha256) == 64 for receipt in receipts)
    assert all(len(receipt.receipt_sha256) == 64 for receipt in receipts)
    assert all(
        guard.verify_capability(
            receipt.capability, boundary=receipt.boundary, arm=receipt.arm
        )
        is receipt
        for receipt in receipts
    )
    ledger = guard.ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 5
    assert [json.loads(line)["sequence"] for line in ledger] == [1, 2, 3, 4, 5]


def test_guard_accepts_only_the_frozen_predecessor_observation_classes(
    tmp_path: Path,
) -> None:
    workspace, paths = _authority_workspace(tmp_path)
    lock = json.loads(paths["base_lock"].read_text(encoding="utf-8"))
    lock["predecessor_custody"]["scientific_observation"] = (
        "pre_development_admission_count_only"
    )
    paths["base_lock"].write_text(json.dumps(lock), encoding="utf-8")

    assert AuthorityGuard(workspace).protocol_id == "s2a_kan_e3_vertical_v8"

    lock["predecessor_custody"]["scientific_observation"] = (
        "inconclusive_infrastructure_with_quarantined_development_outputs"
    )
    paths["base_lock"].write_text(json.dumps(lock), encoding="utf-8")
    assert AuthorityGuard(workspace).protocol_id == "s2a_kan_e3_vertical_v8"

    lock["predecessor_custody"]["scientific_observation"] = "development_seen"
    paths["base_lock"].write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(AuthoritySuperseded, match="custody disposition"):
        AuthorityGuard(workspace)


def test_guard_accepts_current_idea_draft_proposal_role(tmp_path: Path) -> None:
    workspace, paths = _authority_workspace(tmp_path)
    lock = json.loads(paths["base_lock"].read_text(encoding="utf-8"))
    lock["proposal"]["authority"] = "idea_draft"
    paths["base_lock"].write_text(json.dumps(lock), encoding="utf-8")

    assert AuthorityGuard(workspace).protocol_id == "s2a_kan_e3_vertical_v8"


@pytest.mark.parametrize(
    "authority_name",
    [
        "proposal",
        "config",
        "preregistration",
        "directive",
        "incident",
        "predecessor",
    ],
)
def test_one_byte_authority_drift_fails_without_scientific_receipt(
    tmp_path: Path, authority_name: str
) -> None:
    workspace, paths = _authority_workspace(tmp_path)
    guard = AuthorityGuard(workspace)
    guard.revalidate("before_first_label_access")
    before = guard.ledger_path.read_bytes()
    paths[authority_name].write_bytes(paths[authority_name].read_bytes() + b"x")

    with pytest.raises(AuthoritySuperseded, match=authority_name):
        guard.revalidate("before_each_scientific_or_control_arm", arm="kan_e3_selected")

    assert guard.ledger_path.read_bytes() == before


def test_capability_is_instance_boundary_arm_and_live_authority_bound(
    tmp_path: Path,
) -> None:
    workspace, paths = _authority_workspace(tmp_path)
    guard = AuthorityGuard(workspace)
    receipt = guard.revalidate(
        "before_each_scientific_or_control_arm", arm="kan_e3_selected"
    )

    with pytest.raises(TypeError, match="capability string"):
        guard.verify_capability(
            {"capability": receipt.capability},  # type: ignore[arg-type]
            boundary=receipt.boundary,
            arm=receipt.arm,
        )
    with pytest.raises(ValueError, match="boundary or arm"):
        guard.verify_capability(
            receipt.capability,
            boundary="before_each_artifact_publication",
            arm="kan_e3_selected",
        )
    with pytest.raises(ValueError, match="unknown capability"):
        AuthorityGuard(workspace).verify_capability(
            receipt.capability, boundary=receipt.boundary, arm=receipt.arm
        )

    paths["proposal"].write_bytes(paths["proposal"].read_bytes() + b"x")
    with pytest.raises(AuthoritySuperseded, match="proposal"):
        guard.verify_capability(
            receipt.capability, boundary=receipt.boundary, arm=receipt.arm
        )


def test_guard_rejects_archive_as_live_proposal_even_with_identical_bytes(
    tmp_path: Path,
) -> None:
    workspace, paths = _authority_workspace(tmp_path)
    archive = workspace / "governance" / "archives" / "KAN_Alpha_PR.md"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(paths["proposal"].read_bytes())
    lock = json.loads(paths["base_lock"].read_text(encoding="utf-8"))
    lock["proposal"]["path"] = str(archive.relative_to(workspace))
    paths["base_lock"].write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(AuthoritySuperseded, match="live proposal"):
        AuthorityGuard(workspace)


def test_guard_enforces_boundary_arm_contract(tmp_path: Path) -> None:
    workspace, _ = _authority_workspace(tmp_path)
    guard = AuthorityGuard(workspace)

    with pytest.raises(ValueError, match="requires an arm"):
        guard.revalidate("before_each_scientific_or_control_arm")
    with pytest.raises(ValueError, match="unknown scientific arm"):
        guard.revalidate("before_each_scientific_or_control_arm", arm="invented_arm")
    with pytest.raises(ValueError, match="does not accept an arm"):
        guard.revalidate("before_development_opening", arm="kan_e3_selected")
    with pytest.raises(ValueError, match="not frozen"):
        guard.revalidate("after_results")


def test_live_authority_cannot_be_replaced_by_same_bytes_symlink(
    tmp_path: Path,
) -> None:
    workspace, paths = _authority_workspace(tmp_path)
    guard = AuthorityGuard(workspace)
    guard.revalidate("before_first_label_access")
    before = guard.ledger_path.read_bytes()
    archive = workspace / "governance" / "archives" / "proposal.md"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(paths["proposal"].read_bytes())
    paths["proposal"].unlink()
    paths["proposal"].symlink_to(archive)

    with pytest.raises(AuthoritySuperseded, match="proposal"):
        guard.revalidate("before_development_opening")

    assert guard.ledger_path.read_bytes() == before
