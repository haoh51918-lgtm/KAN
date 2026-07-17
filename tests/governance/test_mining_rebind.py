from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml


MINING_KEYS = (
    "mining_run",
    "kan_library",
    "gp_control_library",
    "permutation_control_library",
    "blackbox_control",
    "mechanism_cards",
    "blind_review_package",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _json_sha256(payload: object) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _workspace(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path
    source_protocol = "source_v6"
    target_protocol = "target_v7"
    source_paths = {
        "mining_run": "artifacts/source_mining",
        "kan_library": "factor_libraries/source_kan",
        "gp_control_library": "factor_libraries/source_gp",
        "permutation_control_library": "factor_libraries/source_permutation",
        "blackbox_control": "controls/source_blackbox",
        "mechanism_cards": "mechanism_cards/source_cards",
        "blind_review_package": "reviews/source_blind",
    }
    common = {
        "stage": "S2a",
        "evidence_class": "corrective_adaptive_repeated_development_screen",
        "claim_boundary": {
            "final_claim_allowed": False,
            "graph_unlock_allowed": False,
        },
        "data": {
            "train": ["2016-01-01", "2020-12-31"],
            "validation": ["2021-01-01", "2021-12-31"],
            "development_test": ["2022-01-01", "2025-12-26"],
        },
        "kan_e3": {"total_miner_attempts": 256},
    }
    source_config = {
        "protocol_id": source_protocol,
        **common,
        "artifact_paths": source_paths,
    }
    target_config = {
        "protocol_id": target_protocol,
        **common,
        "claim_boundary": {
            "final_claim_allowed": False,
            "graph_unlock_allowed": True,
        },
        "artifact_paths": {
            "implementation_lock": "prereg/target_implementation.lock.json",
            "development_opening": "governance/openings/target_development.json",
            "evaluations": {},
        },
        "mining_source": {
            "mode": "verified_cross_protocol_rebind",
            "source_protocol_id": source_protocol,
            "source_base_lock": "prereg/source.lock.json",
            "source_implementation_lock": "prereg/source_implementation.lock.json",
            "source_mining_entitlement": "governance/openings/source_mining.json",
            "source_mining_preclaim": "governance/openings/source_preclaim.json",
            "source_artifact_paths": source_paths,
            "rebind_receipt": "governance/openings/target_mining_rebind.json",
            "absence_contract": {
                "source_development_preclaim": (
                    "governance/openings/source_development_preclaim.json"
                ),
                "source_development_opening": (
                    "governance/openings/source_development.json"
                ),
                "target_mining_preclaim": (
                    "governance/openings/target_mining_preclaim.json"
                ),
                "target_mining_entitlement": (
                    "governance/openings/target_mining.json"
                ),
            },
        },
    }
    source_config_path = root / "configs/source.yaml"
    target_config_path = root / "configs/target.yaml"
    source_config_path.parent.mkdir(parents=True)
    source_config_path.write_text(yaml.safe_dump(source_config), encoding="utf-8")
    target_config_path.write_text(yaml.safe_dump(target_config), encoding="utf-8")
    source_implementation = root / "prereg/source_implementation.lock.json"
    target_implementation = root / "prereg/target_implementation.lock.json"
    _write_json(source_implementation, {"protocol_id": source_protocol})
    _write_json(target_implementation, {"protocol_id": target_protocol})
    source_lock = root / "prereg/source.lock.json"
    target_lock = root / "prereg/target.lock.json"
    _write_json(
        source_lock,
        {
            "protocol": {
                "protocol_id": source_protocol,
                "path": "configs/source.yaml",
                "sha256": _sha256(source_config_path),
            }
        },
    )
    _write_json(
        target_lock,
        {
            "protocol": {
                "protocol_id": target_protocol,
                "path": "configs/target.yaml",
                "sha256": _sha256(target_config_path),
            }
        },
    )
    topology_sha256 = "5" * 64
    child_hashes: dict[str, str] = {}
    for key in MINING_KEYS[1:]:
        path = root / source_paths[key]
        path.mkdir(parents=True)
        payload = path / "payload.bin"
        payload.write_bytes(key.encode("utf-8"))
        _write_json(
            path / "manifest.json",
            {
                "topology_key": key,
                "topology_sha256": topology_sha256,
                "files": {"payload.bin": _sha256(payload)},
            },
        )
        child_hashes[key] = _sha256(path / "manifest.json")
    top = root / source_paths["mining_run"]
    top.mkdir(parents=True)
    children = top / "children.json"
    _write_json(children, sorted(child_hashes))
    _write_json(
        top / "manifest.json",
        {
            "schema_version": "mirage_s2a_v2_staging_bundle_v1",
            "role": "mining_top_bundle",
            "topology_key": "mining_run",
            "topology_sha256": topology_sha256,
            "published_child_topology_sha256": topology_sha256,
            "child_manifests": child_hashes,
            "child_manifest_sha256": child_hashes,
            "published_child_paths": {
                key: str((root / source_paths[key]).resolve())
                for key in MINING_KEYS[1:]
            },
            "files": {"children.json": _sha256(children)},
        },
    )
    preclaim = root / "governance/openings/source_preclaim.json"
    entitlement = root / "governance/openings/source_mining.json"
    _write_json(
        preclaim,
        {
            "schema_version": "mirage_topology_preclaim_v2",
            "protocol_id": source_protocol,
            "topology_sha256": topology_sha256,
        },
    )
    authority_payload = {
        "schema_version": "mirage_authority_receipt_v2",
        "protocol_id": source_protocol,
        "guard_instance_sha256": "1" * 64,
        "base_lock_sha256": _sha256(source_lock),
        "sequence": 1,
        "boundary": "before_first_label_access",
        "arm": None,
        "authority_files": {},
        "authority_sha256": "2" * 64,
        "checked_at_unix_ns": 1,
    }
    authority_sha256 = _json_sha256(authority_payload)
    authority_receipt = {
        **authority_payload,
        "receipt_sha256": authority_sha256,
        "capability": "3" * 64,
    }
    _write_json(
        root
        / "governance/authority/source_v6/receipts/00000000000000000001.json",
        authority_receipt,
    )
    _write_json(
        entitlement,
        {
            "schema_version": "mirage_mining_entitlement_v2",
            "protocol_id": source_protocol,
            "state": "consumed_before_first_label_access",
            "topology_sha256": topology_sha256,
            "topology_preclaim_sha256": _sha256(preclaim),
            "base_lock_sha256": _sha256(source_lock),
            "config_sha256": _sha256(source_config_path),
            "implementation_lock_sha256": _sha256(source_implementation),
            "authority_receipt_sha256": authority_sha256,
            "attempt_budget": 256,
        },
    )
    return root, {
        "source_lock": source_lock,
        "target_lock": target_lock,
        "target_implementation": target_implementation,
        "receipt": root / "governance/openings/target_mining_rebind.json",
        "source_kan_payload": root / source_paths["kan_library"] / "payload.bin",
        "target_config": target_config_path,
    }


def _build(root: Path, paths: dict[str, Path]) -> dict[str, object]:
    from mirage_kan.governance.mining_rebind import build_mining_rebind_receipt

    return build_mining_rebind_receipt(
        root,
        target_base_lock_path=paths["target_lock"],
        target_implementation_lock_path=paths["target_implementation"],
    )


def test_rebind_receipt_binds_complete_source_and_target_identities(
    tmp_path: Path,
) -> None:
    root, paths = _workspace(tmp_path)

    receipt = _build(root, paths)

    assert receipt["schema_version"] == "mirage_mining_rebind_receipt_v1"
    assert receipt["state"] == "verified_without_label_access"
    assert receipt["source"]["protocol_id"] == "source_v6"
    assert receipt["target"]["protocol_id"] == "target_v7"
    assert receipt["contract"] == {
        "label_access_performed": False,
        "reselection_performed": False,
        "reordering_performed": False,
        "retuning_performed": False,
        "source_payload_copy_performed": False,
    }
    assert set(receipt["source"]["artifacts"]) == set(MINING_KEYS)
    assert receipt["source"]["entitlement"]["sha256"]


def test_rebind_live_verification_rejects_source_payload_mutation(
    tmp_path: Path,
) -> None:
    from mirage_kan.governance.mining_rebind import verify_mining_rebind_receipt

    root, paths = _workspace(tmp_path)
    receipt = _build(root, paths)
    paths["receipt"].parent.mkdir(parents=True, exist_ok=True)
    _write_json(paths["receipt"], receipt)
    paths["source_kan_payload"].write_bytes(b"mutated")

    with pytest.raises(ValueError, match="source artifact changed"):
        verify_mining_rebind_receipt(
            root,
            target_base_lock_path=paths["target_lock"],
            target_implementation_lock_path=paths["target_implementation"],
        )


def test_rebind_rejects_scientific_config_drift(tmp_path: Path) -> None:
    root, paths = _workspace(tmp_path)
    config = yaml.safe_load(paths["target_config"].read_text(encoding="utf-8"))
    config["kan_e3"]["total_miner_attempts"] = 128
    paths["target_config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    target_lock = json.loads(paths["target_lock"].read_text(encoding="utf-8"))
    target_lock["protocol"]["sha256"] = _sha256(paths["target_config"])
    _write_json(paths["target_lock"], target_lock)

    with pytest.raises(ValueError, match="scientific configuration"):
        _build(root, paths)


def test_rebind_rejects_source_entitlement_or_topology_identity_drift(
    tmp_path: Path,
) -> None:
    root, paths = _workspace(tmp_path)
    entitlement = root / "governance/openings/source_mining.json"
    record = json.loads(entitlement.read_text(encoding="utf-8"))
    record["topology_sha256"] = "9" * 64
    _write_json(entitlement, record)

    with pytest.raises(ValueError, match="entitlement identity"):
        _build(root, paths)


def test_rebind_rejects_late_source_development_opening(tmp_path: Path) -> None:
    root, paths = _workspace(tmp_path)
    _write_json(root / "governance/openings/source_development.json", {})

    with pytest.raises(ValueError, match="forbidden opening exists"):
        _build(root, paths)


def test_rebind_receipt_write_is_exclusive(tmp_path: Path) -> None:
    from mirage_kan.governance.mining_rebind import write_mining_rebind_receipt

    root, paths = _workspace(tmp_path)
    write_mining_rebind_receipt(
        root,
        target_base_lock_path=paths["target_lock"],
        target_implementation_lock_path=paths["target_implementation"],
    )

    with pytest.raises(FileExistsError):
        write_mining_rebind_receipt(
            root,
            target_base_lock_path=paths["target_lock"],
            target_implementation_lock_path=paths["target_implementation"],
        )


def test_rebind_uses_the_real_authority_canonical_byte_contract() -> None:
    from mirage_kan.governance.mining_rebind import _authority_receipt_sha256

    root = Path(__file__).resolve().parents[2]
    receipt_path = (
        root
        / "governance/authority/s2a_kan_e3_vertical_v6/receipts"
        / "00000000000000000001.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_sha256", "capability"}
    }

    assert _authority_receipt_sha256(payload) == receipt["receipt_sha256"]
