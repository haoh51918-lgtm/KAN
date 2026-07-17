from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import mirage_kan.artifacts.topology as topology_module
from mirage_kan.artifacts.topology import TopologyTransaction


class _AuthorityStub:
    def verify_capability(self, capability, *, boundary, arm=None):
        expected = f"{boundary}:{arm or ''}"
        if capability != expected:
            raise ValueError("wrong test authority capability")
        return object()


def _artifact_capability(key: str) -> str:
    return f"before_each_artifact_publication:{key}"


def _publish_child(transaction, key, staging):
    transaction.publish_child(
        key,
        staging,
        authority_guard=_AuthorityStub(),
        authority_capability=_artifact_capability(key),
    )


def _publish_top(transaction, staging):
    kwargs = {
        "authority_guard": _AuthorityStub(),
        "authority_capability": _artifact_capability(transaction.top_key),
    }
    if transaction.phase == "development":
        kwargs["final_decision_capability"] = (
            "before_final_decision_publication:"
        )
    transaction.publish_top_bundle(staging, **kwargs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _topology_workspace(tmp_path: Path) -> Path:
    for parent in (
        "artifacts",
        "factor_libraries",
        "controls",
        "mechanism_cards",
        "reviews",
        "evaluations",
        "governance/openings",
        "governance/recoveries",
        "prereg",
        "configs/experiments",
    ):
        (tmp_path / parent).mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_id": "s2a_kan_e3_vertical_v8",
        "artifact_paths": {
            "mining_run": "artifacts/mining",
            "mining_preclaim": "governance/openings/mining_preclaim.json",
            "mining_entitlement": "governance/openings/mining.json",
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
                "alpha158_replay": "evaluations/alpha158",
                "kan_e3_selected": "evaluations/kan",
                "typed_gp_sr_control": "evaluations/gp",
                "matched_blackbox_control": "evaluations/blackbox",
                "kan_e3_permutation_control": "evaluations/permutation",
            },
            "decision_artifact": "evaluations/decision",
            "report": "reports/report.md",
        },
    }
    config_path = tmp_path / "configs" / "experiments" / "protocol.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    lock = {
        "schema_version": "mirage_s2_prereg_lock_v2",
        "protocol": {
            "protocol_id": config["protocol_id"],
            "path": str(config_path.relative_to(tmp_path)),
            "sha256": _sha256(config_path),
        },
    }
    (tmp_path / "prereg" / "s2a_kan_e3_vertical_v8.lock.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    return tmp_path


def _staging(tmp_path: Path, name: str) -> Path:
    staging = tmp_path / f".{name}.staging"
    staging.mkdir()
    (staging / "payload.json").write_text('{"ok": true}\n', encoding="utf-8")
    (staging / "manifest.json").write_text(
        json.dumps({"files": ["payload.json"]}), encoding="utf-8"
    )
    return staging


def test_frozen_topology_classifies_directories_and_excludes_control_files(
    tmp_path: Path,
) -> None:
    workspace = _topology_workspace(tmp_path)

    mining = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    development = TopologyTransaction.from_frozen_config(workspace, phase="development")

    assert set(mining.targets) == {
        "mining_run",
        "kan_library",
        "gp_control_library",
        "permutation_control_library",
        "blackbox_control",
        "mechanism_cards",
        "blind_review_package",
    }
    assert mining.top_key == "mining_run"
    assert mining.preclaim_path.suffix == ".json"
    assert mining.preclaim_path not in mining.targets.values()
    assert set(development.targets) == {
        "decision_artifact",
        "evaluation:alpha158_replay",
        "evaluation:kan_e3_selected",
        "evaluation:typed_gp_sr_control",
        "evaluation:matched_blackbox_control",
        "evaluation:kan_e3_permutation_control",
    }
    assert development.top_key == "decision_artifact"
    assert all("report" not in str(path) for path in development.targets.values())


def test_preclaim_precedes_claim_all_and_top_bundle_is_published_last(
    tmp_path: Path,
) -> None:
    transaction = TopologyTransaction.from_frozen_config(
        _topology_workspace(tmp_path), phase="mining"
    )

    with pytest.raises(RuntimeError, match="preclaim"):
        transaction.claim_all()
    preclaim = transaction.preclaim()
    transaction.claim_all()
    assert preclaim["topology_sha256"] == transaction.topology_sha256
    assert all(
        (path / ".INCOMPLETE").is_file() for path in transaction.targets.values()
    )
    with pytest.raises(RuntimeError, match="children"):
        _publish_top(transaction, _staging(tmp_path, "early_top"))

    for key in transaction.child_keys:
        _publish_child(transaction, key, _staging(tmp_path, key.replace(":", "_")))
    _publish_top(transaction, _staging(tmp_path, "top"))

    assert all(
        not (path / ".INCOMPLETE").exists() for path in transaction.targets.values()
    )
    assert all(
        (path / "manifest.json").is_file() for path in transaction.targets.values()
    )


@pytest.mark.parametrize("fail_index", range(7))
def test_every_claim_failure_terminalizes_only_owned_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_index: int
) -> None:
    transaction = TopologyTransaction.from_frozen_config(
        _topology_workspace(tmp_path), phase="mining"
    )
    transaction.preclaim()
    real_claim = topology_module.claim_artifact_directory
    calls = 0

    def fail_at(path: Path) -> Path:
        nonlocal calls
        current = calls
        calls += 1
        if current == fail_index:
            raise RuntimeError(f"claim failure {fail_index}")
        return real_claim(path)

    monkeypatch.setattr(topology_module, "claim_artifact_directory", fail_at)
    with pytest.raises(RuntimeError, match="claim failure"):
        transaction.claim_all()

    claimed_before_failure = list(transaction.targets.values())[:fail_index]
    assert all(
        (path / "terminal_failure.json").is_file() for path in claimed_before_failure
    )
    assert all(not (path / ".INCOMPLETE").exists() for path in claimed_before_failure)


def test_no_replace_collision_does_not_modify_foreign_directory(tmp_path: Path) -> None:
    transaction = TopologyTransaction.from_frozen_config(
        _topology_workspace(tmp_path), phase="mining"
    )
    transaction.preclaim()
    foreign = transaction.targets["kan_library"]
    foreign.mkdir()
    (foreign / "owner.txt").write_text("foreign", encoding="utf-8")

    with pytest.raises(FileExistsError, match="replace artifact"):
        transaction.claim_all()

    assert (foreign / "owner.txt").read_text(encoding="utf-8") == "foreign"
    assert not (foreign / "terminal_failure.json").exists()
    assert (transaction.targets["mining_run"] / "terminal_failure.json").is_file()


def test_marker_binding_failure_still_terminalizes_the_claimed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = TopologyTransaction.from_frozen_config(
        _topology_workspace(tmp_path), phase="mining"
    )
    transaction.preclaim()
    monkeypatch.setattr(
        transaction,
        "_bind_claim_marker",
        lambda key: (_ for _ in ()).throw(RuntimeError(f"bind failure {key}")),
    )

    with pytest.raises(RuntimeError, match="bind failure"):
        transaction.claim_all()

    claimed = transaction.targets["mining_run"]
    assert (claimed / "terminal_failure.json").is_file()
    assert not (claimed / ".INCOMPLETE").exists()


@pytest.mark.parametrize(
    ("key", "bad_path", "message"),
    [
        ("kan_library", "factor_libraries/nested/kan", "direct child"),
        ("kan_library", "../escaped", "escapes"),
        ("gp_control_library", "factor_libraries/kan", "alias"),
    ],
)
def test_topology_rejects_non_direct_escape_and_alias_paths(
    tmp_path: Path, key: str, bad_path: str, message: str
) -> None:
    workspace = _topology_workspace(tmp_path)
    config_path = workspace / "configs" / "experiments" / "protocol.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["artifact_paths"][key] = bad_path
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    lock_path = workspace / "prereg" / "s2a_kan_e3_vertical_v8.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["protocol"]["sha256"] = _sha256(config_path)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        TopologyTransaction.from_frozen_config(workspace, phase="mining")


def test_topology_rejects_symlinked_artifact_parent(tmp_path: Path) -> None:
    workspace = _topology_workspace(tmp_path)
    real = workspace / "real_factor_libraries"
    real.mkdir()
    (workspace / "factor_libraries").rmdir()
    (workspace / "factor_libraries").symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        TopologyTransaction.from_frozen_config(workspace, phase="mining")


@pytest.mark.parametrize("fail_key_index", range(6))
def test_every_child_publication_failure_invalidates_all_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_key_index: int
) -> None:
    transaction = TopologyTransaction.from_frozen_config(
        _topology_workspace(tmp_path), phase="mining"
    )
    transaction.preclaim()
    transaction.claim_all()
    real_finalize = topology_module.finalize_claimed_directory
    calls = 0

    def fail_at(*args, **kwargs):
        nonlocal calls
        current = calls
        calls += 1
        if current == fail_key_index:
            raise RuntimeError(f"publication failure {fail_key_index}")
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(topology_module, "finalize_claimed_directory", fail_at)
    with pytest.raises(RuntimeError, match="publication failure"):
        for key in transaction.child_keys:
            _publish_child(
                transaction,
                key,
                _staging(tmp_path, f"publish_{key.replace(':', '_')}")
            )

    assert all(
        (path / "terminal_failure.json").is_file()
        for path in transaction.targets.values()
    )
    assert all(
        not (path / ".INCOMPLETE").exists() for path in transaction.targets.values()
    )


def test_partial_publication_and_double_recovery_are_terminal_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _topology_workspace(tmp_path)
    transaction = TopologyTransaction.from_frozen_config(workspace, phase="development")
    transaction.preclaim()
    transaction.claim_all()
    first_child = transaction.child_keys[0]
    _publish_child(transaction, first_child, _staging(tmp_path, "first_child"))

    recovered = TopologyTransaction.from_frozen_config(workspace, phase="development")
    monkeypatch.setattr(
        topology_module,
        "finalize_claimed_directory",
        lambda *args, **kwargs: pytest.fail("recovery must never publish or rerun"),
    )
    first = recovered.recover({"error": "interrupted"})
    second = recovered.recover({"error": "ignored on idempotent recovery"})

    assert first == second
    assert first["state"] == "terminal_failure"
    assert all(
        (path / "terminal_failure.json").is_file()
        for path in recovered.targets.values()
    )
    assert all(
        not (path / ".INCOMPLETE").exists() for path in recovered.targets.values()
    )
    terminal = json.loads(
        (recovered.targets[first_child] / "terminal_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert "manifest.json" in terminal["partial_files"]


def test_recovery_consumes_paths_never_claimed_before_crash(tmp_path: Path) -> None:
    workspace = _topology_workspace(tmp_path)
    transaction = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    transaction.preclaim()

    receipt = TopologyTransaction.from_frozen_config(workspace, phase="mining").recover(
        {"error": "crash after preclaim"}
    )

    assert receipt["state"] == "terminal_failure"
    assert all(
        (path / "terminal_failure.json").is_file()
        for path in transaction.targets.values()
    )


def test_recovery_rejects_unowned_collision_without_modifying_it(
    tmp_path: Path,
) -> None:
    workspace = _topology_workspace(tmp_path)
    transaction = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    transaction.preclaim()
    foreign = transaction.targets["kan_library"]
    foreign.mkdir()
    owner = foreign / "owner.txt"
    owner.write_text("foreign", encoding="utf-8")

    with pytest.raises(ValueError, match="ownership evidence"):
        TopologyTransaction.from_frozen_config(workspace, phase="mining").recover(
            {"error": "crash"}
        )

    assert owner.read_text(encoding="utf-8") == "foreign"
    assert set(path.name for path in foreign.iterdir()) == {"owner.txt"}


def test_missing_frozen_category_parents_are_created_only_at_preclaim(
    tmp_path: Path,
) -> None:
    workspace = _topology_workspace(tmp_path)
    missing = [
        workspace / "controls",
        workspace / "mechanism_cards",
        workspace / "reviews",
        workspace / "governance" / "recoveries",
    ]
    for path in missing:
        path.rmdir()

    transaction = TopologyTransaction.from_frozen_config(workspace, phase="mining")
    assert all(not path.exists() for path in missing)

    transaction.preclaim()

    assert all(path.is_dir() and not path.is_symlink() for path in missing)


def test_terminalization_attempts_every_owned_target_after_one_cleanup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transaction = TopologyTransaction.from_frozen_config(
        _topology_workspace(tmp_path), phase="mining"
    )
    transaction.preclaim()
    transaction.claim_all()
    real_terminalize = topology_module._terminalize_idempotent
    calls: list[Path] = []

    def fail_first(path: Path, payload):
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("first cleanup failed")
        return real_terminalize(path, payload)

    monkeypatch.setattr(topology_module, "_terminalize_idempotent", fail_first)
    with pytest.raises(RuntimeError, match="terminalize all topology targets"):
        transaction.terminalize({"error": "scientific failure"})

    assert len(calls) == len(transaction.targets)
    assert all(
        (path / "terminal_failure.json").is_file()
        for path in list(transaction.targets.values())[1:]
    )
