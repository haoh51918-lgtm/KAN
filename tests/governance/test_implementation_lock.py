from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mirage_kan.governance import implementation_lock
from mirage_kan.governance.implementation_lock import (
    build_implementation_lock,
    verify_implementation_lock,
    write_implementation_lock,
)


PROTOCOL_ID = "s2a_kan_e3_vertical_v8"


class _FakeDistribution:
    def __init__(
        self,
        root: Path,
        *,
        name: str,
        version: str,
        records: tuple[str, ...] = ("RECORD",),
    ) -> None:
        self.metadata = {"Name": name, "Version": version}
        self.version = version
        self._root = root
        self.files = [
            (
                Path(record)
                if ".dist-info/" in record
                else Path(f"{name.replace('-', '_')}-{version}.dist-info") / record
            )
            for record in records
        ]

    def locate_file(self, path: object) -> Path:
        return self._root / Path(str(path))


@pytest.fixture(autouse=True)
def _locked_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(implementation_lock, "PROTOCOL_ID", PROTOCOL_ID)
    monkeypatch.setattr(
        implementation_lock,
        "BASE_LOCK",
        Path("prereg/s2a_kan_e3_vertical_v8.lock.json"),
    )
    monkeypatch.setattr(
        implementation_lock,
        "DEFAULT_OUTPUT",
        Path("prereg/s2a_kan_e3_vertical_v8_implementation.lock.json"),
    )
    site = tmp_path / "site-packages"
    distributions = (
        _FakeDistribution(site, name="Example_Pkg", version="1.2.3"),
        _FakeDistribution(site, name="Second.Pkg", version="4.5.6"),
    )
    for distribution in distributions:
        for relative in distribution.files:
            record = distribution.locate_file(relative)
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(f"{distribution.metadata['Name']}\n", encoding="utf-8")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: distributions)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("QLIB_DATA_DIR", "/locked/qlib")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("PYTHONPATH", "/locked/src")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    prefix = tmp_path / "fake-venv"
    base_prefix = tmp_path / "fake-base-python"
    site_packages = prefix / "lib/python3.12/site-packages"
    site_packages.mkdir(parents=True)
    base_prefix.mkdir()
    pyvenv = prefix / "pyvenv.cfg"
    pyvenv.write_text("version = 3.12.3\n", encoding="utf-8")
    clean_sys_path = [
        entry
        for entry in implementation_lock.sys.path
        if not entry or "site-packages" not in Path(entry).parts
    ]
    monkeypatch.setattr(implementation_lock.sys, "prefix", str(prefix))
    monkeypatch.setattr(implementation_lock.sys, "base_prefix", str(base_prefix))
    monkeypatch.setattr(
        implementation_lock.sys, "path", [*clean_sys_path, str(site_packages)]
    )
    monkeypatch.setattr(
        implementation_lock.site,
        "getsitepackages",
        lambda: [str(site_packages)],
    )
    monkeypatch.setattr(implementation_lock.site, "ENABLE_USER_SITE", False)
    return pyvenv, site_packages, prefix


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _workspace(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    workspace = tmp_path / "workspace"
    quanta = tmp_path / "QuantaAlpha"
    cache = tmp_path / "pit.parquet"
    baseline = tmp_path / "baseline.json"
    for path in (
        workspace / "configs/data",
        workspace / "configs/evaluation",
        workspace / "configs/experiments",
        workspace / "configs/runtime",
        workspace / "governance/incidents",
        workspace / "prereg",
        workspace / "src/mirage_kan/nested",
        quanta / "configs",
        quanta / "quantaalpha/backtest",
        quanta / "data/qlib/cn_data/calendars",
    ):
        path.mkdir(parents=True, exist_ok=True)

    proposal = workspace / "KAN_Alpha_PR.md"
    preregistration = workspace / "prereg/protocol.md"
    predecessor = workspace / "governance/incidents/predecessor.json"
    config = workspace / "configs/experiments/protocol.yaml"
    data_pin = workspace / "configs/data/pit_cache.json"
    quanta_pin = workspace / "configs/evaluation/quanta_pinned.json"
    pyproject = workspace / "pyproject.toml"
    uv_lock = workspace / "uv.lock"
    runtime_python = workspace / "configs/runtime/python.json"
    runtime_environment = workspace / "configs/runtime/environment.json"
    source_a = workspace / "src/mirage_kan/__init__.py"
    source_b = workspace / "src/mirage_kan/nested/runner.py"
    quanta_config = quanta / "configs/backtest.yaml"
    quanta_runner = quanta / "quantaalpha/backtest/runner.py"
    provider_file = quanta / "data/qlib/cn_data/calendars/day.txt"

    proposal.write_text("proposal\n", encoding="utf-8")
    preregistration.write_text("preregistered\n", encoding="utf-8")
    predecessor.write_text('{"state":"terminal"}\n', encoding="utf-8")
    config.write_text(yaml.safe_dump({"protocol_id": PROTOCOL_ID}), encoding="utf-8")
    pyproject.write_text("[project]\nname='test'\n", encoding="utf-8")
    uv_lock.write_text("version = 1\nrevision = 1\n", encoding="utf-8")
    runtime_python.write_text('{"requires_python":">=3.12"}\n', encoding="utf-8")
    runtime_environment.write_text(
        '{"CUBLAS_WORKSPACE_CONFIG":":4096:8"}\n', encoding="utf-8"
    )
    source_a.write_text('"""package"""\n', encoding="utf-8")
    source_b.write_text("VALUE = 1\n", encoding="utf-8")
    cache.write_bytes(b"real parquet bytes")
    baseline.write_text('{"information_ratio": 0.22}\n', encoding="utf-8")
    quanta_config.write_text(
        yaml.safe_dump({"data": {"provider_uri": "~/.qlib/qlib_data/cn_data"}}),
        encoding="utf-8",
    )
    quanta_runner.write_text("class BacktestRunner: pass\n", encoding="utf-8")
    provider_file.write_text("2020-01-02\n", encoding="utf-8")

    _git(quanta, "init")
    _git(quanta, "config", "user.email", "test@example.com")
    _git(quanta, "config", "user.name", "Test")
    _git(quanta, "add", "configs/backtest.yaml", "quantaalpha/backtest/runner.py")
    _git(quanta, "commit", "-m", "pin")
    commit = _git(quanta, "rev-parse", "HEAD")

    data_pin.write_text(
        json.dumps(
            {
                "cache_path": str(cache),
                "cache_sha256": _sha256(cache),
            }
        ),
        encoding="utf-8",
    )
    quanta_pin.write_text(
        json.dumps(
            {
                "repository": str(quanta),
                "commit": commit,
                "config_sha256": _sha256(quanta_config),
                "runner_sha256": _sha256(quanta_runner),
                "baseline_metric": str(baseline),
                "baseline_metric_sha256": _sha256(baseline),
            }
        ),
        encoding="utf-8",
    )
    base_lock = workspace / "prereg/s2a_kan_e3_vertical_v8.lock.json"
    base_lock.write_text(
        json.dumps(
            {
                "schema_version": "mirage_s2_prereg_lock_v2",
                "protocol": {
                    "protocol_id": PROTOCOL_ID,
                    "path": str(config.relative_to(workspace)),
                    "sha256": _sha256(config),
                },
                "proposal": {
                    "authority": "sole_proposal_authority",
                    "path": str(proposal.relative_to(workspace)),
                    "sha256": _sha256(proposal),
                },
                "preregistration": {
                    "path": str(preregistration.relative_to(workspace)),
                    "sha256": _sha256(preregistration),
                },
                "data": {
                    "cache_path": str(cache),
                    "cache_sha256": _sha256(cache),
                    "config_path": str(data_pin.relative_to(workspace)),
                    "config_sha256": _sha256(data_pin),
                },
                "baseline_metric": {
                    "path": str(baseline),
                    "sha256": _sha256(baseline),
                },
                "quanta": {
                    "commit": commit,
                    "config_sha256": _sha256(quanta_config),
                    "runner_sha256": _sha256(quanta_runner),
                    "pinned_config_path": str(quanta_pin.relative_to(workspace)),
                    "pinned_config_sha256": _sha256(quanta_pin),
                },
                "predecessor_custody": {
                    "protocol_id": "s2a_kan_e3_vertical_v3",
                    "scientific_observation": "pre_development_admission_count_only",
                    "files": {
                        str(predecessor.relative_to(workspace)): _sha256(predecessor)
                    },
                },
                "runtime": {
                    "files": {
                        str(runtime_environment.relative_to(workspace)): _sha256(
                            runtime_environment
                        ),
                        str(runtime_python.relative_to(workspace)): _sha256(
                            runtime_python
                        ),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return workspace, {
        "base_lock": base_lock,
        "source": source_b,
        "cache": cache,
        "baseline": baseline,
        "quanta_runner": quanta_runner,
        "provider": provider_file,
        "uv_lock": uv_lock,
        "predecessor": predecessor,
        "runtime": runtime_environment,
    }


def test_build_write_and_verify_bind_the_complete_execution_closure(
    tmp_path: Path,
) -> None:
    workspace, _ = _workspace(tmp_path)

    payload = build_implementation_lock(workspace)

    assert payload["schema_version"] == "mirage_s2_implementation_lock_v2"
    assert payload["protocol_id"] == PROTOCOL_ID
    assert payload["created_before_science"] is True
    assert payload["scientific_results_observed"] is False
    assert set(payload["files"]) == {
        "configs/data/pit_cache.json",
        "configs/evaluation/quanta_pinned.json",
        "configs/experiments/protocol.yaml",
        "governance/incidents/predecessor.json",
        "prereg/protocol.md",
        "prereg/s2a_kan_e3_vertical_v8.lock.json",
        "pyproject.toml",
        "uv.lock",
        "configs/runtime/environment.json",
        "configs/runtime/python.json",
    }
    assert set(payload["source_tree"]["files"]) == {
        "__init__.py",
        "nested/runner.py",
    }
    assert payload["external_files"]["data_cache"]["path"] == str(
        (tmp_path / "pit.parquet").resolve()
    )
    assert payload["quanta"]["commit"] == _git(
        tmp_path / "QuantaAlpha", "rev-parse", "HEAD"
    )
    assert payload["qlib_provider"]["file_count"] == 1
    assert payload["runtime"]["python"]["executable"]
    assert payload["runtime"]["python"]["executable_sha256"] == _sha256(
        Path(os.path.realpath(os.sys.executable))
    )
    assert payload["runtime"]["python"]["implementation"]
    assert payload["runtime"]["python"]["version"]
    assert payload["runtime"]["python"]["prefix"] == str(Path(os.sys.prefix).resolve())
    assert payload["runtime"]["python"]["base_prefix"] == str(
        Path(os.sys.base_prefix).resolve()
    )
    assert payload["runtime"]["python"]["enable_user_site"] is False
    assert payload["runtime"]["python"]["pyvenv_config"]["path"] == str(
        (Path(os.sys.prefix) / "pyvenv.cfg").resolve()
    )
    assert payload["runtime"]["python"]["site_packages_roots"]
    assert all(
        item["path"].endswith(".pth")
        for item in payload["runtime"]["python"]["pth_files"]
    )
    assert payload["runtime"]["distributions"] == [
        {
            "canonical_name": "example-pkg",
            "declared_name": "Example_Pkg",
            "record_sha256": _sha256(
                tmp_path / "site-packages/Example_Pkg-1.2.3.dist-info/RECORD"
            ),
            "version": "1.2.3",
        },
        {
            "canonical_name": "second-pkg",
            "declared_name": "Second.Pkg",
            "record_sha256": _sha256(
                tmp_path / "site-packages/Second.Pkg-4.5.6.dist-info/RECORD"
            ),
            "version": "4.5.6",
        },
    ]
    assert payload["runtime"]["torch"]["version"]
    assert type(payload["runtime"]["torch"]["cuda_available"]) is bool
    assert type(payload["runtime"]["torch"]["device_count"]) is int
    assert payload["runtime"]["torch"]["device_count"] == len(
        payload["runtime"]["torch"]["devices"]
    )
    assert set(payload["runtime"]["determinism"]) == {
        "cublas_workspace_config",
        "cuda_matmul_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_allow_tf32",
        "deterministic_algorithms_enabled",
    }
    assert payload["runtime"]["environment"] == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "QLIB_DATA_DIR": "/locked/qlib",
    }

    lock_path = write_implementation_lock(workspace)

    assert verify_implementation_lock(workspace) == payload
    assert lock_path.name == "s2a_kan_e3_vertical_v8_implementation.lock.json"
    with pytest.raises(FileExistsError):
        write_implementation_lock(workspace)


def test_build_excludes_idea_and_advisory_docs_from_execution_identity(
    tmp_path: Path,
) -> None:
    workspace, paths = _workspace(tmp_path)
    review = workspace / "Review-from-claude.md"
    review.write_text("advisory review\n", encoding="utf-8")
    base_lock = json.loads(paths["base_lock"].read_text(encoding="utf-8"))
    base_lock["predecessor_custody"]["files"]["Review-from-claude.md"] = _sha256(review)
    paths["base_lock"].write_text(json.dumps(base_lock), encoding="utf-8")

    payload = build_implementation_lock(workspace)

    assert "KAN_Alpha_PR.md" not in payload["files"]
    assert "Review-from-claude.md" not in payload["files"]


def test_verify_ignores_invocation_only_runtime_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _workspace(tmp_path)
    write_implementation_lock(workspace)
    monkeypatch.setenv("PYTHONPATH", "/another/valid/import/root")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "/another/output/location")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "false")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    monkeypatch.setattr(
        implementation_lock.sys,
        "path",
        [
            str(tmp_path / "different-launch-directory"),
            *implementation_lock.sys.path[1:],
        ],
    )

    verify_implementation_lock(workspace)


def test_runtime_identity_binds_cuda_device_and_tf32_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = (
        SimpleNamespace(uuid="GPU-aaaa", total_memory=80_000_000_000),
        SimpleNamespace(uuid="GPU-bbbb", total_memory=40_000_000_000),
    )
    monkeypatch.setattr(implementation_lock, "_distribution_inventory", lambda: [])
    monkeypatch.setattr(implementation_lock.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(implementation_lock.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        implementation_lock.torch.cuda,
        "get_device_name",
        lambda index: ("Accelerator A", "Accelerator B")[index],
    )
    monkeypatch.setattr(
        implementation_lock.torch.cuda,
        "get_device_capability",
        lambda index: ((8, 0), (9, 0))[index],
    )
    monkeypatch.setattr(
        implementation_lock.torch.cuda,
        "get_device_properties",
        lambda index: properties[index],
    )
    monkeypatch.setattr(
        implementation_lock.torch.backends.cuda.matmul, "allow_tf32", False
    )
    monkeypatch.setattr(implementation_lock.torch.backends.cudnn, "allow_tf32", True)

    identity = implementation_lock._runtime_identity()

    assert identity["torch"]["devices"] == [
        {
            "index": 0,
            "name": "Accelerator A",
            "capability": [8, 0],
            "uuid": "GPU-aaaa",
            "total_memory": 80_000_000_000,
        },
        {
            "index": 1,
            "name": "Accelerator B",
            "capability": [9, 0],
            "uuid": "GPU-bbbb",
            "total_memory": 40_000_000_000,
        },
    ]
    assert identity["determinism"]["cuda_matmul_allow_tf32"] is False
    assert identity["determinism"]["cudnn_allow_tf32"] is True


@pytest.mark.parametrize(
    "properties",
    [
        SimpleNamespace(uuid=None, total_memory=1),
        SimpleNamespace(uuid="None", total_memory=1),
        SimpleNamespace(uuid="GPU-cccc", total_memory="80000000000"),
        SimpleNamespace(uuid="GPU-cccc", total_memory=True),
    ],
)
def test_runtime_identity_rejects_coercible_invalid_cuda_properties(
    monkeypatch: pytest.MonkeyPatch, properties: SimpleNamespace
) -> None:
    monkeypatch.setattr(implementation_lock, "_distribution_inventory", lambda: [])
    monkeypatch.setattr(implementation_lock.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(implementation_lock.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        implementation_lock.torch.cuda,
        "get_device_properties",
        lambda _index: properties,
    )
    monkeypatch.setattr(
        implementation_lock.torch.cuda,
        "get_device_name",
        lambda _index: "Accelerator",
    )
    monkeypatch.setattr(
        implementation_lock.torch.cuda,
        "get_device_capability",
        lambda _index: (8, 0),
    )

    with pytest.raises(ValueError, match="invalid CUDA device identity"):
        implementation_lock._runtime_identity()


@pytest.mark.parametrize(
    ("key", "error"),
    [
        ("source", "source-tree"),
        ("cache", "data_cache"),
        ("baseline", "baseline_metric"),
        ("quanta_runner", "Quanta runner"),
        ("provider", "QLib provider"),
        ("uv_lock", "implementation file"),
        ("predecessor", "predecessor custody"),
        ("runtime", "runtime file"),
    ],
)
def test_verify_fails_closed_when_any_live_execution_input_changes(
    tmp_path: Path, key: str, error: str
) -> None:
    workspace, paths = _workspace(tmp_path)
    write_implementation_lock(workspace)
    paths[key].write_bytes(paths[key].read_bytes() + b"drift")

    with pytest.raises(ValueError, match=error):
        verify_implementation_lock(workspace)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda lock: lock.update(schema_version="wrong"), "schema"),
        (lambda lock: lock.update(protocol_id="wrong"), "protocol"),
        (
            lambda lock: lock.update(created_before_science=False),
            "created before science",
        ),
        (
            lambda lock: lock.update(scientific_results_observed=True),
            "scientific-results",
        ),
        (lambda lock: lock["files"].pop("pyproject.toml"), "file set"),
        (lambda lock: lock["files"].update({"extra.txt": "0" * 64}), "file set"),
    ],
)
def test_verify_rejects_malformed_or_incomplete_lock(
    tmp_path: Path, mutation: object, error: str
) -> None:
    workspace, _ = _workspace(tmp_path)
    lock_path = write_implementation_lock(workspace)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    mutation(lock)  # type: ignore[operator]
    lock_path.unlink()
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        verify_implementation_lock(workspace)


def test_build_rejects_pin_disagreement_before_lock_creation(tmp_path: Path) -> None:
    workspace, paths = _workspace(tmp_path)
    base_lock = json.loads(paths["base_lock"].read_text(encoding="utf-8"))
    base_lock["data"]["cache_sha256"] = "f" * 64
    paths["base_lock"].write_text(json.dumps(base_lock), encoding="utf-8")

    with pytest.raises(ValueError, match="data cache pin"):
        build_implementation_lock(workspace)


def test_build_rejects_unrecognized_predecessor_observation(tmp_path: Path) -> None:
    workspace, paths = _workspace(tmp_path)
    base_lock = json.loads(paths["base_lock"].read_text(encoding="utf-8"))
    base_lock["predecessor_custody"]["scientific_observation"] = "development"
    paths["base_lock"].write_text(json.dumps(base_lock), encoding="utf-8")

    with pytest.raises(ValueError, match="predecessor custody disposition"):
        build_implementation_lock(workspace)


def test_build_accepts_quarantined_infrastructure_failure_observation(
    tmp_path: Path,
) -> None:
    workspace, paths = _workspace(tmp_path)
    base_lock = json.loads(paths["base_lock"].read_text(encoding="utf-8"))
    base_lock["predecessor_custody"]["scientific_observation"] = (
        "inconclusive_infrastructure_with_quarantined_development_outputs"
    )
    paths["base_lock"].write_text(json.dumps(base_lock), encoding="utf-8")

    assert build_implementation_lock(workspace)["created_before_science"] is True


def test_verify_rejects_tampered_runtime_identity(tmp_path: Path) -> None:
    workspace, _ = _workspace(tmp_path)
    lock_path = write_implementation_lock(workspace)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["runtime"]["python"]["version"] = "tampered"
    lock_path.unlink()
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime identity"):
        verify_implementation_lock(workspace)


def test_verify_rejects_distribution_record_drift(tmp_path: Path) -> None:
    workspace, _ = _workspace(tmp_path)
    write_implementation_lock(workspace)
    record = tmp_path / "site-packages/Example_Pkg-1.2.3.dist-info/RECORD"
    record.write_text("drifted\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime identity"):
        verify_implementation_lock(workspace)


@pytest.mark.parametrize(
    "name",
    [
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "QLIB_DATA_DIR",
    ],
)
def test_verify_rejects_locked_environment_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    workspace, _ = _workspace(tmp_path)
    write_implementation_lock(workspace)
    monkeypatch.setenv(name, "drifted")

    with pytest.raises(ValueError, match="runtime identity"):
        verify_implementation_lock(workspace)


def test_verify_rejects_pyvenv_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _workspace(tmp_path)
    pyvenv, _, _ = _fake_venv(tmp_path, monkeypatch)
    write_implementation_lock(workspace)
    pyvenv.write_text("version = 3.12.4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime identity"):
        verify_implementation_lock(workspace)


@pytest.mark.parametrize("mutation", ["add", "change"])
def test_verify_rejects_pth_inventory_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    workspace, _ = _workspace(tmp_path)
    _, site_packages, _ = _fake_venv(tmp_path, monkeypatch)
    original = site_packages / "original.pth"
    original.write_text("original\n", encoding="utf-8")
    write_implementation_lock(workspace)
    if mutation == "add":
        (site_packages / "added.pth").write_text("added\n", encoding="utf-8")
    else:
        original.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="runtime identity"):
        verify_implementation_lock(workspace)


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_build_rejects_non_regular_pth_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    workspace, _ = _workspace(tmp_path)
    _, site_packages, _ = _fake_venv(tmp_path, monkeypatch)
    invalid = site_packages / "invalid.pth"
    if kind == "symlink":
        target = tmp_path / "pth-target"
        target.write_text("target\n", encoding="utf-8")
        invalid.symlink_to(target)
    else:
        invalid.mkdir()

    with pytest.raises(ValueError, match="PTH entry"):
        build_implementation_lock(workspace)


def test_build_rejects_external_site_packages_sys_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _workspace(tmp_path)
    _fake_venv(tmp_path, monkeypatch)
    external = tmp_path / "external/site-packages"
    external.mkdir(parents=True)
    monkeypatch.setattr(
        implementation_lock.sys,
        "path",
        [*implementation_lock.sys.path, str(external)],
    )

    with pytest.raises(ValueError, match="external site-packages"):
        build_implementation_lock(workspace)


def test_build_rejects_relative_nonempty_sys_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _workspace(tmp_path)
    _fake_venv(tmp_path, monkeypatch)
    monkeypatch.setattr(
        implementation_lock.sys,
        "path",
        [*implementation_lock.sys.path, "relative-import-root"],
    )

    with pytest.raises(ValueError, match="relative sys.path"):
        build_implementation_lock(workspace)


def test_build_rejects_symlinked_virtual_environment_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _workspace(tmp_path)
    _, _, prefix = _fake_venv(tmp_path, monkeypatch)
    linked_prefix = tmp_path / "linked-venv"
    linked_prefix.symlink_to(prefix, target_is_directory=True)
    monkeypatch.setattr(implementation_lock.sys, "prefix", str(linked_prefix))

    with pytest.raises(ValueError, match="symlinked Python environment"):
        build_implementation_lock(workspace)


def test_build_rejects_symlinked_site_packages_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _workspace(tmp_path)
    _, site_packages, prefix = _fake_venv(tmp_path, monkeypatch)
    linked_site_packages = prefix / "linked-site-packages"
    linked_site_packages.symlink_to(site_packages, target_is_directory=True)
    monkeypatch.setattr(
        implementation_lock.site,
        "getsitepackages",
        lambda: [str(linked_site_packages)],
    )

    with pytest.raises(ValueError, match="symlinked site-packages"):
        build_implementation_lock(workspace)


def test_build_ignores_vendored_record_inside_a_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _workspace(tmp_path)
    site = tmp_path / "vendored-site-packages"
    distribution = _FakeDistribution(
        site,
        name="Outer",
        version="1.0",
        records=("RECORD", "vendor/Inner-2.0.dist-info/RECORD"),
    )
    for relative in distribution.files:
        record = distribution.locate_file(relative)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("record\n", encoding="utf-8")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [distribution])

    payload = build_implementation_lock(workspace)

    assert [item["canonical_name"] for item in payload["runtime"]["distributions"]] == [
        "outer"
    ]


@pytest.mark.parametrize("records", [(), ("RECORD", "Other-1.0.dist-info/RECORD")])
def test_build_rejects_missing_or_duplicate_distribution_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: tuple[str, ...],
) -> None:
    workspace, _ = _workspace(tmp_path)
    site = tmp_path / "invalid-site-packages"
    distribution = _FakeDistribution(
        site, name="Invalid", version="1.0", records=records
    )
    for relative in distribution.files:
        record = distribution.locate_file(relative)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("record\n", encoding="utf-8")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [distribution])

    with pytest.raises(ValueError, match="exactly one .dist-info/RECORD"):
        build_implementation_lock(workspace)


def test_build_rejects_duplicate_canonical_distribution_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _workspace(tmp_path)
    site = tmp_path / "duplicate-site-packages"
    first = _FakeDistribution(site / "a", name="Same_Name", version="1.0")
    second = _FakeDistribution(site / "b", name="same-name", version="2.0")
    for distribution in (first, second):
        record = distribution.locate_file(distribution.files[0])
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("record\n", encoding="utf-8")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [first, second])

    with pytest.raises(ValueError, match="duplicate installed distribution"):
        build_implementation_lock(workspace)


def test_build_rejects_non_regular_distribution_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = _workspace(tmp_path)
    site = tmp_path / "symlink-site-packages"
    distribution = _FakeDistribution(site, name="Invalid", version="1.0")
    record = distribution.locate_file(distribution.files[0])
    target = tmp_path / "record-target"
    target.write_text("record\n", encoding="utf-8")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.symlink_to(target)
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [distribution])

    with pytest.raises(ValueError, match="regular non-symlink file"):
        build_implementation_lock(workspace)
