from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mirage_kan import v8_runtime_launcher as v8_runtime


def _required_environment(workspace: Path) -> dict[str, str]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MLFLOW_ALLOW_FILE_STORE": "true",
        "PYTHONPATH": str((workspace / "src").resolve(strict=True)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "QLIB_DATA_DIR": str(v8_runtime.PINNED_QLIB_DATA),
    }


def _fake_torch() -> SimpleNamespace:
    deterministic = {"enabled": False}
    return SimpleNamespace(
        use_deterministic_algorithms=lambda value: deterministic.update(enabled=value),
        are_deterministic_algorithms_enabled=lambda: deterministic["enabled"],
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=True, allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        ),
    )


def test_establish_runtime_uses_v8_tracking_before_qlib_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    monkeypatch.setattr(v8_runtime, "PINNED_QLIB_DATA", tmp_path / "qlib")
    v8_runtime.PINNED_QLIB_DATA.mkdir()
    for name, value in _required_environment(workspace).items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delitem(sys.modules, "qlib", raising=False)
    imported_at: list[Path] = []
    torch = _fake_torch()

    def fake_import(name: str) -> object:
        if name == "torch":
            return torch
        if name == "qlib":
            imported_at.append(Path.cwd())
            tracking = workspace / v8_runtime.TRACKING_RELATIVE / "mlruns"
            return SimpleNamespace(
                __version__="test",
                config=SimpleNamespace(
                    C={
                        "registered": False,
                        "exp_manager": {"kwargs": {"uri": tracking.as_uri()}},
                    }
                ),
            )
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(v8_runtime.importlib, "import_module", fake_import)
    monkeypatch.chdir(workspace)

    receipt = v8_runtime.establish_runtime(workspace)

    tracking_root = (workspace / v8_runtime.TRACKING_RELATIVE).resolve()
    assert imported_at == [tracking_root]
    assert Path.cwd() == tracking_root
    assert receipt["passed"] is True
    assert receipt["protocol_id"] == "s2a_kan_e3_vertical_v8"
    assert receipt["real_quanta_label_access"] is False


def test_establish_runtime_rejects_external_mlflow_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    monkeypatch.setattr(v8_runtime, "PINNED_QLIB_DATA", tmp_path / "qlib")
    v8_runtime.PINNED_QLIB_DATA.mkdir()
    for name, value in _required_environment(workspace).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid")

    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        v8_runtime.establish_runtime(workspace)


def test_establish_runtime_rejects_tracking_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (workspace / "evaluations").mkdir()
    (workspace / "evaluations/runtime").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(v8_runtime, "PINNED_QLIB_DATA", tmp_path / "qlib")
    v8_runtime.PINNED_QLIB_DATA.mkdir()
    for name, value in _required_environment(workspace).items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(ValueError, match="real directories"):
        v8_runtime.establish_runtime(workspace)


def test_run_development_only_imports_pipeline_after_runtime_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events: list[str] = []
    pending = SimpleNamespace(
        stage_decision=lambda path: events.append(f"stage:{Path(path).name}") or "staged",
        publish_decision=lambda staged: events.append(f"publish:{staged}")
        or (workspace / "decision").mkdir()
        or workspace / "decision",
    )
    pipeline = SimpleNamespace(
        run_s2a_v2_development=lambda root: events.append(f"run:{Path(root).name}")
        or pending
    )
    monkeypatch.setattr(
        v8_runtime,
        "establish_runtime",
        lambda root: events.append("runtime") or {"passed": True},
    )
    monkeypatch.setattr(
        v8_runtime.importlib,
        "import_module",
        lambda name: pipeline
        if name == "mirage_kan.evaluation.v2_pipeline"
        else (_ for _ in ()).throw(AssertionError(name)),
    )

    output = v8_runtime.run_development(workspace)

    assert output == (workspace / "decision").resolve()
    assert events == ["runtime", "run:workspace", "stage:.s2a_v8_decision.staging", "publish:staged"]


def test_environment_contract_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "src").mkdir(parents=True)
    monkeypatch.setattr(v8_runtime, "PINNED_QLIB_DATA", tmp_path / "qlib")
    v8_runtime.PINNED_QLIB_DATA.mkdir()
    for name, value in _required_environment(workspace).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        v8_runtime.establish_runtime(workspace)


def test_establish_runtime_rejects_non_v8_active_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(v8_runtime, "PROTOCOL_ID", "s2a_kan_e3_vertical_v9")
    monkeypatch.delitem(sys.modules, "qlib", raising=False)

    with pytest.raises(ValueError, match="active protocol identities"):
        v8_runtime.establish_runtime(workspace)
