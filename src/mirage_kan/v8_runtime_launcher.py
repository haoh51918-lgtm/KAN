"""Frozen command entry point for the S2a v8 runtime boundary."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import unquote, urlparse


PROTOCOL_ID = "s2a_kan_e3_vertical_v8"
PINNED_ENVIRONMENT = Path(
    "/zju_0012/htq/aaai26_alpha/aaai27_evosci/runtime/s2a_v4_eval/.venv"
)
PINNED_QLIB_DATA = Path(
    "/zju_0012/htq/aaai26_alpha/QuantaAlpha/data/qlib/cn_data"
)
TRACKING_RELATIVE = Path("evaluations/runtime/s2a_v8_tracking")
TRACKING_RECEIPT = Path(
    "runtime/s2a_v8_eval/evidence/tracking_preimport_receipt.json"
)
BASE_LOCK = Path("prereg/s2a_kan_e3_vertical_v8.lock.json")
IMPLEMENTATION_LOCK = Path(
    "prereg/s2a_kan_e3_vertical_v8_implementation.lock.json"
)
REBIND_RECEIPT = Path(
    "governance/openings/s2a_kan_e3_vertical_v8_mining_rebind.json"
)
DECISION_STAGING = Path(".s2a_v8_decision.staging")


def _required_environment(workspace: Path) -> dict[str, str]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MLFLOW_ALLOW_FILE_STORE": "true",
        "PYTHONPATH": str((workspace / "src").resolve(strict=True)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "QLIB_DATA_DIR": str(PINNED_QLIB_DATA),
    }


def _reexec_with_environment(workspace: Path, arguments: Sequence[str]) -> None:
    required = _required_environment(workspace)
    if all(os.environ.get(key) == value for key, value in required.items()) and (
        "MLFLOW_TRACKING_URI" not in os.environ
    ):
        return
    environment = os.environ.copy()
    environment.update(required)
    environment.pop("MLFLOW_TRACKING_URI", None)
    os.chdir(workspace)
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-m",
            "mirage_kan.v8_runtime_launcher",
            *arguments,
        ],
        environment,
    )


def _validate_environment(workspace: Path) -> None:
    required = _required_environment(workspace)
    for name, expected in required.items():
        if os.environ.get(name) != expected:
            raise ValueError(f"runtime environment mismatch: {name}")
    if "MLFLOW_TRACKING_URI" in os.environ:
        raise ValueError("MLFLOW_TRACKING_URI must be absent")
    if not PINNED_QLIB_DATA.resolve(strict=True).is_dir():
        raise ValueError("pinned QLib data directory is missing")


def _configure_torch() -> dict[str, object]:
    torch = importlib.import_module("torch")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = {
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    expected = {
        "cublas_workspace_config": ":4096:8",
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "matmul_allow_tf32": False,
    }
    if state != expected:
        raise RuntimeError("v8 deterministic Torch state did not hold")
    return state


def _prepare_tracking_root(workspace: Path) -> Path:
    current = workspace
    for component in TRACKING_RELATIVE.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise ValueError("tracking path must contain only real directories")
        else:
            current.mkdir()
        resolved = current.resolve(strict=True)
        if not resolved.is_relative_to(workspace):
            raise ValueError("tracking path escapes the workspace")
        current = resolved
    unexpected = {entry.name for entry in current.iterdir()} - {"README.md"}
    if unexpected:
        raise ValueError("v8 tracking root is not clean before development")
    return current


def _file_uri_path(uri: object) -> Path:
    if not isinstance(uri, str):
        raise ValueError("QLib experiment-manager URI is invalid")
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("QLib experiment-manager URI must be a local file URI")
    return Path(unquote(parsed.path)).resolve()


def _registered(config: object) -> bool:
    value = getattr(config, "registered", None)
    if value is None and isinstance(config, Mapping):
        value = config.get("registered")
    return bool(value)


def _verify_active_protocol() -> None:
    from mirage_kan import protocol

    if (
        protocol.PROTOCOL_ID != PROTOCOL_ID
        or protocol.BASE_LOCK != BASE_LOCK
        or protocol.IMPLEMENTATION_LOCK != IMPLEMENTATION_LOCK
    ):
        raise ValueError("active protocol identities are not the frozen v8 identities")


def establish_runtime(workspace: Path | str) -> dict[str, object]:
    """Establish v8 tracking before QLib import, without opening labels."""
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace is not a directory")
    if "qlib" in sys.modules:
        raise ValueError("QLib was imported before the v8 tracking boundary")
    _verify_active_protocol()
    _validate_environment(root)
    torch_state = _configure_torch()
    tracking_root = _prepare_tracking_root(root)
    if (root / DECISION_STAGING).exists() or (root / DECISION_STAGING).is_symlink():
        raise ValueError("v8 decision staging already exists")
    os.chdir(tracking_root)
    qlib = importlib.import_module("qlib")
    config = getattr(getattr(qlib, "config", None), "C", None)
    if config is None:
        raise ValueError("QLib runtime configuration is unavailable")
    try:
        manager_uri = config["exp_manager"]["kwargs"]["uri"]
    except (KeyError, TypeError) as error:
        raise ValueError("QLib experiment-manager configuration is invalid") from error
    expected_mlruns = (tracking_root / "mlruns").resolve()
    passed = (
        _file_uri_path(manager_uri) == expected_mlruns
        and not _registered(config)
        and Path.cwd() == tracking_root
    )
    receipt = {
        "environment": {**_required_environment(root), **torch_state},
        "passed": passed,
        "protocol_id": PROTOCOL_ID,
        "qlib_experiment_manager": {
            "resolved_path": str(_file_uri_path(manager_uri)),
            "uri": manager_uri,
        },
        "qlib_provider_initialized": _registered(config),
        "qlib_version": getattr(qlib, "__version__", None),
        "real_quanta_label_access": False,
        "schema_version": "s2a_v8_tracking_preimport_receipt_v1",
        "tracking_cwd": str(Path.cwd()),
    }
    if not passed:
        raise RuntimeError("v8 tracking pre-import receipt failed")
    return receipt


def _write_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("tracking receipt write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_development(workspace: Path | str) -> Path:
    """Run the one-way v8 development and decision pipeline."""
    root = Path(workspace).resolve(strict=True)
    receipt = establish_runtime(root)
    if not receipt["passed"]:
        raise RuntimeError("v8 runtime was not established")
    pipeline = importlib.import_module("mirage_kan.evaluation.v2_pipeline")
    pending = pipeline.run_s2a_v2_development(root)
    staged = pending.stage_decision(root / DECISION_STAGING)
    return Path(pending.publish_decision(staged)).resolve(strict=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "mode",
        choices=(
            "lock-write",
            "lock-verify",
            "rebind-write",
            "rebind-verify",
            "tracking-receipt",
            "development",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    workspace = Path(args.workspace).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace is not a directory")
    _reexec_with_environment(workspace, arguments)
    if Path(sys.prefix).resolve(strict=True) != PINNED_ENVIRONMENT.resolve(strict=True):
        raise RuntimeError("v8 launcher must use the pinned isolated environment")
    _validate_environment(workspace)
    _verify_active_protocol()

    if args.mode in {"tracking-receipt", "development"}:
        if args.mode == "tracking-receipt":
            payload = establish_runtime(workspace)
            _write_exclusive_json(workspace / TRACKING_RECEIPT, payload)
        else:
            payload = {"output": str(run_development(workspace))}
    else:
        os.chdir(workspace)
        _configure_torch()
        if args.mode in {"lock-write", "lock-verify"}:
            module = importlib.import_module(
                "mirage_kan.governance.implementation_lock"
            )
            if args.mode == "lock-write":
                payload = {"path": str(module.write_implementation_lock(workspace))}
            else:
                payload = module.verify_implementation_lock(workspace)
        else:
            module = importlib.import_module("mirage_kan.governance.mining_rebind")
            kwargs = {
                "target_base_lock_path": BASE_LOCK,
                "target_implementation_lock_path": IMPLEMENTATION_LOCK,
            }
            if args.mode == "rebind-write":
                output = module.write_mining_rebind_receipt(workspace, **kwargs)
                payload = {"path": str(output)}
            else:
                payload = module.verify_mining_rebind_receipt(workspace, **kwargs)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
