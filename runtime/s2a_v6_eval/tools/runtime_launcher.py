"""Establish the frozen v6 runtime before lock, mining, or development imports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse


RUNTIME = Path(__file__).resolve().parents[1]
PINNED_ENVIRONMENT = RUNTIME.parent / "s2a_v4_eval" / ".venv"
PINNED_QLIB_DATA = Path("/zju_0012/htq/aaai26_alpha/QuantaAlpha/data/qlib/cn_data")
TRACKING_RELATIVE = Path("evaluations/runtime/s2a_v6_tracking")


def _required_environment(workspace: Path) -> dict[str, str]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MLFLOW_ALLOW_FILE_STORE": "true",
        "PYTHONPATH": str((workspace / "src").resolve(strict=True)),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "QLIB_DATA_DIR": str(PINNED_QLIB_DATA),
    }


def _reexec_with_environment(workspace: Path) -> None:
    required = _required_environment(workspace)
    if all(os.environ.get(key) == value for key, value in required.items()) and (
        "MLFLOW_TRACKING_URI" not in os.environ
    ):
        return
    environment = os.environ.copy()
    environment.update(required)
    environment.pop("MLFLOW_TRACKING_URI", None)
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def _configure_torch() -> dict[str, object]:
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = {
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
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
        raise RuntimeError("v6 deterministic Torch state did not hold")
    return state


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError("Qlib experiment-manager URI must be a local file URI")
    return Path(unquote(parsed.path)).resolve()


def _prepare_tracking_root(workspace: Path) -> Path:
    current = workspace
    for component in TRACKING_RELATIVE.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise ValueError("tracking path must contain only real directories")
        else:
            current.mkdir()
        if current.is_symlink() or not current.is_dir():
            raise ValueError("tracking path changed during preparation")
        resolved = current.resolve(strict=True)
        if not resolved.is_relative_to(workspace):
            raise ValueError("tracking path escapes the workspace")
        current = resolved
    return current


def _tracking_receipt(
    workspace: Path, torch_state: dict[str, object]
) -> dict[str, object]:
    qlib_loaded_before_receipt = "qlib" in sys.modules
    allow_file_store_before_import = os.environ.get("MLFLOW_ALLOW_FILE_STORE")
    tracking_uri_before_import = os.environ.get("MLFLOW_TRACKING_URI")
    import qlib
    from qlib.config import C

    exp_manager_uri = C["exp_manager"]["kwargs"]["uri"]
    provider_initialized = bool(C.registered)

    import mlflow

    tracking_root = (workspace / TRACKING_RELATIVE).resolve()
    expected_mlruns = (tracking_root / "mlruns").resolve()
    return {
        "environment": {
            **torch_state,
            "mlflow_allow_file_store": os.environ["MLFLOW_ALLOW_FILE_STORE"],
            "mlflow_tracking_uri_env": os.environ.get("MLFLOW_TRACKING_URI"),
            "pythonpath": os.environ["PYTHONPATH"],
            "python_dont_write_bytecode": os.environ["PYTHONDONTWRITEBYTECODE"],
            "qlib_data_dir": os.environ["QLIB_DATA_DIR"],
        },
        "mlflow_raw_default_diagnostic": {
            "authoritative": False,
            "tracking_uri": mlflow.get_tracking_uri(),
            "used_by_receipt": False,
        },
        "passed": (
            not qlib_loaded_before_receipt
            and allow_file_store_before_import == "true"
            and tracking_uri_before_import is None
            and _file_uri_path(exp_manager_uri) == expected_mlruns
            and not provider_initialized
            and Path.cwd() == tracking_root
        ),
        "preimport": {
            "mlflow_allow_file_store": allow_file_store_before_import,
            "mlflow_tracking_uri_env": tracking_uri_before_import,
            "qlib_absent_from_sys_modules": not qlib_loaded_before_receipt,
        },
        "qlib_experiment_manager": {
            "authoritative": True,
            "resolved_path": str(_file_uri_path(exp_manager_uri)),
            "uri": exp_manager_uri,
        },
        "qlib_imported_before_tracking_receipt": qlib_loaded_before_receipt,
        "qlib_provider_initialized": provider_initialized,
        "qlib_version": qlib.__version__,
        "real_quanta_label_access": False,
        "schema_version": "s2a_v6_tracking_preimport_receipt_v1",
        "tracking_cwd": str(Path.cwd()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for name in (
        "lock-build",
        "lock-write",
        "lock-verify",
        "tracking-receipt",
        "mining",
        "development",
    ):
        subparsers.add_parser(name)
    return parser


def main() -> int:
    args = _parser().parse_args()
    workspace = Path(args.workspace).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace is not a directory")
    expected_prefix = PINNED_ENVIRONMENT.resolve(strict=True)
    if Path(sys.prefix).resolve() != expected_prefix:
        raise RuntimeError("runtime launcher must use the pinned isolated environment")

    tracking = args.mode in {"development", "tracking-receipt"}
    _reexec_with_environment(workspace)
    if not PINNED_QLIB_DATA.resolve(strict=True).is_dir():
        raise RuntimeError("pinned Qlib data directory is missing")
    torch_state = _configure_torch()

    if tracking:
        tracking_root = _prepare_tracking_root(workspace)
        os.chdir(tracking_root)
    else:
        os.chdir(workspace)

    if args.mode == "tracking-receipt":
        payload = _tracking_receipt(workspace, torch_state)
        if not payload["passed"]:
            raise RuntimeError("tracking pre-import receipt failed")
    elif args.mode in {"lock-build", "lock-write", "lock-verify"}:
        from mirage_kan.governance.implementation_lock import (
            build_implementation_lock,
            verify_implementation_lock,
            write_implementation_lock,
        )

        if args.mode == "lock-build":
            payload = build_implementation_lock(workspace)
        elif args.mode == "lock-write":
            payload = {"path": str(write_implementation_lock(workspace))}
        else:
            payload = verify_implementation_lock(workspace)
    elif args.mode == "mining":
        from mirage_kan.mining.v2_pipeline import run_s2a_v2_mining

        output = run_s2a_v2_mining(workspace, devices=("cuda:0", "cuda:1"))
        payload = {"command": "mining", "output": str(output)}
    else:
        from mirage_kan.evaluation.v2_pipeline import run_s2a_v2_development

        pending = run_s2a_v2_development(workspace)
        staged = pending.stage_decision(workspace / ".s2a_v6_decision.staging")
        output = pending.publish_decision(staged)
        payload = {"command": "development", "output": str(output)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

