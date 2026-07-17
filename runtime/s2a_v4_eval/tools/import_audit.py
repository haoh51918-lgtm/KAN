"""Emit the v4 import-closure and module-source compatibility receipt."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import site
import sys
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1]
MODULES = (
    "mirage_kan.artifacts.v2_bundle",
    "mirage_kan.evaluation.quanta",
    "mirage_kan.evaluation.v2_decision_assembler",
    "mirage_kan.evaluation.v2_pipeline",
    "mirage_kan.evaluation.v2_runner",
    "mirage_kan.mining.e3",
    "mirage_kan.mining.e3_runner",
    "mirage_kan.mining.mlp_control",
    "mirage_kan.mining.v2_pipeline",
)
VERSION_IMPORTS = (
    "lightgbm",
    "mlflow",
    "numpy",
    "pandas",
    "pyarrow",
    "qlib",
    "scipy",
    "torch",
    "yaml",
)
QUANTA_RUNNER = Path(
    "/zju_0012/htq/aaai26_alpha/QuantaAlpha/quantaalpha/backtest/runner.py"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def load_quanta_runner() -> tuple[object, Path]:
    path = QUANTA_RUNNER.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("_s2a_v4_quanta_runner_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create pinned Quanta runner import spec")
    module = importlib.util.module_from_spec(spec)
    original_sys_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module, path


def main() -> int:
    prefix = Path(sys.prefix).resolve(strict=True)
    expected_prefix = (RUNTIME / ".venv").resolve(strict=True)
    site_packages = tuple(
        Path(path).resolve(strict=True) for path in site.getsitepackages()
    )
    path_entries = [
        str(Path(path).resolve()) if path else str(Path.cwd()) for path in sys.path
    ]
    inherited_site_entries = [
        path
        for path in path_entries
        if "site-packages" in Path(path).parts
        and not any(within(Path(path), root) for root in site_packages)
    ]

    pth_files = []
    external_pth_paths = []
    for root in site_packages:
        for path in sorted(root.glob("*.pth")):
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            for line in lines:
                if not line.startswith("import "):
                    candidate = (root / line).resolve()
                    if not any(within(candidate, allowed) for allowed in site_packages):
                        external_pth_paths.append(str(candidate))
            pth_files.append(
                {
                    "path": str(path.relative_to(prefix)),
                    "sha256": file_sha256(path),
                    "size": path.stat().st_size,
                }
            )

    imported_modules = {}
    for name in MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        imported_modules[name] = {
            "file": str(path),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
    quanta_runner, quanta_path = load_quanta_runner()
    imported_modules["quantaalpha.backtest.runner"] = {
        "class_available": hasattr(quanta_runner, "BacktestRunner"),
        "file": str(quanta_path),
        "import_mode": "isolated_file_spec",
        "sha256": file_sha256(quanta_path),
        "size": quanta_path.stat().st_size,
    }
    versions = {}
    for name in VERSION_IMPORTS:
        module = importlib.import_module(name)
        versions[name] = {
            "file": str(Path(module.__file__).resolve(strict=True)),
            "version": getattr(module, "__version__", None),
        }

    import torch
    from qlib.config import C

    qlib_provider_initialized = bool(C.registered)

    passed = (
        prefix == expected_prefix
        and not inherited_site_entries
        and not external_pth_paths
        and site.ENABLE_USER_SITE is False
        and torch.cuda.is_available()
        and torch.cuda.device_count() == 2
        and not qlib_provider_initialized
        and hasattr(quanta_runner, "BacktestRunner")
        and os.environ.get("PYTHONPATH") == str((RUNTIME.parents[1] / "src").resolve())
        and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    )
    payload = {
        "command": "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/zju_0012/htq/aaai26_alpha/aaai27_evosci/src runtime/s2a_v4_eval/.venv/bin/python runtime/s2a_v4_eval/tools/import_audit.py",
        "cuda": {
            "available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "runtime": torch.version.cuda,
        },
        "imported_modules": imported_modules,
        "isolation": {
            "external_pth_paths": external_pth_paths,
            "inherited_site_package_entries": inherited_site_entries,
            "only_v4_site_packages": not inherited_site_entries
            and not external_pth_paths,
            "pth_files": pth_files,
            "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "user_site_enabled": site.ENABLE_USER_SITE,
        },
        "passed": passed,
        "python": {
            "base_prefix": sys.base_prefix,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "sys_path": path_entries,
            "version": sys.version,
        },
        "qlib_provider_initialized": qlib_provider_initialized,
        "real_quanta_label_access": False,
        "schema_version": "s2a_v4_import_closure_v1",
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "versions": versions,
    }
    if not passed:
        raise RuntimeError(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
