"""Implementation-closure lock support for the frozen S2a protocol family."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import site
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping

import yaml
import torch

from mirage_kan.data.pit import sha256_file
from mirage_kan.protocol import BASE_LOCK, IMPLEMENTATION_LOCK, PROTOCOL_ID

SCHEMA_VERSION = "mirage_s2_implementation_lock_v2"
DEFAULT_OUTPUT = IMPLEMENTATION_LOCK
CONTROL_DOCUMENT_PATHS = frozenset(
    {
        "AGENTS.md",
        "KAN_Alpha_PR.md",
        "README.md",
        "Review-from-claude.md",
        "docs/research/MIRAGE_KAN_LIVING_MANUAL.md",
        "experiments/pipeline_tracker.md",
        "plans/paper_plan.md",
        "plans/todos.md",
        "research_request.md",
    }
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "created_before_science",
        "scientific_results_observed",
        "files",
        "source_tree",
        "external_files",
        "quanta",
        "qlib_provider",
        "runtime",
    }
)


def _canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_identity(path: Path) -> dict[str, object]:
    state = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": state.st_size,
    }


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a mapping")
    return value


def _string(mapping: Mapping[str, object], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} lacks {key}")
    return value


def _sha256(mapping: Mapping[str, object], key: str, *, label: str) -> str:
    value = _string(mapping, key, label=label)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} has an invalid {key}")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _workspace_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is missing")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} path is not workspace-relative")
    path = root / raw
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} does not exist") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the workspace") from error
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} is not a regular non-symlink file")
    return resolved


def _external_file(path_value: object, *, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is missing")
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} does not exist") from error
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} is not a regular non-symlink file")
    return resolved


def _source_tree_identity(package_root: Path) -> dict[str, object]:
    try:
        root = package_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("source-tree root is missing") from error
    if package_root.is_symlink() or not root.is_dir():
        raise ValueError("source-tree root is not a regular directory")
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "source-tree contains a symlink or non-regular Python file"
            )
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not files:
        raise ValueError("source-tree has no Python files")
    digest = hashlib.sha256()
    for relative, file_hash in files.items():
        digest.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
    return {"tree_sha256": digest.hexdigest(), "files": files}


def _regular_file_tree_identity(root_path: Path) -> dict[str, object]:
    """Hash a large regular-file tree concurrently in canonical path order."""
    root = root_path.resolve(strict=True)
    entries: list[tuple[str, Path, int, int]] = []
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names.sort()
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise ValueError("QLib provider contains a symlinked directory")
        for name in sorted(filenames):
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("QLib provider contains a non-regular file or symlink")
            state = path.stat()
            entries.append(
                (
                    path.relative_to(root).as_posix(),
                    path,
                    state.st_size,
                    state.st_mtime_ns,
                )
            )
    entries.sort(key=lambda entry: entry[0])

    def stable_hash(entry: tuple[str, Path, int, int]) -> str:
        _, path, size, mtime_ns = entry
        file_hash = sha256_file(path)
        state = path.stat()
        if path.is_symlink() or state.st_size != size or state.st_mtime_ns != mtime_ns:
            raise ValueError("QLib provider changed while it was being hashed")
        return file_hash

    workers = min(32, len(entries)) or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashes = list(executor.map(stable_hash, entries))
    content_digest = hashlib.sha256()
    stat_digest = hashlib.sha256()
    total_bytes = 0
    for (relative, _, size, mtime_ns), file_hash in zip(entries, hashes, strict=True):
        content_digest.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
        stat_digest.update(f"{relative}\0{size}\0{mtime_ns}\n".encode("utf-8"))
        total_bytes += size
    return {
        "path": str(root),
        "tree_sha256": content_digest.hexdigest(),
        "stat_inventory_sha256": stat_digest.hexdigest(),
        "file_count": len(entries),
        "total_bytes": total_bytes,
    }


def _git_head(repository: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot verify the pinned Quanta commit") from error
    if len(result) != 40 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError("pinned Quanta HEAD is not a full commit identity")
    return result


def _fixed_files(
    root: Path, base_lock_path: Path, base_lock: Mapping[str, object]
) -> dict[str, Path]:
    proposal = _mapping(base_lock.get("proposal"), label="base proposal pin")
    protocol = _mapping(base_lock.get("protocol"), label="base protocol pin")
    preregistration = _mapping(
        base_lock.get("preregistration"), label="base preregistration pin"
    )
    data = _mapping(base_lock.get("data"), label="base data pin")
    quanta = _mapping(base_lock.get("quanta"), label="base Quanta pin")
    custody = _mapping(
        base_lock.get("predecessor_custody"), label="predecessor custody"
    )
    runtime = _mapping(base_lock.get("runtime"), label="base runtime pin")
    if proposal.get("authority") not in {"sole_proposal_authority", "idea_draft"}:
        raise ValueError("base lock has an invalid proposal classification")
    if proposal.get("path") != "KAN_Alpha_PR.md":
        raise ValueError("base lock proposal path is invalid")
    specifications = {
        BASE_LOCK.as_posix(): (BASE_LOCK.as_posix(), sha256_file(base_lock_path)),
        _string(preregistration, "path", label="base preregistration pin"): (
            preregistration.get("path"),
            preregistration.get("sha256"),
        ),
        _string(protocol, "path", label="base protocol pin"): (
            protocol.get("path"),
            protocol.get("sha256"),
        ),
        _string(data, "config_path", label="base data pin"): (
            data.get("config_path"),
            data.get("config_sha256"),
        ),
        _string(quanta, "pinned_config_path", label="base Quanta pin"): (
            quanta.get("pinned_config_path"),
            quanta.get("pinned_config_sha256"),
        ),
        "pyproject.toml": ("pyproject.toml", None),
        "uv.lock": ("uv.lock", None),
    }
    predecessor_protocol = custody.get("protocol_id")
    if (
        not isinstance(predecessor_protocol, str)
        or not predecessor_protocol
        or predecessor_protocol == PROTOCOL_ID
        or custody.get("scientific_observation")
        not in {
            "none",
            "pre_development_admission_count_only",
            "inconclusive_infrastructure_with_quarantined_development_outputs",
            "successful_mining_development_unopened_exact_rebind",
            "terminal_preopening_rebind_software_failure",
        }
    ):
        raise ValueError("predecessor custody disposition is invalid")
    custody_files = _mapping(custody.get("files"), label="predecessor custody files")
    if not custody_files:
        raise ValueError("predecessor custody file set is empty")
    custody_paths: set[str] = set()
    for relative in sorted(custody_files):
        if not isinstance(relative, str) or not relative:
            raise ValueError("predecessor custody path is invalid")
        if relative in CONTROL_DOCUMENT_PATHS:
            continue
        if relative in specifications:
            raise ValueError(
                "predecessor custody aliases an active implementation file"
            )
        specifications[relative] = (
            relative,
            _sha256(custody_files, relative, label="predecessor custody files"),
        )
        custody_paths.add(relative)
    runtime_files = _mapping(runtime.get("files"), label="base runtime files")
    if not runtime_files:
        raise ValueError("base runtime file set is empty")
    runtime_paths: set[str] = set()
    for relative in sorted(runtime_files):
        if not isinstance(relative, str) or not relative:
            raise ValueError("runtime file path is invalid")
        if relative in specifications:
            raise ValueError("runtime file aliases another implementation file")
        specifications[relative] = (
            relative,
            _sha256(runtime_files, relative, label="base runtime files"),
        )
        runtime_paths.add(relative)
    files: dict[str, Path] = {}
    for relative, (raw_path, expected_hash) in specifications.items():
        label = (
            f"predecessor custody file {relative}"
            if relative in custody_paths
            else (
                f"runtime file {relative}"
                if relative in runtime_paths
                else f"implementation file {relative}"
            )
        )
        path = _workspace_file(root, raw_path, label=label)
        if path.relative_to(root).as_posix() != relative:
            raise ValueError(f"{label} has a non-canonical path")
        if expected_hash is not None:
            if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
                if relative in custody_paths:
                    raise ValueError(f"predecessor custody hash mismatch: {relative}")
                if relative in runtime_paths:
                    raise ValueError(f"runtime file hash mismatch: {relative}")
                raise ValueError(f"base pin hash mismatch: {relative}")
        files[relative] = path
    return files


def _canonical_distribution_name(name: str) -> str:
    canonical = re.sub(r"[-_.]+", "-", name).lower()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", canonical):
        raise ValueError("installed distribution has an invalid declared name")
    return canonical


def _distribution_inventory() -> list[dict[str, str]]:
    """Inventory every installed distribution through its immutable RECORD."""
    inventory: list[dict[str, str]] = []
    names: set[str] = set()
    try:
        distributions = list(importlib.metadata.distributions())
    except Exception as error:
        raise ValueError("cannot enumerate installed distributions") from error
    if not distributions:
        raise ValueError("installed distribution inventory is empty")
    for distribution in distributions:
        declared_name = distribution.metadata.get("Name")
        version = distribution.metadata.get("Version")
        if not isinstance(declared_name, str) or not declared_name:
            raise ValueError("installed distribution lacks a declared name")
        if not isinstance(version, str) or not version:
            raise ValueError(
                f"installed distribution lacks an exact version: {declared_name}"
            )
        canonical_name = _canonical_distribution_name(declared_name)
        if canonical_name in names:
            raise ValueError(f"duplicate installed distribution: {canonical_name}")
        names.add(canonical_name)
        files = distribution.files
        if files is None:
            raise ValueError(
                f"installed distribution has exactly one .dist-info/RECORD requirement: "
                f"{declared_name}"
            )
        records = [
            path
            for path in files
            if len(Path(str(path)).parts) == 2
            and Path(str(path)).name == "RECORD"
            and Path(str(path)).parent.name.endswith(".dist-info")
        ]
        if len(records) != 1:
            raise ValueError(
                f"installed distribution must have exactly one .dist-info/RECORD: "
                f"{declared_name}"
            )
        raw_record = Path(distribution.locate_file(records[0]))
        try:
            record = raw_record.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"distribution RECORD does not exist: {declared_name}"
            ) from error
        if raw_record.is_symlink() or not record.is_file():
            raise ValueError(
                f"distribution RECORD is not a regular non-symlink file: "
                f"{declared_name}"
            )
        inventory.append(
            {
                "canonical_name": canonical_name,
                "declared_name": declared_name,
                "version": version,
                "record_sha256": sha256_file(record),
            }
        )
    inventory.sort(key=lambda item: item["canonical_name"])
    return inventory


def _has_symlink_component(path: Path) -> bool:
    if not path.is_absolute():
        return True
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _python_environment_identity() -> dict[str, object]:
    """Capture the active virtual environment and its import search surface."""
    raw_prefix = Path(sys.prefix).expanduser()
    if _has_symlink_component(raw_prefix):
        raise ValueError(
            "active virtual environment has a symlinked Python environment prefix"
        )
    try:
        prefix = raw_prefix.resolve(strict=True)
        base_prefix = Path(sys.base_prefix).resolve(strict=True)
    except OSError as error:
        raise ValueError("cannot resolve the Python environment prefixes") from error
    if not prefix.is_dir() or not base_prefix.is_dir():
        raise ValueError("Python environment prefix is not a directory")

    pyvenv_path = prefix / "pyvenv.cfg"
    try:
        pyvenv = pyvenv_path.resolve(strict=True)
    except OSError as error:
        raise ValueError("active virtual environment lacks pyvenv.cfg") from error
    if pyvenv_path.is_symlink() or not pyvenv.is_file():
        raise ValueError("pyvenv.cfg is not a regular non-symlink file")

    try:
        configured_site_packages = site.getsitepackages()
    except Exception as error:
        raise ValueError(
            "cannot enumerate virtual-environment site-packages"
        ) from error
    site_packages_roots: list[Path] = []
    for raw_root in configured_site_packages:
        if not isinstance(raw_root, str) or not raw_root:
            raise ValueError("virtual-environment site-packages path is invalid")
        raw_site_packages = Path(raw_root).expanduser()
        if _has_symlink_component(raw_site_packages):
            raise ValueError("virtual environment has symlinked site-packages")
        try:
            root = raw_site_packages.resolve(strict=True)
            root.relative_to(prefix)
        except (OSError, ValueError) as error:
            raise ValueError(
                "site-packages root is outside the active virtual environment"
            ) from error
        if not root.is_dir():
            raise ValueError(
                "virtual-environment site-packages root is not a directory"
            )
        site_packages_roots.append(root)
    site_packages_roots = sorted(set(site_packages_roots))
    if not site_packages_roots:
        raise ValueError("virtual-environment site-packages root set is empty")

    for raw_entry in sys.path:
        if not isinstance(raw_entry, str):
            raise ValueError("sys.path contains a non-string entry")
        if raw_entry == "":
            continue
        raw_path = Path(raw_entry).expanduser()
        if not raw_path.is_absolute():
            raise ValueError("sys.path contains a relative sys.path entry")
        entry = raw_path.resolve(strict=False)
        raw_parts = raw_path.parts
        if ("site-packages" in raw_parts or "site-packages" in entry.parts) and not any(
            entry == root or entry.is_relative_to(root) for root in site_packages_roots
        ):
            raise ValueError("sys.path contains external site-packages")

    pth_files: list[dict[str, object]] = []
    for root in site_packages_roots:
        for raw_path in sorted(root.glob("*.pth")):
            try:
                path = raw_path.resolve(strict=True)
            except OSError as error:
                raise ValueError("cannot resolve PTH entry") from error
            if raw_path.is_symlink() or not path.is_file():
                raise ValueError("PTH entry is not a regular non-symlink file")
            identity = _sha256_identity(path)
            identity["path"] = path.relative_to(prefix).as_posix()
            pth_files.append(identity)

    return {
        "prefix": str(prefix),
        "base_prefix": str(base_prefix),
        "enable_user_site": site.ENABLE_USER_SITE,
        "pyvenv_config": _sha256_identity(pyvenv),
        "site_packages_roots": [str(root) for root in site_packages_roots],
        "pth_files": pth_files,
    }


def _runtime_identity() -> dict[str, object]:
    """Capture the exact Python, package, accelerator, and determinism runtime."""
    try:
        python_executable = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise ValueError("cannot resolve the Python executable") from error
    if not python_executable.is_file():
        raise ValueError("resolved Python executable is not a regular file")

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    if not cuda_available and device_count != 0:
        raise ValueError("Torch reports CUDA devices while CUDA is unavailable")
    devices = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        raw_uuid = properties.uuid
        raw_total_memory = properties.total_memory
        uuid = "" if raw_uuid is None else str(raw_uuid).strip()
        if (
            not uuid
            or uuid.casefold() in {"none", "null"}
            or type(raw_total_memory) is not int
            or raw_total_memory <= 0
        ):
            raise ValueError("Torch reports an invalid CUDA device identity")
        total_memory = raw_total_memory
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "uuid": uuid,
                "total_memory": total_memory,
            }
        )
    python_environment = _python_environment_identity()
    return {
        "python": {
            "executable": str(python_executable),
            "executable_sha256": sha256_file(python_executable),
            "implementation": sys.implementation.name,
            "version": sys.version,
            **python_environment,
        },
        "distributions": _distribution_inventory(),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": cuda_available,
            "device_count": device_count,
            "devices": devices,
        },
        "determinism": {
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
                "QLIB_DATA_DIR",
            )
        },
    }


def _live_payload(workspace: Path | str) -> dict[str, object]:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("implementation-lock workspace is not a directory")
    base_lock_path = _workspace_file(root, BASE_LOCK.as_posix(), label="base lock")
    base_lock = _read_json(base_lock_path, label="base lock")
    if base_lock.get("schema_version") != "mirage_s2_prereg_lock_v2":
        raise ValueError("unsupported base-lock schema")
    protocol = _mapping(base_lock.get("protocol"), label="base protocol pin")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("base lock has the wrong protocol")
    files = _fixed_files(root, base_lock_path, base_lock)

    config_path = files[_string(protocol, "path", label="base protocol pin")]
    config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _mapping(config_value, label="protocol config")
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol config has the wrong protocol")
    configured_output = _mapping(
        config.get("artifact_paths", {}), label="artifact paths"
    ).get("implementation_lock", DEFAULT_OUTPUT.as_posix())
    if configured_output != DEFAULT_OUTPUT.as_posix():
        raise ValueError("protocol config has the wrong implementation-lock path")

    data = _mapping(base_lock.get("data"), label="base data pin")
    data_pin_path = files[_string(data, "config_path", label="base data pin")]
    data_pin = _read_json(data_pin_path, label="data pin")
    cache = _external_file(data.get("cache_path"), label="data_cache")
    cache_hash = _sha256(data, "cache_sha256", label="base data pin")
    if (
        data_pin.get("cache_path") != str(cache)
        or data_pin.get("cache_sha256") != cache_hash
    ):
        raise ValueError("data cache pin disagrees with the base lock")
    if sha256_file(cache) != cache_hash:
        raise ValueError("data_cache pin hash mismatch")

    baseline_pin = _mapping(base_lock.get("baseline_metric"), label="base baseline pin")
    baseline = _external_file(baseline_pin.get("path"), label="baseline_metric")
    baseline_hash = _sha256(baseline_pin, "sha256", label="base baseline pin")
    if sha256_file(baseline) != baseline_hash:
        raise ValueError("baseline_metric pin hash mismatch")

    quanta_base = _mapping(base_lock.get("quanta"), label="base Quanta pin")
    quanta_pin_path = files[
        _string(quanta_base, "pinned_config_path", label="base Quanta pin")
    ]
    quanta_pin = _read_json(quanta_pin_path, label="Quanta pin")
    repository_value = _string(quanta_pin, "repository", label="Quanta pin")
    repository = Path(repository_value).expanduser().resolve(strict=True)
    if not repository.is_dir() or Path(repository_value).is_symlink():
        raise ValueError("Quanta repository is not a regular directory")
    commit = _string(quanta_pin, "commit", label="Quanta pin")
    if commit != _string(quanta_base, "commit", label="base Quanta pin"):
        raise ValueError("Quanta commit pins disagree")
    if _git_head(repository) != commit:
        raise ValueError("Quanta commit differs from the pin")
    quanta_config = _external_file(
        str(repository / "configs/backtest.yaml"), label="Quanta config"
    )
    quanta_runner = _external_file(
        str(repository / "quantaalpha/backtest/runner.py"), label="Quanta runner"
    )
    config_hash = _sha256(quanta_pin, "config_sha256", label="Quanta pin")
    runner_hash = _sha256(quanta_pin, "runner_sha256", label="Quanta pin")
    if config_hash != _sha256(
        quanta_base, "config_sha256", label="base Quanta pin"
    ) or runner_hash != _sha256(quanta_base, "runner_sha256", label="base Quanta pin"):
        raise ValueError("Quanta config or runner pins disagree")
    if sha256_file(quanta_config) != config_hash:
        raise ValueError("Quanta config hash mismatch")
    if sha256_file(quanta_runner) != runner_hash:
        raise ValueError("Quanta runner hash mismatch")
    if (
        quanta_pin.get("baseline_metric") != str(baseline)
        or quanta_pin.get("baseline_metric_sha256") != baseline_hash
    ):
        raise ValueError("baseline_metric Quanta pin disagrees with the base lock")

    provider = repository / "data/qlib/cn_data"
    if provider.is_symlink() or not provider.is_dir():
        raise ValueError("QLib provider tree is missing or is a symlink")
    quanta_config_value = yaml.safe_load(quanta_config.read_text(encoding="utf-8"))
    quanta_config_mapping = _mapping(quanta_config_value, label="Quanta config")
    quanta_data = _mapping(
        quanta_config_mapping.get("data"), label="Quanta data config"
    )
    configured_provider_uri = _string(
        quanta_data, "provider_uri", label="Quanta data config"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_before_science": True,
        "scientific_results_observed": False,
        "files": {relative: sha256_file(path) for relative, path in files.items()},
        "source_tree": _source_tree_identity(root / "src/mirage_kan"),
        "external_files": {
            "data_cache": _sha256_identity(cache),
            "baseline_metric": _sha256_identity(baseline),
        },
        "quanta": {
            "repository": str(repository),
            "commit": commit,
            "config": {
                "path": str(quanta_config),
                "sha256": config_hash,
            },
            "runner": {
                "path": str(quanta_runner),
                "sha256": runner_hash,
            },
            "configured_provider_uri": configured_provider_uri,
        },
        "qlib_provider": _regular_file_tree_identity(provider),
        "runtime": _runtime_identity(),
    }


def build_implementation_lock(workspace: Path | str) -> dict[str, object]:
    """Build, but do not publish, the configured prospective implementation lock."""
    return _live_payload(workspace)


def _lock_path(root: Path, lock_path: Path | str | None) -> Path:
    relative = DEFAULT_OUTPUT if lock_path is None else Path(lock_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("implementation lock output must be workspace-relative")
    if relative != DEFAULT_OUTPUT:
        raise ValueError("implementation lock output is not the frozen protocol path")
    path = root / relative
    if path.parent.resolve(strict=True) != (root / "prereg").resolve(strict=True):
        raise ValueError("implementation lock output escapes prereg")
    if path.is_symlink():
        raise ValueError("implementation lock output is a symlink")
    return path


def _write_exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        view = memoryview(_canonical_bytes(payload))
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("implementation-lock write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def write_implementation_lock(
    workspace: Path | str, lock_path: Path | str | None = None
) -> Path:
    """Create the frozen lock exactly once with ``O_EXCL`` semantics."""
    root = Path(workspace).resolve(strict=True)
    destination = _lock_path(root, lock_path)
    payload = build_implementation_lock(root)
    _write_exclusive(destination, payload)
    return destination


def _validate_lock_schema(lock: Mapping[str, object]) -> None:
    if set(lock) != _TOP_LEVEL_KEYS:
        raise ValueError("implementation lock has the wrong schema field set")
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported implementation-lock schema")
    if lock.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("implementation lock has the wrong protocol")
    if lock.get("created_before_science") is not True:
        raise ValueError("implementation lock was not created before science")
    if lock.get("scientific_results_observed") is not False:
        raise ValueError("implementation lock scientific-results state is invalid")


def verify_implementation_lock(
    workspace: Path | str, lock_path: Path | str | None = None
) -> dict[str, object]:
    """Strictly rehash and verify the complete live S2a v2 execution closure."""
    root = Path(workspace).resolve(strict=True)
    path = _lock_path(root, lock_path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("implementation lock is not a regular file")
    lock = _read_json(path, label="implementation lock")
    _validate_lock_schema(lock)
    live = _live_payload(root)

    locked_files = lock.get("files")
    live_files = live["files"]
    if not isinstance(locked_files, Mapping) or set(locked_files) != set(live_files):
        raise ValueError("implementation lock expected file set differs")
    if locked_files != live_files:
        raise ValueError("implementation file hash mismatch")
    if lock.get("source_tree") != live["source_tree"]:
        raise ValueError("implementation source-tree identity mismatch")

    locked_external = _mapping(
        lock.get("external_files"), label="implementation external files"
    )
    live_external = _mapping(live["external_files"], label="live external files")
    if set(locked_external) != {"data_cache", "baseline_metric"}:
        raise ValueError("implementation external file set differs")
    for key in ("data_cache", "baseline_metric"):
        if locked_external.get(key) != live_external.get(key):
            raise ValueError(f"implementation {key} identity mismatch")

    locked_quanta = _mapping(lock.get("quanta"), label="implementation Quanta")
    live_quanta = _mapping(live["quanta"], label="live Quanta")
    if locked_quanta.get("commit") != live_quanta.get("commit"):
        raise ValueError("Quanta commit identity mismatch")
    if locked_quanta.get("config") != live_quanta.get("config"):
        raise ValueError("Quanta config identity mismatch")
    if locked_quanta.get("runner") != live_quanta.get("runner"):
        raise ValueError("Quanta runner identity mismatch")
    if locked_quanta != live_quanta:
        raise ValueError("Quanta execution identity mismatch")
    if lock.get("qlib_provider") != live["qlib_provider"]:
        raise ValueError("QLib provider tree identity mismatch")
    if lock.get("runtime") != live["runtime"]:
        raise ValueError("implementation runtime identity mismatch")
    if lock != live:
        raise ValueError("implementation lock differs from the live closure")
    return lock


__all__ = [
    "build_implementation_lock",
    "verify_implementation_lock",
    "write_implementation_lock",
]
