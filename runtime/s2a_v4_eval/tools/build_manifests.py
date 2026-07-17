"""Build categorized, hash-complete manifests for the frozen v4 runtime."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import canonicalize_name

from fetch_wheelhouse import choose_artifacts, validate


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests"
ROOT_FILES = (
    ROOT / ".python-version",
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "requirements.lock",
    ROOT / "uv.lock",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, category: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"manifest input is not a regular file: {path}")
    return {
        "category": category,
        "path": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "size": path.stat().st_size,
    }


def json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def tsv_text(rows: list[dict[str, object]], fields: tuple[str, ...]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in rows)
    return stream.getvalue()


def runtime_files() -> list[dict[str, object]]:
    paths = [
        *((path, "frozen_root") for path in ROOT_FILES),
        *((path, "closure_tool") for path in sorted((ROOT / "tools").glob("*.py"))),
        *((path, "evidence") for path in sorted((ROOT / "evidence").glob("*"))),
    ]
    return [file_record(path, category) for path, category in paths]


def wheelhouse_files() -> list[dict[str, object]]:
    rows = []
    for artifact in choose_artifacts():
        path = ROOT / "wheelhouse" / artifact.filename
        validate(path, artifact)
        rows.append(
            {
                "applies_on_linux_runtime": artifact.applies,
                "category": "wheel" if artifact.filename.endswith(".whl") else "sdist",
                "distribution": artifact.name,
                "path": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
                "version": artifact.version,
            }
        )
    return rows


def installed_distributions() -> tuple[list[dict[str, object]], dict[str, object]]:
    prefix = (ROOT / ".venv").resolve(strict=True)
    site_packages = prefix / "lib/python3.12/site-packages"
    rows = []
    for dist_info in sorted(site_packages.glob("*.dist-info")):
        metadata_path = dist_info / "METADATA"
        record_path = dist_info / "RECORD"
        if (
            dist_info.is_symlink()
            or not metadata_path.is_file()
            or not record_path.is_file()
        ):
            raise ValueError(f"incomplete installed distribution: {dist_info}")
        metadata = BytesParser().parsebytes(metadata_path.read_bytes())
        rows.append(
            {
                "distribution": canonicalize_name(metadata["Name"]),
                "record_path": str(record_path.relative_to(prefix)),
                "record_sha256": file_sha256(record_path),
                "record_size": record_path.stat().st_size,
                "version": metadata["Version"],
            }
        )
    if len({row["distribution"] for row in rows}) != len(rows):
        raise ValueError("installed environment contains duplicate normalized names")

    logical_python = ROOT / ".venv/bin/python"
    resolved_python = logical_python.resolve(strict=True)
    python = {
        "logical_executable": str(logical_python.relative_to(ROOT)),
        "resolved_executable": str(resolved_python),
        "resolved_executable_sha256": file_sha256(resolved_python),
        "resolved_executable_size": resolved_python.stat().st_size,
        "version": "3.12.3",
    }
    return rows, python


def publish(name: str, content: str) -> None:
    path = MANIFESTS / name
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    if MANIFESTS.is_symlink() or not MANIFESTS.is_dir():
        raise ValueError("manifests destination must be a real directory")
    if any(MANIFESTS.iterdir()):
        raise ValueError("manifests destination must be empty")
    for temporary in (ROOT / ".wheelhouse_staging", ROOT / ".download_fallback"):
        if temporary.exists() or temporary.is_symlink():
            raise ValueError(f"acquisition-only duplicate still exists: {temporary}")

    runtime = runtime_files()
    wheels = wheelhouse_files()
    installed, python = installed_distributions()
    if (
        len(wheels) != 223
        or sum(row["applies_on_linux_runtime"] for row in wheels) != 217
    ):
        raise ValueError("wheelhouse counts differ from the frozen closure")
    if len(installed) != 217:
        raise ValueError("installed distribution count differs from the frozen closure")

    runtime_json = json_text(
        {
            "file_count": len(runtime),
            "files": runtime,
            "schema_version": "s2a_v4_runtime_files_v1",
        }
    )
    wheels_json = json_text(
        {
            "applicable_distribution_count": 217,
            "artifact_count": len(wheels),
            "artifacts": wheels,
            "schema_version": "s2a_v4_wheelhouse_files_v1",
            "total_bytes": sum(row["size"] for row in wheels),
        }
    )
    installed_json = json_text(
        {
            "distribution_count": len(installed),
            "distributions": installed,
            "python": python,
            "schema_version": "s2a_v4_installed_distributions_v1",
        }
    )
    machine = {
        "installed_distributions.json": installed_json,
        "installed_distributions.tsv": tsv_text(
            installed,
            ("distribution", "version", "record_path", "record_size", "record_sha256"),
        ),
        "runtime_files.json": runtime_json,
        "runtime_files.tsv": tsv_text(runtime, ("category", "path", "size", "sha256")),
        "wheelhouse_files.json": wheels_json,
        "wheelhouse_files.tsv": tsv_text(
            wheels,
            (
                "category",
                "distribution",
                "version",
                "applies_on_linux_runtime",
                "path",
                "size",
                "sha256",
            ),
        ),
    }
    for name, content in machine.items():
        publish(name, content)

    machine_hashes = {name: file_sha256(MANIFESTS / name) for name in sorted(machine)}
    summary = {
        "categories": {
            "evidence_files": sum(row["category"] == "evidence" for row in runtime),
            "frozen_root_files": sum(
                row["category"] == "frozen_root" for row in runtime
            ),
            "runtime_closure_tools": sum(
                row["category"] == "closure_tool" for row in runtime
            ),
        },
        "installed_distribution_count": len(installed),
        "machine_manifest_sha256": machine_hashes,
        "passed": True,
        "schema_version": "s2a_v4_closure_summary_v1",
        "wheelhouse": {
            "applicable_distribution_count": 217,
            "artifact_count": len(wheels),
            "sdist_count": sum(row["category"] == "sdist" for row in wheels),
            "total_bytes": sum(row["size"] for row in wheels),
            "wheel_count": sum(row["category"] == "wheel" for row in wheels),
        },
    }
    summary_text = json_text(summary)
    publish("closure_summary.json", summary_text)
    summary_sha256 = file_sha256(MANIFESTS / "closure_summary.json")
    human = f"""# S2a v4 runtime closure manifest

- Status: PASS
- Frozen root files: {summary["categories"]["frozen_root_files"]}
- Runtime closure tools: {summary["categories"]["runtime_closure_tools"]}
- Evidence files: {summary["categories"]["evidence_files"]}
- Wheelhouse: 223 artifacts (222 wheels, 1 sdist), 4,884,582,030 bytes
- Linux-applicable installed distributions: 217
- Python: 3.12.3
- Torch/CUDA/cuDNN: 2.9.1+cu129 / 12.9 / 9.10.2.21
- Closure summary SHA256: `{summary_sha256}`

The JSON and TSV files in this directory are the machine-readable categorized
manifests. `runtime_files` covers frozen roots, closure tools, and evidence;
`wheelhouse_files` covers every selected locked artifact; and
`installed_distributions` binds every installed `.dist-info/RECORD`.
"""
    publish("MANIFEST.md", human)
    descriptor = os.open(MANIFESTS, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(
        f"runtime_files={len(runtime)} wheelhouse_artifacts={len(wheels)} "
        f"installed_distributions={len(installed)} bytes={summary['wheelhouse']['total_bytes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
