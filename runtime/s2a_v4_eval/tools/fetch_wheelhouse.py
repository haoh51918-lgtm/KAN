"""Fetch one hash-locked artifact for every distribution in requirements.lock."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import shutil
import stat
import sys
import tomllib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from packaging.markers import Marker
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.lock"
UV_LOCK = ROOT / "uv.lock"


@dataclass(frozen=True)
class Requirement:
    name: str
    version: str
    marker: str | None
    hashes: frozenset[str]


@dataclass(frozen=True)
class Artifact:
    name: str
    version: str
    filename: str
    url: str
    sha256: str
    size: int | None
    applies: bool


def read_requirements() -> list[Requirement]:
    text = REQUIREMENTS.read_text(encoding="utf-8")
    starts = list(
        re.finditer(
            r"(?m)^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^ ;\\]+)(?: ; ([^\\\n]+))? \\\\?$",
            text,
        )
    )
    requirements = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.start() : end]
        hashes = frozenset(re.findall(r"--hash=sha256:([0-9a-f]{64})", block))
        if not hashes:
            raise ValueError(f"no hashes found for {match.group(1)}")
        requirements.append(
            Requirement(
                name=canonicalize_name(match.group(1)),
                version=match.group(2),
                marker=match.group(3),
                hashes=hashes,
            )
        )
    if not requirements:
        raise ValueError("requirements.lock contains no distributions")
    return requirements


def choose_artifacts() -> list[Artifact]:
    requirements = read_requirements()
    lock = tomllib.loads(UV_LOCK.read_text(encoding="utf-8"))
    packages = {canonicalize_name(item["name"]): item for item in lock["package"]}
    tag_rank = {tag: rank for rank, tag in enumerate(sys_tags())}
    selected = []

    for requirement in requirements:
        package = packages.get(requirement.name)
        if package is None or package["version"] != requirement.version:
            raise ValueError(
                f"uv.lock mismatch for {requirement.name}=={requirement.version}"
            )
        applies = requirement.marker is None or Marker(requirement.marker).evaluate()
        wheels = []
        for wheel in package.get("wheels", []):
            filename = urllib.parse.unquote(
                urllib.parse.urlsplit(wheel["url"]).path.rsplit("/", 1)[-1]
            )
            _, _, _, tags = parse_wheel_filename(filename)
            ranks = [tag_rank[tag] for tag in tags if tag in tag_rank]
            wheels.append((min(ranks) if ranks else None, filename, wheel))

        compatible = [item for item in wheels if item[0] is not None]
        if compatible:
            _, filename, source = min(compatible, key=lambda item: (item[0], item[1]))
        elif not applies and wheels:
            preferred = [item for item in wheels if "win_amd64" in item[1]] or wheels
            _, filename, source = min(preferred, key=lambda item: item[1])
        elif "sdist" in package:
            source = package["sdist"]
            filename = urllib.parse.unquote(
                urllib.parse.urlsplit(source["url"]).path.rsplit("/", 1)[-1]
            )
        else:
            raise ValueError(
                f"no usable artifact for {requirement.name}=={requirement.version}"
            )
        if (
            not filename
            or filename in {".", ".."}
            or "/" in filename
            or "\\" in filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"unsafe artifact filename for {requirement.name}")

        digest = source["hash"].removeprefix("sha256:")
        if digest not in requirement.hashes:
            raise ValueError(
                f"selected hash is absent from requirements.lock for {requirement.name}"
            )
        selected.append(
            Artifact(
                name=requirement.name,
                version=requirement.version,
                filename=filename,
                url=source["url"],
                sha256=digest,
                size=source.get("size"),
                applies=applies,
            )
        )

    if len({item.name for item in selected}) != len(selected):
        raise ValueError("requirements.lock contains duplicate normalized names")
    if len({item.filename for item in selected}) != len(selected):
        raise ValueError("selected artifacts contain duplicate filenames")
    return selected


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(path: Path, artifact: Artifact) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"not a regular file: {path}")
    if artifact.size is not None and path.stat().st_size != artifact.size:
        raise ValueError(f"size mismatch: {path}")
    if file_sha256(path) != artifact.sha256:
        raise ValueError(f"SHA256 mismatch: {path}")


def validate_destination(destination: Path, expected: set[str]) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"destination is not a regular directory: {destination}")
    unexpected = []
    for entry in os.scandir(destination):
        mode = entry.stat(follow_symlinks=False).st_mode
        if entry.is_symlink() or not stat.S_ISREG(mode) or entry.name not in expected:
            unexpected.append(entry.name)
    if unexpected:
        raise ValueError(f"unexpected destination entries: {sorted(unexpected)}")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_file(destination: Path, source: Path, artifact: Artifact) -> None:
    validate(source, artifact)
    final = destination / artifact.filename
    descriptor = os.open(final, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
            shutil.copyfileobj(input_stream, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        validate(final, artifact)
        fsync_directory(destination)
    except BaseException:
        final.unlink(missing_ok=True)
        raise


def fetch(destination: Path, artifact: Artifact) -> str:
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / artifact.filename
    if final.exists():
        validate(final, artifact)
        return "reused"

    temporary = destination / f".{artifact.filename}.{os.getpid()}.part"
    try:
        request = urllib.request.Request(
            artifact.url, headers={"User-Agent": "s2a-v4-runtime-closure/1"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("xb") as output,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        validate(temporary, artifact)
        publish_file(destination, temporary, artifact)
        return "downloaded"
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "destination", nargs="?", type=Path, default=ROOT / ".wheelhouse_staging"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--copy-from", type=Path)
    args = parser.parse_args()
    artifacts = choose_artifacts()

    expected = {artifact.filename for artifact in artifacts}
    if args.copy_from is not None:
        if args.destination.exists():
            raise ValueError("copy destination already exists")
        args.destination.mkdir(parents=True)
        fsync_directory(args.destination.parent)
        validate_destination(args.copy_from, expected)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            actions = list(
                executor.map(
                    lambda item: (
                        publish_file(
                            args.destination,
                            args.copy_from / item.filename,
                            item,
                        ),
                        "published",
                    )[1],
                    artifacts,
                )
            )
        fsync_directory(args.destination)
    elif args.destination.exists():
        validate_destination(args.destination, expected)
    if args.copy_from is not None:
        pass
    elif args.verify_only:
        for artifact in artifacts:
            validate(args.destination / artifact.filename, artifact)
        validate_destination(args.destination, expected)
        actions = ["verified"] * len(artifacts)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            actions = list(
                executor.map(lambda item: fetch(args.destination, item), artifacts)
            )
        for artifact in artifacts:
            validate(args.destination / artifact.filename, artifact)

    total_size = sum(
        (args.destination / artifact.filename).stat().st_size for artifact in artifacts
    )
    applicable = sum(artifact.applies for artifact in artifacts)
    wheels = sum(artifact.filename.endswith(".whl") for artifact in artifacts)
    print(
        f"artifacts={len(artifacts)} applicable={applicable} wheels={wheels} "
        f"sdists={len(artifacts) - wheels} bytes={total_size} "
        f"downloaded={actions.count('downloaded')} reused={actions.count('reused')} "
        f"verified={actions.count('verified')} published={actions.count('published')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
