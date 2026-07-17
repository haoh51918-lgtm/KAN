"""Canonical source identity helpers for experiment manifests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from mirage_kan.data.pit import sha256_file


def source_tree_identity(package_root: Path | str) -> dict[str, object]:
    """Hash every project Python source and their ordered identity list."""
    root = Path(package_root).resolve(strict=True)
    files = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*.py"))
    }
    digest = hashlib.sha256()
    for name, file_hash in files.items():
        digest.update(f"{name}\0{file_hash}\n".encode("utf-8"))
    return {"tree_sha256": digest.hexdigest(), "files": files}


def regular_file_tree_identity(root_path: Path | str) -> dict[str, object]:
    """Hash a complete flat identity stream for a regular-file-only tree."""
    root = Path(root_path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"identity root is not a directory: {root}")
    content_digest = hashlib.sha256()
    stat_digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names.sort()
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise ValueError("identity tree contains a symlinked directory")
        for name in sorted(filenames):
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("identity tree contains a non-regular file or symlink")
            relative = path.relative_to(root).as_posix()
            state = path.stat()
            file_hash = sha256_file(path)
            content_digest.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
            stat_digest.update(
                f"{relative}\0{state.st_size}\0{state.st_mtime_ns}\n".encode("utf-8")
            )
            file_count += 1
            total_bytes += state.st_size
    return {
        "path": str(root),
        "tree_sha256": content_digest.hexdigest(),
        "stat_inventory_sha256": stat_digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def regular_file_tree_stat_identity(root_path: Path | str) -> dict[str, object]:
    """Recheck a previously content-hashed tree without rereading file bodies."""
    root = Path(root_path).resolve(strict=True)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for directory, directory_names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names.sort()
        for name in directory_names:
            if (directory_path / name).is_symlink():
                raise ValueError("identity tree contains a symlinked directory")
        for name in sorted(filenames):
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("identity tree contains a non-regular file or symlink")
            relative = path.relative_to(root).as_posix()
            state = path.stat()
            digest.update(
                f"{relative}\0{state.st_size}\0{state.st_mtime_ns}\n".encode("utf-8")
            )
            file_count += 1
            total_bytes += state.st_size
    return {
        "path": str(root),
        "stat_inventory_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }
