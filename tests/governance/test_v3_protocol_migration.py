from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
V2_CONFIG = ROOT / "configs/experiments/s2a_kan_e3_vertical_v2.yaml"
V3_CONFIG = ROOT / "configs/experiments/s2a_kan_e3_vertical_v3.yaml"


def _config(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _path_values(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            path
            for child in value.values()
            for path in _path_values(child)
        }
    if isinstance(value, str):
        return {value}
    raise AssertionError("artifact_paths must contain only mappings and strings")


def test_v3_changes_no_scientific_setting_from_terminal_v2() -> None:
    v2 = _config(V2_CONFIG)
    v3 = _config(V3_CONFIG)

    for config in (v2, v3):
        del config["protocol_id"]
        del config["artifact_paths"]

    assert v3 == v2


def test_v3_writable_paths_are_disjoint_from_terminal_v2() -> None:
    v2_paths = _path_values(_config(V2_CONFIG)["artifact_paths"])
    v3_paths = _path_values(_config(V3_CONFIG)["artifact_paths"])

    assert v2_paths.isdisjoint(v3_paths)


def test_active_source_has_no_terminal_v2_protocol_literal() -> None:
    forbidden = (
        "s2a_kan_e3_vertical_v2",
        "prereg/s2a_kan_e3_vertical_v2.lock.json",
        "mirage_kan_s2a_v2",
    )
    for path in (ROOT / "src/mirage_kan").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(literal in source for literal in forbidden), path
