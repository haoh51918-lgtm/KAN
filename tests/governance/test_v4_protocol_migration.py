from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import yaml

from mirage_kan.protocol import BASE_LOCK, IMPLEMENTATION_LOCK, PROTOCOL_ID


ROOT = Path(__file__).resolve().parents[2]
V2_CONFIG = ROOT / "configs/experiments/s2a_kan_e3_vertical_v2.yaml"
V3_CONFIG = ROOT / "configs/experiments/s2a_kan_e3_vertical_v3.yaml"
V4_CONFIG = ROOT / "configs/experiments/s2a_kan_e3_vertical_v4.yaml"
V5_CONFIG = ROOT / "configs/experiments/s2a_kan_e3_vertical_v5.yaml"
V6_CONFIG = ROOT / "configs/experiments/s2a_kan_e3_vertical_v6.yaml"
V7_CONFIG = ROOT / "configs/experiments/s2a_kan_e3_vertical_v7.yaml"
V8_CONFIG = ROOT / "configs/experiments/s2a_kan_e3_vertical_v8.yaml"


def _config(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _path_values(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {path for child in value.values() for path in _path_values(child)}
    if isinstance(value, str):
        return {value}
    raise AssertionError("artifact_paths must contain only mappings and strings")


def test_v4_changes_only_the_two_adaptive_library_size_scalars() -> None:
    v3 = _config(V3_CONFIG)
    v4 = _config(V4_CONFIG)

    assert v3["admission"]["minimum_library_size"] == 8
    assert v4["admission"]["minimum_library_size"] == 6
    assert v3["s2a_decision"]["integrity"]["production_library_size_minimum"] == 8
    assert v4["s2a_decision"]["integrity"]["production_library_size_minimum"] == 6

    normalized_v4 = deepcopy(v4)
    normalized_v4["protocol_id"] = v3["protocol_id"]
    normalized_v4["artifact_paths"] = v3["artifact_paths"]
    normalized_v4["admission"]["minimum_library_size"] = 8
    normalized_v4["s2a_decision"]["integrity"]["production_library_size_minimum"] = 8

    assert normalized_v4 == v3


def test_v4_writable_paths_are_v4_named_and_disjoint_from_v2_and_v3() -> None:
    v2_paths = _path_values(_config(V2_CONFIG)["artifact_paths"])
    v3_paths = _path_values(_config(V3_CONFIG)["artifact_paths"])
    v4_paths = _path_values(_config(V4_CONFIG)["artifact_paths"])

    assert all("v4" in path for path in v4_paths)
    assert v4_paths.isdisjoint(v2_paths)
    assert v4_paths.isdisjoint(v3_paths)


def test_v5_changes_only_identity_paths_and_evidence_class_from_v4() -> None:
    v4 = _config(V4_CONFIG)
    v5 = _config(V5_CONFIG)

    normalized_v5 = deepcopy(v5)
    normalized_v5["protocol_id"] = v4["protocol_id"]
    normalized_v5["artifact_paths"] = v4["artifact_paths"]
    normalized_v5["evidence_class"] = v4["evidence_class"]

    assert normalized_v5 == v4
    assert v5["evidence_class"] == "corrective_adaptive_repeated_development_screen"


def test_v5_writable_paths_are_named_and_disjoint_from_all_predecessors() -> None:
    predecessor_paths = set().union(
        *(
            _path_values(_config(path)["artifact_paths"])
            for path in (V2_CONFIG, V3_CONFIG, V4_CONFIG)
        )
    )
    v5_paths = _path_values(_config(V5_CONFIG)["artifact_paths"])

    assert all("v5" in path for path in v5_paths)
    assert v5_paths.isdisjoint(predecessor_paths)


def test_v6_changes_only_identity_and_paths_from_v5() -> None:
    v5 = _config(V5_CONFIG)
    v6 = _config(V6_CONFIG)

    normalized_v6 = deepcopy(v6)
    normalized_v6["protocol_id"] = v5["protocol_id"]
    normalized_v6["artifact_paths"] = v5["artifact_paths"]

    assert normalized_v6 == v5


def test_v6_writable_paths_are_named_and_disjoint_from_all_predecessors() -> None:
    predecessor_paths = set().union(
        *(
            _path_values(_config(path)["artifact_paths"])
            for path in (V2_CONFIG, V3_CONFIG, V4_CONFIG, V5_CONFIG)
        )
    )
    v6_paths = _path_values(_config(V6_CONFIG)["artifact_paths"])

    assert all("v6" in path for path in v6_paths)
    assert v6_paths.isdisjoint(predecessor_paths)


def test_active_protocol_identity_is_v8() -> None:
    assert PROTOCOL_ID == "s2a_kan_e3_vertical_v8"
    assert BASE_LOCK == Path("prereg/s2a_kan_e3_vertical_v8.lock.json")
    assert IMPLEMENTATION_LOCK == Path(
        "prereg/s2a_kan_e3_vertical_v8_implementation.lock.json"
    )


def test_v7_changes_only_identity_graph_authorization_and_mining_custody() -> None:
    v6 = _config(V6_CONFIG)
    v7 = _config(V7_CONFIG)

    normalized_v7 = deepcopy(v7)
    normalized_v7["protocol_id"] = v6["protocol_id"]
    normalized_v7["claim_boundary"]["graph_unlock_allowed"] = False
    normalized_v7["artifact_paths"] = v6["artifact_paths"]
    normalized_v7.pop("mining_source")

    assert normalized_v7 == v6
    assert v7["claim_boundary"]["graph_unlock_allowed"] is True
    assert v7["mining_source"]["mode"] == "verified_cross_protocol_rebind"


def test_v7_writable_paths_are_disjoint_but_source_paths_equal_v6() -> None:
    predecessor_paths = set().union(
        *(
            _path_values(_config(path)["artifact_paths"])
            for path in (V2_CONFIG, V3_CONFIG, V4_CONFIG, V5_CONFIG, V6_CONFIG)
        )
    )
    v6 = _config(V6_CONFIG)
    v7 = _config(V7_CONFIG)
    writable = _path_values(v7["artifact_paths"])
    source = v7["mining_source"]["source_artifact_paths"]

    assert all("v7" in path for path in writable)
    assert writable.isdisjoint(predecessor_paths)
    assert source == {
        key: v6["artifact_paths"][key]
        for key in (
            "mining_run",
            "kan_library",
            "gp_control_library",
            "permutation_control_library",
            "blackbox_control",
            "mechanism_cards",
            "blind_review_package",
        )
    }


def test_v8_changes_only_successor_identity_and_paths_from_v7() -> None:
    v7 = _config(V7_CONFIG)
    v8 = _config(V8_CONFIG)
    normalized = deepcopy(v8)
    normalized["protocol_id"] = v7["protocol_id"]
    normalized["artifact_paths"] = v7["artifact_paths"]
    normalized["mining_source"]["rebind_receipt"] = v7["mining_source"][
        "rebind_receipt"
    ]
    normalized["mining_source"]["absence_contract"] = v7["mining_source"][
        "absence_contract"
    ]

    assert normalized == v7
