from __future__ import annotations

from mirage_kan.seed import seed_wiring_programs


def test_seed_library_is_small_valid_and_deterministic() -> None:
    first = seed_wiring_programs()
    second = seed_wiring_programs()
    assert 1 <= len(first) <= 8
    assert list(first) == list(second)
    assert [node.identity for node in first.values()] == [
        node.identity for node in second.values()
    ]
    assert all(node.contract().causal for node in first.values())
