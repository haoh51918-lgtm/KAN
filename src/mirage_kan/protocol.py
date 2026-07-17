"""Active exact-rebind corrective S2a protocol identities."""

from pathlib import Path

PROTOCOL_ID = "s2a_kan_e3_vertical_v8"
BASE_LOCK = Path("prereg/s2a_kan_e3_vertical_v8.lock.json")
IMPLEMENTATION_LOCK = Path("prereg/s2a_kan_e3_vertical_v8_implementation.lock.json")

__all__ = ["BASE_LOCK", "IMPLEMENTATION_LOCK", "PROTOCOL_ID"]
