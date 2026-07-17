"""Frozen S1 Gate A capacity-harness public seams."""

from .data import (
    GateAReplication,
    generate_gate_a_replication,
    save_gate_a_replication,
)
from .e5 import (
    E5Atom,
    E5ExecutableModel,
    E5SearchResult,
    E5SearchSettings,
    E5Structure,
    load_e5_export,
    save_e5_search,
    search_e5,
)
from .models import (
    CAPACITY_SPEC,
    FreeSplineKAN,
    MatchedMLP,
    SymbolicKAN,
    SymbolicResidualKAN,
)
from .posthoc import E2Symbolification, symbolify_e1, symbolify_e1_checkpoint
from .symbolic import (
    HardAnalyticalKAN,
    PRIMITIVE_NAMES,
    fidelity_metrics,
    load_hard_export,
    save_hard_export,
)
from .training import TrainingSettings, evaluate_test_once, train_and_select

__all__ = [
    "CAPACITY_SPEC",
    "FreeSplineKAN",
    "GateAReplication",
    "MatchedMLP",
    "SymbolicKAN",
    "SymbolicResidualKAN",
    "E2Symbolification",
    "E5Atom",
    "E5ExecutableModel",
    "E5SearchResult",
    "E5SearchSettings",
    "E5Structure",
    "HardAnalyticalKAN",
    "PRIMITIVE_NAMES",
    "TrainingSettings",
    "evaluate_test_once",
    "generate_gate_a_replication",
    "fidelity_metrics",
    "load_hard_export",
    "load_e5_export",
    "save_gate_a_replication",
    "save_hard_export",
    "save_e5_search",
    "search_e5",
    "symbolify_e1",
    "symbolify_e1_checkpoint",
    "train_and_select",
]
