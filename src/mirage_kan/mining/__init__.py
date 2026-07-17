"""Deterministic, typed S2a factor mining and prospective selection."""

from .core import (
    PROFILES,
    CandidateScore,
    MiningAttempt,
    ScoringRun,
    SelectionResult,
    ast_depth,
    ast_node_count,
    generate_attempts,
    greedy_select,
    permute_labels_within_date,
    score_attempts,
    select_random_control,
)

__all__ = [
    "PROFILES",
    "CandidateScore",
    "MiningAttempt",
    "ScoringRun",
    "SelectionResult",
    "ast_depth",
    "ast_node_count",
    "generate_attempts",
    "greedy_select",
    "permute_labels_within_date",
    "score_attempts",
    "select_random_control",
]
