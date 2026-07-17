"""Batch execution backends for validated factor ASTs."""

from .torch import evaluate_torch

__all__ = ["evaluate_torch"]
