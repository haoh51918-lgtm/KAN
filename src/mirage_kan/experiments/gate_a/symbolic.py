"""Frozen analytical primitives and shared symbolic export interfaces for Gate A."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

PRIMITIVE_NAMES = (
    "Identity",
    "Abs",
    "Square",
    "SignedLog1p",
    "Tanh",
    "Clip(-1,1)",
    "PositiveHinge(0)",
    "NegativeHinge(0)",
)

def _finite_input(values: torch.Tensor) -> torch.Tensor:
    if not values.is_floating_point():
        values = values.to(dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("analytical primitives require finite inputs")
    return values


def evaluate_primitive(name: str, values: torch.Tensor) -> torch.Tensor:
    """Execute one frozen primitive as finite-domain Torch code."""
    x = _finite_input(values)
    if name == "Identity":
        result = x
    elif name == "Abs":
        result = torch.abs(x)
    elif name == "Square":
        result = torch.square(x)
    elif name == "SignedLog1p":
        result = torch.sign(x) * torch.log1p(torch.abs(x))
    elif name == "Tanh":
        result = torch.tanh(x)
    elif name == "Clip(-1,1)":
        result = torch.clamp(x, -1.0, 1.0)
    elif name == "PositiveHinge(0)":
        result = torch.relu(x)
    elif name == "NegativeHinge(0)":
        result = torch.relu(-x)
    else:
        raise KeyError(f"unknown frozen analytical primitive: {name}")
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"primitive produced non-finite output: {name}")
    return result


def evaluate_all_primitives(values: torch.Tensor) -> torch.Tensor:
    """Evaluate the frozen primitive library on a final primitive axis."""
    return torch.stack(
        tuple(evaluate_primitive(name, values) for name in PRIMITIVE_NAMES), dim=-1
    )


def residual_energy_ratio(
    residual_outputs: torch.Tensor,
    full_outputs: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return proposal rho with its explicit epsilon-stabilized denominator."""
    if residual_outputs.shape != full_outputs.shape or residual_outputs.ndim < 1:
        raise ValueError("residual and full outputs must have the same non-scalar shape")
    if epsilon <= 0:
        raise ValueError("residual-energy epsilon must be positive")
    numerator = torch.mean(torch.square(residual_outputs), dim=0)
    denominator = torch.mean(torch.square(full_outputs), dim=0)
    return torch.where(
        denominator == 0,
        torch.where(
            numerator == 0,
            torch.zeros_like(numerator),
            numerator / epsilon,
        ),
        numerator / (denominator + epsilon),
    )


def fidelity_metrics(
    soft_predictions: torch.Tensor, hard_predictions: torch.Tensor
) -> dict[str, float]:
    """Compute soft-hard Pearson, NRMSE, and maximum absolute error."""
    soft = soft_predictions.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    hard = hard_predictions.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    if soft.shape != hard.shape or soft.numel() < 2:
        raise ValueError("soft and hard predictions need the same shape and two samples")
    if not torch.isfinite(soft).all() or not torch.isfinite(hard).all():
        raise ValueError("fidelity predictions must be finite")
    centered_soft = soft - soft.mean()
    centered_hard = hard - hard.mean()
    scale = torch.sqrt(
        torch.sum(torch.square(centered_soft))
        * torch.sum(torch.square(centered_hard))
    )
    if float(scale) == 0:
        pearson = 1.0 if torch.equal(soft, hard) else 0.0
    else:
        pearson = float(torch.sum(centered_soft * centered_hard) / scale)
    denominator = torch.std(soft, correction=0)
    rmse = torch.sqrt(torch.mean(torch.square(soft - hard)))
    if float(denominator) == 0:
        nrmse = 0.0 if float(rmse) == 0 else float("inf")
    else:
        nrmse = float(rmse / denominator)
    return {
        "pearson": pearson,
        "nrmse": nrmse,
        "max_absolute_error": float(torch.max(torch.abs(soft - hard))),
    }


def _source_metadata(source_name: str) -> dict[str, Any]:
    window = None
    if source_name.startswith("Return(Close,"):
        window = int(source_name.removeprefix("Return(Close,").removesuffix(")"))
    elif "TsMean(Volume," in source_name:
        window = int(source_name.split("TsMean(Volume,", 1)[1].split(")", 1)[0])
    return {"source": source_name, "window": window}


@dataclass(frozen=True)
class SymbolicShapeReport:
    """Shared model-shape outputs on the sealed grid."""

    grid: torch.Tensor
    soft_edges: torch.Tensor
    hard_edges: torch.Tensor
    residual_edges: torch.Tensor | None


class HardAnalyticalKAN(nn.Module):
    """Independently executable additive analytical model with no soft state."""

    def __init__(
        self,
        *,
        primitive_indices: torch.Tensor,
        affine_parameters: torch.Tensor,
        output_bias: torch.Tensor,
        source_names: Sequence[str],
    ) -> None:
        super().__init__()
        indices = primitive_indices.detach().to(dtype=torch.long).reshape(-1)
        affine = affine_parameters.detach().to(dtype=torch.float64)
        bias = output_bias.detach().to(dtype=torch.float64).reshape(1)
        if affine.shape != (indices.numel(), 4):
            raise ValueError("hard affine parameters must have shape (edges, 4)")
        if len(source_names) != indices.numel():
            raise ValueError("one source identity is required per hard edge")
        if bool(((indices < 0) | (indices >= len(PRIMITIVE_NAMES))).any()):
            raise ValueError("hard primitive index is outside the frozen dictionary")
        self.source_names = tuple(source_names)
        self.register_buffer("primitive_indices", indices.clone())
        self.register_buffer("affine_parameters", affine.clone())
        self.register_buffer("output_bias", bias.clone())

    @property
    def input_count(self) -> int:
        return int(self.primitive_indices.numel())

    @property
    def parameter_count(self) -> int:
        return 0

    def edge_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_count:
            raise ValueError(f"inputs must have shape (samples, {self.input_count})")
        outputs = []
        for edge, primitive_index in enumerate(self.primitive_indices.tolist()):
            input_scale, input_bias, output_scale, output_bias = self.affine_parameters[
                edge
            ]
            transformed = input_scale * inputs[:, edge] + input_bias
            primitive = evaluate_primitive(PRIMITIVE_NAMES[primitive_index], transformed)
            outputs.append(output_scale * primitive + output_bias)
        return torch.stack(outputs, dim=1)

    def edge_curves(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 1:
            raise ValueError("curve points must be one-dimensional")
        return self.edge_outputs(points[:, None].expand(-1, self.input_count))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.edge_outputs(inputs).sum(dim=1, keepdim=True) + self.output_bias

    def selected_metadata(self) -> list[dict[str, Any]]:
        return [
            {
                "edge": edge,
                "primitive": PRIMITIVE_NAMES[index],
                **_source_metadata(source),
            }
            for edge, (index, source) in enumerate(
                zip(self.primitive_indices.tolist(), self.source_names, strict=True)
            )
        ]

    def canonical_description(self) -> dict[str, Any]:
        edges = []
        for metadata, affine in zip(
            self.selected_metadata(), self.affine_parameters.tolist(), strict=True
        ):
            edges.append(
                {
                    **metadata,
                    "affine": [float(value) for value in affine],
                }
            )
        return {
            "type": "AdditiveSymbolicKAN",
            "edges": edges,
            "output_bias": float(self.output_bias.item()),
        }

    def canonical_serialization(self) -> str:
        return json.dumps(
            self.canonical_description(), sort_keys=True, separators=(",", ":")
        )

    @property
    def description_length(self) -> int:
        return len(self.canonical_serialization().encode("utf-8"))

    def complexity(self) -> dict[str, int]:
        return {
            "ast_node_count": 1 + self.input_count * 4,
            "ast_depth": 4,
            "free_constants": self.input_count * 4 + 1,
            "serialized_description_length": self.description_length,
        }


def save_hard_export(
    model: HardAnalyticalKAN,
    output_directory: Path | str,
    *,
    arm: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Publish a reconstructable hard checkpoint, then its manifest, no-replace."""
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    checkpoint_path = output / "hard_model.pt"
    torch.save(
        {
            "primitive_indices": model.primitive_indices.detach().cpu(),
            "affine_parameters": model.affine_parameters.detach().cpu(),
            "output_bias": model.output_bias.detach().cpu(),
            "source_names": model.source_names,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path = output / "manifest.json"
    temporary = output / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "arm": arm,
                "canonical": model.canonical_description(),
                "complexity": model.complexity(),
                "selected_metadata": model.selected_metadata(),
                "metadata": metadata or {},
                "paths": {"checkpoint": str(checkpoint_path)},
                "checkpoint_sha256": checkpoint_sha256,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest_path


def load_hard_export(manifest_path: Path | str) -> HardAnalyticalKAN:
    """Reconstruct a hard analytical model from its published checkpoint."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    checkpoint_path = Path(manifest["paths"]["checkpoint"])
    actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual_sha256 != manifest["checkpoint_sha256"]:
        raise ValueError("hard symbolic checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return HardAnalyticalKAN(
        primitive_indices=checkpoint["primitive_indices"],
        affine_parameters=checkpoint["affine_parameters"],
        output_bias=checkpoint["output_bias"],
        source_names=checkpoint["source_names"],
    )
