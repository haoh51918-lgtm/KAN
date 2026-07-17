"""E2 train-only post-hoc symbolification of selected E1 spline edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import scipy
import torch
from scipy.optimize import least_squares
from torch import nn

from .data import FEATURE_NAMES
from .symbolic import (
    HardAnalyticalKAN,
    PRIMITIVE_NAMES,
    SymbolicShapeReport,
    evaluate_primitive,
    fidelity_metrics,
)


@dataclass(frozen=True)
class E2FitSettings:
    """Prospectively fixed independent primitive-fit budget and bounds."""

    method: str = "trf"
    loss: str = "linear"
    max_nfev: int = 500
    lower_bound: float = -8.0
    upper_bound: float = 8.0
    initial_parameters: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.0)


@dataclass(frozen=True)
class E2PrimitiveFit:
    edge: int
    primitive: str
    affine_parameters: tuple[float, float, float, float]
    train_edge_mse: float
    function_evaluations: int
    success: bool


@dataclass(frozen=True)
class E2Symbolification:
    """E1 soft reference and its independently executable analytical export."""

    source_model: nn.Module
    hard_model: HardAnalyticalKAN
    all_fits: tuple[E2PrimitiveFit, ...]
    selected_fits: tuple[E2PrimitiveFit, ...]
    fit_manifest: dict[str, object]

    @property
    def fit_count(self) -> int:
        return len(self.all_fits)

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.source_model.parameters()
            if parameter.requires_grad
        )

    def soft_predictions(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.source_model(inputs)

    def hard_predictions(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.hard_model.to(inputs.device)(inputs)

    def edge_curves(self, points: torch.Tensor) -> SymbolicShapeReport:
        tiled = points[:, None].expand(-1, self.hard_model.input_count)
        with torch.no_grad():
            soft_edges = self.source_model.edge_outputs(tiled)
            hard_edges = self.hard_model.to(points.device).edge_outputs(tiled)
        return SymbolicShapeReport(
            grid=points,
            soft_edges=soft_edges,
            hard_edges=hard_edges,
            residual_edges=None,
        )

    def fidelity(self, inputs: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            return fidelity_metrics(
                self.soft_predictions(inputs), self.hard_predictions(inputs)
            )

    def source_mass(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            edge_outputs = self.source_model.edge_outputs(inputs)
            centered = edge_outputs - torch.mean(edge_outputs, dim=0, keepdim=True)
            energies = torch.mean(torch.square(centered), dim=0)
        total = torch.sum(energies)
        if bool(total > 0):
            return energies / total
        return torch.zeros_like(energies)

    def selected_source_metadata(self, inputs: torch.Tensor) -> dict[str, object]:
        masses = self.source_mass(inputs)
        if not bool(torch.any(masses > 0)):
            return {
                "edge": None,
                "primitive": None,
                "source": None,
                "window": None,
                "selected_input_mass": 0.0,
                "selection_status": "zero_contribution_energy",
            }
        index = int(torch.argmax(masses).detach().cpu())
        metadata = self.hard_model.selected_metadata()[index]
        return {
            **metadata,
            "selected_input_mass": float(masses[index].detach().cpu()),
            "selection_status": "selected",
        }


def _fit_one_family(
    edge: int,
    primitive_index: int,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    settings: E2FitSettings,
) -> E2PrimitiveFit:
    name = PRIMITIVE_NAMES[primitive_index]
    x = inputs.detach().to(dtype=torch.float64, device="cpu")
    y = outputs.detach().to(dtype=torch.float64, device="cpu")

    def residual(parameters: object) -> object:
        values = torch.as_tensor(parameters, dtype=torch.float64)
        input_scale, input_bias, output_scale, output_bias = values
        prediction = output_scale * evaluate_primitive(
            name, input_scale * x + input_bias
        ) + output_bias
        return (prediction - y).numpy()

    fitted = least_squares(
        residual,
        x0=settings.initial_parameters,
        bounds=(settings.lower_bound, settings.upper_bound),
        method=settings.method,
        loss=settings.loss,
        max_nfev=settings.max_nfev,
    )
    parameters = tuple(float(value) for value in fitted.x)
    errors = torch.as_tensor(residual(fitted.x), dtype=torch.float64)
    return E2PrimitiveFit(
        edge=edge,
        primitive=name,
        affine_parameters=parameters,
        train_edge_mse=float(torch.mean(torch.square(errors))),
        function_evaluations=int(fitted.nfev),
        success=bool(fitted.success),
    )


def symbolify_e1(
    model: nn.Module,
    train_features: torch.Tensor,
    *,
    source_names: Sequence[str] | None = None,
    settings: E2FitSettings = E2FitSettings(),
) -> E2Symbolification:
    """Fit all eight families to each E1 edge using only training edge pairs."""
    if train_features.ndim != 2 or train_features.shape[1] < 1:
        raise ValueError("E2 train features must have shape (samples, edges)")
    if not bool(torch.isfinite(train_features).all()):
        raise ValueError("E2 train features must be finite")
    names = tuple(source_names or FEATURE_NAMES[: train_features.shape[1]])
    if len(names) != train_features.shape[1]:
        raise ValueError("one frozen source identity is required per E2 edge")
    model.eval()
    with torch.no_grad():
        edge_targets = model.edge_outputs(train_features).detach()
    if edge_targets.shape != train_features.shape:
        raise ValueError("E1 edge outputs must match its train feature matrix")
    if not bool(torch.isfinite(edge_targets).all()):
        raise ValueError("E1 train edge outputs must be finite")

    all_fits = []
    selected = []
    for edge in range(train_features.shape[1]):
        edge_fits = [
            _fit_one_family(
                edge,
                primitive_index,
                train_features[:, edge],
                edge_targets[:, edge],
                settings,
            )
            for primitive_index in range(len(PRIMITIVE_NAMES))
        ]
        all_fits.extend(edge_fits)
        selected.append(
            min(
                enumerate(edge_fits),
                key=lambda item: (item[1].train_edge_mse, item[0]),
            )[1]
        )

    primitive_indices = torch.tensor(
        [PRIMITIVE_NAMES.index(fit.primitive) for fit in selected], dtype=torch.long
    )
    affine = torch.tensor(
        [fit.affine_parameters for fit in selected], dtype=torch.float64
    )
    output_bias = model.output_bias.detach().to(dtype=torch.float64, device="cpu")
    hard = HardAnalyticalKAN(
        primitive_indices=primitive_indices,
        affine_parameters=affine,
        output_bias=output_bias,
        source_names=names,
    )
    return E2Symbolification(
        source_model=model,
        hard_model=hard,
        all_fits=tuple(all_fits),
        selected_fits=tuple(selected),
        fit_manifest={
            "optimizer": "scipy.optimize.least_squares",
            "scipy_version": scipy.__version__,
            "settings": asdict(settings),
            "fits_per_edge": len(PRIMITIVE_NAMES),
            "total_fits": len(all_fits),
            "selection_data": "train_inputs_and_e1_edges_only",
            "selection_metric": "train_edge_mse",
            "tie_break": "primitive_dictionary_order",
            "e1_or_target_retraining": False,
        },
    )


def symbolify_e1_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    train_features: torch.Tensor,
    *,
    source_names: Sequence[str] | None = None,
    settings: E2FitSettings = E2FitSettings(),
) -> E2Symbolification:
    """Load a selected E1 state and symbolify it without any optimizer step."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return symbolify_e1(
        model,
        train_features,
        source_names=source_names,
        settings=settings,
    )
