"""E1 cubic B-spline KAN and parameter-matched C6 capacity control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from .data import FEATURE_NAMES
from .symbolic import (
    HardAnalyticalKAN,
    PRIMITIVE_NAMES,
    SymbolicShapeReport,
    evaluate_primitive,
    fidelity_metrics,
    residual_energy_ratio,
)

PYKAN_REFERENCE = {
    "commit": "ecde4ec3274d3bef1ad737479cf126aed38ab530",
    "spline_py_sha256": "16443d4b5e14b7096d585fbd2b32a6f26d142f3b0556428a0ac9845b8563a8a0",
    "kan_layer_py_sha256": "fd1a408b0ee0a8b83eefcde193b30f7323c52172a72e21280e15430cc6742b8d",
    "reuse": "independent_dependency_free_cox_de_boor_implementation",
}


@dataclass(frozen=True)
class CapacityModelSpec:
    """Prospective E3/E4 architecture accounting used to freeze C6 capacity."""

    input_count: int = 6
    output_count: int = 1
    spline_degree: int = 3
    spline_grid_intervals: int = 16
    spline_grid_minimum: float = -6.0
    spline_grid_maximum: float = 6.0
    analytical_primitives_per_edge: int = 8
    affine_parameters_per_primitive: int = 4
    gate_parameters_per_edge: int = 8
    residual_scale_parameters_per_edge: int = 1
    c6_hidden_width: int = 45

    @property
    def spline_coefficients_per_edge(self) -> int:
        return self.spline_grid_intervals + self.spline_degree

    @property
    def e4_parameter_count(self) -> int:
        per_edge = (
            self.gate_parameters_per_edge
            + self.analytical_primitives_per_edge
            * self.affine_parameters_per_primitive
            + self.spline_coefficients_per_edge
            + self.residual_scale_parameters_per_edge
        )
        return self.input_count * self.output_count * per_edge + self.output_count


CAPACITY_SPEC = CapacityModelSpec()


def cubic_bspline_basis(x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Evaluate a degree-three B-spline basis by the Cox--de Boor recurrence."""
    if knots.ndim != 1 or knots.numel() < 5:
        raise ValueError("knots must be a one-dimensional cubic knot vector")
    if not bool(torch.all(knots[1:] > knots[:-1])):
        raise ValueError("knots must be strictly increasing")
    values = x.unsqueeze(-1)
    basis = ((values >= knots[:-1]) & (values < knots[1:])).to(x.dtype)
    for degree in range(1, 4):
        left_denominator = knots[degree:-1] - knots[: -(degree + 1)]
        right_denominator = knots[degree + 1 :] - knots[1:-degree]
        left = (
            (values - knots[: -(degree + 1)])
            / left_denominator
            * basis[..., :-1]
        )
        right = (
            (knots[degree + 1 :] - values)
            / right_denominator
            * basis[..., 1:]
        )
        basis = left + right
    return basis


class FreeSplineKAN(nn.Module):
    """Single-layer additive KAN whose input edges are free cubic B-splines."""

    spline_degree = 3

    def __init__(
        self,
        input_count: int = CAPACITY_SPEC.input_count,
        *,
        grid_intervals: int = CAPACITY_SPEC.spline_grid_intervals,
        grid_range: tuple[float, float] = (
            CAPACITY_SPEC.spline_grid_minimum,
            CAPACITY_SPEC.spline_grid_maximum,
        ),
    ) -> None:
        super().__init__()
        if input_count < 1 or grid_intervals < 1 or grid_range[0] >= grid_range[1]:
            raise ValueError("invalid additive spline architecture")
        self.input_count = input_count
        self.grid_intervals = grid_intervals
        step = (grid_range[1] - grid_range[0]) / grid_intervals
        internal = torch.linspace(
            grid_range[0],
            grid_range[1],
            grid_intervals + 1,
            dtype=torch.float64,
        )
        left = internal[0] - step * torch.arange(3, 0, -1, dtype=torch.float64)
        right = internal[-1] + step * torch.arange(1, 4, dtype=torch.float64)
        knots = torch.cat((left, internal, right))
        self.register_buffer("knots", knots)
        coefficient_count = knots.numel() - self.spline_degree - 1
        self.coefficients = nn.Parameter(
            torch.zeros(input_count, coefficient_count, dtype=torch.float64)
        )
        self.output_bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def edge_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return every input-to-output spline edge for each sample."""
        if inputs.ndim != 2 or inputs.shape[1] != self.input_count:
            raise ValueError(f"inputs must have shape (samples, {self.input_count})")
        basis = cubic_bspline_basis(inputs, self.knots)
        return torch.einsum("bic,ic->bi", basis, self.coefficients)

    def edge_curves(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluate every learned input edge on one shared one-dimensional grid."""
        if points.ndim != 1:
            raise ValueError("curve points must be one-dimensional")
        tiled = points[:, None].expand(-1, self.input_count)
        return self.edge_outputs(tiled)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Sum additive edge outputs and the output bias."""
        return self.edge_outputs(inputs).sum(dim=1, keepdim=True) + self.output_bias


@dataclass(frozen=True)
class SymbolicModelSettings:
    """Prospectively fixed symbolic optimization choices."""

    initial_temperature: float = 2.0
    final_temperature: float = 0.1
    entropy_weight: float = 0.001
    sparsity_weight: float = 0.001
    spline_residual_energy_weight: float = 0.01
    primitive_output_scale_initialization: float = 0.01
    residual_scale_initialization: float = 0.01


SYMBOLIC_SETTINGS = SymbolicModelSettings()


class _SymbolicKANBase(nn.Module):
    """Shared E3/E4 analytical path; E4 alone adds the residual spline path."""

    spline_degree = 3

    def __init__(
        self,
        input_count: int = CAPACITY_SPEC.input_count,
        *,
        source_names: Sequence[str] | None = None,
        include_spline_residual: bool,
        settings: SymbolicModelSettings = SYMBOLIC_SETTINGS,
    ) -> None:
        super().__init__()
        if input_count < 1 or input_count > len(FEATURE_NAMES):
            raise ValueError("symbolic input count is outside the frozen feature set")
        self.input_count = input_count
        self.source_names = tuple(source_names or FEATURE_NAMES[:input_count])
        if len(self.source_names) != input_count:
            raise ValueError("one frozen source identity is required per edge")
        self.include_spline_residual = include_spline_residual
        self.settings = settings
        primitive_count = len(PRIMITIVE_NAMES)
        self.gate_logits = nn.Parameter(
            torch.zeros(input_count, primitive_count, dtype=torch.float64)
        )
        affine = torch.zeros(input_count, primitive_count, 4, dtype=torch.float64)
        affine[..., 0] = 1.0
        affine[..., 2] = settings.primitive_output_scale_initialization
        self.affine_parameters = nn.Parameter(affine)
        if include_spline_residual:
            step = (
                CAPACITY_SPEC.spline_grid_maximum
                - CAPACITY_SPEC.spline_grid_minimum
            ) / CAPACITY_SPEC.spline_grid_intervals
            internal = torch.linspace(
                CAPACITY_SPEC.spline_grid_minimum,
                CAPACITY_SPEC.spline_grid_maximum,
                CAPACITY_SPEC.spline_grid_intervals + 1,
                dtype=torch.float64,
            )
            left = internal[0] - step * torch.arange(3, 0, -1, dtype=torch.float64)
            right = internal[-1] + step * torch.arange(1, 4, dtype=torch.float64)
            knots = torch.cat((left, internal, right))
            self.register_buffer("knots", knots)
            self.coefficients = nn.Parameter(
                torch.zeros(
                    input_count,
                    CAPACITY_SPEC.spline_coefficients_per_edge,
                    dtype=torch.float64,
                )
            )
            self.residual_scales = nn.Parameter(
                torch.full(
                    (input_count,),
                    settings.residual_scale_initialization,
                    dtype=torch.float64,
                )
            )
        self.output_bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self._training_temperature = settings.initial_temperature

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def temperature(self, *, step: int, max_steps: int) -> float:
        """Linearly anneal temperature using optimizer-step progress."""
        if max_steps < 1 or step < 0:
            raise ValueError("temperature schedule needs non-negative valid progress")
        fraction = min(step, max_steps) / max_steps
        return (
            self.settings.initial_temperature
            + fraction
            * (self.settings.final_temperature - self.settings.initial_temperature)
        )

    def set_training_progress(self, *, step: int, max_steps: int) -> None:
        self._training_temperature = self.temperature(step=step, max_steps=max_steps)

    def set_inference_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("inference gate temperature must be positive")
        self._training_temperature = float(temperature)

    def gate_outputs(self, *, temperature: float | None = None) -> torch.Tensor:
        value = self._training_temperature if temperature is None else temperature
        if value <= 0:
            raise ValueError("gate temperature must be positive")
        return torch.softmax(self.gate_logits / value, dim=-1)

    def primitive_edge_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return per-sample, per-edge, per-primitive affine outputs."""
        if inputs.ndim != 2 or inputs.shape[1] != self.input_count:
            raise ValueError(f"inputs must have shape (samples, {self.input_count})")
        input_scale = self.affine_parameters[..., 0]
        input_bias = self.affine_parameters[..., 1]
        output_scale = self.affine_parameters[..., 2]
        output_bias = self.affine_parameters[..., 3]
        transformed = inputs[:, :, None] * input_scale + input_bias
        primitives = torch.stack(
            tuple(
                evaluate_primitive(name, transformed[..., index])
                for index, name in enumerate(PRIMITIVE_NAMES)
            ),
            dim=-1,
        )
        return primitives * output_scale + output_bias

    def analytical_edge_outputs(
        self, inputs: torch.Tensor, *, temperature: float | None = None
    ) -> torch.Tensor:
        gates = self.gate_outputs(temperature=temperature)
        return torch.sum(self.primitive_edge_outputs(inputs) * gates, dim=-1)

    def residual_edge_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.include_spline_residual:
            return torch.zeros_like(inputs)
        basis = cubic_bspline_basis(inputs, self.knots)
        spline = torch.einsum("bic,ic->bi", basis, self.coefficients)
        return spline * self.residual_scales

    def edge_outputs(
        self, inputs: torch.Tensor, *, temperature: float | None = None
    ) -> torch.Tensor:
        return self.analytical_edge_outputs(
            inputs, temperature=temperature
        ) + self.residual_edge_outputs(inputs)

    def soft_predictions(
        self, inputs: torch.Tensor, *, temperature: float | None = None
    ) -> torch.Tensor:
        return (
            self.edge_outputs(inputs, temperature=temperature).sum(dim=1, keepdim=True)
            + self.output_bias
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.soft_predictions(inputs)

    def harden(self) -> HardAnalyticalKAN:
        """Deterministically select one primitive per edge; ties use library order."""
        indices = torch.argmax(self.gate_logits.detach(), dim=-1)
        edge_positions = torch.arange(self.input_count, device=indices.device)
        affine = self.affine_parameters.detach()[edge_positions, indices]
        return HardAnalyticalKAN(
            primitive_indices=indices,
            affine_parameters=affine,
            output_bias=self.output_bias,
            source_names=self.source_names,
        )

    def hard_predictions(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.harden().to(inputs.device)(inputs)

    def fidelity(self, inputs: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            return fidelity_metrics(
                self.soft_predictions(inputs), self.hard_predictions(inputs)
            )

    def source_mass(self, inputs: torch.Tensor) -> torch.Tensor:
        """Contribution-energy mass per frozen source, without target access."""
        edge_outputs = self.edge_outputs(inputs)
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
        metadata = self.harden().selected_metadata()[index]
        return {
            **metadata,
            "selected_input_mass": float(masses[index].detach().cpu()),
            "selection_status": "selected",
        }

    def residual_energy(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.residual_edge_outputs(inputs)
        return residual_energy_ratio(residual, self.edge_outputs(inputs))

    def regularization_terms(
        self, inputs: torch.Tensor, *, temperature: float | None = None
    ) -> dict[str, torch.Tensor]:
        gates = self.gate_outputs(temperature=temperature)
        entropy = -torch.sum(gates * torch.log(gates.clamp_min(1e-15)), dim=-1)
        entropy = torch.mean(entropy / math.log(len(PRIMITIVE_NAMES)))
        sparsity = torch.mean(1.0 - torch.max(gates, dim=-1).values)
        residual_outputs = self.residual_edge_outputs(inputs)
        raw_residual_energy = torch.mean(torch.square(residual_outputs))
        residual_ratio = torch.mean(
            residual_energy_ratio(residual_outputs, self.edge_outputs(inputs))
        )
        return {
            "gate_entropy": entropy,
            "gate_sparsity": sparsity,
            "spline_residual_raw_energy": raw_residual_energy,
            "spline_residual_ratio": residual_ratio,
        }

    def training_penalty(
        self, inputs: torch.Tensor, *, step: int, max_steps: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self.set_training_progress(step=step, max_steps=max_steps)
        terms = self.regularization_terms(inputs)
        total = (
            self.settings.entropy_weight * terms["gate_entropy"]
            + self.settings.sparsity_weight * terms["gate_sparsity"]
            + self.settings.spline_residual_energy_weight
            * terms["spline_residual_ratio"]
        )
        accounting = {
            name: float(value.detach().cpu()) for name, value in terms.items()
        }
        accounting["temperature"] = self._training_temperature
        accounting["total"] = float(total.detach().cpu())
        return total, accounting

    def shape_report(self) -> SymbolicShapeReport:
        device = self.output_bias.device
        grid = torch.linspace(-4.0, 4.0, 801, dtype=torch.float64, device=device)
        inputs = grid[:, None].expand(-1, self.input_count)
        hard = self.harden().to(device)
        residual = self.residual_edge_outputs(inputs)
        return SymbolicShapeReport(
            grid=grid,
            soft_edges=self.edge_outputs(inputs),
            hard_edges=hard.edge_outputs(inputs),
            residual_edges=residual if self.include_spline_residual else None,
        )

    def training_metadata(self) -> dict[str, object]:
        return {
            "gate_temperature_schedule": {
                "kind": "linear_by_optimizer_step",
                "initial": self.settings.initial_temperature,
                "final": self.settings.final_temperature,
            },
            "penalty_weights": {
                "gate_entropy": self.settings.entropy_weight,
                "gate_sparsity": self.settings.sparsity_weight,
                "spline_residual_ratio": self.settings.spline_residual_energy_weight,
            },
            "unpenalized_accounting": ["spline_residual_raw_energy"],
        }


class SymbolicKAN(_SymbolicKANBase):
    """E3 pure Symbolic-KAN with train-time primitive gates and no spline path."""

    def __init__(
        self,
        input_count: int = CAPACITY_SPEC.input_count,
        *,
        source_names: Sequence[str] | None = None,
        settings: SymbolicModelSettings = SYMBOLIC_SETTINGS,
    ) -> None:
        super().__init__(
            input_count,
            source_names=source_names,
            include_spline_residual=False,
            settings=settings,
        )


class SymbolicResidualKAN(_SymbolicKANBase):
    """E4 analytical E3 path plus a scaled free cubic B-spline on every edge."""

    def __init__(
        self,
        input_count: int = CAPACITY_SPEC.input_count,
        *,
        source_names: Sequence[str] | None = None,
        settings: SymbolicModelSettings = SYMBOLIC_SETTINGS,
    ) -> None:
        super().__init__(
            input_count,
            source_names=source_names,
            include_spline_residual=True,
            settings=settings,
        )


class MatchedMLP(nn.Module):
    """One-hidden-layer SiLU MLP frozen to the declared E4 parameter count."""

    def __init__(
        self, spec: CapacityModelSpec = CAPACITY_SPEC, *, initialization_seed: int = 0
    ) -> None:
        super().__init__()
        self.spec = spec
        self.hidden_width = spec.c6_hidden_width
        self.initialization_seed = initialization_seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(initialization_seed)
            self.network = nn.Sequential(
                nn.Linear(spec.input_count, self.hidden_width),
                nn.SiLU(),
                nn.Linear(self.hidden_width, spec.output_count),
            )
        self.double()
        if self.parameter_match_relative_gap > 0.10:
            raise ValueError("C6 parameter count is not within 10% of declared E4")

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def parameter_match_relative_gap(self) -> float:
        """Return the absolute relative trainable-parameter gap from E4."""
        return abs(self.parameter_count - self.spec.e4_parameter_count) / self.spec.e4_parameter_count

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict one numerical response per row."""
        return self.network(inputs)
