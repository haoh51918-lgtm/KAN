"""Cross-seed residual eligibility, governed promotion, and hard refitting."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import least_squares
from torch import nn

from .data import FEATURE_NAMES
from .symbolic import HardAnalyticalKAN, PRIMITIVE_NAMES, evaluate_primitive

PROMOTION_SPEC_PATH = Path("configs/model_specs/s1_gate_a_promotion_v0.json")
FAMILY_ORDER = (
    "asymmetric_exponential_saturation_v1",
    "asymmetric_rational_saturation_v1",
    "two_sided_tanh_v1",
)
REQUIRED_SOURCE = "Return(Close,5)"


def _spec() -> dict[str, Any]:
    value = json.loads(PROMOTION_SPEC_PATH.read_text(encoding="utf-8"))
    if tuple(item["id"] for item in value["families"]) != FAMILY_ORDER:
        raise ValueError("promotion family order differs from prospective spec")
    return value


def _family_spec(family_id: str) -> dict[str, Any]:
    for value in _spec()["families"]:
        if value["id"] == family_id:
            return value
    raise KeyError(f"unknown promotion family: {family_id}")


def evaluate_promotion_family(
    family_id: str, x: np.ndarray | Sequence[float], parameters: np.ndarray | Sequence[float]
) -> np.ndarray:
    """Execute one of the three frozen continuous monotone family equations."""
    values = np.asarray(x, dtype=np.float64)
    params = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 1 or params.shape != (4,):
        raise ValueError("promotion evaluation needs one-dimensional x and four constants")
    if not np.isfinite(values).all() or not np.isfinite(params).all():
        raise ValueError("promotion family inputs and constants must be finite")
    a_neg, k_neg, a_pos, k_pos = params
    if family_id == "asymmetric_exponential_saturation_v1":
        result = np.empty_like(values)
        negative = values < 0
        result[negative] = -a_neg * (1.0 - np.exp(k_neg * values[negative]))
        result[~negative] = a_pos * (
            1.0 - np.exp(-k_pos * values[~negative])
        )
    elif family_id == "asymmetric_rational_saturation_v1":
        result = np.empty_like(values)
        negative = values < 0
        negative_x = -values[negative]
        result[negative] = -a_neg * negative_x / (1.0 + k_neg * negative_x)
        positive_x = values[~negative]
        result[~negative] = a_pos * positive_x / (1.0 + k_pos * positive_x)
    elif family_id == "two_sided_tanh_v1":
        result = np.empty_like(values)
        negative = values < 0
        result[negative] = -a_neg * np.tanh(k_neg * (-values[negative]))
        result[~negative] = a_pos * np.tanh(k_pos * values[~negative])
    else:
        raise KeyError(f"unknown promotion family: {family_id}")
    if not np.isfinite(result).all():
        raise FloatingPointError("promotion family produced non-finite values")
    return result


def _primitive_identity(family_id: str) -> str:
    spec = _spec()
    family = _family_spec(family_id)
    payload = {
        "family": family,
        "governance": spec["governance"],
        "schema_version": spec["schema_version"],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"{family_id}_{digest}"


@dataclass(frozen=True)
class PromotedPrimitive:
    """One executable generic family with fitted continuous constants."""

    family_id: str
    parameters: np.ndarray
    primitive_id: str

    @classmethod
    def create(
        cls, family_id: str, parameters: np.ndarray | Sequence[float]
    ) -> "PromotedPrimitive":
        family = _family_spec(family_id)
        params = np.asarray(parameters, dtype=np.float64).copy()
        bounds = np.asarray(family["bounds"], dtype=np.float64)
        if params.shape != (4,) or not np.isfinite(params).all():
            raise ValueError("promoted primitive needs four finite constants")
        if np.any(params < bounds[:, 0]) or np.any(params > bounds[:, 1]):
            raise ValueError("promoted primitive constants violate frozen bounds")
        params.setflags(write=False)
        return cls(family_id, params, _primitive_identity(family_id))

    def evaluate(self, x: np.ndarray | Sequence[float]) -> np.ndarray:
        return evaluate_promotion_family(self.family_id, x, self.parameters)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "parameters_float64_hex": [float(value).hex() for value in self.parameters],
            "primitive_id": self.primitive_id,
            "spec_sha256": hashlib.sha256(PROMOTION_SPEC_PATH.read_bytes()).hexdigest(),
        }

    def canonical_serialization(self) -> str:
        return json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_canonical(cls, value: str) -> "PromotedPrimitive":
        payload = json.loads(value)
        if payload["spec_sha256"] != hashlib.sha256(PROMOTION_SPEC_PATH.read_bytes()).hexdigest():
            raise ValueError("promoted primitive spec hash mismatch")
        primitive = cls.create(
            payload["family_id"],
            [float.fromhex(item) for item in payload["parameters_float64_hex"]],
        )
        if primitive.primitive_id != payload["primitive_id"]:
            raise ValueError("promoted primitive ID mismatch")
        return primitive


@dataclass(frozen=True)
class ResidualShape:
    seed: int
    source: str | None
    centered_contribution_mass: float
    standardized_z: np.ndarray
    residual_values: np.ndarray

    def __post_init__(self) -> None:
        z = np.asarray(self.standardized_z, dtype=np.float64).copy()
        values = np.asarray(self.residual_values, dtype=np.float64).copy()
        if z.shape != (801,) or values.shape != z.shape:
            raise ValueError("residual shape must use the frozen 801-point grid")
        frozen_grid = np.linspace(-4.0, 4.0, 801)
        if not np.array_equal(z, frozen_grid) or not np.isfinite(values).all():
            raise ValueError(
                "residual shape must use the exact frozen 801-point grid and finite values"
            )
        z.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "standardized_z", z)
        object.__setattr__(self, "residual_values", values)


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    eligible_seeds: tuple[int, ...]
    correlations: dict[str, float]
    exact_source_seeds: tuple[int, ...]
    sign_flip_allowed: bool = False


def _aligned(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    scale = float(np.std(centered, ddof=0))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("residual shape alignment needs positive finite variation")
    return centered / scale


def assess_residual_eligibility(
    shapes: Sequence[ResidualShape], *, minimum_correlation: float = 0.95
) -> EligibilityResult:
    """Find the largest sign-preserving pairwise-complete exact-source subset."""
    if len({shape.seed for shape in shapes}) != len(shapes):
        raise ValueError("residual eligibility needs unique seeds")
    frozen_grid = np.linspace(-4.0, 4.0, 801)
    if any(not np.array_equal(shape.standardized_z, frozen_grid) for shape in shapes):
        raise ValueError("eligibility requires the identical frozen standardized-z grid")
    exact = tuple(sorted(shape.seed for shape in shapes if shape.source == REQUIRED_SOURCE))
    by_seed = {shape.seed: shape for shape in shapes}
    correlations: dict[str, float] = {}
    for left, right in itertools.combinations(sorted(by_seed), 2):
        try:
            rho = float(
                np.corrcoef(
                    _aligned(by_seed[left].residual_values),
                    _aligned(by_seed[right].residual_values),
                )[0, 1]
            )
        except ValueError:
            rho = -1.0
        correlations[f"{left}:{right}"] = rho
    selected: tuple[int, ...] = ()
    for size in range(len(exact), 1, -1):
        candidates = []
        for subset in itertools.combinations(exact, size):
            if all(
                correlations[f"{min(a, b)}:{max(a, b)}"] >= minimum_correlation
                for a, b in itertools.combinations(subset, 2)
            ):
                candidates.append(subset)
        if candidates:
            selected = min(candidates)
            break
    return EligibilityResult(bool(selected), selected, correlations, exact, False)


def extract_residual_shape(
    model: nn.Module,
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    *,
    seed: int,
    source_names: Sequence[str] = FEATURE_NAMES,
) -> ResidualShape:
    """Freeze the selected E4 residual from train/validation features only."""
    combined = torch.cat((train_features, validation_features), dim=0)
    with torch.no_grad():
        masses = model.source_mass(combined)
        if not bool(torch.any(masses > 0)):
            source_index = None
            source = None
            mass = 0.0
            residual = torch.zeros(801, dtype=torch.float64, device=combined.device)
        else:
            source_index = int(torch.argmax(masses).cpu())
            source = tuple(source_names)[source_index]
            mass = float(masses[source_index].cpu())
            z = torch.linspace(-4.0, 4.0, 801, dtype=torch.float64, device=combined.device)
            grid_inputs = torch.zeros(
                801, combined.shape[1], dtype=torch.float64, device=combined.device
            )
            grid_inputs[:, source_index] = z
            residual = model.residual_edge_outputs(grid_inputs)[:, source_index]
    return ResidualShape(
        int(seed),
        source,
        mass,
        np.linspace(-4.0, 4.0, 801),
        residual.detach().cpu().numpy(),
    )


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_residual_shape(shape: ResidualShape, output_directory: Path | str) -> Path:
    """Persist a frozen residual array before any family fitting, manifest last."""
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    array_path = output / "residual_shape.npz"
    with array_path.open("xb") as stream:
        np.savez(
            stream,
            standardized_z=shape.standardized_z,
            residual_values=shape.residual_values,
        )
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {
        "schema_version": 1,
        "seed": shape.seed,
        "source": shape.source,
        "centered_contribution_mass": shape.centered_contribution_mass,
        "selection_data": "train_plus_validation_features_only",
        "test_access": False,
        "array_path": str(array_path),
        "array_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
    }
    manifest_path = output / "manifest.json"
    _exclusive_json(manifest_path, manifest)
    return manifest_path


def _fit_family(shape: ResidualShape, family_id: str) -> dict[str, Any]:
    family = _family_spec(family_id)
    bounds = np.asarray(family["bounds"], dtype=np.float64)
    best = None
    for amplitude, rate in itertools.product((0.25, 1.0), repeat=2):
        fitted = least_squares(
            lambda parameters: evaluate_promotion_family(
                family_id, shape.standardized_z, parameters
            )
            - shape.residual_values,
            x0=np.array([amplitude, rate, amplitude, rate]),
            bounds=(bounds[:, 0], bounds[:, 1]),
            method="trf",
            loss="linear",
            max_nfev=2000,
        )
        prediction = evaluate_promotion_family(
            family_id, shape.standardized_z, fitted.x
        )
        denominator = float(np.std(shape.residual_values, ddof=0))
        nrmse = (
            float(np.sqrt(np.mean(np.square(prediction - shape.residual_values))))
            / denominator
            if denominator > 0
            else float("inf")
        )
        candidate = (nrmse, int(fitted.nfev), fitted)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return {
        "parameters": [float(value) for value in best[2].x],
        "nrmse": best[0],
        "function_evaluations": best[1],
        "continuous_optimization": True,
        "success": bool(best[2].success),
    }


def _duplicate_audit(
    family_id: str,
    seed_fits: dict[int, dict[str, Any]],
    frozen_shapes: dict[int, ResidualShape],
) -> dict[str, Any]:
    governance = _spec()["governance"]
    z = np.linspace(*governance["audit_grid"])
    settings = governance["non_duplication_fit"]
    lower = np.asarray(settings["lower_bounds"], dtype=np.float64)
    upper = np.asarray(settings["upper_bounds"], dtype=np.float64)
    starts = tuple(np.asarray(start, dtype=np.float64) for start in settings["starts"])
    best_by_curve: dict[str, dict[str, Any]] = {}
    total_function_evaluations = 0
    estimated_flops = 0
    for seed, fit in seed_fits.items():
        curves = (
            (
                "fitted_candidate_family",
                z,
                evaluate_promotion_family(family_id, z, fit["parameters"]),
            ),
            (
                "frozen_residual_shape",
                frozen_shapes[seed].standardized_z,
                frozen_shapes[seed].residual_values,
            ),
        )
        for curve_kind, curve_grid, curve in curves:
            scale = float(np.std(curve, ddof=0))
            for primitive_order, name in enumerate(PRIMITIVE_NAMES):
                for start_order, start in enumerate(starts):
                    def residual(parameters: np.ndarray) -> np.ndarray:
                        input_scale, input_bias, output_scale, output_bias = parameters
                        primitive = evaluate_primitive(
                            name,
                            torch.as_tensor(input_scale * curve_grid + input_bias),
                        ).numpy()
                        return output_scale * primitive + output_bias - curve

                    fitted = least_squares(
                        residual,
                        x0=start,
                        bounds=(lower, upper),
                        method=settings["method"],
                        loss=settings["loss"],
                        max_nfev=int(settings["max_nfev"]),
                    )
                    total_function_evaluations += int(fitted.nfev)
                    estimated_flops += int(fitted.nfev) * len(curve_grid) * 16
                    approximation = curve + residual(fitted.x)
                    nrmse = float(
                        np.sqrt(np.mean(np.square(approximation - curve))) / scale
                    )
                    centered_curve = curve - np.mean(curve)
                    centered_approximation = approximation - np.mean(approximation)
                    correlation_scale = float(
                        np.sqrt(
                            np.sum(np.square(centered_curve))
                            * np.sum(np.square(centered_approximation))
                        )
                    )
                    rho = (
                        float(
                            abs(
                                np.sum(centered_curve * centered_approximation)
                                / correlation_scale
                            )
                        )
                        if correlation_scale > 0
                        else 0.0
                    )
                    candidate = {
                        "absolute_pearson": rho,
                        "affine_fit_nrmse": nrmse,
                        "primitive": name,
                        "audited_curve": curve_kind,
                        "seed": int(seed),
                        "parameters": [float(value) for value in fitted.x],
                        "function_evaluations": int(fitted.nfev),
                        "success": bool(fitted.success),
                        "primitive_order": primitive_order,
                        "start_order": start_order,
                    }
                    key = (
                        nrmse,
                        -rho,
                        primitive_order,
                        start_order,
                        seed,
                        curve_kind,
                    )
                    current = best_by_curve.get(curve_kind)
                    if current is None or key < current["_key"]:
                        best_by_curve[curve_kind] = {**candidate, "_key": key}
    candidate_best = best_by_curve["fitted_candidate_family"]
    residual_best = best_by_curve["frozen_residual_shape"]
    candidate_best.pop("_key")
    residual_best.pop("_key")
    candidate_duplicate = (
        candidate_best["absolute_pearson"]
        >= governance["duplicate_pearson_absolute_minimum"]
        and candidate_best["affine_fit_nrmse"]
        <= governance["duplicate_affine_fit_nrmse_maximum"]
    )
    residual_already_in_dictionary = (
        residual_best["absolute_pearson"]
        >= governance["existing_dictionary_residual_pearson_minimum"]
        and residual_best["affine_fit_nrmse"]
        <= governance["existing_dictionary_residual_nrmse_maximum"]
    )
    return {
        **candidate_best,
        "candidate_fit": candidate_best,
        "frozen_residual_fit": residual_best,
        "candidate_duplicate": candidate_duplicate,
        "residual_already_in_dictionary": residual_already_in_dictionary,
        "duplicate": candidate_duplicate or residual_already_in_dictionary,
        "fit_semantics": settings["semantics"],
        "total_function_evaluations": total_function_evaluations,
        "estimated_flops": estimated_flops,
    }


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    status: str
    family_id: str | None
    primitive_id: str | None
    seed_fits: dict[int, dict[str, Any]]
    governance: dict[str, Any]
    candidate_audits: tuple[dict[str, Any], ...]


def fit_governed_promotion(
    frozen_residual_shapes: Sequence[ResidualShape],
    eligibility: EligibilityResult,
) -> PromotionResult:
    """Fit frozen generic families; this API has no target or test-data argument."""
    if not eligibility.eligible:
        return PromotionResult(
            False, "ineligible_residual_shapes", None, None, {},
            {"all_passed": False}, (),
        )
    shapes = {shape.seed: shape for shape in frozen_residual_shapes}
    spec = _spec()
    governance = spec["governance"]
    audits = []
    for order, family_id in enumerate(FAMILY_ORDER):
        fits = {seed: _fit_family(shapes[seed], family_id) for seed in eligibility.eligible_seeds}
        nrmse_values = [fit["nrmse"] for fit in fits.values()]
        audit_grid = np.linspace(*governance["audit_grid"])
        domain = True
        continuity = True
        monotonic = True
        stability = True
        reconstructable = True
        for fit in fits.values():
            primitive = PromotedPrimitive.create(family_id, fit["parameters"])
            values = primitive.evaluate(audit_grid)
            domain &= bool(np.isfinite(values).all())
            continuity &= bool(
                abs(np.ptp(primitive.evaluate(np.array([-1e-12, 1e-12]))))
                <= governance["boundary_continuity_absolute_tolerance"]
            )
            differences = np.diff(values)
            monotonic &= bool(np.min(differences) >= -governance["monotonic_first_difference_tolerance"])
            stability &= bool(np.max(np.abs(values)) <= governance["maximum_absolute_output"])
            slopes = differences / np.diff(audit_grid)
            stability &= bool(np.max(np.abs(slopes)) <= governance["maximum_absolute_numerical_slope"])
            restored = PromotedPrimitive.from_canonical(primitive.canonical_serialization())
            reconstructable &= bool(np.array_equal(restored.evaluate(audit_grid), values))
        duplicate = _duplicate_audit(family_id, fits, shapes)
        low_complexity = bool(
            np.median(nrmse_values)
            <= governance["median_eligible_seed_residual_shape_nrmse_maximum"]
            and max(nrmse_values)
            <= governance["each_eligible_seed_residual_shape_nrmse_maximum"]
        )
        checks = {
            "domain": domain,
            "boundary_continuity": continuity,
            "monotonicity": monotonic,
            "stability": stability,
            "non_duplication": not duplicate["duplicate"],
            "reconstructability": reconstructable,
            "low_complexity_approximation": low_complexity,
        }
        family_spec = _family_spec(family_id)
        description_length = (
            8 * family_spec["base_ast_nodes"]
            + 32 * 4
            + len(_primitive_identity(family_id).encode("utf-8")) * 8
        )
        audits.append(
            {
                "family_id": family_id,
                "family_order": order,
                "seed_fits": fits,
                "median_seed_residual_shape_nrmse": float(np.median(nrmse_values)),
                "maximum_seed_residual_shape_nrmse": float(max(nrmse_values)),
                "description_length_bits": description_length,
                "duplicate_audit": duplicate,
                "checks": checks,
                "all_passed": all(checks.values()),
            }
        )
    eligible_candidates = [audit for audit in audits if audit["all_passed"]]
    if not eligible_candidates:
        return PromotionResult(
            False, "no_governed_candidate", None, None, {},
            {"all_passed": False}, tuple(audits),
        )
    selected = min(
        eligible_candidates,
        key=lambda audit: (
            audit["median_seed_residual_shape_nrmse"],
            audit["description_length_bits"],
            audit["family_order"],
        ),
    )
    return PromotionResult(
        True,
        "promoted",
        selected["family_id"],
        _primitive_identity(selected["family_id"]),
        selected["seed_fits"],
        {"all_passed": True, "selected_audit": selected},
        tuple(audits),
    )


class PromotedHardModel(nn.Module):
    """Independent fixed circuit after continuous-only output affine refitting."""

    def __init__(
        self,
        analytical: HardAnalyticalKAN,
        promoted: PromotedPrimitive,
        source_index: int,
        output_scales: np.ndarray | Sequence[float],
        intercept: float,
    ) -> None:
        super().__init__()
        if not 0 <= source_index < analytical.input_count:
            raise ValueError("promoted source index is outside the hard model")
        scales = torch.as_tensor(output_scales, dtype=torch.float64).reshape(-1)
        if scales.numel() != analytical.input_count + 1:
            raise ValueError("output refit needs one scale per edge plus promoted residual")
        self.source_names = analytical.source_names
        self.source_index = int(source_index)
        self.promoted = promoted
        self.register_buffer("primitive_indices", analytical.primitive_indices.clone())
        self.register_buffer(
            "analytical_affine_parameters", analytical.affine_parameters.clone()
        )
        self.register_buffer("output_scales", scales.clone())
        self.register_buffer("intercept", torch.tensor(float(intercept), dtype=torch.float64))

    def _analytical(self) -> HardAnalyticalKAN:
        return HardAnalyticalKAN(
            primitive_indices=self.primitive_indices,
            affine_parameters=self.analytical_affine_parameters,
            output_bias=torch.zeros(1, dtype=torch.float64, device=self.primitive_indices.device),
            source_names=self.source_names,
        ).to(self.primitive_indices.device)

    def _promoted_output(self, inputs: torch.Tensor) -> torch.Tensor:
        params = torch.tensor(
            self.promoted.parameters.copy(), dtype=torch.float64, device=inputs.device
        )
        x = inputs[:, self.source_index]
        a_neg, k_neg, a_pos, k_pos = params
        if self.promoted.family_id == "asymmetric_exponential_saturation_v1":
            negative_x = torch.clamp(x, max=0.0)
            positive_x = torch.clamp(x, min=0.0)
            residual = torch.where(
                x < 0,
                -a_neg * (1 - torch.exp(k_neg * negative_x)),
                a_pos * (1 - torch.exp(-k_pos * positive_x)),
            )
        elif self.promoted.family_id == "asymmetric_rational_saturation_v1":
            negative_x = torch.clamp(-x, min=0.0)
            positive_x = torch.clamp(x, min=0.0)
            residual = torch.where(
                x < 0,
                -a_neg * negative_x / (1 + k_neg * negative_x),
                a_pos * positive_x / (1 + k_pos * positive_x),
            )
        else:
            negative_x = torch.clamp(-x, min=0.0)
            positive_x = torch.clamp(x, min=0.0)
            residual = torch.where(
                x < 0,
                -a_neg * torch.tanh(k_neg * negative_x),
                a_pos * torch.tanh(k_pos * positive_x),
            )
        return residual

    def source_contributions(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return centered-mass-compatible contributions grouped by input source."""
        contributions = (
            self._analytical().edge_outputs(inputs) * self.output_scales[:-1]
        )
        contributions = contributions.clone()
        contributions[:, self.source_index] += (
            self.output_scales[-1] * self._promoted_output(inputs)
        )
        return contributions

    def selected_source_metadata(self, inputs: torch.Tensor) -> dict[str, Any]:
        """Report the promoted source and its centered contribution mass."""
        contributions = self.source_contributions(inputs)
        centered = contributions - torch.mean(contributions, dim=0, keepdim=True)
        energies = torch.mean(torch.square(centered), dim=0)
        total = torch.sum(energies)
        if not bool(total > 0):
            return {
                "source": None,
                "window": None,
                "selected_input_mass": 0.0,
                "selection_status": "zero_contribution_energy",
            }
        masses = energies / total
        index = self.source_index
        source = self.source_names[index]
        window = (
            int(source.removeprefix("Return(Close,").removesuffix(")"))
            if source.startswith("Return(Close,")
            else (20 if "TsMean(Volume,20)" in source else None)
        )
        return {
            "source": source,
            "window": window,
            "selected_input_mass": float(masses[index].detach().cpu()),
            "selection_status": "promoted_source",
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        edges = self._analytical().edge_outputs(inputs)
        residual = self._promoted_output(inputs)
        prediction = (
            edges * self.output_scales[:-1]
        ).sum(dim=1) + self.output_scales[-1] * residual + self.intercept
        return prediction[:, None]

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self(torch.as_tensor(features, dtype=torch.float64)).numpy().reshape(-1)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "type": "PromotedHardModel",
            "source_names": list(self.source_names),
            "source_index": self.source_index,
            "primitive_indices": self.primitive_indices.tolist(),
            "analytical_affine_float64_hex": [
                [float(value).hex() for value in row]
                for row in self.analytical_affine_parameters.tolist()
            ],
            "promoted": self.promoted.canonical_payload(),
            "output_scales_float64_hex": [float(value).hex() for value in self.output_scales],
            "intercept_float64_hex": float(self.intercept).hex(),
        }

    def canonical_serialization(self) -> str:
        return json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))

    def complexity(self) -> dict[str, int]:
        serialized = self.canonical_serialization()
        return {
            "ast_node_count": 2 + len(self.source_names) * 4 + _family_spec(self.promoted.family_id)["base_ast_nodes"],
            "ast_depth": 5,
            "free_constants": int(self.analytical_affine_parameters.numel() + self.output_scales.numel() + 5),
            "serialized_description_length": len(serialized.encode("utf-8")),
        }


def refit_promoted_hard_model(
    hard_model: HardAnalyticalKAN,
    promoted_primitive: PromotedPrimitive,
    *,
    source_index: int,
    train_features: np.ndarray,
    train_noisy_target: np.ndarray,
    validation_features: np.ndarray,
    validation_clean_truth: np.ndarray,
) -> tuple[PromotedHardModel, dict[str, Any]]:
    """Refit fixed circuit output scales/intercept with no test or discrete search."""
    train_x = np.asarray(train_features, dtype=np.float64)
    validation_x = np.asarray(validation_features, dtype=np.float64)
    x = np.concatenate((train_x, validation_x), axis=0)
    y = np.concatenate(
        (
            np.asarray(train_noisy_target, dtype=np.float64).reshape(-1),
            np.asarray(validation_clean_truth, dtype=np.float64).reshape(-1),
        )
    )
    if x.shape != (len(y), hard_model.input_count) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("hard refit requires finite aligned train/validation arrays")
    with torch.no_grad():
        edges = hard_model.edge_outputs(torch.as_tensor(x)).numpy()
    residual = promoted_primitive.evaluate(x[:, source_index])
    design = np.column_stack((edges, residual, np.ones(len(x))))
    fitted, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    model = PromotedHardModel(
        hard_model,
        promoted_primitive,
        source_index,
        fitted[:-1],
        float(fitted[-1]),
    )
    return model, {
        "fit": "continuous_output_affine_scales_and_intercept_only",
        "rows": {"train_noisy": len(train_x), "validation_clean": len(validation_x)},
        "design_columns": int(design.shape[1]),
        "least_squares_rank": int(rank),
        "discrete_reselection": False,
        "analytical_input_affines_changed": False,
        "all_free_spline_paths_closed_by_hardening": True,
        "test_access": False,
    }


def _model_from_payload(payload: dict[str, Any]) -> PromotedHardModel:
    analytical = HardAnalyticalKAN(
        primitive_indices=torch.tensor(payload["primitive_indices"], dtype=torch.long),
        affine_parameters=torch.tensor(
            [[float.fromhex(value) for value in row] for row in payload["analytical_affine_float64_hex"]],
            dtype=torch.float64,
        ),
        output_bias=torch.zeros(1, dtype=torch.float64),
        source_names=payload["source_names"],
    )
    promoted = PromotedPrimitive.from_canonical(
        json.dumps(payload["promoted"], sort_keys=True, separators=(",", ":"))
    )
    return PromotedHardModel(
        analytical,
        promoted,
        int(payload["source_index"]),
        [float.fromhex(value) for value in payload["output_scales_float64_hex"]],
        float.fromhex(payload["intercept_float64_hex"]),
    )


def save_promoted_hard_model(
    model: PromotedHardModel,
    output_directory: Path | str,
    *,
    evidence: dict[str, Any],
) -> Path:
    """Publish canonical executable and evidence with a final exclusive manifest."""
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    model_path = output / "promoted_hard_model.json"
    with model_path.open("x", encoding="utf-8") as stream:
        stream.write(model.canonical_serialization() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    evidence_path = output / "refit_evidence.json"
    _exclusive_json(evidence_path, evidence)
    manifest_path = output / "manifest.json"
    _exclusive_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "executable_promoted_hard_model",
            "primitive_id": model.promoted.primitive_id,
            "complexity": model.complexity(),
            "paths": {"model": str(model_path), "evidence": str(evidence_path)},
            "sha256": {
                "model": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "evidence": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            },
            "publication": {"no_replace": True, "manifest_last": True},
        },
    )
    return manifest_path


def save_no_promotion_status(
    output_directory: Path | str, result: PromotionResult
) -> Path:
    """Persist an explicit semi-symbolic state when governance cannot promote."""
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    manifest = output / "manifest.json"
    family_fit_flops = int(
        sum(
            fit["function_evaluations"]
            for audit in result.candidate_audits
            for fit in audit.get("seed_fits", {}).values()
        )
        * 801
        * 8
    )
    nondup_audit_flops = int(
        sum(
            audit.get("duplicate_audit", {}).get("estimated_flops", 0)
            for audit in result.candidate_audits
        )
    )
    _exclusive_json(
        manifest,
        {
            "schema_version": 1,
            "status": "semi_symbolic_no_promotion",
            "reason": result.status,
            "gate_a_executable_promotion": False,
            "candidate_audits": list(result.candidate_audits),
            "accounting": {
                "family_fit_flops": family_fit_flops,
                "nondup_audit_flops": nondup_audit_flops,
                "total_estimated_flops": family_fit_flops + nondup_audit_flops,
            },
            "publication": {"no_replace": True, "manifest_last": True},
        },
    )
    return manifest


def load_promoted_hard_model(manifest_path: Path | str) -> PromotedHardModel:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    for name in ("model", "evidence"):
        path = Path(manifest["paths"][name])
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"][name]:
            raise ValueError(f"promoted hard export hash mismatch: {name}")
    payload = json.loads(Path(manifest["paths"]["model"]).read_text(encoding="utf-8"))
    return _model_from_payload(payload)
