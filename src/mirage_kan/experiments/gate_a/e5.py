"""Deterministic typed symbolic regression for the sealed Gate A E5 arm."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import FEATURE_NAMES
from .symbolic import PRIMITIVE_NAMES


def _source_window(source: str) -> int | None:
    if source.startswith("Return(Close,"):
        return int(source.removeprefix("Return(Close,").removesuffix(")"))
    if "TsMean(Volume," in source:
        return int(source.split("TsMean(Volume,", 1)[1].split(")", 1)[0])
    return None


def _require_finite(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def evaluate_e5_primitive(name: str, values: np.ndarray) -> np.ndarray:
    """Execute one frozen real-to-real primitive without imputation or clipping."""
    x = _require_finite(values, "primitive input")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if name == "Identity":
            result = x
        elif name == "Abs":
            result = np.abs(x)
        elif name == "Square":
            result = np.square(x)
        elif name == "SignedLog1p":
            result = np.sign(x) * np.log1p(np.abs(x))
        elif name == "Tanh":
            result = np.tanh(x)
        elif name == "Clip(-1,1)":
            result = np.clip(x, -1.0, 1.0)
        elif name == "PositiveHinge(0)":
            result = np.maximum(x, 0.0)
        elif name == "NegativeHinge(0)":
            result = np.maximum(-x, 0.0)
        else:
            raise KeyError(f"unknown frozen E5 primitive: {name}")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"E5 primitive produced non-finite output: {name}")
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, order=True)
class E5Atom:
    """One typed unary expression over a frozen candidate source."""

    source_index: int
    source: str
    primitive: str

    def validate(self) -> None:
        if self.source_index < 0 or self.source_index >= len(FEATURE_NAMES):
            raise ValueError("E5 source index is outside the frozen candidate inputs")
        if self.source != FEATURE_NAMES[self.source_index]:
            raise ValueError("E5 source identity does not match its frozen index")
        if self.primitive not in PRIMITIVE_NAMES:
            raise ValueError("E5 primitive is outside the frozen dictionary")

    def canonical_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "op": "Primitive",
            "name": self.primitive,
            "child": {
                "op": "Input",
                "source_index": self.source_index,
                "source": self.source,
                "value_type": "finite_real",
            },
            "value_type": "finite_real",
        }

    @property
    def ordering_key(self) -> tuple[int, int]:
        self.validate()
        return self.source_index, PRIMITIVE_NAMES.index(self.primitive)


@dataclass(frozen=True)
class E5Structure:
    """A canonical sparse additive set of distinct unary typed atoms."""

    atoms: tuple[E5Atom, ...]

    def canonical_atoms(self) -> tuple[E5Atom, ...]:
        for atom in self.atoms:
            atom.validate()
        return tuple(sorted(set(self.atoms), key=lambda atom: atom.ordering_key))

    def validate(self) -> None:
        atoms = self.canonical_atoms()
        if not 1 <= len(atoms) <= 3:
            raise ValueError("E5 structure must contain one to three distinct terms")
        if self.ast_node_count > 15 or self.ast_depth > 5:
            raise ValueError("E5 structure exceeds the sealed AST ceiling")

    @property
    def ast_node_count(self) -> int:
        return 2 + 4 * len(self.canonical_atoms())

    @property
    def ast_depth(self) -> int:
        return 4

    def canonical_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "op": "SparseAdditiveStructure",
            "terms": [atom.canonical_payload() for atom in self.canonical_atoms()],
            "intercept": "free_float64",
            "term_coefficients": "one_free_float64_each",
            "value_type": "finite_real",
        }

    def canonical_serialization(self) -> str:
        return json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )

    @property
    def identity(self) -> str:
        serialized = self.canonical_serialization().encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True)
class E5ExecutableModel:
    """Torch-free concrete AST with fitted float64 constants."""

    structure: E5Structure
    coefficients: np.ndarray
    intercept: float

    def __post_init__(self) -> None:
        self.structure.validate()
        coefficients = _require_finite(self.coefficients, "E5 coefficients").reshape(-1)
        atoms = self.structure.canonical_atoms()
        if coefficients.shape != (len(atoms),):
            raise ValueError("one E5 coefficient is required per canonical term")
        if not np.isfinite(self.intercept):
            raise ValueError("E5 intercept must be finite")
        immutable_coefficients = coefficients.copy()
        immutable_coefficients.setflags(write=False)
        object.__setattr__(self, "structure", E5Structure(atoms))
        object.__setattr__(self, "coefficients", immutable_coefficients)
        object.__setattr__(self, "intercept", float(self.intercept))

    def _term_outputs(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("E5 features must be a two-dimensional matrix")
        if len(matrix) == 0:
            raise ValueError("E5 execution requires at least one sample")
        maximum_source = max(
            atom.source_index for atom in self.structure.canonical_atoms()
        )
        if matrix.shape[1] <= maximum_source:
            raise ValueError("E5 features do not contain all referenced source columns")
        outputs = []
        for atom in self.structure.canonical_atoms():
            outputs.append(
                evaluate_e5_primitive(
                    atom.primitive, matrix[:, atom.source_index]
                )
            )
        return np.column_stack(outputs)

    def evaluate(self, features: np.ndarray) -> np.ndarray:
        """Execute the concrete AST and reject any non-finite input or output."""
        terms = self._term_outputs(features)
        result = terms @ self.coefficients + self.intercept
        return _require_finite(result, "E5 executable output")

    def canonical_payload(self) -> dict[str, Any]:
        children: list[dict[str, Any]] = [
            {
                "op": "Constant",
                "role": "intercept",
                "value_hex": self.intercept.hex(),
                "value_type": "finite_real",
            }
        ]
        for atom, coefficient in zip(
            self.structure.canonical_atoms(), self.coefficients, strict=True
        ):
            children.append(
                {
                    "op": "Mul",
                    "children": [
                        {
                            "op": "Constant",
                            "role": "term_coefficient",
                            "value_hex": float(coefficient).hex(),
                            "value_type": "finite_real",
                        },
                        atom.canonical_payload(),
                    ],
                    "value_type": "finite_real",
                }
            )
        return {"op": "Add", "children": children, "value_type": "finite_real"}

    def canonical_serialization(self) -> str:
        """Return reconstructable canonical JSON with exact float64 constants."""
        return json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_canonical_serialization(cls, serialized: str) -> "E5ExecutableModel":
        """Reconstruct and validate an independently executable concrete AST."""
        payload = json.loads(serialized)
        if payload.get("op") != "Add" or payload.get("value_type") != "finite_real":
            raise ValueError("E5 concrete AST root must be finite-real Add")
        children = payload.get("children")
        if not isinstance(children, list) or len(children) < 2:
            raise ValueError("E5 concrete Add must contain intercept and terms")
        intercept_node = children[0]
        if (
            intercept_node.get("op") != "Constant"
            or intercept_node.get("role") != "intercept"
        ):
            raise ValueError("E5 concrete AST must begin with its intercept")
        atoms = []
        coefficients = []
        for term in children[1:]:
            if term.get("op") != "Mul" or len(term.get("children", [])) != 2:
                raise ValueError("E5 concrete terms must be binary Mul nodes")
            coefficient_node, primitive_node = term["children"]
            input_node = primitive_node.get("child", {})
            if (
                coefficient_node.get("op") != "Constant"
                or coefficient_node.get("role") != "term_coefficient"
                or primitive_node.get("op") != "Primitive"
                or input_node.get("op") != "Input"
            ):
                raise ValueError("malformed E5 concrete term")
            coefficients.append(float.fromhex(coefficient_node["value_hex"]))
            atoms.append(
                E5Atom(
                    int(input_node["source_index"]),
                    str(input_node["source"]),
                    str(primitive_node["name"]),
                )
            )
        return cls(
            E5Structure(tuple(atoms)),
            np.asarray(coefficients, dtype=np.float64),
            float.fromhex(intercept_node["value_hex"]),
        )

    def complexity(self) -> dict[str, int]:
        serialization = self.canonical_serialization()
        return {
            "ast_node_count": self.structure.ast_node_count,
            "ast_depth": self.structure.ast_depth,
            "free_constants": len(self.coefficients) + 1,
            "serialized_description_length": len(serialization.encode("utf-8")),
        }

    def source_metadata(self) -> list[dict[str, Any]]:
        """Return source and fixed-window identities in canonical term order."""
        return [
            {
                "source_index": atom.source_index,
                "source": atom.source,
                "window": _source_window(atom.source),
            }
            for atom in self.structure.canonical_atoms()
        ]

    def source_mass(self, features: np.ndarray) -> dict[str, Any]:
        """Aggregate centered contribution energy by source with explicit zero state."""
        term_outputs = self._term_outputs(features) * self.coefficients[None, :]
        source_contributions = np.zeros(
            (len(term_outputs), len(FEATURE_NAMES)), dtype=np.float64
        )
        for term_index, atom in enumerate(self.structure.canonical_atoms()):
            source_contributions[:, atom.source_index] += term_outputs[:, term_index]
        centered = source_contributions - source_contributions.mean(
            axis=0, keepdims=True
        )
        source_energies = np.mean(np.square(centered), axis=0)
        total = float(source_energies.sum())
        if total == 0.0:
            masses = np.zeros_like(source_energies)
            status = "zero_contribution_energy"
        else:
            masses = source_energies / total
            status = "defined"
        return {
            "status": status,
            "masses": [
                {
                    "source_index": index,
                    "source": source,
                    "window": _source_window(source),
                    "mass": float(masses[index]),
                }
                for index, source in enumerate(FEATURE_NAMES)
            ],
        }


@dataclass(frozen=True)
class E5SearchSettings:
    """Prospectively frozen E5 search and hard-ceiling settings."""

    max_distinct_valid_evaluations: int = 12_000
    max_depth: int = 5
    max_nodes: int = 15
    two_term_parent_beam_size: int = 32
    complexity_tolerance_nrmse: float = 0.005
    seed: int = 0

    def validate(self) -> None:
        if not 1 <= self.max_distinct_valid_evaluations <= 12_000:
            raise ValueError("E5 valid-AST budget must be in [1, 12000]")
        if not 1 <= self.max_depth <= 5:
            raise ValueError("E5 depth ceiling must be in [1, 5]")
        if not 1 <= self.max_nodes <= 15:
            raise ValueError("E5 node ceiling must be in [1, 15]")
        if self.two_term_parent_beam_size < 1:
            raise ValueError("E5 two-term parent beam must be positive")
        if self.complexity_tolerance_nrmse != 0.005:
            raise ValueError("E5 selection tolerance is frozen at 0.005 NRMSE")


@dataclass(frozen=True)
class E5CandidateEvaluation:
    """One successfully fitted structural candidate."""

    attempt_index: int
    model: E5ExecutableModel
    validation_clean_nrmse: float
    structure_identity: str
    least_squares_rank: int


@dataclass(frozen=True)
class E5SearchResult:
    """Selected E5 executable plus its complete attempted-candidate evidence."""

    selected_model: E5ExecutableModel
    selected_validation_nrmse: float
    candidates: tuple[E5CandidateEvaluation, ...]
    ledger: tuple[dict[str, Any], ...]
    accounting: dict[str, int | float | bool]
    settings: E5SearchSettings
    source_names: tuple[str, ...]
    selected_train_source_mass: dict[str, Any]
    generation_mode: str


def _all_frozen_atoms(source_names: Sequence[str]) -> tuple[E5Atom, ...]:
    return tuple(
        E5Atom(source_index, source, primitive)
        for source_index, source in enumerate(source_names)
        for primitive in PRIMITIVE_NAMES
    )


def _structure_design(
    structure: E5Structure, features: np.ndarray
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"E5 features must have shape (samples, {len(FEATURE_NAMES)})"
        )
    columns = [
        evaluate_e5_primitive(atom.primitive, matrix[:, atom.source_index])
        for atom in structure.canonical_atoms()
    ]
    return np.column_stack(columns)


def _complexity_order(candidate: E5CandidateEvaluation) -> tuple[Any, ...]:
    complexity = candidate.model.complexity()
    return (
        complexity["ast_node_count"],
        complexity["ast_depth"],
        complexity["free_constants"],
        complexity["serialized_description_length"],
        candidate.validation_clean_nrmse,
        candidate.structure_identity,
    )


def _select_candidate(
    candidates: Sequence[E5CandidateEvaluation], tolerance: float
) -> E5CandidateEvaluation:
    if not candidates:
        raise RuntimeError("E5 search produced no valid fitted candidate")
    best_nrmse = min(candidate.validation_clean_nrmse for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.validation_clean_nrmse <= best_nrmse + tolerance
    ]
    return min(eligible, key=_complexity_order)


def search_e5(
    train_features: np.ndarray,
    train_noisy_target: np.ndarray,
    validation_features: np.ndarray,
    validation_clean_truth: np.ndarray,
    *,
    source_names: Sequence[str] = FEATURE_NAMES,
    settings: E5SearchSettings = E5SearchSettings(),
    candidate_proposals: Sequence[E5Structure] | None = None,
) -> E5SearchResult:
    """Fit and validation-select genuine typed ASTs without any test-data seam."""
    settings.validate()
    names = tuple(source_names)
    if names != FEATURE_NAMES:
        raise ValueError("E5 source identities must equal the frozen candidate inputs")
    train_matrix = np.asarray(train_features, dtype=np.float64)
    validation_matrix = np.asarray(validation_features, dtype=np.float64)
    if train_matrix.ndim != 2 or train_matrix.shape[1] != len(names):
        raise ValueError("E5 train features must contain the six frozen sources")
    if validation_matrix.ndim != 2 or validation_matrix.shape[1] != len(names):
        raise ValueError("E5 validation features must contain the six frozen sources")
    train_target = _require_finite(
        train_noisy_target, "E5 train noisy target"
    ).reshape(-1)
    validation_truth = _require_finite(
        validation_clean_truth, "E5 validation clean truth"
    ).reshape(-1)
    if len(train_target) != len(train_matrix) or len(validation_truth) != len(
        validation_matrix
    ):
        raise ValueError("E5 feature and target row counts must agree")
    validation_scale = float(np.std(validation_truth, ddof=0))
    if not np.isfinite(validation_scale) or validation_scale <= 0:
        raise ValueError("E5 validation clean truth must have positive finite scale")

    ledger: list[dict[str, Any]] = []
    candidates: list[E5CandidateEvaluation] = []
    seen: dict[str, int] = {}
    fit_attempts = 0
    successful_fits = 0
    valid_evaluations = 0
    automatic_budget_exhausted = False
    start = time.perf_counter()

    def record(status: str, **values: Any) -> None:
        ledger.append(
            {
                "attempt_index": len(ledger) + 1,
                "status": status,
                **values,
            }
        )

    def attempt(proposal: E5Structure) -> bool:
        nonlocal fit_attempts, successful_fits, valid_evaluations
        attempt_index = len(ledger) + 1
        try:
            normalized = E5Structure(proposal.canonical_atoms())
            normalized.validate()
            if normalized.ast_depth > settings.max_depth:
                raise ValueError("candidate exceeds configured E5 depth ceiling")
            if normalized.ast_node_count > settings.max_nodes:
                raise ValueError("candidate exceeds configured E5 node ceiling")
            identity = normalized.identity
        except (KeyError, TypeError, ValueError) as error:
            record(
                "invalid_ast",
                error=type(error).__name__ + ": " + str(error),
                proposed_terms=[
                    {
                        "source_index": atom.source_index,
                        "source": atom.source,
                        "primitive": atom.primitive,
                    }
                    for atom in proposal.atoms
                ],
            )
            return True
        if identity in seen:
            record(
                "duplicate",
                canonical_structural_identity=identity,
                duplicate_of_attempt=seen[identity],
            )
            return True
        seen[identity] = attempt_index
        if valid_evaluations >= settings.max_distinct_valid_evaluations:
            record(
                "budget_exhausted",
                canonical_structural_identity=identity,
                full_fit_performed=False,
            )
            return False
        try:
            train_terms = _structure_design(normalized, train_matrix)
            _structure_design(normalized, validation_matrix)
        except (FloatingPointError, ValueError) as error:
            record(
                "invalid_execution",
                canonical_structural_identity=identity,
                error=type(error).__name__ + ": " + str(error),
            )
            return True
        design = np.column_stack(
            (train_terms, np.ones(len(train_terms), dtype=np.float64))
        )
        fit_attempts += 1
        try:
            fitted, _, rank, singular_values = np.linalg.lstsq(
                design, train_target, rcond=None
            )
            if not np.isfinite(fitted).all() or not np.isfinite(singular_values).all():
                raise FloatingPointError("least-squares fit produced non-finite values")
            model = E5ExecutableModel(normalized, fitted[:-1], float(fitted[-1]))
            validation_prediction = model.evaluate(validation_matrix)
            nrmse = float(
                np.sqrt(np.mean(np.square(validation_prediction - validation_truth)))
                / validation_scale
            )
            if not np.isfinite(nrmse):
                raise FloatingPointError("validation NRMSE is non-finite")
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            record(
                "fit_failure",
                canonical_structural_identity=identity,
                error=type(error).__name__ + ": " + str(error),
            )
            return True
        successful_fits += 1
        valid_evaluations += 1
        candidate = E5CandidateEvaluation(
            attempt_index=attempt_index,
            model=model,
            validation_clean_nrmse=nrmse,
            structure_identity=identity,
            least_squares_rank=int(rank),
        )
        candidates.append(candidate)
        record(
            "evaluated",
            canonical_structural_identity=identity,
            validation_clean_nrmse=nrmse,
            complexity=model.complexity(),
            fit={
                "target": "train_noisy_target_only",
                "method": "numpy.linalg.lstsq",
                "design_columns": int(design.shape[1]),
                "rank": int(rank),
                "train_rows": int(len(train_matrix)),
                "validation_rows": int(len(validation_matrix)),
                "estimated_fit_flops": int(2 * design.shape[0] * design.shape[1] ** 2),
            },
            fitted_model=model.canonical_payload(),
        )
        return True

    if candidate_proposals is not None:
        for proposal in candidate_proposals:
            attempt(proposal)
    else:
        atoms = _all_frozen_atoms(names)
        first_two_phases = itertools.chain(
            (E5Structure((atom,)) for atom in atoms),
            (
                E5Structure(pair)
                for pair in itertools.combinations(atoms, 2)
            ),
        )
        for proposal in first_two_phases:
            if not attempt(proposal):
                automatic_budget_exhausted = True
                break
        if not automatic_budget_exhausted:
            pair_candidates = [
                candidate
                for candidate in candidates
                if len(candidate.model.structure.canonical_atoms()) == 2
            ]
            parent_beam = sorted(
                pair_candidates,
                key=lambda candidate: (
                    candidate.validation_clean_nrmse,
                    *_complexity_order(candidate),
                ),
            )[: settings.two_term_parent_beam_size]
            for parent in parent_beam:
                parent_atoms = parent.model.structure.canonical_atoms()
                last_key = parent_atoms[-1].ordering_key
                for atom in atoms:
                    if atom.ordering_key <= last_key:
                        continue
                    if not attempt(E5Structure((*parent_atoms, atom))):
                        automatic_budget_exhausted = True
                        break
                if automatic_budget_exhausted:
                    break

    selected = _select_candidate(candidates, settings.complexity_tolerance_nrmse)
    status_counts = {
        status: sum(entry["status"] == status for entry in ledger)
        for status in (
            "evaluated",
            "duplicate",
            "invalid_ast",
            "invalid_execution",
            "fit_failure",
            "budget_exhausted",
        )
    }
    accounting: dict[str, int | float | bool] = {
        "attempted_candidates": len(ledger),
        "distinct_valid_ast_evaluations": valid_evaluations,
        "duplicate_attempts": status_counts["duplicate"],
        "invalid_ast_attempts": status_counts["invalid_ast"],
        "invalid_execution_attempts": status_counts["invalid_execution"],
        "fit_attempts": fit_attempts,
        "successful_fits": successful_fits,
        "fit_failures": status_counts["fit_failure"],
        "budget_exhausted_attempts": status_counts["budget_exhausted"],
        "budget_exhausted": status_counts["budget_exhausted"] > 0,
        "wall_clock_seconds": time.perf_counter() - start,
    }
    return E5SearchResult(
        selected_model=selected.model,
        selected_validation_nrmse=selected.validation_clean_nrmse,
        candidates=tuple(candidates),
        ledger=tuple(ledger),
        accounting=accounting,
        settings=settings,
        source_names=names,
        selected_train_source_mass=selected.model.source_mass(train_matrix),
        generation_mode=(
            "deterministic_frozen_generator"
            if candidate_proposals is None
            else "caller_supplied_audit_proposals"
        ),
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exclusive_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def save_e5_search(
    result: E5SearchResult,
    output_directory: Path | str,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Publish ledger and selected executable before a no-replace final manifest."""
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    ledger_path = output / "ledger.jsonl"
    selected_model_path = output / "selected_model.json"
    ledger_text = "".join(
        json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
        for entry in result.ledger
    )
    _exclusive_text(ledger_path, ledger_text)
    _exclusive_text(
        selected_model_path,
        result.selected_model.canonical_serialization() + "\n",
    )
    spec_path = Path("configs/model_specs/s1_gate_a_e5_v0.json")
    if not spec_path.is_file():
        raise FileNotFoundError("prospective E5 model spec is missing")
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": 1,
        "arm": "E5",
        "implementation": "deterministic_sparse_additive_typed_symbolic_regression",
        "numpy_version": np.__version__,
        "model_spec": {
            "path": str(spec_path),
            "resolved_path": str(spec_path.resolve()),
            "sha256": _sha256_path(spec_path),
        },
        "settings": asdict(result.settings),
        "generation_mode": result.generation_mode,
        "source_names": list(result.source_names),
        "selection": {
            "dataset": "validation",
            "target": "clean_truth",
            "metric": "nrmse",
            "within_0.005_tiebreak": "lower_executable_complexity",
            "selected_value": result.selected_validation_nrmse,
            "canonical_structural_identity": result.selected_model.structure.identity,
        },
        "selected": {
            "canonical": result.selected_model.canonical_payload(),
            "complexity": result.selected_model.complexity(),
            "source_metadata": result.selected_model.source_metadata(),
            "source_mass": result.selected_train_source_mass,
        },
        "accounting": result.accounting,
        "metadata": metadata or {},
        "paths": {
            "ledger": str(ledger_path),
            "selected_model": str(selected_model_path),
            "manifest": str(manifest_path),
        },
        "sha256": {
            "ledger": _sha256_path(ledger_path),
            "selected_model": _sha256_path(selected_model_path),
        },
    }
    temporary = output / "manifest.json.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)
    directory_descriptor = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return manifest_path


def load_e5_export(manifest_path: Path | str) -> E5ExecutableModel:
    """Verify hashes and reconstruct a published Torch-free E5 executable."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("arm") != "E5" or manifest.get("schema_version") != 1:
        raise ValueError("unsupported E5 export manifest")
    model_spec_path = Path(
        manifest["model_spec"].get(
            "resolved_path", manifest["model_spec"]["path"]
        )
    )
    if not model_spec_path.is_file():
        raise FileNotFoundError("E5 export model spec is unavailable")
    if _sha256_path(model_spec_path) != manifest["model_spec"]["sha256"]:
        raise ValueError("E5 export model-spec hash mismatch")
    for name in ("ledger", "selected_model"):
        path = Path(manifest["paths"][name])
        if _sha256_path(path) != manifest["sha256"][name]:
            raise ValueError(f"E5 export hash mismatch: {name}")
    model = E5ExecutableModel.from_canonical_serialization(
        Path(manifest["paths"]["selected_model"])
        .read_text(encoding="utf-8")
        .strip()
    )
    if model.structure.identity != manifest["selection"][
        "canonical_structural_identity"
    ]:
        raise ValueError("E5 selected structural identity mismatch")
    if model.canonical_payload() != manifest["selected"]["canonical"]:
        raise ValueError("E5 selected canonical payload mismatch")
    return model
