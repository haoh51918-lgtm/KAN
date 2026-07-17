"""Identity-checked adapter to QuantaAlpha's real precomputed-factor path."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from mirage_kan.data.pit import sha256_file
from mirage_kan.identities import regular_file_tree_stat_identity


class QuantaIdentityError(RuntimeError):
    """Raised before import when pinned Quanta source identities do not match."""


_BACKTEST_PATCH_LOCK = threading.Lock()


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class QuantaAdapter:
    """Narrow adapter that verifies and invokes the pinned BacktestRunner internals."""

    def __init__(
        self,
        repository: Path | str,
        *,
        expected_commit: str,
        expected_config_sha256: str,
        expected_runner_sha256: str,
        expected_provider_identity: dict[str, object] | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        repo = Path(repository).resolve(strict=True)
        config_path = repo / "configs" / "backtest.yaml"
        runner_path = repo / "quantaalpha" / "backtest" / "runner.py"
        observed_config = sha256_file(config_path)
        observed_runner = sha256_file(runner_path)
        if observed_config != expected_config_sha256:
            raise QuantaIdentityError(
                f"Quanta config SHA-256 mismatch: {observed_config}"
            )
        if observed_runner != expected_runner_sha256:
            raise QuantaIdentityError(
                f"Quanta runner SHA-256 mismatch: {observed_runner}"
            )
        observed_commit = _git_output(repo, "rev-parse", "HEAD")
        if observed_commit != expected_commit:
            raise QuantaIdentityError(
                f"Quanta commit mismatch: expected {expected_commit}, got {observed_commit}"
            )

        dirty = _git_output(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "quantaalpha",
            "configs/backtest.yaml",
        )
        if dirty:
            raise QuantaIdentityError(
                "Quanta execution closure is dirty or has untracked package files"
            )
        tracked_tree = _git_output(repo, "rev-parse", "HEAD:quantaalpha")

        existing = sys.modules.get("quantaalpha.backtest.runner")
        if existing is not None:
            existing_path = Path(existing.__file__).resolve(strict=True)
            if existing_path != runner_path:
                raise QuantaIdentityError(
                    f"quantaalpha package was already imported from {existing_path}"
                )
            module = existing
        else:
            original_sys_path = list(sys.path)
            sys.path.insert(0, str(repo))
            try:
                module = importlib.import_module("quantaalpha.backtest.runner")
            finally:
                sys.path[:] = original_sys_path
        imported_path = Path(module.__file__).resolve(strict=True)
        if imported_path != runner_path:
            raise QuantaIdentityError(
                f"pinned runner import resolved to unexpected path: {imported_path}"
            )
        self.runner = module.BacktestRunner(str(config_path))
        self._expected_provider_identity = expected_provider_identity
        if output_dir is not None:
            destination = Path(output_dir).resolve()
            destination.mkdir(parents=True, exist_ok=True)
            self.runner.config["experiment"]["output_dir"] = str(destination)
        self.identity = {
            "verified": True,
            "repository": str(repo),
            "commit": observed_commit,
            "config_path": str(config_path),
            "config_sha256": observed_config,
            "runner_path": str(runner_path),
            "runner_sha256": observed_runner,
            "package_tree_object": tracked_tree,
            "execution_closure_clean": True,
            "imported_runner_path": str(imported_path),
            "model_protocol": "LightGBM num_boost_round=500 early_stopping_round=50",
            "portfolio_protocol": "TopkDropout topk=50 n_drop=5 cost-aware open-price",
        }

    def evaluate_panel(
        self,
        panel: pd.DataFrame,
        *,
        experiment_name: str,
        recorder_name: str,
        output_name: str,
        capture_report: bool = False,
    ) -> dict[str, Any]:
        """Call Quanta's actual computed-factor dataset, training, and backtest path."""
        if not self.identity.get("verified"):
            raise QuantaIdentityError("Quanta identities were not verified")
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError("factor panel must have a MultiIndex")
        if list(panel.index.names) != ["datetime", "instrument"]:
            raise ValueError(
                "factor panel index must be ordered and named datetime, instrument"
            )
        if panel.empty or panel.shape[1] == 0:
            raise ValueError("factor panel must contain rows and factors")
        self.runner._init_qlib()
        self._verify_effective_provider()
        original_segments = self._normalize_computed_factor_segments()
        try:
            dataset = self.runner._create_dataset_with_computed_factors({}, panel)
        finally:
            segments = self.runner.config["dataset"]["segments"]
            segments.clear()
            segments.update(original_segments)

        def invoke() -> dict[str, Any]:
            return self.runner._train_and_backtest(
                dataset, experiment_name, recorder_name, output_name=output_name
            )

        return self._with_exact_backtest_capture(invoke) if capture_report else invoke()

    def _normalize_computed_factor_segments(self) -> dict[str, object]:
        """Make pinned YAML ranges usable by Quanta's precomputed-data handler."""
        try:
            segments = self.runner.config["dataset"]["segments"]
        except (KeyError, TypeError) as error:
            raise QuantaIdentityError(
                "pinned Quanta config lacks computed-factor dataset segments"
            ) from error
        if not isinstance(segments, dict):
            raise QuantaIdentityError(
                "pinned Quanta computed-factor segments are not a mapping"
            )
        original = dict(segments)
        normalized: dict[str, tuple[str, str]] = {}
        recorded: dict[str, list[str]] = {}
        for name in ("train", "valid", "test"):
            value = segments.get(name)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or not all(isinstance(item, str) and item for item in value)
            ):
                raise QuantaIdentityError(
                    f"pinned Quanta {name} segment is not a two-date sequence"
                )
            start, end = value
            try:
                start_timestamp = pd.Timestamp(start)
                end_timestamp = pd.Timestamp(end)
            except (TypeError, ValueError) as error:
                raise QuantaIdentityError(
                    f"pinned Quanta {name} segment contains an invalid date"
                ) from error
            if start_timestamp > end_timestamp:
                raise QuantaIdentityError(
                    f"pinned Quanta {name} segment is reverse ordered"
                )
            normalized[name] = (start, end)
            recorded[name] = [start, end]
        segments.update(normalized)
        self.identity["computed_factor_segments"] = recorded
        return original

    def initialize_and_verify_provider(self) -> None:
        """Initialize Qlib and prove the effective provider before result access."""
        if not self.identity.get("verified"):
            raise QuantaIdentityError("Quanta identities were not verified")
        self.runner._init_qlib()
        self._verify_effective_provider()

    def evaluate_alpha158(
        self,
        *,
        experiment_name: str,
        output_name: str,
        capture_report: bool = False,
    ) -> dict[str, Any]:
        """Run the pinned runner's official Alpha158 path without changing it."""
        if not self.identity.get("verified"):
            raise QuantaIdentityError("Quanta identities were not verified")

        def invoke() -> dict[str, Any]:
            return self.runner.run(
                factor_source="alpha158",
                experiment_name=experiment_name,
                output_name=output_name,
            )

        metrics = (
            self._with_exact_backtest_capture(invoke) if capture_report else invoke()
        )
        self._verify_effective_provider()
        return metrics

    def _verify_effective_provider(self) -> None:
        """Bind the recorded run to Qlib's effective provider, including env override."""
        if "repository" not in self.identity:
            return
        selected = os.path.expanduser(
            os.environ.get("QLIB_DATA_DIR")
            or os.environ.get("QLIB_PROVIDER_URI")
            or str(self.runner.config["data"]["provider_uri"])
        )
        locked = self._expected_provider_identity
        expected = str(locked["path"]) if locked is not None else selected
        if locked is not None and Path(selected).expanduser().resolve(
            strict=True
        ) != Path(expected).resolve(strict=True):
            raise QuantaIdentityError(
                "selected Qlib provider is not the implementation-lock provider"
            )
        qlib = importlib.import_module("qlib")
        config = getattr(qlib, "config", None)
        container = getattr(config, "C", None)
        observed = None
        if container is not None:
            data_path_manager = getattr(container, "dpm", None)
            if data_path_manager is not None:
                observed = data_path_manager.get_data_uri()
            else:
                try:
                    observed = container["provider_uri"]
                except (KeyError, TypeError):
                    observed = getattr(container, "provider_uri", None)
        if isinstance(observed, dict):
            observed = (
                observed.get("__DEFAULT_FREQ")
                or observed.get("uri")
                or observed.get("default")
            )
        if observed is None:
            raise QuantaIdentityError("cannot verify Qlib's effective provider URI")
        expected_path = Path(str(expected)).expanduser().resolve(strict=True)
        observed_path = Path(str(observed)).expanduser().resolve(strict=True)
        if observed_path != expected_path:
            raise QuantaIdentityError(
                "Qlib effective provider differs from the pinned/env-selected provider"
            )
        if locked is not None:
            stat_identity = regular_file_tree_stat_identity(observed_path)
            for key in ("stat_inventory_sha256", "file_count", "total_bytes"):
                if stat_identity[key] != locked[key]:
                    raise QuantaIdentityError(
                        f"Qlib provider changed after lock verification: {key}"
                    )
        self.identity["effective_qlib_provider"] = str(observed_path)
        if locked is not None:
            self.identity["qlib_provider_tree_sha256"] = locked["tree_sha256"]

    def _with_exact_backtest_capture(
        self, invoke: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        """Intercept one exact Qlib call, restoring the module function in all cases."""
        import importlib

        with _BACKTEST_PATCH_LOCK:
            self._captured_report_df = None
            self._captured_prediction_signal = None
            backtest_module = importlib.import_module("qlib.backtest")
            original = backtest_module.backtest
            calls = 0
            captured_report: pd.DataFrame | None = None
            captured_signal: pd.Series | None = None

            def intercept(*args: object, **kwargs: object) -> object:
                nonlocal calls, captured_report, captured_signal
                calls += 1
                if calls > 1:
                    raise RuntimeError(
                        "pinned runner invoked Qlib backtest more than once"
                    )
                result = original(*args, **kwargs)
                try:
                    portfolio_metric_dict = result[0]
                    report = portfolio_metric_dict["1day"][0]
                    strategy = kwargs["strategy"]
                    signal = strategy["kwargs"]["signal"]
                except (KeyError, IndexError, TypeError) as error:
                    raise RuntimeError(
                        "cannot capture exact Qlib report and prediction signal"
                    ) from error
                if not isinstance(report, pd.DataFrame):
                    raise RuntimeError("captured Qlib report_df is not a DataFrame")
                if not isinstance(signal, pd.Series):
                    raise RuntimeError(
                        "captured Qlib prediction signal is not a Series"
                    )
                captured_report = report.copy(deep=True)
                captured_signal = signal.copy(deep=True)
                return result

            backtest_module.backtest = intercept
            try:
                metrics = invoke()
            finally:
                backtest_module.backtest = original
            if calls != 1 or captured_report is None or captured_signal is None:
                raise RuntimeError(
                    "pinned runner did not complete exactly one Qlib backtest"
                )
            self._captured_report_df = captured_report
            self._captured_prediction_signal = captured_signal
            return metrics

    def write_portfolio_diagnostics(self, destination: Path | str) -> dict[str, str]:
        """Publish exact report-backed daily diagnostics into an existing staging dir."""
        report = getattr(self, "_captured_report_df", None)
        signal = getattr(self, "_captured_prediction_signal", None)
        if not isinstance(report, pd.DataFrame) or not isinstance(signal, pd.Series):
            raise RuntimeError("no exact Qlib backtest capture is available")
        required = {"return", "bench", "cost", "turnover"}
        missing = sorted(required.difference(report.columns))
        if missing:
            raise RuntimeError(f"exact Qlib report lacks required columns: {missing}")
        try:
            report_values = report.loc[:, sorted(required)].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "exact Qlib report diagnostics are not numeric"
            ) from error
        if not np.isfinite(report_values).all():
            raise RuntimeError("exact Qlib report contains non-finite diagnostics")
        if (
            not isinstance(signal.index, pd.MultiIndex)
            or "datetime" not in signal.index.names
        ):
            raise RuntimeError("prediction signal lacks a datetime MultiIndex level")
        output = Path(destination).resolve(strict=True)
        paths = {
            "qlib_report.parquet": output / "qlib_report.parquet",
            "portfolio_daily.parquet": output / "portfolio_daily.parquet",
            "prediction_coverage.parquet": output / "prediction_coverage.parquet",
        }
        for path in paths.values():
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"refusing to replace Quanta diagnostic: {path}")

        report.to_parquet(paths["qlib_report.parquet"])
        portfolio_return = report["return"]
        benchmark_return = report["bench"]
        realized_cost_for_excess = report["cost"]
        daily = pd.DataFrame(
            {
                "daily_excess_return": (
                    portfolio_return - benchmark_return - realized_cost_for_excess
                ),
                "turnover": report["turnover"],
                "realized_cost": report["cost"],
            },
            index=report.index,
        )
        daily.to_parquet(paths["portfolio_daily.parquet"])

        finite = pd.Series(
            np.isfinite(signal.to_numpy(dtype=float)), index=signal.index
        )
        coverage = pd.DataFrame(
            {
                "total_predictions": signal.groupby(level="datetime", sort=True).size(),
                "finite_predictions": finite.groupby(level="datetime", sort=True)
                .sum()
                .astype(int),
            }
        )
        coverage["prediction_coverage"] = (
            coverage["finite_predictions"] / coverage["total_predictions"]
        )
        coverage.to_parquet(paths["prediction_coverage.parquet"])
        return {name: sha256_file(path) for name, path in paths.items()}

    @staticmethod
    def baseline_link(metric_path: Path | str) -> dict[str, Any]:
        """Hash and describe the historical Alpha158 metric without reinterpretation."""
        path = Path(metric_path).resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        best_iteration = payload.get("best_iter", payload.get("best_iteration"))
        if best_iteration is None:
            raise ValueError(
                "Alpha158 metric does not record its observed best iteration"
            )
        return {
            "role": "provisional_historical_baseline_link",
            "path": str(path),
            "sha256": sha256_file(path),
            "training_regime": "500_round_cap_early_stopping_50",
            "observed_best_iteration": int(best_iteration),
            "fixed_14_regime": False,
            "reported_metrics": payload,
        }
