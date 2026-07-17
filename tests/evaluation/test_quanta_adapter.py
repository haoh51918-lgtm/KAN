from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pandas as pd
import pytest

from mirage_kan.evaluation.quanta import QuantaAdapter, QuantaIdentityError
from mirage_kan.data.pit import sha256_file
from mirage_kan.identities import regular_file_tree_identity


class FakeRunner:
    def __init__(self) -> None:
        self.calls = []
        self.config = {
            "dataset": {
                "segments": {
                    "train": ["2016-01-01", "2020-12-31"],
                    "valid": ["2021-01-01", "2021-12-31"],
                    "test": ["2022-01-01", "2025-12-26"],
                }
            }
        }

    def _init_qlib(self) -> None:
        self.calls.append(("init",))

    def _create_dataset_with_computed_factors(
        self, factor_expressions, computed_factors
    ):
        self.dataset_segments = dict(self.config["dataset"]["segments"])
        self.calls.append(("dataset", factor_expressions, computed_factors))
        return "dataset"

    def _train_and_backtest(self, dataset, exp_name, rec_name, output_name=None):
        self.calls.append(("evaluate", dataset, exp_name, rec_name, output_name))
        return {"information_ratio": 1.0}


def test_adapter_calls_real_runner_path_with_no_qlib_expressions(tmp_path) -> None:
    runner = FakeRunner()
    panel = pd.DataFrame(
        {"f": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("2020-01-01", "A")], names=["datetime", "instrument"]
        ),
    )
    adapter = QuantaAdapter.__new__(QuantaAdapter)
    adapter.runner = runner
    adapter.identity = {"verified": True}
    metrics = adapter.evaluate_panel(
        panel, experiment_name="s0", recorder_name="s0", output_name="seed"
    )
    assert metrics["information_ratio"] == 1.0
    assert runner.calls[1][0:2] == ("dataset", {})
    assert runner.calls[2] == ("evaluate", "dataset", "s0", "s0", "seed")


def test_computed_factor_path_normalizes_yaml_segment_lists_without_changing_dates(
    tmp_path,
) -> None:
    runner = FakeRunner()
    runner.config = {
        "dataset": {
            "segments": {
                "train": ["2016-01-01", "2020-12-31"],
                "valid": ["2021-01-01", "2021-12-31"],
                "test": ["2022-01-01", "2025-12-26"],
            }
        }
    }
    panel = pd.DataFrame(
        {"f": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("2020-01-01", "A")], names=["datetime", "instrument"]
        ),
    )
    adapter = QuantaAdapter.__new__(QuantaAdapter)
    adapter.runner = runner
    adapter.identity = {"verified": True}

    adapter.evaluate_panel(
        panel, experiment_name="s0", recorder_name="s0", output_name="seed"
    )

    assert runner.dataset_segments == {
        "train": ("2016-01-01", "2020-12-31"),
        "valid": ("2021-01-01", "2021-12-31"),
        "test": ("2022-01-01", "2025-12-26"),
    }
    assert runner.config["dataset"]["segments"] == {
        "train": ["2016-01-01", "2020-12-31"],
        "valid": ["2021-01-01", "2021-12-31"],
        "test": ["2022-01-01", "2025-12-26"],
    }
    assert adapter.identity["computed_factor_segments"] == {
        "train": ["2016-01-01", "2020-12-31"],
        "valid": ["2021-01-01", "2021-12-31"],
        "test": ["2022-01-01", "2025-12-26"],
    }


def test_adapter_rejects_identity_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "quantaalpha" / "backtest").mkdir(parents=True)
    (repo / "configs" / "backtest.yaml").write_text("x")
    (repo / "quantaalpha" / "backtest" / "runner.py").write_text("x")
    with pytest.raises(QuantaIdentityError):
        QuantaAdapter(
            repo,
            expected_commit="0" * 40,
            expected_config_sha256="1" * 64,
            expected_runner_sha256="2" * 64,
        )


def _pinned_fake_quanta_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "quantaalpha" / "backtest").mkdir(parents=True)
    (repo / "quantaalpha" / "__init__.py").write_text("")
    (repo / "quantaalpha" / "backtest" / "__init__.py").write_text("")
    (repo / "configs" / "backtest.yaml").write_text(
        "data:\n  provider_uri: .\nexperiment:\n  output_dir: .\n"
    )
    (repo / "quantaalpha" / "backtest" / "runner.py").write_text(
        "class BacktestRunner:\n"
        "    def __init__(self, config_path):\n"
        "        self.config = {'data': {'provider_uri': '.'}, "
        "'experiment': {'output_dir': '.'}}\n"
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


def test_adapter_uses_real_package_import_and_rejects_dirty_dependency(
    tmp_path: Path,
) -> None:
    repo, commit = _pinned_fake_quanta_repository(tmp_path)
    config = repo / "configs" / "backtest.yaml"
    runner = repo / "quantaalpha" / "backtest" / "runner.py"
    try:
        adapter = QuantaAdapter(
            repo,
            expected_commit=commit,
            expected_config_sha256=sha256_file(config),
            expected_runner_sha256=sha256_file(runner),
        )
        assert adapter.runner.__class__.__module__ == "quantaalpha.backtest.runner"
        assert adapter.identity["imported_runner_path"] == str(runner.resolve())
        assert adapter.identity["execution_closure_clean"] is True

        dependency = repo / "quantaalpha" / "backtest" / "dependency.py"
        dependency.write_text("DIRTY = True\n")
        with pytest.raises(QuantaIdentityError, match="execution closure"):
            QuantaAdapter(
                repo,
                expected_commit=commit,
                expected_config_sha256=sha256_file(config),
                expected_runner_sha256=sha256_file(runner),
            )
    finally:
        for name in list(sys.modules):
            if name == "quantaalpha" or name.startswith("quantaalpha."):
                sys.modules.pop(name, None)


def test_effective_provider_must_equal_content_verified_lock(
    tmp_path: Path, monkeypatch
) -> None:
    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "calendar.txt").write_text("2022-01-01")
    identity = regular_file_tree_identity(provider)
    qlib = types.ModuleType("qlib")
    qlib.config = types.SimpleNamespace(C={"provider_uri": str(provider)})
    monkeypatch.setitem(sys.modules, "qlib", qlib)
    monkeypatch.setenv("QLIB_DATA_DIR", str(provider))
    adapter = QuantaAdapter.__new__(QuantaAdapter)
    adapter.identity = {"verified": True, "repository": "/pinned/quanta"}
    adapter.runner = types.SimpleNamespace(
        config={"data": {"provider_uri": str(tmp_path / "wrong")}}
    )
    adapter._expected_provider_identity = identity

    adapter._verify_effective_provider()

    assert adapter.identity["effective_qlib_provider"] == str(provider)
    assert adapter.identity["qlib_provider_tree_sha256"] == identity["tree_sha256"]
    monkeypatch.setenv("QLIB_DATA_DIR", str(tmp_path))
    with pytest.raises(QuantaIdentityError, match="implementation-lock provider"):
        adapter._verify_effective_provider()


def test_effective_provider_supports_qlib_frequency_mapping(
    tmp_path: Path, monkeypatch
) -> None:
    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "calendar.txt").write_text("2022-01-01")
    identity = regular_file_tree_identity(provider)

    class FakeDataPathManager:
        def get_data_uri(self, freq=None):
            assert freq is None
            return provider

    class FakeQlibConfig:
        dpm = FakeDataPathManager()

        def __getitem__(self, key):
            assert key == "provider_uri"
            return {"__DEFAULT_FREQ": str(provider)}

    qlib = types.ModuleType("qlib")
    qlib.config = types.SimpleNamespace(C=FakeQlibConfig())
    monkeypatch.setitem(sys.modules, "qlib", qlib)
    monkeypatch.setenv("QLIB_DATA_DIR", str(provider))
    adapter = QuantaAdapter.__new__(QuantaAdapter)
    adapter.identity = {"verified": True, "repository": "/pinned/quanta"}
    adapter.runner = types.SimpleNamespace(
        config={"data": {"provider_uri": str(tmp_path / "wrong")}}
    )
    adapter._expected_provider_identity = identity

    adapter._verify_effective_provider()

    assert adapter.identity["effective_qlib_provider"] == str(provider)


def test_baseline_link_preserves_early_stopping_semantics(tmp_path: Path) -> None:
    metric = tmp_path / "metric.json"
    metric.write_text(json.dumps({"metrics": {"Rank IC": 0.03}, "best_iteration": 14}))
    link = QuantaAdapter.baseline_link(metric)
    assert link["training_regime"] == "500_round_cap_early_stopping_50"
    assert link["observed_best_iteration"] == 14


def _install_fake_qlib(monkeypatch, report: pd.DataFrame):
    qlib = types.ModuleType("qlib")
    backtest_module = types.ModuleType("qlib.backtest")
    calls = []

    def original_backtest(*args, **kwargs):
        calls.append((args, kwargs))
        return {"1day": (report, pd.DataFrame())}, {"indicator": "exact"}

    backtest_module.backtest = original_backtest
    qlib.backtest = backtest_module
    monkeypatch.setitem(sys.modules, "qlib", qlib)
    monkeypatch.setitem(sys.modules, "qlib.backtest", backtest_module)
    return backtest_module, original_backtest, calls


class CapturingRunner(FakeRunner):
    def _train_and_backtest(self, dataset, exp_name, rec_name, output_name=None):
        from qlib.backtest import backtest

        dates = pd.date_range("2022-01-03", periods=2)
        signal = pd.Series(
            [0.1, float("nan"), 0.2, 0.3],
            index=pd.MultiIndex.from_product(
                [dates, ["A", "B"]], names=["datetime", "instrument"]
            ),
        )
        backtest(
            strategy={"kwargs": {"signal": signal}},
            executor={"class": "SimulatorExecutor"},
        )
        return {"information_ratio": 1.0}


def test_exact_backtest_capture_calls_once_publishes_diagnostics_and_restores(
    tmp_path, monkeypatch
) -> None:
    report = pd.DataFrame(
        {
            "return": [0.02, -0.01],
            "bench": [0.01, -0.02],
            "cost": [0.001, 0.002],
            "turnover": [0.3, 0.4],
        },
        index=pd.date_range("2022-01-03", periods=2, name="date"),
    )
    module, original, calls = _install_fake_qlib(monkeypatch, report)
    adapter = QuantaAdapter.__new__(QuantaAdapter)
    adapter.runner = CapturingRunner()
    adapter.identity = {"verified": True}
    panel = pd.DataFrame(
        {"f": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("2020-01-01", "A")], names=["datetime", "instrument"]
        ),
    )

    adapter.evaluate_panel(
        panel,
        experiment_name="s2a",
        recorder_name="s2a",
        output_name="selected",
        capture_report=True,
    )
    files = adapter.write_portfolio_diagnostics(tmp_path)

    assert len(calls) == 1
    assert module.backtest is original
    stored_report = pd.read_parquet(tmp_path / "qlib_report.parquet")
    pd.testing.assert_frame_equal(
        stored_report, report, check_exact=True, check_freq=False
    )
    daily = pd.read_parquet(tmp_path / "portfolio_daily.parquet")
    assert daily["daily_excess_return"].tolist() == pytest.approx([0.009, 0.008])
    assert daily["turnover"].tolist() == [0.3, 0.4]
    assert daily["realized_cost"].tolist() == [0.001, 0.002]
    coverage = pd.read_parquet(tmp_path / "prediction_coverage.parquet")
    assert coverage["finite_predictions"].tolist() == [1, 2]
    assert coverage["prediction_coverage"].tolist() == [0.5, 1.0]
    assert set(files) == {
        "qlib_report.parquet",
        "portfolio_daily.parquet",
        "prediction_coverage.parquet",
    }


def test_portfolio_diagnostics_reject_nonfinite_report_without_writing(
    tmp_path,
) -> None:
    adapter = QuantaAdapter.__new__(QuantaAdapter)
    adapter._captured_report_df = pd.DataFrame(
        {
            "return": [0.01, float("nan")],
            "bench": [0.0, 0.0],
            "cost": [0.001, 0.001],
            "turnover": [0.2, 0.3],
        },
        index=pd.date_range("2022-01-03", periods=2, name="date"),
    )
    adapter._captured_prediction_signal = pd.Series(
        [0.1, 0.2],
        index=pd.MultiIndex.from_product(
            [[pd.Timestamp("2022-01-03")], ["A", "B"]],
            names=["datetime", "instrument"],
        ),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        adapter.write_portfolio_diagnostics(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_backtest_interception_restores_original_when_runner_raises(
    monkeypatch,
) -> None:
    report = pd.DataFrame(
        {"return": [0.0], "bench": [0.0], "cost": [0.0], "turnover": [0.0]}
    )
    module, original, _ = _install_fake_qlib(monkeypatch, report)

    class FailingRunner(CapturingRunner):
        def _train_and_backtest(self, *args, **kwargs):
            super()._train_and_backtest(*args, **kwargs)
            raise RuntimeError("after exact backtest")

    adapter = QuantaAdapter.__new__(QuantaAdapter)
    adapter.runner = FailingRunner()
    adapter.identity = {"verified": True}
    adapter._captured_report_df = pd.DataFrame({"stale": [1]})
    adapter._captured_prediction_signal = pd.Series([1.0])
    panel = pd.DataFrame(
        {"f": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("2020-01-01", "A")], names=["datetime", "instrument"]
        ),
    )
    with pytest.raises(RuntimeError, match="after exact backtest"):
        adapter.evaluate_panel(
            panel,
            experiment_name="s2a",
            recorder_name="s2a",
            output_name="selected",
            capture_report=True,
        )
    assert module.backtest is original
    assert adapter._captured_report_df is None
    assert adapter._captured_prediction_signal is None


def test_backtest_interception_is_process_serialized(monkeypatch) -> None:
    report = pd.DataFrame(
        {"return": [0.0], "bench": [0.0], "cost": [0.0], "turnover": [0.0]}
    )
    _install_fake_qlib(monkeypatch, report)
    state = {"active": 0, "maximum": 0}
    state_lock = threading.Lock()

    class SlowRunner(CapturingRunner):
        def _train_and_backtest(self, *args, **kwargs):
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            try:
                result = super()._train_and_backtest(*args, **kwargs)
                time.sleep(0.03)
                return result
            finally:
                with state_lock:
                    state["active"] -= 1

    panel = pd.DataFrame(
        {"f": [1.0]},
        index=pd.MultiIndex.from_tuples(
            [("2020-01-01", "A")], names=["datetime", "instrument"]
        ),
    )
    adapters = []
    for _ in range(2):
        adapter = QuantaAdapter.__new__(QuantaAdapter)
        adapter.runner = SlowRunner()
        adapter.identity = {"verified": True}
        adapters.append(adapter)
    errors = []

    def run(adapter):
        try:
            adapter.evaluate_panel(
                panel,
                experiment_name="s2a",
                recorder_name="s2a",
                output_name="concurrent",
                capture_report=True,
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(adapter,)) for adapter in adapters]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert state["maximum"] == 1


def test_official_alpha158_replay_uses_unchanged_runner_call(monkeypatch) -> None:
    report = pd.DataFrame(
        {"return": [0.0], "bench": [0.0], "cost": [0.0], "turnover": [0.0]}
    )
    _, _, calls = _install_fake_qlib(monkeypatch, report)

    class OfficialRunner:
        def __init__(self) -> None:
            self.calls = []

        def run(self, **kwargs):
            from qlib.backtest import backtest

            self.calls.append(kwargs)
            signal = pd.Series(
                [0.1],
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp("2022-01-03"), "A")],
                    names=["datetime", "instrument"],
                ),
            )
            backtest(strategy={"kwargs": {"signal": signal}})
            return {"information_ratio": 0.1}

    adapter = QuantaAdapter.__new__(QuantaAdapter)
    adapter.runner = OfficialRunner()
    adapter.identity = {"verified": True}
    metrics = adapter.evaluate_alpha158(
        experiment_name="s2a_alpha158_replay",
        output_name="alpha158_replay",
        capture_report=True,
    )

    assert metrics == {"information_ratio": 0.1}
    assert len(calls) == 1
    assert adapter.runner.calls == [
        {
            "factor_source": "alpha158",
            "experiment_name": "s2a_alpha158_replay",
            "output_name": "alpha158_replay",
        }
    ]
