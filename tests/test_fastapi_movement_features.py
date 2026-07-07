"""Tests for movement features, leakage boundaries, and import safety."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from financial_news_intelligence.api.movement_runtime import MovementRuntime


def test_single_event_features_use_only_text_time_and_sentiment() -> None:
    """Prepare one event, run feature creation, and check named formulas."""

    result = MovementRuntime._sentiment_event_features(
        text="NVIDIA filed a quarterly report.",
        sentiment={
            "label": "Bullish",
            "prob_bearish": 0.10,
            "prob_neutral": 0.20,
            "prob_bullish": 0.70,
        },
        hours_to_open=4.5,
    )

    assert result["article_count"] == 1
    assert result["net_sentiment_mean"] == pytest.approx(0.60)
    assert result["bullish_event_share"] == 1.0
    assert result["hours_to_open_mean"] == 4.5
    assert "reaction_return" not in result
    assert "target_close" not in result


def test_target_session_accepts_only_one_held_out_test_row() -> None:
    """Prepare one test row, map publication time, and check accepted session."""

    import pandas as pd

    runtime = MovementRuntime.__new__(MovementRuntime)
    runtime.prices = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "session_date": [pd.Timestamp("2020-01-02")],
        }
    )
    runtime.model_table = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "target_session_date": [pd.Timestamp("2020-01-02")],
            "split": ["test"],
        }
    )

    target_date, _ = runtime.target_session(
        "AAPL",
        pd.Timestamp("2020-01-01T12:00:00Z").to_pydatetime(),
    )

    assert target_date == pd.Timestamp("2020-01-02")


def test_target_session_rejects_train_row_to_prevent_reference_leakage() -> None:
    """Prepare one train row, map publication time, and check fail-closed rejection."""

    import pandas as pd

    from financial_news_intelligence.api.errors import ApiProblem

    runtime = MovementRuntime.__new__(MovementRuntime)
    runtime.prices = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "session_date": [pd.Timestamp("2020-01-02")],
        }
    )
    runtime.model_table = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "target_session_date": [pd.Timestamp("2020-01-02")],
            "split": ["train"],
        }
    )

    with pytest.raises(ApiProblem) as captured:
        runtime.target_session(
            "AAPL",
            pd.Timestamp("2020-01-01T12:00:00Z").to_pydatetime(),
        )

    assert captured.value.error_code == "historical_feature_row_unavailable"


def test_movement_runtime_blocks_crashing_optional_pyarrow_import(
    tmp_path: Path,
) -> None:
    """Prepare a crashing fake pyarrow, import movement runtime, and check safety."""

    # The fake package represents the native pyarrow installation that crashed
    # on the audited Mac. If the movement runtime does not block the optional
    # import before pandas starts, the child process terminates with SIGSEGV.
    fake_package = tmp_path / "pyarrow"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text(
        "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n",
        encoding="utf-8",
    )

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [
                    str(project_root / "runtime_shims"),
                    str(project_root / "src"),
                    str(tmp_path),
                ]
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import financial_news_intelligence.api.movement_runtime; "
                "assert 'pyarrow' not in sys.modules; "
                "print('PASSED: pyarrow remained unloaded.')"
            ),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED: pyarrow remained unloaded." in result.stdout




def test_bundle_loader_uses_minimal_path_without_loading_pyarrow(
    tmp_path: Path,
) -> None:
    """Prepare a simple bundle, run the loader, and check pyarrow stays absent."""

    import importlib.metadata
    import joblib
    import platform

    model_path = tmp_path / "movement_bundle.joblib"
    bundle = {
        "pipeline": {"kind": "test"},
        "champion_name": "stability_soft_vote_rf_sgd",
        "numeric_features": ["feature_a"],
        "categorical_features": ["ticker"],
        "label_order": ["Down", "Flat", "Up"],
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "joblib": importlib.metadata.version("joblib"),
        },
        "decision_policy": {"kind": "test"},
    }
    joblib.dump(bundle, model_path)

    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        "from pathlib import Path; "
        "from financial_news_intelligence.api.movement_bundle_loader import "
        "load_verified_movement_bundle; "
        f"bundle=load_verified_movement_bundle(Path({str(model_path)!r})); "
        "assert bundle['champion_name']=='stability_soft_vote_rf_sgd'; "
        "assert 'financial_news_intelligence.api.movement_runtime' not in sys.modules; "
        "assert 'pyarrow' not in sys.modules; "
        "print('PASSED: bundle loaded before movement runtime and without pyarrow.')"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [
                    str(project_root / "runtime_shims"),
                    str(project_root / "src"),
                ]
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "PASSED: bundle loaded before movement runtime and without pyarrow."
        in result.stdout
    )


def test_bundle_loader_rejects_runtime_version_change(tmp_path: Path) -> None:
    """Prepare changed runtime evidence and check fail-closed rejection."""

    import joblib

    from financial_news_intelligence.api.errors import ApiProblem
    from financial_news_intelligence.api.movement_bundle_loader import (
        load_verified_movement_bundle,
    )

    model_path = tmp_path / "movement_bundle.joblib"
    joblib.dump(
        {
            "pipeline": {"kind": "test"},
            "champion_name": "stability_soft_vote_rf_sgd",
            "numeric_features": ["feature_a"],
            "categorical_features": ["ticker"],
            "label_order": ["Down", "Flat", "Up"],
            "runtime_versions": {
                "python": "0.0.0",
                "numpy": "0.0.0",
                "pandas": "0.0.0",
                "scikit_learn": "0.0.0",
                "joblib": "0.0.0",
            },
            "decision_policy": {"kind": "test"},
        },
        model_path,
    )

    with pytest.raises(ApiProblem) as captured:
        load_verified_movement_bundle(model_path)

    assert captured.value.error_code == "movement_runtime_changed"
    assert "Runtime versions differ" in captured.value.why_failed


def test_movement_runtime_uses_preloaded_bundle_without_reloading_joblib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepare a validated bundle and check no joblib reload occurs."""

    from types import SimpleNamespace

    import pandas as pd

    preloaded_bundle = {
        "pipeline": {"kind": "test"},
        "champion_name": "stability_soft_vote_rf_sgd",
        "numeric_features": ["feature_a"],
        "categorical_features": ["ticker"],
        "text_features": [],
        "label_order": ["Down", "Flat", "Up"],
        "runtime_versions": {},
        "decision_policy": {"kind": "test"},
    }
    artifact_paths = SimpleNamespace(
        movement_model=Path("unused.joblib"),
        movement_model_table=Path("model_table.csv"),
        movement_test_predictions=Path("predictions.csv"),
        foundation_news=Path("news.csv"),
        foundation_prices=Path("prices.csv"),
        global_drivers=Path("drivers.csv"),
        sentiment_phrases=Path("phrases.csv"),
    )
    monkeypatch.setattr(pd, "read_csv", lambda _path: pd.DataFrame())
    monkeypatch.setattr(MovementRuntime, "_normalize_frames", lambda self: None)
    monkeypatch.setattr(MovementRuntime, "_validate_schemas", lambda self: None)

    runtime = MovementRuntime(artifact_paths, preloaded_bundle)

    assert runtime.bundle is preloaded_bundle
    assert runtime.champion_name == "stability_soft_vote_rf_sgd"


def test_bundle_loader_rejects_distribution_outside_active_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prepare an outside package path, inspect metadata, and fail closed."""

    from types import SimpleNamespace

    from financial_news_intelligence.api.errors import ApiProblem
    from financial_news_intelligence.api import movement_bundle_loader

    active_environment = tmp_path / "project-venv"
    outside_packages = tmp_path / "global-site-packages"
    active_environment.mkdir()
    outside_packages.mkdir()

    fake_distribution = SimpleNamespace(
        version="1.0.0",
        locate_file=lambda _relative: outside_packages,
    )
    monkeypatch.setattr(movement_bundle_loader.sys, "prefix", str(active_environment))
    monkeypatch.setattr(
        movement_bundle_loader.importlib.metadata,
        "distribution",
        lambda _name: fake_distribution,
    )

    with pytest.raises(ApiProblem) as captured:
        movement_bundle_loader._installed_runtime_versions()

    assert captured.value.error_code == "movement_runtime_dependency_outside_venv"
    assert str(active_environment) in captured.value.why_failed
    assert str(outside_packages) in captured.value.why_failed
