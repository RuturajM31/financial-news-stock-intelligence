"""Focused tests for foundation features, splits, training, and artifacts."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from financial_news_intelligence.models.movement_artifacts import (
    cleanup_temporary_outputs,
    resolve_outputs,
    write_json,
)
from financial_news_intelligence.models.movement_dataset import (
    MovementDatasetError,
    SplitConfig,
    aggregate_events,
    assign_chronological_splits,
    build_model_table,
    build_prior_price_features,
    feature_columns,
    filter_events_by_split,
)
from financial_news_intelligence.models.movement_training import (
    CandidateDefinition,
    MovementTrainingError,
    TrainingConfig,
    _candidate_fold_summary,
    _candidate_sample_weight,
    _classifier_for_definition,
    _fit_candidate,
    _rolling_validation_frames,
    _split_development_confirmation,
    _validation_gate_report,
    candidate_definitions,
    classification_metrics,
    evaluate_quality_gates,
    fit_stable_global_policy,
    global_importance,
    per_ticker_metrics,
    rank_confirmation_tournament,
    rank_validation_results,
    select_terminal_shortlist,
    train_and_evaluate,
)


class WarningClassifier(BaseEstimator, ClassifierMixin):
    """Emit one convergence warning so fail-closed fitting can be tested."""

    def fit(self, features: object, target: object) -> "WarningClassifier":
        """Prepare input, run warning emission, and check through the caller."""

        warnings.warn(
            "synthetic convergence failure",
            ConvergenceWarning,
            stacklevel=2,
        )
        self.classes_ = np.asarray(["Down", "Flat", "Up"], dtype=object)
        return self

    def predict_proba(self, features: object) -> np.ndarray:
        """Prepare rows, run fixed probabilities, and check valid shape."""

        row_count = len(features)
        return np.tile(np.asarray([[0.34, 0.33, 0.33]]), (row_count, 1))


def make_foundation_frames(
    date_count: int = 330,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare deterministic event and adjusted-price evidence."""

    dates = pd.bdate_range("2015-01-02", periods=date_count + 30)
    tickers = ("AAA", "BBB", "CCC")
    price_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(tickers):
        closes = 100 + ticker_index * 10 + np.linspace(0, 20, len(dates))
        closes += np.sin(np.arange(len(dates)) / 7.0)
        for index, session_date in enumerate(dates):
            price_rows.append(
                {
                    "ticker": ticker,
                    "session_date": session_date.date().isoformat(),
                    "close": float(closes[index]),
                    "volume": float(1_000_000 + index * 100),
                    "source_provider": "tiingo_eod",
                }
            )
        for index, session_date in enumerate(dates[25 : 25 + date_count]):
            label = ("Down", "Flat", "Up")[(index + ticker_index) % 3]
            reaction = {"Down": -0.02, "Flat": 0.0, "Up": 0.02}[label]
            probabilities = {
                "Down": (0.90, 0.08, 0.02),
                "Flat": (0.05, 0.90, 0.05),
                "Up": (0.02, 0.08, 0.90),
            }[label]
            open_time = pd.Timestamp(session_date, tz="UTC") + pd.Timedelta(
                hours=14
            )
            event_rows.append(
                {
                    "article_id": f"{ticker}-{index}",
                    "ticker": ticker,
                    "company": f"{ticker} Corporation",
                    "published_at_utc": (
                        open_time - pd.Timedelta(hours=4)
                    ).isoformat(),
                    "text": f"{ticker} verified {label} filing event {index}",
                    "source_name": "SEC EDGAR",
                    "source_url": f"https://sec.example/{ticker}/{index}",
                    "target_session_date": session_date.date().isoformat(),
                    "session_open_utc": open_time.isoformat(),
                    "prob_bearish": probabilities[0],
                    "prob_neutral": probabilities[1],
                    "prob_bullish": probabilities[2],
                    "sentiment_label": (
                        "Bearish"
                        if label == "Down"
                        else "Bullish"
                        if label == "Up"
                        else "Neutral"
                    ),
                    "reaction_return": reaction,
                    "movement_label": label,
                }
            )
    return pd.DataFrame(event_rows), pd.DataFrame(price_rows)


def make_split_table(date_count: int = 330) -> pd.DataFrame:
    """Prepare one complete split model table for training tests."""

    news, prices = make_foundation_frames(date_count)
    table, _ = assign_chronological_splits(build_model_table(news, prices))
    return table


def focused_training_config(**overrides: object) -> TrainingConfig:
    """Return a fast deterministic config for synthetic focused tests.

    Production defaults are never changed. The smaller estimator count, text
    vocabulary, permutation count, and terminal shortlist keep repeated test
    fits fast while unit tests separately verify the full v8 defaults.
    """

    values: dict[str, object] = {
        "forest_estimators": 5,
        "permutation_repeats": 1,
        "text_max_features": 100,
        "terminal_shortlist_size": 2,
        "terminal_shortlist_max_per_family": 1,
        "focused_test_mode": True,
    }
    values.update(overrides)
    return TrainingConfig(**values)


@pytest.fixture(scope="module")
def strong_training_result() -> dict[str, object]:
    """Prepare one reusable end-to-end synthetic movement result.

    A single module-scoped fit avoids loading and refitting native estimators
    repeatedly inside one pytest process. Separate unit tests cover failure
    paths, ranking rules, gates, and evidence writing without another full fit.
    """

    table = make_split_table()
    numeric, categorical, text_features = feature_columns(table)
    return train_and_evaluate(
        table,
        numeric,
        categorical,
        text_features,
        focused_training_config(forest_estimators=10),
    )


def test_aggregate_events_uses_one_ticker_session_grain() -> None:
    """Prepare duplicate events, run aggregation, and check one-row grain."""

    # Prepare data.
    news, _ = make_foundation_frames(30)
    duplicate = news.iloc[[0]].copy()
    duplicate["article_id"] = "duplicate-event"
    # Run function.
    result = aggregate_events(pd.concat([news, duplicate], ignore_index=True))
    # Check result.
    assert not result.duplicated(["ticker", "target_session_date"]).any()
    assert result.iloc[0]["article_count"] == 2


def test_aggregate_events_rejects_future_leakage() -> None:
    """Prepare a late event, run aggregation, and check rejection."""

    # Prepare data.
    news, _ = make_foundation_frames(30)
    news.loc[0, "published_at_utc"] = news.loc[0, "session_open_utc"]
    # Run function and check result.
    with pytest.raises(MovementDatasetError, match="future leakage"):
        aggregate_events(news)


def test_aggregate_events_rejects_conflicting_company_names() -> None:
    """Prepare conflicting names, run aggregation, and check rejection."""

    # Prepare data.
    news, _ = make_foundation_frames(30)
    duplicate = news.iloc[[0]].copy()
    duplicate["article_id"] = "other-company"
    duplicate["company"] = "Different Company"
    # Run function and check result.
    with pytest.raises(MovementDatasetError, match="company names"):
        aggregate_events(pd.concat([news, duplicate], ignore_index=True))


def test_prior_price_features_are_shifted() -> None:
    """Prepare monotonic prices, run features, and check target data is absent."""

    # Prepare data.
    _, prices = make_foundation_frames(30)
    # Run function.
    result = build_prior_price_features(prices)
    # Check result.
    assert "close" not in result.columns
    assert "volume" not in result.columns
    assert result["prior_return_1d"].notna().any()


def test_prior_price_features_include_longer_lagged_windows() -> None:
    """Prepare prices, run lagging, and check expanded pre-session evidence."""

    # Prepare data.
    _, prices = make_foundation_frames(90)
    # Run function.
    result = build_prior_price_features(prices)
    # Check result.
    expected = {
        "prior_return_60d",
        "prior_volatility_60d",
        "prior_close_position_20d",
        "prior_drawdown_20d",
        "prior_momentum_5_20",
    }
    assert expected.issubset(result.columns)
    assert "close" not in result.columns


def test_prior_price_features_reject_wrong_provider() -> None:
    """Prepare a wrong provider, run features, and check fail-closed behavior."""

    # Prepare data.
    _, prices = make_foundation_frames(30)
    prices["source_provider"] = "unverified"
    # Run function and check result.
    with pytest.raises(MovementDatasetError, match="provider"):
        build_prior_price_features(prices)


def test_build_model_table_joins_one_to_one() -> None:
    """Prepare foundation frames, run the join, and check unique grain."""

    # Prepare data.
    news, prices = make_foundation_frames(60)
    # Run function.
    result = build_model_table(news, prices)
    # Check result.
    assert len(result) == 60 * 3
    assert not result.duplicated(["ticker", "target_session_date"]).any()


def test_chronological_splits_are_ordered_and_purged() -> None:
    """Prepare many dates, run splitting, and check temporal separation."""

    # Prepare data.
    news, prices = make_foundation_frames(330)
    table = build_model_table(news, prices)
    # Run function.
    split_table, report = assign_chronological_splits(table)
    # Check result.
    assert pd.Timestamp(report["train"]["end_date"]) < pd.Timestamp(
        report["validation"]["start_date"]
    )
    assert pd.Timestamp(report["validation"]["end_date"]) < pd.Timestamp(
        report["test"]["start_date"]
    )
    assert report["purged"]["unique_dates"] == 2
    assert set(split_table["split"]) == {"train", "validation", "test"}


def test_chronological_splits_reject_small_dataset() -> None:
    """Prepare too few dates, run splitting, and check the minimum gate."""

    # Prepare data.
    news, prices = make_foundation_frames(60)
    table = build_model_table(news, prices)
    # Run function and check result.
    with pytest.raises(MovementDatasetError, match="Need 300 dates"):
        assign_chronological_splits(table)


def test_feature_columns_exclude_targets_and_prices() -> None:
    """Prepare a split table, run selection, and check forbidden evidence."""

    # Prepare data.
    table = make_split_table()
    # Run function.
    numeric, categorical, text_features = feature_columns(table)
    # Check result.
    selected = set(numeric + categorical + text_features)
    assert "movement_label" not in selected
    assert "reaction_return" not in selected
    assert categorical == ["ticker"]
    assert text_features == ["event_text"]


def test_filter_events_by_split_excludes_test_period() -> None:
    """Prepare split evidence, run filtering, and check reference-only dates."""

    # Prepare data.
    news, prices = make_foundation_frames(330)
    table, _ = assign_chronological_splits(build_model_table(news, prices))
    # Run function.
    result = filter_events_by_split(news, table, {"train", "validation"})
    # Check result.
    maximum_reference = result["target_session_date"].max()
    minimum_test = table.loc[
        table["split"] == "test",
        "target_session_date",
    ].min()
    assert pd.Timestamp(maximum_reference) < pd.Timestamp(minimum_test)


def test_classification_metrics_include_all_classes() -> None:
    """Prepare labels, run metrics, and check complete class evidence."""

    # Prepare data.
    actual = pd.Series(["Down", "Flat", "Up", "Up"])
    predicted = np.array(["Down", "Flat", "Flat", "Up"])
    # Run function.
    result = classification_metrics(actual, predicted)
    # Check result.
    assert set(result["per_class"]) == {"Down", "Flat", "Up"}
    assert result["predicted_class_count"] == 3
    assert len(result["confusion_matrix"]) == 3


def test_validation_ranking_uses_macro_f1_first() -> None:
    """Prepare candidate evidence, run ranking, and check selection priority."""

    # Prepare data.
    results = [
        {
            "model_name": "a",
            "status": "passed",
            "latency_ms_per_record": 1.0,
            "metrics": {"macro_f1": 0.7, "weighted_f1": 0.9, "accuracy": 0.9},
        },
        {
            "model_name": "b",
            "status": "passed",
            "latency_ms_per_record": 2.0,
            "metrics": {"macro_f1": 0.8, "weighted_f1": 0.8, "accuracy": 0.8},
        },
    ]
    # Run function.
    result = rank_validation_results(results)
    # Check result.
    assert result[0]["model_name"] == "b"


def test_train_and_evaluate_passes_strong_synthetic_signal(
    strong_training_result: dict[str, object],
) -> None:
    """Prepare one shared fit, run retrieval, and check every quality gate."""

    # Prepare data.
    result = strong_training_result
    # Run function: the module-scoped fixture executes the end-to-end fit once.
    status = result["status"]
    # Check result.
    assert status == "passed"
    assert result["champion_name"] != "prior_baseline"
    assert result["quality_gates"]["status"] == "passed"
    assert result["test_used_for_selection"] is False
    assert result["test_metrics"]["predicted_class_count"] == 3
    assert set(result["test_predictions"].columns) == {
        "record_id",
        "ticker",
        "target_session_date",
        "actual_movement",
        "predicted_movement",
        "prob_down",
        "prob_flat",
        "prob_up",
    }


def test_quality_gates_reject_prior_baseline_champion() -> None:
    """Prepare baseline evidence, run gates, and check rejection."""

    # Prepare data.
    metrics = classification_metrics(
        pd.Series(["Down", "Flat", "Up"] * 5),
        np.array(["Down", "Flat", "Up"] * 5),
    )
    ranking = [
        {"model_name": "prior_baseline", "metrics": metrics},
    ]
    ticker_metrics = [
        {
            "ticker": "AAA",
            "record_count": 15,
            "macro_f1": 1.0,
            "weighted_f1": 1.0,
            "accuracy": 1.0,
            "predicted_class_count": 3,
            "actual_class_support": {},
            "predicted_class_support": {},
        }
    ]
    # Run function and check result.
    with pytest.raises(MovementTrainingError, match="baseline"):
        evaluate_quality_gates(
            "prior_baseline",
            ranking,
            metrics,
            metrics,
            ticker_metrics,
            TrainingConfig(),
        )


def test_quality_gates_reject_missing_prediction_class() -> None:
    """Prepare one-class predictions, run gates, and check rejection."""

    # Prepare data.
    actual = pd.Series(["Down", "Flat", "Up"] * 10)
    weak = classification_metrics(actual, np.array(["Flat"] * len(actual)))
    ranking = [
        {
            "model_name": "balanced_logistic_regression",
            "metrics": {**weak, "macro_f1": 0.50},
        },
        {
            "model_name": "prior_baseline",
            "metrics": {**weak, "macro_f1": 0.30},
        },
    ]
    # Run function and check result.
    with pytest.raises(MovementTrainingError, match="three movement classes"):
        evaluate_quality_gates(
            "balanced_logistic_regression",
            ranking,
            weak,
            weak,
            [],
            TrainingConfig(minimum_test_macro_f1=0.0, minimum_test_weighted_f1=0.0),
        )


def test_per_ticker_metrics_reports_every_ticker() -> None:
    """Prepare predictions, run per-ticker metrics, and check coverage."""

    # Prepare data.
    predictions = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3 + ["BBB"] * 3,
            "actual_movement": ["Down", "Flat", "Up"] * 2,
            "predicted_movement": ["Down", "Flat", "Up"] * 2,
        }
    )
    # Run function.
    result = per_ticker_metrics(predictions)
    # Check result.
    assert [row["ticker"] for row in result] == ["AAA", "BBB"]
    assert all(row["macro_f1"] == 1.0 for row in result)


def test_global_importance_reports_learned_champion(
    strong_training_result: dict[str, object],
) -> None:
    """Prepare the shared champion, run importance, and check signal."""

    # Prepare data.
    learned_pipeline = strong_training_result["champion_pipeline"]
    assert isinstance(learned_pipeline, Pipeline)
    # Run function.
    importance = global_importance(learned_pipeline)
    # Check result.
    assert importance["importance"].gt(0).any()


def test_validation_gate_rejects_before_terminal_or_audit() -> None:
    """Prepare strict development evidence, run gate, and check rejection."""

    # Prepare data: the candidate is learned and stable, but an intentionally
    # impossible validation threshold must stop the protocol before later data.
    metrics = {
        "macro_f1": 0.70,
        "weighted_f1": 0.70,
        "accuracy": 0.70,
        "predicted_class_count": 3,
    }
    ranking = [
        {
            "model_name": "candidate",
            "metrics": metrics,
            "minimum_fold_macro_f1": 0.60,
            "minimum_fold_weighted_f1": 0.60,
            "minimum_fold_predicted_class_count": 3,
        },
        {
            "model_name": "prior_baseline",
            "metrics": {**metrics, "macro_f1": 0.20},
        },
    ]
    config = TrainingConfig(minimum_validation_macro_f1=1.01)
    # Run function.
    report = _validation_gate_report("candidate", ranking, config)
    # Check result.
    assert report["status"] == "failed"
    assert any("macro F1" in item for item in report["failures"])


def test_atomic_json_write_removes_temporary_file(tmp_path: Path) -> None:
    """Prepare an output path, run atomic writing, and check no temp remains."""

    # Prepare data.
    output_path = tmp_path / "artifact.json"
    # Run function.
    write_json(output_path, {"status": "passed"})
    # Check result.
    assert output_path.exists()
    assert list(tmp_path.glob("artifact.json.strike_tmp.*")) == []


def test_cleanup_temporary_outputs_removes_only_controlled_files(
    tmp_path: Path,
) -> None:
    """Prepare controlled temp files, run cleanup, and check unrelated safety."""

    # Prepare data.
    outputs = resolve_outputs(tmp_path)
    controlled = outputs["movement_metrics"].with_name(
        outputs["movement_metrics"].name + ".strike_tmp.123"
    )
    controlled.parent.mkdir(parents=True, exist_ok=True)
    controlled.write_text("partial", encoding="utf-8")
    unrelated = controlled.parent / "unrelated.tmp"
    unrelated.write_text("keep", encoding="utf-8")
    # Run function.
    removed = cleanup_temporary_outputs(tmp_path)
    # Check result.
    assert controlled.relative_to(tmp_path).as_posix() in removed
    assert unrelated.exists()


def test_full_candidate_search_covers_multiple_model_families() -> None:
    """Prepare the full profile, run definitions, and check search diversity."""

    # Prepare data.
    config = TrainingConfig()
    # Run function.
    definitions = candidate_definitions(config)
    # Check result.
    families = {definition.model_family for definition in definitions}
    assert len(definitions) >= 15
    assert {
        "dummy",
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "rbf_svc",
        "sgd_log_loss",
        "stability_soft_vote",
    }.issubset(families)
    assert "calibrated_linear_svc" not in families


def test_failed_audit_gate_returns_complete_diagnostics() -> None:
    """Prepare strict audit evidence, run gates, and check failure report."""

    # Prepare data.
    actual = pd.Series(["Down", "Flat", "Up"] * 5)
    metrics = classification_metrics(actual, actual.to_numpy())
    ranking = [
        {"model_name": "candidate", "metrics": metrics},
        {"model_name": "prior_baseline", "metrics": metrics},
    ]
    ticker_metrics = [
        {
            "ticker": "AAA",
            "record_count": 15,
            "macro_f1": 1.0,
            "weighted_f1": 1.0,
            "accuracy": 1.0,
            "predicted_class_count": 3,
            "actual_class_support": {},
            "predicted_class_support": {},
        }
    ]
    config = TrainingConfig(minimum_test_weighted_f1=1.01)
    # Run function.
    report = evaluate_quality_gates(
        "candidate",
        ranking,
        metrics,
        metrics,
        ticker_metrics,
        config,
        raise_on_failure=False,
    )
    # Check result.
    assert report["status"] == "failed"
    assert report["test_champion_weighted_f1"] == 1.0
    assert any("weighted F1" in item for item in report["failures"])


def _load_runner_module() -> object:
    """Load the project runner so external diagnostic writing can be tested."""

    script_path = Path(__file__).parents[1] / "scripts" / "run_movement_intelligence.py"
    specification = importlib.util.spec_from_file_location(
        "movement_intelligence_test_runner",
        script_path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load runner: {script_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_runner_reads_offset_grid_from_typed_training_config() -> None:
    """Prepare a config, run offset loading, and check dataclass access."""

    # Prepare data.
    runner = _load_runner_module()
    config = TrainingConfig(
        decision_global_offsets=(-0.2, 0.0, 0.3),
    )
    # Run function.
    offsets = runner._allowed_decision_global_offsets(config)
    # Check result.
    assert offsets == {-0.2, 0.0, 0.3}


def test_runner_validates_global_importance_before_intelligence() -> None:
    """Prepare importance, run movement validation, and check the safe frame."""

    # Prepare data.
    runner = _load_runner_module()
    records = [
        {"feature": "one", "importance": 0.4, "method": "native"},
        {"feature": "two", "importance": 0.3, "method": "native"},
        {"feature": "three", "importance": 0.2, "method": "native"},
    ]
    # Run function.
    frame = runner._validate_global_importance_records(records)
    # Check result.
    assert frame["feature"].tolist() == ["one", "two", "three"]
    assert frame["importance"].sum() == pytest.approx(0.9)


def test_runner_rejects_insufficient_global_importance() -> None:
    """Prepare weak importance, run validation, and check fail-closed behavior."""

    # Prepare data.
    runner = _load_runner_module()
    records = [
        {"feature": "one", "importance": 0.4, "method": "native"},
        {"feature": "two", "importance": 0.0, "method": "native"},
        {"feature": "three", "importance": 0.0, "method": "native"},
    ]
    # Run function and check result.
    with pytest.raises(runner.CombinedRunError, match="nonzero drivers"):
        runner._validate_global_importance_records(records)


def test_external_diagnostics_are_owner_only_and_licence_safe(
    tmp_path: Path,
) -> None:
    """Prepare diagnostics, run writing, and check privacy and permissions."""

    # Prepare data.
    runner = _load_runner_module()
    result = {
        "status": "quality_failed",
        "champion_name": "candidate",
        "candidate_results": [
            {
                "model_name": "candidate",
                "model_family": "test",
                "parameters": {"value": 1},
                "status": "passed",
                "metrics": {
                    "accuracy": 0.4,
                    "macro_f1": 0.3,
                    "weighted_f1": 0.4,
                    "predicted_class_count": 3,
                },
                "training_seconds": 1.0,
                "latency_ms_per_record": 2.0,
                "error": None,
            }
        ],
        "validation_gates": {"status": "passed", "failures": []},
        "quality_gates": {"status": "failed", "failures": ["gate"]},
        "test_metrics": {"weighted_f1": 0.39},
        "baseline_test_metrics": {"weighted_f1": 0.35},
        "per_ticker_test_metrics": [],
        "test_used_for_selection": False,
        "test_evaluation_count": 1,
        "training_config": {},
        "runtime_versions": {},
        "test_predictions": pd.DataFrame(
            {
                "record_id": [1],
                "ticker": ["AAA"],
                "target_session_date": ["2020-01-02"],
                "actual_movement": ["Flat"],
                "predicted_movement": ["Flat"],
                "prob_down": [0.2],
                "prob_flat": [0.6],
                "prob_up": [0.2],
            }
        ),
    }
    # Run function.
    summary_path = runner.write_external_diagnostics(
        tmp_path,
        result,
        {"train": {"rows": 10}},
    )
    # Check result.
    assert summary_path is not None and summary_path.exists()
    assert not (tmp_path / "movement_failed_test_predictions.csv").read_text().find("close") >= 0
    for file_path in tmp_path.iterdir():
        assert file_path.stat().st_mode & 0o077 == 0


def test_external_diagnostics_reject_restricted_price_columns(
    tmp_path: Path,
) -> None:
    """Prepare raw prices, run diagnostics, and check redistribution rejection."""

    # Prepare data.
    runner = _load_runner_module()
    result = {
        "status": "quality_failed",
        "candidate_results": [],
        "test_predictions": pd.DataFrame(
            {
                "record_id": [1],
                "ticker": ["AAA"],
                "close": [100.0],
            }
        ),
    }
    # Run function and check result.
    with pytest.raises(runner.CombinedRunError, match="restricted columns"):
        runner.write_external_diagnostics(tmp_path, result, {})


def test_decision_policy_makes_labels_match_saved_probabilities() -> None:
    """Prepare mismatched scores, run policy fitting, and check alignment."""

    # Prepare data: the raw probability winner is deliberately different from
    # an external classifier label, reproducing the real SVC artifact problem.
    from financial_news_intelligence.models.movement_training import (
        fit_decision_policy,
    )

    probabilities = np.asarray(
        [
            [0.42, 0.10, 0.48],
            [0.41, 0.15, 0.44],
            [0.22, 0.55, 0.23],
            [0.20, 0.25, 0.55],
            [0.60, 0.20, 0.20],
            [0.18, 0.22, 0.60],
        ],
        dtype=float,
    )
    actual = pd.Series(["Down", "Down", "Flat", "Up", "Down", "Up"])
    tickers = pd.Series(["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"])
    config = TrainingConfig(
        decision_global_offsets=(-0.3, 0.0, 0.3),
        decision_ticker_offsets=(0.0,),
        decision_ticker_minimum_records=99,
    )

    # Run function: fit the decision rule using validation labels only.
    policy, adjusted, predicted, diagnostics = fit_decision_policy(
        probabilities,
        actual,
        tickers,
        config,
    )

    # Check result: the exported label is always the largest exported
    # probability, and the diagnostic mismatch count remains exactly zero.
    expected = np.asarray(("Down", "Flat", "Up"), dtype=object)[
        adjusted.argmax(axis=1)
    ]
    assert np.array_equal(predicted, expected)
    assert diagnostics["prediction_probability_mismatch_count"] == 0
    assert policy["fit_split"] == "validation"


def test_prior_event_history_excludes_current_reaction() -> None:
    """Prepare event history, run feature creation, and check the shift."""

    # Prepare data: three ordered events have visibly different reaction values.
    from financial_news_intelligence.models.movement_dataset import (
        add_prior_event_history_features,
    )

    events = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "target_session_date": pd.to_datetime(
                ["2019-01-02", "2019-01-03", "2019-01-04"]
            ),
            "reaction_return": [0.10, -0.20, 0.30],
            "movement_label": ["Up", "Down", "Up"],
        }
    )

    # Run function: create rolling history after the mandatory one-row shift.
    result = add_prior_event_history_features(events)

    # Check result: the third row can use 0.10 and -0.20, but never its own
    # current 0.30 reaction value.
    third = result.sort_values("target_session_date").iloc[2]
    assert third["prior_event_return_mean_5"] == pytest.approx(-0.05)
    assert third["prior_up_share_20"] == pytest.approx(0.50)
    assert third["prior_down_share_20"] == pytest.approx(0.50)


def test_market_regime_features_use_shifted_returns() -> None:
    """Prepare ticker prices, run regime features, and check prior-only use."""

    # Prepare data: two tickers share dates but have different closing paths.
    dates = pd.bdate_range("2019-01-02", periods=8)
    rows: list[dict[str, object]] = []
    for ticker, closes in (
        ("AAA", [100, 101, 102, 104, 103, 105, 106, 108]),
        ("BBB", [200, 198, 199, 201, 202, 204, 203, 205]),
    ):
        for date, close in zip(dates, closes, strict=True):
            rows.append(
                {
                    "ticker": ticker,
                    "session_date": date,
                    "close": close,
                    "volume": 1_000_000,
                    "source_provider": "tiingo_eod",
                }
            )

    # Run function: calculate ticker and cross-sectional prior features.
    result = build_prior_price_features(pd.DataFrame(rows))

    # Check result: both tickers share the same market aggregate for one date,
    # while the relative return equals ticker prior return minus that aggregate.
    selected_date = dates[4]
    selected = result[result["session_date"] == selected_date]
    assert selected["prior_market_return_1d"].nunique(dropna=False) == 1
    for _, row in selected.iterrows():
        assert row["prior_relative_return_1d"] == pytest.approx(
            row["prior_return_1d"] - row["prior_market_return_1d"]
        )


def test_saved_prediction_validator_rejects_label_probability_mismatch() -> None:
    """Prepare inconsistent output, run validation, and check rejection."""

    # Prepare data: the saved label says Down while Up has the largest
    # probability, matching the defect observed in the real v3 evidence.
    runner = _load_runner_module()
    predictions = pd.DataFrame(
        {
            "record_id": [1],
            "ticker": ["AAA"],
            "target_session_date": ["2020-01-02"],
            "actual_movement": ["Down"],
            "predicted_movement": ["Down"],
            "prob_down": [0.42],
            "prob_flat": [0.10],
            "prob_up": [0.48],
        }
    )

    # Run function and check result: semantic inconsistency must fail closed.
    with pytest.raises(
        runner.CombinedRunError,
        match="highest saved probability",
    ):
        runner._validate_minimal_predictions(predictions)


def test_production_config_disables_ticker_offsets() -> None:
    """Prepare default config, run policy inspection, and check no ticker tuning."""

    # Prepare data.
    config = TrainingConfig()
    # Run function.
    ticker_offsets_enabled = config.enable_ticker_offsets
    # Check result.
    assert ticker_offsets_enabled is False
    assert config.decision_ticker_offsets == (0.0,)


def test_validation_ranking_prefers_stable_rolling_candidate() -> None:
    """Prepare rolling evidence, run ranking, and check gates precede recency."""

    # Prepare data: candidate A has the best recent score but violates the
    # unchanged weakest-fold weighted-F1 gate. Candidate B is fully eligible.
    results = [
        {
            "model_name": "candidate_a",
            "status": "passed",
            "minimum_fold_macro_f1": 0.29,
            "minimum_fold_weighted_f1": 0.28,
            "minimum_fold_predicted_class_count": 3,
            "latest_fold_weighted_f1": 0.60,
            "recency_weighted_fold_weighted_f1": 0.55,
            "recency_weighted_fold_macro_f1": 0.53,
            "median_fold_weighted_f1": 0.50,
            "median_fold_macro_f1": 0.49,
            "weighted_f1_std": 0.12,
            "metrics": {
                "macro_f1": 0.52,
                "weighted_f1": 0.53,
                "accuracy": 0.53,
            },
        },
        {
            "model_name": "candidate_b",
            "status": "passed",
            "minimum_fold_macro_f1": 0.35,
            "minimum_fold_weighted_f1": 0.40,
            "minimum_fold_predicted_class_count": 3,
            "latest_fold_weighted_f1": 0.48,
            "recency_weighted_fold_weighted_f1": 0.46,
            "recency_weighted_fold_macro_f1": 0.45,
            "median_fold_weighted_f1": 0.46,
            "median_fold_macro_f1": 0.44,
            "weighted_f1_std": 0.03,
            "metrics": {
                "macro_f1": 0.45,
                "weighted_f1": 0.46,
                "accuracy": 0.46,
            },
        },
    ]
    # Run function.
    ranked = rank_validation_results(results)
    # Check result.
    assert ranked[0]["model_name"] == "candidate_b"


def test_validation_ranking_uses_recent_folds_after_stability_gates() -> None:
    """Prepare stable candidates, run ranking, and check recency priority."""

    # Prepare data: both candidates pass unchanged stability gates. Candidate B
    # has the stronger old minimum, while candidate A generalizes better in the
    # later development periods and therefore has the higher fixed recency score.
    common = {
        "status": "passed",
        "minimum_fold_macro_f1": 0.31,
        "minimum_fold_predicted_class_count": 3,
        "median_fold_macro_f1": 0.40,
        "weighted_f1_std": 0.04,
        "metrics": {
            "macro_f1": 0.40,
            "weighted_f1": 0.41,
            "accuracy": 0.41,
        },
    }
    results = [
        {
            **common,
            "model_name": "recent_candidate",
            "minimum_fold_weighted_f1": 0.31,
            "latest_fold_weighted_f1": 0.49,
            "recency_weighted_fold_weighted_f1": 0.45,
            "recency_weighted_fold_macro_f1": 0.43,
            "median_fold_weighted_f1": 0.42,
        },
        {
            **common,
            "model_name": "old_minimum_candidate",
            "minimum_fold_weighted_f1": 0.36,
            "latest_fold_weighted_f1": 0.41,
            "recency_weighted_fold_weighted_f1": 0.40,
            "recency_weighted_fold_macro_f1": 0.39,
            "median_fold_weighted_f1": 0.40,
        },
    ]
    # Run function.
    ranked = rank_validation_results(results)
    # Check result.
    assert ranked[0]["model_name"] == "recent_candidate"


def test_training_records_rolling_protocol_and_known_audit(
    strong_training_result: dict[str, object],
) -> None:
    """Prepare the shared result, run retrieval, and check disclosure."""

    # Prepare data.
    result = strong_training_result
    # Run function.
    protocol = result["evaluation_protocol"]
    # Check result.
    assert protocol == (
        "purged_four_fold_recency_ranking_plus_oof_policy_calibration_"
        "plus_terminal_development_tournament_plus_known_historical_audit"
    )
    assert result["historical_audit_pristine"] is False
    assert result["development_confirmation_pristine"] is False
    assert result["development_confirmation_known_from_prior_run"] is True
    assert result["historical_audit_used_for_selection"] is False
    assert result["historical_audit_evaluation_count"] == 1
    assert 1 <= result["development_confirmation_evaluation_count"] <= 5
    assert result["development_confirmation_used_for_candidate_selection"] is True
    assert len(result["development_confirmation_tournament"]) == (
        result["development_confirmation_evaluation_count"]
    )
    assert result["development_confirmation_gates"]["status"] == "passed"
    assert len(result["rolling_fold_reports"]) == 4
    assert result["decision_policy"]["fit_split"] == "selection_oof"
    assert result["decision_policy"]["ticker_logit_offsets"] == {}
    assert (
        result["decision_policy_calibration"][
            "historical_audit_used_for_selection"
        ]
        is False
    )


def test_external_diagnostics_preserve_complete_failure_tables(
    tmp_path: Path,
) -> None:
    """Prepare full evidence, run diagnostic writing, and check every failure table."""

    # Prepare data.
    runner = _load_runner_module()
    actual = pd.Series(["Down", "Flat", "Up"] * 3)
    predicted = np.asarray(["Down", "Flat", "Up"] * 3, dtype=object)
    metrics = classification_metrics(actual, predicted)
    predictions = pd.DataFrame(
        {
            "record_id": range(1, 10),
            "ticker": ["AAA"] * 9,
            "target_session_date": pd.bdate_range(
                "2020-01-02",
                periods=9,
            ).date.astype(str),
            "actual_movement": actual,
            "predicted_movement": predicted,
            "prob_down": [0.8, 0.1, 0.1] * 3,
            "prob_flat": [0.1, 0.8, 0.1] * 3,
            "prob_up": [0.1, 0.1, 0.8] * 3,
        }
    )
    result = {
        "status": "quality_failed",
        "champion_name": "candidate",
        "candidate_results": [
            {
                "model_name": "candidate",
                "model_family": "test",
                "parameters": {},
                "status": "passed",
                "convergence_status": "converged",
                "convergence_diagnostics": [
                    {
                        "candidate_name": "candidate",
                        "fit_stage": "rolling_fold_1",
                        "estimator_name": "classifier",
                        "estimator_class": "SyntheticClassifier",
                        "configured_max_iter": 100,
                        "observed_iterations": [4],
                        "status": "converged",
                    }
                ],
                "metrics": metrics,
                "minimum_fold_macro_f1": 1.0,
                "minimum_fold_weighted_f1": 1.0,
                "mean_fold_macro_f1": 1.0,
                "median_fold_macro_f1": 1.0,
                "mean_fold_weighted_f1": 1.0,
                "median_fold_weighted_f1": 1.0,
                "macro_f1_std": 0.0,
                "weighted_f1_std": 0.0,
                "minimum_fold_predicted_class_count": 3,
                "fold_metrics": [
                    {
                        "fold_name": "rolling_fold_1",
                        "metrics": metrics,
                        "training_seconds": 0.1,
                        "latency_ms_per_record": 0.1,
                        "train_start_date": "2018-01-01",
                        "train_end_date": "2018-06-30",
                        "validation_start_date": "2018-07-02",
                        "validation_end_date": "2018-09-30",
                        "convergence_status": "converged",
                    }
                ],
            }
        ],
        "decision_policy_candidates": [
            {
                "flat_offset": 0.0,
                "up_offset": 0.0,
                "is_identity": True,
                "qualifies": False,
            }
        ],
        "decision_policy_oof_predictions": predictions.rename(
            columns={
                "predicted_movement": "predicted_movement",
            }
        ).assign(fold_name="rolling_fold_1"),
        "decision_policy_calibration": {
            "status": "identity_fallback",
            "historical_audit_used_for_selection": False,
        },
        "development_confirmation_metrics": metrics,
        "baseline_development_confirmation_metrics": metrics,
        "development_confirmation_gates": {
            "status": "passed",
            "used_for_candidate_selection": True,
            "failures": [],
        },
        "development_confirmation_tournament": [
            {
                "model_name": "candidate",
                "model_family": "test",
                "status": "passed",
                "metrics": metrics,
                "confirmation_gates": {
                    "status": "passed",
                    "used_for_candidate_selection": True,
                    "failures": [],
                },
            }
        ],
        "development_confirmation_evaluation_count": 1,
        "development_confirmation_used_for_candidate_selection": True,
        "development_confirmation_predictions": predictions.copy(),
        "test_metrics": metrics,
        "historical_audit_metrics": metrics,
        "baseline_test_metrics": metrics,
        "per_ticker_test_metrics": per_ticker_metrics(predictions),
        "test_predictions": predictions,
        "test_used_for_selection": False,
        "test_evaluation_count": 1,
        "historical_audit_evaluation_count": 1,
        "historical_audit_pristine": False,
        "development_confirmation_pristine": False,
        "development_confirmation_known_from_prior_run": True,
        "evaluation_protocol": "rolling",
    }
    # Run function.
    runner.write_external_diagnostics(tmp_path, result, {})
    # Check result.
    expected_files = {
        "movement_candidate_validation.csv",
        "movement_candidate_fold_metrics.csv",
        "movement_convergence_diagnostics.csv",
        "movement_policy_calibration.csv",
        "movement_policy_oof_predictions.csv",
        "movement_development_confirmation_metrics.csv",
        "movement_development_confirmation_predictions.csv",
        "movement_development_confirmation_tournament.csv",
        "movement_diagnostic_summary.json",
        "movement_failed_test_predictions.csv",
        "movement_test_metrics.csv",
        "movement_test_class_metrics.csv",
        "movement_test_confusion_matrix.csv",
        "movement_test_ticker_metrics.csv",
        "movement_diagnostic_manifest.json",
    }
    assert expected_files == {path.name for path in tmp_path.iterdir()}
    assert all((path.stat().st_mode & 0o077) == 0 for path in tmp_path.iterdir())

def test_default_logistic_solver_avoids_liblinear() -> None:
    """Prepare default config, run classifier build, and check LBFGS use."""

    # Prepare data.
    config = TrainingConfig()
    definition = CandidateDefinition(
        "balanced_logistic_c_1_0",
        "logistic_regression",
        {"C": 1.0},
    )
    # Run function.
    classifier = _classifier_for_definition(definition, config)
    # Check result.
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 5000
    assert config.convergence_warnings_are_errors is True


def test_soft_vote_has_fixed_diverse_development_weights() -> None:
    """Prepare the three-family vote, run build, and check frozen design."""

    # Prepare data: component names and weights are declared before terminal or
    # historical-audit labels are available.
    config = TrainingConfig(forest_estimators=10)
    definition = next(
        item
        for item in candidate_definitions(config)
        if item.model_name == "stability_soft_vote_rf_et_sgd"
    )
    # Run function.
    classifier = _classifier_for_definition(definition, config)
    # Check result.
    assert classifier.weights == [2.0, 1.0, 1.0]
    assert [name for name, _ in classifier.estimators] == [
        "component_1",
        "component_2",
        "component_3",
    ]
    assert definition.parameters["components"] == [
        "balanced_random_forest_leaf_4_depth_12",
        "balanced_extra_trees_leaf_4_depth_12",
        "balanced_sgd_log_loss_alpha_0_0001",
    ]


def test_terminal_development_tournament_is_purged_from_selection() -> None:
    """Prepare development dates, run split, and check isolation and use."""

    # Prepare data.
    table = make_split_table()
    development = table[table["split"].isin({"train", "validation"})]
    config = TrainingConfig(
        development_confirmation_ratio=0.15,
        minimum_confirmation_dates=30,
    )
    # Run function.
    selection, confirmation, report = _split_development_confirmation(
        development,
        config,
    )
    # Check result.
    assert selection["target_session_date"].max() < (
        confirmation["target_session_date"].min()
    )
    assert report["used_for_candidate_selection"] is True
    assert report["pristine"] is False
    assert report["known_from_prior_v6_evidence"] is True
    assert len(report["purged_dates"]) == config.rolling_purge_dates


def test_convergence_warning_rejects_candidate() -> None:
    """Prepare warning estimator, run fit, and check fail-closed rejection."""

    # Prepare data.
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "movement_label": ["Down", "Flat", "Up"],
        }
    )
    model = Pipeline(
        [
            ("preprocessor", "passthrough"),
            ("classifier", WarningClassifier()),
        ]
    )
    definition = CandidateDefinition(
        "warning_candidate",
        "synthetic",
        {},
    )
    # Run function and check result.
    with pytest.raises(MovementTrainingError, match="failed to converge"):
        _fit_candidate(
            model,
            frame,
            ["x"],
            {},
            definition,
            "test_fold",
            TrainingConfig(),
        )


def test_openmp_classifier_distinguishes_intel_and_llvm() -> None:
    """Prepare native rows, run classification, and check conflict evidence."""

    # Prepare data.
    script_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "check_movement_runtime.py"
    )
    specification = importlib.util.spec_from_file_location(
        "movement_runtime_test_module",
        script_path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    pools = [
        {
            "user_api": "openmp",
            "filepath": "/tmp/libiomp5.dylib",
            "prefix": "libiomp",
        },
        {
            "user_api": "openmp",
            "filepath": "/tmp/libomp.dylib",
            "prefix": "libomp",
        },
    ]
    # Run function.
    report = module.runtime_status_from_pools(pools)
    # Check result.
    assert report["status"] == "conflict"
    assert report["openmp_families"] == ["intel", "llvm"]


def test_external_diagnostics_record_convergence_status(
    tmp_path: Path,
) -> None:
    """Prepare convergence evidence, run diagnostics, and check saved rows."""

    # Prepare data.
    runner = _load_runner_module()
    result = {
        "status": "validation_failed",
        "candidate_results": [
            {
                "model_name": "candidate",
                "model_family": "logistic_regression",
                "status": "failed",
                "convergence_status": "failed",
                "convergence_diagnostics": [],
                "error": "failed to converge",
            }
        ],
    }
    # Run function.
    runner.write_external_diagnostics(tmp_path, result, {})
    saved = pd.read_csv(tmp_path / "movement_candidate_validation.csv")
    # Check result.
    assert saved.loc[0, "convergence_status"] == "failed"
    assert (
        tmp_path / "movement_convergence_diagnostics.csv"
    ).exists()


def test_historical_audit_labels_cannot_change_candidate_selection() -> None:
    """Prepare changed audit labels, run split creation, and check isolation."""

    # Prepare data: alter only historical-audit labels. Development rows and
    # dates must remain identical because all selection folds are built before
    # the audit block is ever passed to prediction or quality-gate functions.
    original = make_split_table()
    altered = original.copy()
    audit_mask = altered["split"].eq("test")
    altered.loc[audit_mask, "movement_label"] = np.resize(
        np.asarray(["Up", "Down", "Flat"], dtype=object),
        int(audit_mask.sum()),
    )
    config = focused_training_config()
    original_development = original[
        original["split"].isin({"train", "validation"})
    ].copy()
    altered_development = altered[
        altered["split"].isin({"train", "validation"})
    ].copy()

    # Run function.
    original_selection, original_terminal, original_report = (
        _split_development_confirmation(original_development, config)
    )
    altered_selection, altered_terminal, altered_report = (
        _split_development_confirmation(altered_development, config)
    )
    original_folds = _rolling_validation_frames(original_selection, config)
    altered_folds = _rolling_validation_frames(altered_selection, config)

    # Check result: changed audit labels cannot alter any row, date, or report
    # used by rolling ranking or the terminal development tournament.
    pd.testing.assert_frame_equal(original_selection, altered_selection)
    pd.testing.assert_frame_equal(original_terminal, altered_terminal)
    assert original_report == altered_report
    assert len(original_folds) == len(altered_folds) == 4
    for original_fold, altered_fold in zip(original_folds, altered_folds):
        assert original_fold[0] == altered_fold[0]
        pd.testing.assert_frame_equal(original_fold[1], altered_fold[1])
        pd.testing.assert_frame_equal(original_fold[2], altered_fold[2])
        assert original_fold[3] == altered_fold[3]


def make_policy_oof_predictions() -> pd.DataFrame:
    """Prepare two-fold OOF probabilities with stable Up overprediction."""

    rows: list[dict[str, object]] = []
    templates = [
        ("Down", [0.40, 0.20, 0.40]),
        ("Flat", [0.30, 0.31, 0.39]),
        ("Up", [0.25, 0.25, 0.50]),
    ]
    for fold_name in ("rolling_fold_1", "rolling_fold_2"):
        for repeat in range(12):
            for actual, probabilities in templates:
                rows.append(
                    {
                        "fold_name": fold_name,
                        "ticker": "AAA" if repeat % 2 == 0 else "BBB",
                        "actual_movement": actual,
                        "prob_down": probabilities[0],
                        "prob_flat": probabilities[1],
                        "prob_up": probabilities[2],
                    }
                )
    return pd.DataFrame(rows)


def test_stable_global_policy_uses_oof_evidence_only() -> None:
    """Prepare OOF scores, run policy fitting, and check audited isolation."""

    # Prepare data.
    oof = make_policy_oof_predictions()
    config = TrainingConfig(
        decision_global_offsets=(-0.2, -0.1, 0.0, 0.1, 0.2),
        decision_policy_minimum_macro_improvement=0.001,
    )
    # Run function.
    policy, report, candidates, predictions = fit_stable_global_policy(
        oof,
        config,
    )
    # Check result.
    assert policy["fit_split"] == "selection_oof"
    assert report["historical_audit_used_for_selection"] is False
    assert report["candidate_policy_count"] == 25
    assert len(candidates) == 25
    assert len(predictions) == len(oof)
    assert "adjusted_prob_up" in predictions.columns


def test_stable_global_policy_improves_supported_class_bias() -> None:
    """Prepare biased OOF scores, run calibration, and check improvement."""

    # Prepare data.
    oof = make_policy_oof_predictions()
    config = TrainingConfig(
        decision_global_offsets=(-0.2, -0.1, 0.0, 0.1, 0.2),
        decision_policy_minimum_macro_improvement=0.001,
    )
    # Run function.
    _, report, _, _ = fit_stable_global_policy(oof, config)
    # Check result.
    assert report["status"] == "adjusted"
    assert (
        report["selected_metrics"]["macro_f1"]
        > report["identity_metrics"]["macro_f1"]
    )
    assert (
        report["selected_metrics"]["weighted_f1"]
        >= report["identity_metrics"]["weighted_f1"]
    )


def test_stable_global_policy_falls_back_without_gain() -> None:
    """Prepare perfect OOF scores, run calibration, and check identity fallback."""

    # Prepare data.
    rows = []
    for fold_name in ("rolling_fold_1", "rolling_fold_2"):
        for actual in ("Down", "Flat", "Up"):
            probability = {
                "Down": [0.9, 0.05, 0.05],
                "Flat": [0.05, 0.9, 0.05],
                "Up": [0.05, 0.05, 0.9],
            }[actual]
            rows.append(
                {
                    "fold_name": fold_name,
                    "ticker": "AAA",
                    "actual_movement": actual,
                    "prob_down": probability[0],
                    "prob_flat": probability[1],
                    "prob_up": probability[2],
                }
            )
    oof = pd.DataFrame(rows)
    # Run function.
    policy, report, _, _ = fit_stable_global_policy(
        oof,
        TrainingConfig(
            decision_global_offsets=(-0.1, 0.0, 0.1),
        ),
    )
    # Check result.
    assert report["status"] == "identity_fallback"
    assert policy["global_logit_offsets"] == {
        "Down": 0.0,
        "Flat": 0.0,
        "Up": 0.0,
    }


def test_stable_global_policy_rejects_material_single_fold_damage() -> None:
    """Prepare uneven folds, run policy fitting, and check per-fold protection."""

    # Prepare data: the adjustment helps a large biased fold but damages a
    # smaller fold whose identity decisions are already correct.
    rows: list[dict[str, object]] = []
    for _ in range(20):
        for actual, probabilities in (
            ("Down", [0.45, 0.20, 0.35]),
            ("Flat", [0.30, 0.31, 0.39]),
            ("Up", [0.20, 0.25, 0.55]),
        ):
            rows.append(
                {
                    "fold_name": "rolling_fold_1",
                    "ticker": "AAA",
                    "actual_movement": actual,
                    "prob_down": probabilities[0],
                    "prob_flat": probabilities[1],
                    "prob_up": probabilities[2],
                }
            )
    for _ in range(3):
        for actual, probabilities in (
            ("Down", [0.70, 0.15, 0.15]),
            ("Flat", [0.15, 0.70, 0.15]),
            ("Up", [0.15, 0.15, 0.70]),
            ("Down", [0.36, 0.34, 0.30]),
            ("Flat", [0.30, 0.36, 0.34]),
            ("Up", [0.30, 0.32, 0.38]),
        ):
            rows.append(
                {
                    "fold_name": "rolling_fold_2",
                    "ticker": "BBB",
                    "actual_movement": actual,
                    "prob_down": probabilities[0],
                    "prob_flat": probabilities[1],
                    "prob_up": probabilities[2],
                }
            )
    oof = pd.DataFrame(rows)
    config = TrainingConfig(
        decision_global_offsets=(-0.2, 0.0, 0.2),
        decision_policy_minimum_macro_improvement=0.001,
        decision_policy_maximum_fold_weighted_drop=0.015,
    )

    # Run function.
    _, _, candidates, _ = fit_stable_global_policy(oof, config)

    # Check result: pooled gain cannot hide a material decline in one fold.
    damaged = next(
        row
        for row in candidates
        if row["flat_offset"] == 0.2 and row["up_offset"] == -0.2
    )
    assert damaged["macro_f1_improvement"] > 0.0
    assert damaged["maximum_fold_weighted_f1_drop"] > 0.015
    assert damaged["qualifies"] is False


def test_recency_tree_candidates_use_training_dates_only() -> None:
    """Prepare dated rows, run weighting, and check deterministic decay."""

    # Prepare data: newer rows should receive larger normalized weights.
    frame = pd.DataFrame(
        {
            "target_session_date": [
                "2020-01-01",
                "2020-12-31",
                "2021-12-31",
            ],
            "ticker": ["AAA", "AAA", "AAA"],
        }
    )
    definition = CandidateDefinition(
        "recent_tree",
        "random_forest",
        {"sample_weight_mode": "recency_365"},
    )
    # Run function.
    weights = _candidate_sample_weight(frame, definition)
    # Check result.
    assert weights is not None
    assert weights[0] < weights[1] < weights[2]
    assert np.isclose(weights.mean(), 1.0)


def test_fold_summary_uses_fixed_chronological_recency_weights() -> None:
    """Prepare fold metrics, run summary, and check later-fold weighting."""

    # Prepare data: chronological weights are 1, 2, 3, and 4.
    fold_metrics = [
        {
            "metrics": {
                "macro_f1": value,
                "weighted_f1": value,
                "predicted_class_count": 3,
            }
        }
        for value in (0.30, 0.35, 0.45, 0.50)
    ]
    # Run function.
    summary = _candidate_fold_summary(fold_metrics, TrainingConfig())
    # Check result.
    expected = np.average([0.30, 0.35, 0.45, 0.50], weights=[1, 2, 3, 4])
    assert np.isclose(summary["recency_weighted_fold_weighted_f1"], expected)
    assert summary["latest_fold_weighted_f1"] == 0.50


def test_terminal_shortlist_enforces_family_diversity() -> None:
    """Prepare ranked candidates, run shortlist, and check family limits."""

    # Prepare data: three tree rows lead, but only two may enter. All rows pass
    # the unchanged rolling gates used by shortlist admission.
    rows = [
        {
            "model_name": "prior_baseline",
            "model_family": "dummy",
            "status": "passed",
            "metrics": {
                "macro_f1": 0.20,
                "weighted_f1": 0.25,
                "accuracy": 0.25,
                "predicted_class_count": 1,
            },
        }
    ]
    for name, family, score in (
        ("tree_a", "random_forest", 0.50),
        ("tree_b", "random_forest", 0.49),
        ("tree_c", "random_forest", 0.48),
        ("linear_a", "sgd_log_loss", 0.47),
        ("linear_b", "logistic_regression", 0.46),
    ):
        rows.append(
            {
                "model_name": name,
                "model_family": family,
                "status": "passed",
                "minimum_fold_macro_f1": 0.31,
                "minimum_fold_weighted_f1": 0.32,
                "minimum_fold_predicted_class_count": 3,
                "recency_weighted_fold_weighted_f1": score,
                "latest_fold_weighted_f1": score,
                "median_fold_weighted_f1": score,
                "recency_weighted_fold_macro_f1": score,
                "median_fold_macro_f1": score,
                "weighted_f1_std": 0.02,
                "metrics": {
                    "macro_f1": score,
                    "weighted_f1": score,
                    "accuracy": score,
                    "predicted_class_count": 3,
                },
            }
        )
    config = TrainingConfig(
        terminal_shortlist_size=4,
        terminal_shortlist_max_per_family=2,
    )
    # Run function.
    shortlist = select_terminal_shortlist(rows, config)
    # Check result.
    assert [row["model_name"] for row in shortlist] == [
        "tree_a",
        "tree_b",
        "linear_a",
        "linear_b",
    ]


def test_terminal_tournament_selects_best_passing_weighted_f1() -> None:
    """Prepare terminal results, run ranking, and check failed rows excluded."""

    # Prepare data.
    rows = [
        {
            "model_name": "failed_high_score",
            "status": "gate_failed",
            "metrics": {"weighted_f1": 0.60, "macro_f1": 0.60},
            "confirmation_gates": {"status": "failed"},
        },
        {
            "model_name": "passing_a",
            "status": "passed",
            "metrics": {"weighted_f1": 0.45, "macro_f1": 0.44},
            "confirmation_gates": {"status": "passed"},
            "recency_weighted_fold_weighted_f1": 0.42,
            "latest_fold_weighted_f1": 0.44,
            "minimum_fold_weighted_f1": 0.31,
        },
        {
            "model_name": "passing_b",
            "status": "passed",
            "metrics": {"weighted_f1": 0.47, "macro_f1": 0.42},
            "confirmation_gates": {"status": "passed"},
            "recency_weighted_fold_weighted_f1": 0.41,
            "latest_fold_weighted_f1": 0.43,
            "minimum_fold_weighted_f1": 0.32,
        },
    ]
    # Run function.
    ranking = rank_confirmation_tournament(rows)
    # Check result.
    assert [row["model_name"] for row in ranking] == [
        "passing_b",
        "passing_a",
    ]


def test_default_v8_protocol_preserves_quality_gates() -> None:
    """Prepare default config, run inspection, and check unchanged gates."""

    # Prepare data.
    config = TrainingConfig()
    # Run function.
    thresholds = {
        "validation_macro": config.minimum_validation_macro_f1,
        "test_macro": config.minimum_test_macro_f1,
        "test_weighted": config.minimum_test_weighted_f1,
        "confirmation_dates": config.minimum_confirmation_dates,
    }
    # Check result.
    assert thresholds == {
        "validation_macro": 0.34,
        "test_macro": 0.30,
        "test_weighted": 0.40,
        "confirmation_dates": 60,
    }
    assert config.rolling_validation_folds == 4
    assert config.rolling_recency_weight_power == 1.0
    assert config.development_confirmation_ratio == 0.09
    assert config.terminal_shortlist_size == 5
    assert config.terminal_shortlist_max_per_family == 2
