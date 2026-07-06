"""Focused tests for explainability, retrieval, scenarios, and provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_news_intelligence.intelligence.historical_intelligence import (
    HistoricalIntelligenceError,
    company_context,
    earlier_only_matches,
    sentiment_phrases,
)
from financial_news_intelligence.intelligence.investment_scenarios import (
    InvestmentScenarioError,
    build_scenarios,
)
from financial_news_intelligence.intelligence.movement_explainability import (
    MovementExplainabilityError,
    global_drivers,
    local_perturbation_drivers,
)
from financial_news_intelligence.intelligence.provenance import (
    ProvenanceError,
    build_provenance_report,
)
from financial_news_intelligence.models.movement_dataset import (
    SplitConfig,
    assign_chronological_splits,
    build_model_table,
    feature_columns,
)
from financial_news_intelligence.models.movement_training import (
    TrainingConfig,
    global_importance,
    train_and_evaluate,
)
from test_movement_pipeline import make_foundation_frames


def make_event_history() -> pd.DataFrame:
    """Prepare reference events with all classes, sentiments, and tickers."""

    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2015-01-02", periods=180)
    tickers = ("AAA", "BBB", "CCC")
    for ticker_index, ticker in enumerate(tickers):
        for index, session_date in enumerate(dates):
            movement = ("Down", "Flat", "Up")[(index + ticker_index) % 3]
            sentiment = {
                "Down": "Bearish",
                "Flat": "Neutral",
                "Up": "Bullish",
            }[movement]
            phrase = {
                "Down": "risk decline weak loss",
                "Flat": "stable unchanged ordinary report",
                "Up": "growth gain strong expansion",
            }[movement]
            rows.append(
                {
                    "article_id": f"{ticker}-{index}",
                    "ticker": ticker,
                    "company": (
                        f"{ticker} Corporation"
                        if index % 7
                        else f"{ticker} Corp"
                    ),
                    "target_session_date": session_date,
                    "text": f"{ticker} {phrase} filing {index % 5}",
                    "source_name": "SEC EDGAR",
                    "source_url": f"https://sec.example/{ticker}/{index}",
                    "sentiment_label": sentiment,
                    "prob_bearish": {
                        "Down": 0.80,
                        "Flat": 0.20,
                        "Up": 0.10,
                    }[movement],
                    "prob_neutral": {
                        "Down": 0.10,
                        "Flat": 0.60,
                        "Up": 0.10,
                    }[movement],
                    "prob_bullish": {
                        "Down": 0.10,
                        "Flat": 0.20,
                        "Up": 0.80,
                    }[movement],
                    "movement_label": movement,
                    "reaction_return": {
                        "Down": -0.02,
                        "Flat": 0.0,
                        "Up": 0.02,
                    }[movement],
                }
            )
    return pd.DataFrame(rows)


def make_predictions() -> pd.DataFrame:
    """Prepare two historical-audit predictions after the reference window."""

    return pd.DataFrame(
        {
            "record_id": [1, 2],
            "ticker": ["AAA", "BBB"],
            "target_session_date": ["2016-01-20", "2016-01-21"],
            "actual_movement": ["Up", "Down"],
            "predicted_movement": ["Up", "Down"],
            "prob_down": [0.10, 0.75],
            "prob_flat": [0.15, 0.15],
            "prob_up": [0.75, 0.10],
        }
    )


def make_query_events() -> pd.DataFrame:
    """Prepare SEC text mapped to each test prediction date."""

    return pd.DataFrame(
        {
            "article_id": ["query-a", "query-b"],
            "ticker": ["AAA", "BBB"],
            "target_session_date": ["2016-01-20", "2016-01-21"],
            "text": ["AAA growth gain filing", "BBB risk decline filing"],
            "source_url": ["https://sec.example/query/a", "https://sec.example/query/b"],
            "movement_label": ["Up", "Down"],
            "reaction_return": [0.02, -0.02],
        }
    )


def make_fitted_model() -> tuple[
    object,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
]:
    """Prepare a fitted champion and aligned saved test predictions."""

    news, prices = make_foundation_frames(330)
    model_table, _ = assign_chronological_splits(
        build_model_table(news, prices),
        SplitConfig(
            train_ratio=0.82,
            validation_ratio=0.14,
            test_ratio=0.04,
        ),
    )
    numeric, categorical, text_features = feature_columns(model_table)
    result = train_and_evaluate(
        model_table,
        numeric,
        categorical,
        text_features,
        TrainingConfig(
            forest_estimators=30,
            permutation_repeats=2,
            focused_test_mode=True,
        ),
    )
    return (
        result["champion_pipeline"],
        model_table,
        result["test_predictions"],
        numeric,
        categorical,
        text_features,
    )


def test_global_drivers_require_nonzero_features() -> None:
    """Prepare zero importance, run ranking, and check fail-closed behavior."""

    # Prepare data.
    importance = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "importance": [0.0, 0.0, 0.0],
            "method": ["test"] * 3,
        }
    )
    # Run function and check result.
    with pytest.raises(MovementExplainabilityError, match="nonzero"):
        global_drivers(importance)


def test_global_drivers_rank_meaningful_features() -> None:
    """Prepare importance, run ranking, and check descending order."""

    # Prepare data.
    importance = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "importance": [0.1, 0.5, 0.2],
            "method": ["test"] * 3,
        }
    )
    # Run function.
    result = global_drivers(importance, top_n=3)
    # Check result.
    assert result["feature"].tolist() == ["b", "c", "a"]
    assert result["rank"].tolist() == [1, 2, 3]


def test_local_drivers_cover_every_test_record() -> None:
    """Prepare a champion, run local explanations, and check full coverage."""

    # Prepare data.
    champion, table, predictions, numeric, categorical, text_features = make_fitted_model()
    # Run function.
    result = local_perturbation_drivers(
        champion,
        table,
        predictions,
        numeric,
        categorical,
        text_features,
    )
    # Check result.
    counts = result.groupby("record_id").size()
    assert set(counts.index) == set(predictions["record_id"])
    assert counts.eq(5).all()
    assert result.groupby("record_id")["absolute_effect"].max().gt(0).all()


def test_local_drivers_use_reference_scope() -> None:
    """Prepare a champion, run local explanations, and check scope markers."""

    # Prepare data.
    champion, table, predictions, numeric, categorical, text_features = make_fitted_model()
    # Run function.
    result = local_perturbation_drivers(
        champion,
        table,
        predictions,
        numeric,
        categorical,
        text_features,
    )
    # Check result.
    assert set(result["reference_scope"]) == {"train_validation_only"}


def test_sentiment_phrases_cover_all_three_classes() -> None:
    """Prepare reference text, run phrase extraction, and check class coverage."""

    # Prepare data.
    events = make_event_history()
    # Run function.
    result = sentiment_phrases(events, top_n_per_class=5)
    # Check result.
    counts = result.groupby("sentiment_label").size()
    assert set(counts.index) == {"Bearish", "Neutral", "Bullish"}
    assert counts.ge(3).all()
    assert set(result["method"]) == {
        "reference_only_class_mean_tfidf_difference"
    }
    assert not result["probability_fallback_used"].any()


def test_sentiment_phrases_reject_empty_vocabulary() -> None:
    """Prepare stop-word text, run phrase extraction, and check clear failure."""

    # Prepare data.
    events = make_event_history().head(9).copy()
    events["text"] = "the and or"
    # Run function and check result.
    with pytest.raises(HistoricalIntelligenceError, match="vocabulary"):
        sentiment_phrases(events)


def test_sentiment_phrases_use_probabilities_for_missing_hard_class() -> None:
    """Prepare missing hard labels, run fallback, and check three-class output."""

    # Prepare data: remove Neutral hard labels while preserving verified soft
    # probabilities that were already present on the reference events.
    events = make_event_history()
    events.loc[events["sentiment_label"] == "Neutral", "sentiment_label"] = (
        "Bullish"
    )

    # Run function.
    result = sentiment_phrases(events, top_n_per_class=5)

    # Check result: the absent hard class is represented only through explicit
    # probability-weighted reference evidence, never through fabricated labels.
    counts = result.groupby("sentiment_label", observed=True).size()
    neutral_rows = result[result["sentiment_label"] == "Neutral"]
    assert set(counts.index) == {"Bearish", "Neutral", "Bullish"}
    assert counts.ge(3).all()
    assert neutral_rows["hard_label_record_count"].eq(0).all()
    assert neutral_rows["effective_reference_weight"].gt(0).all()
    assert neutral_rows["probability_fallback_used"].all()
    assert set(result["method"]) == {
        "reference_only_probability_weighted_tfidf_difference"
    }


def test_sentiment_phrases_reject_invalid_probability_fallback() -> None:
    """Prepare invalid soft probabilities, run fallback, and check rejection."""

    # Prepare data: remove one hard class and corrupt one saved probability row.
    events = make_event_history()
    events.loc[events["sentiment_label"] == "Neutral", "sentiment_label"] = (
        "Bullish"
    )
    events.loc[events.index[0], "prob_neutral"] = 0.95

    # Run function and check result.
    with pytest.raises(HistoricalIntelligenceError, match="sum to one"):
        sentiment_phrases(events)


def test_historical_matches_are_reference_only_and_earlier() -> None:
    """Prepare history and queries, run matching, and check time boundaries."""

    # Prepare data.
    reference = make_event_history()
    predictions = make_predictions()
    queries = make_query_events()
    # Run function.
    result = earlier_only_matches(reference, queries, predictions)
    # Check result.
    assert set(result["record_id"]) == {1, 2}
    assert result.groupby("record_id").size().eq(5).all()
    assert (
        pd.to_datetime(result["historical_session_date"])
        < pd.to_datetime(result["query_session_date"])
    ).all()
    assert set(result["candidate_scope"]) == {"train_validation_reference_only"}


def test_historical_matches_reject_missing_ticker_history() -> None:
    """Prepare missing history, run matching, and check coverage rejection."""

    # Prepare data.
    reference = make_event_history().query("ticker != 'BBB'")
    predictions = make_predictions()
    queries = make_query_events()
    # Run function and check result.
    with pytest.raises(HistoricalIntelligenceError, match="same-ticker"):
        earlier_only_matches(reference, queries, predictions)


def test_company_context_handles_name_variants() -> None:
    """Prepare name variants, run context, and check one row per ticker."""

    # Prepare data.
    events = make_event_history()
    # Run function.
    result = company_context(events, {"AAA", "BBB", "CCC"})
    # Check result.
    assert len(result) == 3
    assert not result["ticker"].duplicated().any()
    assert result["company_name_variants"].ge(2).all()
    assert "no invented fundamentals" in result.iloc[0]["limitation"].lower()


def test_company_context_rejects_missing_ticker() -> None:
    """Prepare incomplete coverage, run context, and check rejection."""

    # Prepare data.
    events = make_event_history().query("ticker != 'CCC'")
    # Run function and check result.
    with pytest.raises(HistoricalIntelligenceError, match="coverage"):
        company_context(events, {"AAA", "BBB", "CCC"})


def test_scenarios_cover_every_prediction() -> None:
    """Prepare reference reactions, run scenarios, and check one-row coverage."""

    # Prepare data.
    events = make_event_history()
    predictions = make_predictions()
    # Run function.
    result = build_scenarios(events, predictions)
    # Check result.
    assert set(result["record_id"]) == {1, 2}
    assert result["record_id"].is_unique
    assert (result["downside_return"] <= result["upside_return"]).all()
    assert result["disclaimer"].str.contains("not investment advice").all()


def test_scenarios_reject_invalid_probabilities() -> None:
    """Prepare invalid probabilities, run scenarios, and check rejection."""

    # Prepare data.
    predictions = make_predictions()
    predictions.loc[0, "prob_up"] = 0.9
    # Run function and check result.
    with pytest.raises(InvestmentScenarioError, match="sum to one"):
        build_scenarios(make_event_history(), predictions)


def test_scenarios_reject_insufficient_same_ticker_history() -> None:
    """Prepare short history, run scenarios, and check minimum coverage gate."""

    # Prepare data.
    events = make_event_history().groupby("ticker").head(5)
    # Run function and check result.
    with pytest.raises(InvestmentScenarioError, match="same-ticker"):
        build_scenarios(events, make_predictions())


def test_provenance_requires_passed_quality_gates() -> None:
    """Prepare failed quality evidence, run provenance, and check rejection."""

    # Prepare data.
    movement = {
        "status": "trained_and_evaluated",
        "quality_gates": {"status": "failed"},
    }
    # Run function and check result.
    with pytest.raises(ProvenanceError, match="quality"):
        build_provenance_report({"status": "foundation_verified"}, movement)


def test_provenance_records_licence_and_no_deployment() -> None:
    """Prepare verified summaries, run provenance, and check licence gates."""

    # Prepare data.
    movement = {
        "status": "trained_and_evaluated",
        "quality_champion": "balanced_logistic_regression",
        "quality_gates": {"status": "passed"},
        "random_seed": 42,
        "runtime_versions": {"python": "3.10.9"},
        "training_config": {"random_seed": 42},
        "foundation_manifest_sha256": "abc",
    }
    # Run function.
    result = build_provenance_report(
        {"status": "foundation_verified"},
        movement,
    )
    # Check result.
    licence = result["licence_boundary"]
    assert licence["raw_tiingo_values_publicly_redistributable"] is False
    assert result["deployment_changed"] is False
    assert "not financial" in result["mandatory_disclaimer"].lower()


def test_model_native_importance_supports_global_driver_pipeline() -> None:
    """Prepare a fitted model, run importance and ranking, and check evidence."""

    # Prepare data.
    champion, table, _, numeric, categorical, text_features = make_fitted_model()
    # Run function.
    validation = table[table["split"] == "validation"]
    result = global_drivers(
        global_importance(
            champion,
            validation,
            numeric + categorical + text_features,
            TrainingConfig(
                permutation_repeats=2,
                focused_test_mode=True,
            ),
        )
    )
    # Check result.
    assert result["importance"].gt(0).sum() >= 3
