"""Diagnose, improve, select, gate, and evaluate movement classifiers.

Purpose
-------
Search a deterministic set of learned classifiers using development evidence.
Purged expanding folds rank candidates inside an early development block.
A pre-registered diverse shortlist then competes on a purged terminal
development block before the known historical audit is entered exactly once.

Inputs
------
The input table comes from ``movement_dataset`` and has one row per ticker and
target session. Numeric, categorical, and SEC filing-text feature lists are
explicitly supplied. The verified Down, Flat, and Up label is the target.

Outputs and downstream use
--------------------------
The returned mapping contains candidate diagnostics, validation gates, the
fitted champion, model-agnostic validation importance, historical-audit
predictions, per-class and per-ticker metrics, runtime versions, and final
quality-gate evidence. The runner persists diagnostics even when the model is
rejected, while explainability starts only after every gate passes.

Limitations
-----------
Candidate search improves the chance of finding a useful model but cannot
promise that the historical signal is strong enough. Quality thresholds are
never reduced automatically after failure.
"""

from __future__ import annotations

import json
import math
import platform
from itertools import product
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from financial_news_intelligence.models.movement_dataset import LABEL_ORDER


class MovementTrainingError(RuntimeError):
    """Raised when fitting, selection, prediction, or evidence is unsafe."""


@dataclass(frozen=True)
class TrainingConfig:
    """Store deterministic search settings and unchanged quality gates.

    Historical-audit thresholds remain unchanged. V8 fields control
    recency-aware rolling selection, a pre-registered terminal development
    tournament, OOF policy calibration, convergence enforcement, and the
    non-Liblinear solver.
    """

    random_seed: int = 42
    logistic_max_iterations: int = 5000
    logistic_solver: str = "lbfgs"
    logistic_tolerance: float = 1e-4
    forest_estimators: int = 500
    text_max_features: int = 750
    permutation_repeats: int = 8
    minimum_validation_macro_f1: float = 0.34
    minimum_validation_improvement: float = 0.015
    minimum_test_macro_f1: float = 0.30
    minimum_test_weighted_f1: float = 0.40
    minimum_class_support: int = 5
    minimum_predicted_classes: int = 3
    per_ticker_minimum_records: int = 8
    minimum_per_ticker_macro_f1: float = 0.10
    probability_tolerance: float = 1e-6
    decision_global_offsets: tuple[float, ...] = (
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
    )
    decision_ticker_offsets: tuple[float, ...] = (0.0,)
    decision_ticker_minimum_records: int = 12
    decision_ticker_minimum_improvement: float = 0.015
    decision_policy_minimum_macro_improvement: float = 0.01
    decision_policy_minimum_weighted_improvement: float = 0.0
    decision_policy_maximum_fold_weighted_drop: float = 0.015
    decision_policy_minimum_improved_fold_ratio: float = 0.50
    enable_ticker_offsets: bool = False
    rolling_validation_folds: int = 4
    rolling_initial_train_ratio: float = 0.55
    rolling_purge_dates: int = 1
    rolling_recency_weight_power: float = 1.0
    development_confirmation_ratio: float = 0.09
    minimum_confirmation_dates: int = 60
    terminal_shortlist_size: int = 5
    terminal_shortlist_max_per_family: int = 2
    minimum_fold_macro_f1: float = 0.25
    minimum_fold_weighted_f1: float = 0.30
    convergence_warnings_are_errors: bool = True
    focused_test_mode: bool = False


@dataclass(frozen=True)
class CandidateDefinition:
    """Describe one deterministic candidate without hiding its parameters."""

    model_name: str
    model_family: str
    parameters: Mapping[str, Any]


def _preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    text_features: list[str],
    config: TrainingConfig,
) -> ColumnTransformer:
    """Create training-only preprocessing for numbers, ticker, and SEC text."""

    # Every candidate receives exactly the same preprocessing. This makes the
    # validation ranking a comparison of classifiers rather than a comparison
    # of different feature-cleaning rules.
    transformers: list[tuple[str, Any, Any]] = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        )
    ]
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    if text_features:
        if len(text_features) != 1:
            raise MovementTrainingError(
                "Exactly one aggregated SEC text feature is supported."
            )
        transformers.append(
            (
                "filing_text",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=config.text_max_features,
                    sublinear_tf=True,
                ),
                text_features[0],
            )
        )
    return ColumnTransformer(transformers, remainder="drop")



def candidate_definitions(
    config: TrainingConfig | None = None,
) -> list[CandidateDefinition]:
    """Return the fixed development-only candidate space in stable order.

    Calibrated LinearSVC remains excluded because every such v5 candidate
    was weak across development folds and the family emitted unresolved
    Liblinear convergence warnings. V8 adds only evidence-backed candidates:
    recency-weighted tree variants and fixed diverse soft votes. Their
    definitions are frozen before the historical audit is evaluated.
    """

    active = config or TrainingConfig()
    definitions: list[CandidateDefinition] = [
        CandidateDefinition("prior_baseline", "dummy", {"strategy": "prior"})
    ]

    # Logistic models now use the documented multinomial-capable LBFGS solver.
    # Numeric columns are standardized inside each training fold, while sparse
    # ticker and text columns remain inside the same leakage-safe pipeline.
    for value in (0.05, 0.20, 1.0, 5.0):
        definitions.append(
            CandidateDefinition(
                f"balanced_logistic_c_{str(value).replace('.', '_')}",
                "logistic_regression",
                {"C": value},
            )
        )

    for value in (0.02, 0.05, 0.10, 0.20, 1.0):
        definitions.append(
            CandidateDefinition(
                f"unweighted_logistic_c_{str(value).replace('.', '_')}",
                "logistic_regression",
                {"C": value, "class_weight": "none"},
            )
        )

    # Recency weights use only dates already visible inside the current
    # training fold. Validation, confirmation, and audit dates remain excluded.
    for half_life in (365, 730):
        definitions.append(
            CandidateDefinition(
                f"recent_balanced_logistic_c_0_05_h{half_life}",
                "logistic_regression",
                {
                    "C": 0.05,
                    "sample_weight_mode": f"recency_{half_life}",
                },
            )
        )

    for leaf_size, maximum_depth in ((1, None), (2, None), (4, 12), (8, 8)):
        depth_name = "none" if maximum_depth is None else str(maximum_depth)
        definitions.append(
            CandidateDefinition(
                f"balanced_random_forest_leaf_{leaf_size}_depth_{depth_name}",
                "random_forest",
                {
                    "min_samples_leaf": leaf_size,
                    "max_depth": maximum_depth,
                    "n_estimators": active.forest_estimators,
                },
            )
        )
        definitions.append(
            CandidateDefinition(
                f"balanced_extra_trees_leaf_{leaf_size}_depth_{depth_name}",
                "extra_trees",
                {
                    "min_samples_leaf": leaf_size,
                    "max_depth": maximum_depth,
                    "n_estimators": active.forest_estimators,
                },
            )
        )

    # The v7.3 evidence showed a clear temporal rotation in the strongest
    # family: logistic led the oldest fold, SGD led fold three, and ExtraTrees
    # led the latest fold. These tree variants apply deterministic decay using
    # training-fold dates only, so no validation or audit row influences a
    # sample weight.
    for half_life in (365, 730):
        definitions.extend(
            [
                CandidateDefinition(
                    "recent_balanced_random_forest_leaf_4_depth_12_"
                    f"h{half_life}",
                    "random_forest",
                    {
                        "min_samples_leaf": 4,
                        "max_depth": 12,
                        "n_estimators": active.forest_estimators,
                        "sample_weight_mode": f"recency_{half_life}",
                    },
                ),
                CandidateDefinition(
                    "recent_balanced_extra_trees_leaf_4_depth_12_"
                    f"h{half_life}",
                    "extra_trees",
                    {
                        "min_samples_leaf": 4,
                        "max_depth": 12,
                        "n_estimators": active.forest_estimators,
                        "sample_weight_mode": f"recency_{half_life}",
                    },
                ),
            ]
        )

    for c_value, gamma_value in (
        (0.5, "scale"),
        (1.0, "scale"),
        (2.0, "scale"),
        (2.0, 0.05),
    ):
        gamma_name = str(gamma_value).replace(".", "_")
        definitions.append(
            CandidateDefinition(
                f"balanced_rbf_svc_c_{str(c_value).replace('.', '_')}"
                f"_g_{gamma_name}",
                "rbf_svc",
                {"C": c_value, "gamma": gamma_value},
            )
        )

    for alpha_value in (0.0001, 0.001):
        definitions.append(
            CandidateDefinition(
                f"balanced_sgd_log_loss_alpha_"
                f"{str(alpha_value).replace('.', '_')}",
                "sgd_log_loss",
                {"alpha": alpha_value},
            )
        )

    definitions.extend(
        [
            CandidateDefinition(
                "ticker_balanced_logistic_c_1_0",
                "logistic_regression",
                {"C": 1.0, "sample_weight_mode": "ticker_balanced"},
            ),
            CandidateDefinition(
                "ticker_balanced_extra_trees_leaf_2_depth_none",
                "extra_trees",
                {
                    "min_samples_leaf": 2,
                    "max_depth": None,
                    "n_estimators": active.forest_estimators,
                    "sample_weight_mode": "ticker_balanced",
                },
            ),
            CandidateDefinition(
                "ticker_balanced_rbf_svc_c_2_0_g_scale",
                "rbf_svc",
                {
                    "C": 2.0,
                    "gamma": "scale",
                    "sample_weight_mode": "ticker_balanced",
                },
            ),
        ]
    )

    # Fold leadership rotated across random forest, ExtraTrees, and SGD in
    # v7.3. These fixed votes test whether probability averaging reduces that
    # period-specific variance. The weights are declared in source and are not
    # tuned on confirmation or historical-audit labels.
    definitions.extend(
        [
            CandidateDefinition(
                "stability_soft_vote_rf_sgd",
                "stability_soft_vote",
                {
                    "components": [
                        "balanced_random_forest_leaf_4_depth_12",
                        "balanced_sgd_log_loss_alpha_0_0001",
                    ],
                    "weights": [1.0, 1.0],
                },
            ),
            CandidateDefinition(
                "stability_soft_vote_rf_et",
                "stability_soft_vote",
                {
                    "components": [
                        "balanced_random_forest_leaf_4_depth_12",
                        "balanced_extra_trees_leaf_4_depth_12",
                    ],
                    "weights": [1.0, 1.0],
                },
            ),
            CandidateDefinition(
                "stability_soft_vote_rf_et_sgd",
                "stability_soft_vote",
                {
                    "components": [
                        "balanced_random_forest_leaf_4_depth_12",
                        "balanced_extra_trees_leaf_4_depth_12",
                        "balanced_sgd_log_loss_alpha_0_0001",
                    ],
                    "weights": [2.0, 1.0, 1.0],
                },
            ),
        ]
    )

    if active.focused_test_mode:
        # Focused tests exercise the complete time-order, gate, policy, and
        # tournament workflow with two learned families. Separate unit tests
        # validate every additional production candidate, including recency
        # weighting and soft voting. This keeps package verification fast
        # without changing the production search space.
        approved_names = {
            "prior_baseline",
            "balanced_logistic_c_1_0",
            "balanced_random_forest_leaf_4_depth_12",
        }
        return [
            definition
            for definition in definitions
            if definition.model_name in approved_names
        ]
    return definitions



def _classifier_for_definition(
    definition: CandidateDefinition,
    config: TrainingConfig,
) -> Any:
    """Create one classifier from a transparent candidate definition."""

    parameters = dict(definition.parameters)
    family = definition.model_family
    if family == "dummy":
        return DummyClassifier(strategy="prior")

    class_weight_value = parameters.get("class_weight", "balanced")
    class_weight = None if class_weight_value == "none" else class_weight_value
    if family == "logistic_regression":
        return LogisticRegression(
            C=float(parameters["C"]),
            max_iter=config.logistic_max_iterations,
            tol=config.logistic_tolerance,
            class_weight=class_weight,
            random_state=config.random_seed,
            solver=config.logistic_solver,
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_depth=parameters["max_depth"],
            max_features="sqrt",
            class_weight=(
                "balanced_subsample" if class_weight is not None else None
            ),
            random_state=config.random_seed,
            n_jobs=1,
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(parameters["n_estimators"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_depth=parameters["max_depth"],
            max_features="sqrt",
            class_weight=class_weight,
            random_state=config.random_seed,
            n_jobs=1,
        )
    if family == "rbf_svc":
        return SVC(
            C=float(parameters["C"]),
            gamma=parameters["gamma"],
            kernel="rbf",
            class_weight=class_weight,
            probability=True,
            random_state=config.random_seed,
        )
    if family == "sgd_log_loss":
        return SGDClassifier(
            loss="log_loss",
            alpha=float(parameters["alpha"]),
            class_weight=class_weight,
            max_iter=3000,
            tol=1e-4,
            random_state=config.random_seed,
        )
    if family == "stability_soft_vote":
        # The shared preprocessor is fitted once per fold. Component names and
        # weights are frozen in the candidate definition, which keeps the vote
        # reproducible and prevents confirmation or audit labels from tuning it.
        full_config = replace(config, focused_test_mode=False)
        catalog = {
            item.model_name: item
            for item in candidate_definitions(full_config)
            if item.model_family != "stability_soft_vote"
        }
        component_names = [str(value) for value in parameters["components"]]
        weights = [float(value) for value in parameters["weights"]]
        if len(component_names) != len(weights) or not component_names:
            raise MovementTrainingError(
                "Soft-vote components and weights are invalid."
            )
        estimators: list[tuple[str, Any]] = []
        for component_index, component_name in enumerate(component_names):
            component = catalog.get(component_name)
            if component is None:
                raise MovementTrainingError(
                    f"Unknown soft-vote component: {component_name}"
                )
            estimator_name = f"component_{component_index + 1}"
            estimators.append(
                (
                    estimator_name,
                    _classifier_for_definition(component, config),
                )
            )
        return VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=weights,
            n_jobs=1,
            flatten_transform=True,
        )

    raise MovementTrainingError(f"Unknown candidate family: {family}")


def build_candidates(
    numeric_features: list[str],
    categorical_features: list[str],
    text_features: list[str] | None = None,
    config: TrainingConfig | None = None,
) -> dict[str, Pipeline]:
    """Create the complete deterministic candidate set."""

    active = config or TrainingConfig()
    approved_text = list(text_features or [])
    candidates: dict[str, Pipeline] = {}
    for definition in candidate_definitions(active):
        candidates[definition.model_name] = Pipeline(
            [
                (
                    "preprocessor",
                    _preprocessor(
                        numeric_features,
                        categorical_features,
                        approved_text,
                        active,
                    ),
                ),
                (
                    "classifier",
                    _classifier_for_definition(definition, active),
                ),
            ]
        )
    return candidates


def _definition_lookup(config: TrainingConfig) -> dict[str, CandidateDefinition]:
    """Index candidate metadata by its stable model name."""

    return {item.model_name: item for item in candidate_definitions(config)}



def _aligned_probabilities(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """Return model probabilities in the fixed Down, Flat, Up order.

    Some classifiers, especially support-vector models, can return a label from
    ``predict`` that differs from the largest calibrated probability. The
    package therefore treats one aligned probability matrix as the only input
    to the final decision policy.
    """

    probabilities = np.asarray(model.predict_proba(frame[feature_names]), dtype=float)
    class_names = [str(value) for value in model.named_steps["classifier"].classes_]
    if set(class_names) != set(LABEL_ORDER):
        raise MovementTrainingError("Probability class order changed.")
    aligned = probabilities[:, [class_names.index(label) for label in LABEL_ORDER]]
    if aligned.ndim != 2 or aligned.shape != (len(frame), len(LABEL_ORDER)):
        raise MovementTrainingError("Probability matrix shape is invalid.")
    if not np.isfinite(aligned).all() or (aligned < 0.0).any():
        raise MovementTrainingError("Probability matrix contains invalid values.")
    row_sums = aligned.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise MovementTrainingError("Probability matrix does not sum to one.")
    return aligned


def _zero_decision_policy() -> dict[str, Any]:
    """Return the identity policy used by the transparent prior baseline."""

    return {
        "label_order": list(LABEL_ORDER),
        "fit_split": "identity",
        "global_logit_offsets": {label: 0.0 for label in LABEL_ORDER},
        "ticker_logit_offsets": {},
    }


def _apply_decision_policy(
    probabilities: np.ndarray,
    tickers: pd.Series | np.ndarray,
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frozen validation-only logit offsets and return aligned decisions."""

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(LABEL_ORDER):
        raise MovementTrainingError("Decision probabilities have an invalid shape.")
    ticker_values = pd.Series(tickers).astype(str).str.upper().to_numpy()
    if len(ticker_values) != len(matrix):
        raise MovementTrainingError("Ticker and probability row counts differ.")

    global_offsets = policy.get("global_logit_offsets")
    ticker_offsets = policy.get("ticker_logit_offsets", {})
    if not isinstance(global_offsets, Mapping) or not isinstance(
        ticker_offsets, Mapping
    ):
        raise MovementTrainingError("Decision policy is incomplete.")
    global_vector = np.asarray(
        [float(global_offsets[label]) for label in LABEL_ORDER],
        dtype=float,
    )

    # Add offsets in log-probability space, then normalize. The saved
    # probabilities and saved labels now come from the same decision scores.
    log_scores = np.log(np.clip(matrix, 1e-12, 1.0)) + global_vector
    for row_index, ticker in enumerate(ticker_values):
        local = ticker_offsets.get(ticker, {})
        if local:
            log_scores[row_index] += np.asarray(
                [float(local.get(label, 0.0)) for label in LABEL_ORDER],
                dtype=float,
            )
    log_scores -= log_scores.max(axis=1, keepdims=True)
    adjusted = np.exp(log_scores)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    predicted = np.asarray(LABEL_ORDER, dtype=object)[adjusted.argmax(axis=1)]
    return adjusted, predicted


def _decision_objective(
    actual: pd.Series,
    predicted: np.ndarray,
    offset_magnitude: float,
) -> tuple[float, float, float, int, float]:
    """Rank one validation decision rule with deterministic tie breakers."""

    metrics = classification_metrics(actual, predicted)
    return (
        float(metrics["macro_f1"]),
        float(metrics["weighted_f1"]),
        float(metrics["accuracy"]),
        int(metrics["predicted_class_count"]),
        -float(offset_magnitude),
    )



def fit_stable_global_policy(
    oof_predictions: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    """Fit one global class policy from selection-period OOF predictions.

    Inputs
    ------
    ``oof_predictions`` has one row per validation observation produced by a
    model that was trained only on earlier dates. The required grain is
    candidate-fold-record, with a fold name, ticker, actual label, and aligned
    Down/Flat/Up probabilities.

    Logic
    -----
    Candidate offsets are evaluated on pooled out-of-fold evidence and on every
    fold separately. A non-identity policy qualifies only when it improves
    pooled macro and weighted F1, predicts all classes in every fold, improves
    at least half of the folds, and does not materially damage the weakest
    fold. No fold may hide behind a stronger fold. The historical audit is not accepted as an input.

    Outputs and downstream use
    --------------------------
    Return the frozen policy, a summary report, the complete candidate table,
    and OOF predictions under the selected policy. The policy is then checked
    on the reserved terminal development-confirmation block before any known
    historical-audit evaluation. The block is known from v6 evidence, so it is
    a rejection check rather than a statistically pristine holdout.

    Limitations
    -----------
    Global offsets can correct broad class bias but cannot create missing model
    signal. Identity is the fail-safe fallback when no stable adjustment
    qualifies.
    """

    required = {
        "fold_name",
        "ticker",
        "actual_movement",
        "prob_down",
        "prob_flat",
        "prob_up",
    }
    missing = sorted(required - set(oof_predictions.columns))
    if missing:
        raise MovementTrainingError(
            f"OOF decision-policy evidence is missing columns: {missing}"
        )

    frame = oof_predictions.copy().reset_index(drop=True)
    if frame.empty or frame["fold_name"].nunique() < 2:
        raise MovementTrainingError(
            "OOF decision-policy evidence requires at least two folds."
        )
    probabilities = frame[
        ["prob_down", "prob_flat", "prob_up"]
    ].to_numpy(float)
    actual = frame["actual_movement"].astype(str).reset_index(drop=True)
    tickers = frame["ticker"].astype(str).str.upper().reset_index(drop=True)
    fold_names = frame["fold_name"].astype(str).reset_index(drop=True)

    identity_policy = _zero_decision_policy()
    identity_adjusted, identity_predicted = _apply_decision_policy(
        probabilities,
        tickers,
        identity_policy,
    )
    identity_metrics = classification_metrics(actual, identity_predicted)
    identity_fold_metrics: dict[str, dict[str, Any]] = {}
    for fold_name in sorted(fold_names.unique()):
        mask = fold_names.eq(fold_name).to_numpy()
        identity_fold_metrics[fold_name] = classification_metrics(
            actual[mask].reset_index(drop=True),
            identity_predicted[mask],
        )

    candidate_rows: list[dict[str, Any]] = []
    qualified_rows: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for flat_offset, up_offset in product(
        config.decision_global_offsets,
        repeat=2,
    ):
        policy = _zero_decision_policy()
        policy["fit_split"] = "selection_oof"
        policy["global_logit_offsets"] = {
            "Down": 0.0,
            "Flat": float(flat_offset),
            "Up": float(up_offset),
        }
        adjusted, predicted = _apply_decision_policy(
            probabilities,
            tickers,
            policy,
        )
        pooled = classification_metrics(actual, predicted)

        fold_metrics: list[dict[str, Any]] = []
        improved_folds = 0
        for fold_name in sorted(fold_names.unique()):
            mask = fold_names.eq(fold_name).to_numpy()
            metrics = classification_metrics(
                actual[mask].reset_index(drop=True),
                predicted[mask],
            )
            baseline = identity_fold_metrics[fold_name]
            if metrics["weighted_f1"] > baseline["weighted_f1"]:
                improved_folds += 1
            fold_metrics.append(
                {
                    "fold_name": fold_name,
                    "macro_f1": float(metrics["macro_f1"]),
                    "weighted_f1": float(metrics["weighted_f1"]),
                    "predicted_class_count": int(
                        metrics["predicted_class_count"]
                    ),
                    "identity_weighted_f1": float(
                        baseline["weighted_f1"]
                    ),
                }
            )

        macro_values = np.asarray(
            [row["macro_f1"] for row in fold_metrics],
            dtype=float,
        )
        weighted_values = np.asarray(
            [row["weighted_f1"] for row in fold_metrics],
            dtype=float,
        )
        minimum_classes = min(
            row["predicted_class_count"] for row in fold_metrics
        )
        improved_ratio = improved_folds / len(fold_metrics)
        macro_improvement = float(
            pooled["macro_f1"] - identity_metrics["macro_f1"]
        )
        weighted_improvement = float(
            pooled["weighted_f1"] - identity_metrics["weighted_f1"]
        )
        maximum_allowed_drop = (
            config.decision_policy_maximum_fold_weighted_drop
        )
        fold_weighted_drops = [
            float(identity_fold_metrics[row["fold_name"]]["weighted_f1"])
            - float(row["weighted_f1"])
            for row in fold_metrics
        ]
        maximum_fold_weighted_drop = max(fold_weighted_drops)

        is_identity = (
            float(flat_offset) == 0.0 and float(up_offset) == 0.0
        )
        qualifies = bool(
            not is_identity
            and macro_improvement
            >= config.decision_policy_minimum_macro_improvement
            and weighted_improvement
            >= config.decision_policy_minimum_weighted_improvement
            and maximum_fold_weighted_drop <= maximum_allowed_drop
            and minimum_classes >= config.minimum_predicted_classes
            and improved_ratio
            >= config.decision_policy_minimum_improved_fold_ratio
        )

        magnitude = abs(float(flat_offset)) + abs(float(up_offset))
        row = {
            "flat_offset": float(flat_offset),
            "up_offset": float(up_offset),
            "is_identity": is_identity,
            "qualifies": qualifies,
            "pooled_accuracy": float(pooled["accuracy"]),
            "pooled_macro_f1": float(pooled["macro_f1"]),
            "pooled_weighted_f1": float(pooled["weighted_f1"]),
            "pooled_predicted_class_count": int(
                pooled["predicted_class_count"]
            ),
            "macro_f1_improvement": macro_improvement,
            "weighted_f1_improvement": weighted_improvement,
            "minimum_fold_macro_f1": float(macro_values.min()),
            "median_fold_macro_f1": float(np.median(macro_values)),
            "minimum_fold_weighted_f1": float(weighted_values.min()),
            "median_fold_weighted_f1": float(np.median(weighted_values)),
            "minimum_fold_predicted_class_count": int(minimum_classes),
            "maximum_fold_weighted_f1_drop": float(
                maximum_fold_weighted_drop
            ),
            "improved_fold_count": int(improved_folds),
            "improved_fold_ratio": float(improved_ratio),
            "offset_magnitude": float(magnitude),
            "fold_metrics_json": json.dumps(
                fold_metrics,
                sort_keys=True,
            ),
        }
        candidate_rows.append(row)
        if qualifies:
            objective = (
                float(weighted_values.min()),
                float(np.median(weighted_values)),
                float(np.median(macro_values)),
                float(pooled["macro_f1"]),
                float(pooled["weighted_f1"]),
                -float(magnitude),
                -float(flat_offset),
                -float(up_offset),
            )
            qualified_rows.append((objective, row))

    if qualified_rows:
        selected_row = max(qualified_rows, key=lambda item: item[0])[1]
        status = "adjusted"
    else:
        selected_row = next(
            row for row in candidate_rows if row["is_identity"]
        )
        status = "identity_fallback"

    selected_policy = _zero_decision_policy()
    selected_policy["fit_split"] = "selection_oof"
    selected_policy["global_logit_offsets"] = {
        "Down": 0.0,
        "Flat": float(selected_row["flat_offset"]),
        "Up": float(selected_row["up_offset"]),
    }
    selected_adjusted, selected_predicted = _apply_decision_policy(
        probabilities,
        tickers,
        selected_policy,
    )
    selected_oof = frame.copy()
    selected_oof["predicted_movement"] = selected_predicted
    for index, label in enumerate(LABEL_ORDER):
        selected_oof[f"adjusted_prob_{label.lower()}"] = (
            selected_adjusted[:, index]
        )

    report = {
        "status": status,
        "fit_split": "selection_oof",
        "historical_audit_used_for_selection": False,
        "candidate_policy_count": int(len(candidate_rows)),
        "qualified_policy_count": int(len(qualified_rows)),
        "selected_global_logit_offsets": dict(
            selected_policy["global_logit_offsets"]
        ),
        "identity_metrics": identity_metrics,
        "selected_metrics": classification_metrics(
            actual,
            selected_predicted,
        ),
        "selected_policy_evidence": dict(selected_row),
    }
    return selected_policy, report, candidate_rows, selected_oof


def fit_decision_policy(
    probabilities: np.ndarray,
    actual: pd.Series,
    tickers: pd.Series,
    config: TrainingConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit class and optional ticker offsets using validation evidence only.

    The Down offset is fixed at zero because adding the same constant to all
    logits changes nothing. Flat and Up offsets are searched on a small named
    grid. Optional ticker offsets are accepted only when enough validation rows
    exist and the ticker macro F1 improves by the configured minimum.
    """

    actual_series = pd.Series(actual).astype(str).reset_index(drop=True)
    ticker_series = pd.Series(tickers).astype(str).str.upper().reset_index(drop=True)
    if len(actual_series) != len(probabilities):
        raise MovementTrainingError("Decision-policy validation lengths differ.")

    best_policy = _zero_decision_policy()
    best_policy["fit_split"] = "validation"
    best_adjusted, best_predicted = _apply_decision_policy(
        probabilities,
        ticker_series,
        best_policy,
    )
    best_objective = _decision_objective(actual_series, best_predicted, 0.0)
    for flat_offset, up_offset in product(
        config.decision_global_offsets,
        repeat=2,
    ):
        candidate = _zero_decision_policy()
        candidate["fit_split"] = "validation"
        candidate["global_logit_offsets"] = {
            "Down": 0.0,
            "Flat": float(flat_offset),
            "Up": float(up_offset),
        }
        adjusted, predicted = _apply_decision_policy(
            probabilities,
            ticker_series,
            candidate,
        )
        magnitude = abs(float(flat_offset)) + abs(float(up_offset))
        objective = _decision_objective(actual_series, predicted, magnitude)
        if objective > best_objective:
            best_policy = candidate
            best_adjusted = adjusted
            best_predicted = predicted
            best_objective = objective

    # Ticker-specific offsets are a small validation-only correction. They are
    # skipped for thin tickers and never fitted from the historical audit.
    ticker_policy: dict[str, dict[str, float]] = {}
    if not config.enable_ticker_offsets:
        best_policy["ticker_logit_offsets"] = ticker_policy
        best_adjusted, best_predicted = _apply_decision_policy(
            probabilities,
            ticker_series,
            best_policy,
        )
        confidence = best_adjusted.max(axis=1)
        sorted_probabilities = np.sort(best_adjusted, axis=1)
        diagnostics = {
            "mean_max_probability": float(confidence.mean()),
            "mean_top_two_margin": float(
                np.mean(sorted_probabilities[:, -1] - sorted_probabilities[:, -2])
            ),
            "ticker_policy_count": 0,
            "prediction_probability_mismatch_count": 0,
        }
        return best_policy, best_adjusted, best_predicted, diagnostics

    for ticker in sorted(ticker_series.unique()):
        mask = ticker_series.eq(ticker).to_numpy()
        if int(mask.sum()) < config.decision_ticker_minimum_records:
            continue
        ticker_actual = actual_series[mask].reset_index(drop=True)
        if ticker_actual.nunique() < 2:
            continue
        base_metrics = classification_metrics(ticker_actual, best_predicted[mask])
        local_best_macro = float(base_metrics["macro_f1"])
        local_best: dict[str, float] | None = None
        for flat_offset, up_offset in product(
            config.decision_ticker_offsets,
            repeat=2,
        ):
            candidate = dict(best_policy)
            candidate["ticker_logit_offsets"] = {
                ticker: {
                    "Down": 0.0,
                    "Flat": float(flat_offset),
                    "Up": float(up_offset),
                }
            }
            _, local_predictions = _apply_decision_policy(
                probabilities[mask],
                ticker_series[mask],
                candidate,
            )
            local_metrics = classification_metrics(ticker_actual, local_predictions)
            local_macro = float(local_metrics["macro_f1"])
            if local_macro > local_best_macro:
                local_best_macro = local_macro
                local_best = candidate["ticker_logit_offsets"][ticker]
        improvement = local_best_macro - float(base_metrics["macro_f1"])
        if (
            local_best is not None
            and improvement >= config.decision_ticker_minimum_improvement
        ):
            ticker_policy[ticker] = local_best

    best_policy["ticker_logit_offsets"] = ticker_policy
    best_adjusted, best_predicted = _apply_decision_policy(
        probabilities,
        ticker_series,
        best_policy,
    )
    confidence = best_adjusted.max(axis=1)
    sorted_probabilities = np.sort(best_adjusted, axis=1)
    diagnostics = {
        "mean_max_probability": float(confidence.mean()),
        "mean_top_two_margin": float(
            np.mean(sorted_probabilities[:, -1] - sorted_probabilities[:, -2])
        ),
        "ticker_policy_count": int(len(ticker_policy)),
        "prediction_probability_mismatch_count": 0,
    }
    return best_policy, best_adjusted, best_predicted, diagnostics


def classification_metrics(
    actual: pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Return aggregate, per-class, support, and confusion evidence."""

    labels = list(LABEL_ORDER)
    actual_series = pd.Series(actual).astype(str).reset_index(drop=True)
    predicted_array = np.asarray(predicted, dtype=object)
    if len(actual_series) != len(predicted_array) or len(actual_series) == 0:
        raise MovementTrainingError("Metric inputs have different or empty lengths.")
    if not set(actual_series).issubset(labels):
        raise MovementTrainingError("Actual labels contain an unknown class.")
    if not set(predicted_array).issubset(labels):
        raise MovementTrainingError("Predictions contain an unknown class.")

    report = classification_report(
        actual_series,
        predicted_array,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(actual_series, predicted_array)),
        "macro_precision": float(
            precision_score(
                actual_series,
                predicted_array,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                actual_series,
                predicted_array,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                actual_series,
                predicted_array,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual_series,
                predicted_array,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            actual_series,
            predicted_array,
            labels=labels,
        ).astype(int).tolist(),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
                "predicted_support": int((predicted_array == label).sum()),
            }
            for label in labels
        },
        "record_count": int(len(actual_series)),
        "predicted_class_count": int(len(set(predicted_array))),
    }


def per_ticker_metrics(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    """Calculate complete test metrics independently for each ticker."""

    required = {"ticker", "actual_movement", "predicted_movement"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise MovementTrainingError(
            f"Per-ticker predictions are missing columns: {missing}"
        )

    rows: list[dict[str, Any]] = []
    for ticker, ticker_frame in predictions.groupby("ticker", observed=True):
        metrics = classification_metrics(
            ticker_frame["actual_movement"],
            ticker_frame["predicted_movement"].to_numpy(),
        )
        rows.append(
            {
                "ticker": str(ticker),
                "record_count": metrics["record_count"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "accuracy": metrics["accuracy"],
                "predicted_class_count": metrics["predicted_class_count"],
                "actual_class_support": {
                    label: metrics["per_class"][label]["support"]
                    for label in LABEL_ORDER
                },
                "predicted_class_support": {
                    label: metrics["per_class"][label]["predicted_support"]
                    for label in LABEL_ORDER
                },
            }
        )
    return sorted(rows, key=lambda row: row["ticker"])


def rank_validation_results(
    results: list[dict[str, Any]],
    config: TrainingConfig | None = None,
) -> list[dict[str, Any]]:
    """Rank successful candidates using development-period evidence only.

    V7.3 selected the random forest mainly because its oldest fold had the
    strongest minimum weighted F1, even though ExtraTrees and SGD led later
    folds. V8 first requires unchanged per-fold stability gates, then ranks by
    a fixed chronological weighting that gives later development periods more
    influence without reading confirmation or historical-audit labels.

    Runtime latency remains diagnostic only because machine load is not a
    deterministic model-quality signal. Stable model name is the final
    tie-breaker.
    """

    active = config or TrainingConfig()
    successful = [row for row in results if row.get("status") == "passed"]
    if not successful:
        raise MovementTrainingError(
            "No validation candidate completed successfully."
        )

    uses_rolling_evidence = any(
        "minimum_fold_weighted_f1" in row for row in successful
    )
    if uses_rolling_evidence:
        def ranking_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
            """Return one deterministic development-only ranking tuple."""

            minimum_macro = float(
                row.get("minimum_fold_macro_f1", -1.0)
            )
            minimum_weighted = float(
                row.get("minimum_fold_weighted_f1", -1.0)
            )
            minimum_classes = int(
                row.get("minimum_fold_predicted_class_count", 0)
            )
            stability_failed = (
                minimum_macro < active.minimum_fold_macro_f1
                or minimum_weighted < active.minimum_fold_weighted_f1
                or minimum_classes < active.minimum_predicted_classes
            )
            return (
                stability_failed,
                -float(
                    row.get(
                        "recency_weighted_fold_weighted_f1",
                        row.get("mean_fold_weighted_f1", -1.0),
                    )
                ),
                -float(
                    row.get(
                        "latest_fold_weighted_f1",
                        row.get("median_fold_weighted_f1", -1.0),
                    )
                ),
                -float(row.get("median_fold_weighted_f1", -1.0)),
                -minimum_weighted,
                -float(
                    row.get(
                        "recency_weighted_fold_macro_f1",
                        row.get("mean_fold_macro_f1", -1.0),
                    )
                ),
                -float(row.get("median_fold_macro_f1", -1.0)),
                float(row.get("weighted_f1_std", 999.0)),
                str(row["model_name"]),
            )

        return sorted(successful, key=ranking_key)

    return sorted(
        successful,
        key=lambda row: (
            -row["metrics"]["macro_f1"],
            -row["metrics"]["weighted_f1"],
            -row["metrics"]["accuracy"],
            row["model_name"],
        ),
    )


def _validation_gate_report(
    champion_name: str,
    validation_ranking: list[dict[str, Any]],
    config: TrainingConfig,
) -> dict[str, Any]:
    """Decide whether validation evidence is strong enough to unlock test."""

    by_name = {row["model_name"]: row["metrics"] for row in validation_ranking}
    baseline = by_name.get("prior_baseline")
    champion = by_name.get(champion_name)
    if not isinstance(baseline, dict) or not isinstance(champion, dict):
        raise MovementTrainingError("Validation baseline or champion is missing.")

    improvement = float(champion["macro_f1"] - baseline["macro_f1"])
    failures: list[str] = []
    if champion_name == "prior_baseline":
        failures.append("A learned model did not beat the prior baseline.")
    if champion["macro_f1"] < config.minimum_validation_macro_f1:
        failures.append("Validation macro F1 is below the minimum gate.")
    if improvement < config.minimum_validation_improvement:
        failures.append("Validation improvement over baseline is too small.")
    if champion["predicted_class_count"] < config.minimum_predicted_classes:
        failures.append("Validation predictions do not cover all three classes.")

    champion_row = next(
        row for row in validation_ranking if row["model_name"] == champion_name
    )
    minimum_fold_macro = champion_row.get("minimum_fold_macro_f1")
    minimum_fold_weighted = champion_row.get("minimum_fold_weighted_f1")
    minimum_fold_classes = champion_row.get(
        "minimum_fold_predicted_class_count"
    )
    if (
        minimum_fold_macro is not None
        and float(minimum_fold_macro) < config.minimum_fold_macro_f1
    ):
        failures.append(
            "A rolling validation fold has macro F1 below the stability gate."
        )
    if (
        minimum_fold_weighted is not None
        and float(minimum_fold_weighted) < config.minimum_fold_weighted_f1
    ):
        failures.append(
            "A rolling validation fold has weighted F1 below the stability gate."
        )
    if (
        minimum_fold_classes is not None
        and int(minimum_fold_classes) < config.minimum_predicted_classes
    ):
        failures.append(
            "A rolling validation fold does not predict all three classes."
        )

    return {
        "status": "passed" if not failures else "failed",
        "champion_name": champion_name,
        "champion_macro_f1": champion["macro_f1"],
        "champion_weighted_f1": champion["weighted_f1"],
        "baseline_macro_f1": baseline["macro_f1"],
        "macro_f1_improvement": improvement,
        "minimum_fold_macro_f1": minimum_fold_macro,
        "minimum_fold_weighted_f1": minimum_fold_weighted,
        "minimum_fold_predicted_class_count": minimum_fold_classes,
        "failures": failures,
    }


def evaluate_quality_gates(
    champion_name: str,
    validation_ranking: list[dict[str, Any]],
    test_metrics: Mapping[str, Any],
    baseline_test_metrics: Mapping[str, Any],
    ticker_metrics: list[dict[str, Any]],
    config: TrainingConfig,
    *,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    """Apply unchanged test, baseline, class, and per-ticker gates."""

    validation_report = _validation_gate_report(
        champion_name,
        validation_ranking,
        config,
    )
    validation_by_name = {
        row["model_name"]: row["metrics"] for row in validation_ranking
    }
    champion_validation = validation_by_name[champion_name]
    baseline_validation = validation_by_name["prior_baseline"]
    validation_improvement = float(
        champion_validation["macro_f1"] - baseline_validation["macro_f1"]
    )
    test_improvement = float(
        test_metrics["macro_f1"] - baseline_test_metrics["macro_f1"]
    )

    failures = list(validation_report["failures"])
    if test_metrics["macro_f1"] < config.minimum_test_macro_f1:
        failures.append("Historical-audit macro F1 is below the minimum gate.")
    if test_metrics["weighted_f1"] < config.minimum_test_weighted_f1:
        failures.append("Historical-audit weighted F1 is below the minimum gate.")
    if test_metrics["predicted_class_count"] < config.minimum_predicted_classes:
        failures.append("The champion did not predict all three movement classes.")

    for label in LABEL_ORDER:
        class_evidence = test_metrics["per_class"][label]
        if class_evidence["support"] < config.minimum_class_support:
            failures.append(f"Historical-audit support is too small for class {label}.")
        if class_evidence["predicted_support"] == 0:
            failures.append(f"The champion never predicted class {label}.")

    weak_tickers: list[str] = []
    for row in ticker_metrics:
        if row["record_count"] < config.per_ticker_minimum_records:
            continue
        if row["macro_f1"] < config.minimum_per_ticker_macro_f1:
            weak_tickers.append(row["ticker"])
    if weak_tickers:
        failures.append(
            "Per-ticker macro F1 is below the minimum for: "
            + ", ".join(sorted(weak_tickers))
        )

    report = {
        "status": "passed" if not failures else "failed",
        "thresholds": asdict(config),
        "validation_champion_macro_f1": champion_validation["macro_f1"],
        "validation_baseline_macro_f1": baseline_validation["macro_f1"],
        "validation_macro_f1_improvement": validation_improvement,
        "test_champion_macro_f1": test_metrics["macro_f1"],
        "test_champion_weighted_f1": test_metrics["weighted_f1"],
        "test_baseline_macro_f1": baseline_test_metrics["macro_f1"],
        "test_baseline_weighted_f1": baseline_test_metrics["weighted_f1"],
        "test_macro_f1_improvement": test_improvement,
        "weak_tickers": sorted(weak_tickers),
        "failures": failures,
    }
    if failures and raise_on_failure:
        raise MovementTrainingError(
            "Movement quality gates failed: " + " | ".join(failures)
        )
    return report


def _predict_output(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_names: list[str],
    decision_policy: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Create one probability-consistent, licence-safe prediction table."""

    raw_probabilities = _aligned_probabilities(model, frame, feature_names)
    policy = decision_policy or _zero_decision_policy()
    probabilities, predicted = _apply_decision_policy(
        raw_probabilities,
        frame["ticker"],
        policy,
    )

    # Raw and adjusted market prices are intentionally absent. The table keeps
    # only identifiers, labels, predictions, and the exact probabilities used
    # to choose each saved label.
    output = frame[["ticker", "target_session_date", "movement_label"]].copy()
    output = output.rename(columns={"movement_label": "actual_movement"})
    output.insert(0, "record_id", np.arange(1, len(output) + 1))
    output["predicted_movement"] = predicted
    for column_index, label in enumerate(LABEL_ORDER):
        output[f"prob_{label.lower()}"] = probabilities[:, column_index]
    probability_columns = ["prob_down", "prob_flat", "prob_up"]
    if not np.allclose(
        output[probability_columns].sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise MovementTrainingError("Prediction probabilities do not sum to one.")
    probability_labels = np.asarray(LABEL_ORDER, dtype=object)[
        output[probability_columns].to_numpy().argmax(axis=1)
    ]
    if not np.array_equal(probability_labels, predicted):
        raise MovementTrainingError(
            "Saved labels do not match the highest saved probability."
        )
    return output.reset_index(drop=True)


def global_importance(
    model: Pipeline,
    validation_frame: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    config: TrainingConfig | None = None,
) -> pd.DataFrame:
    """Return validation-only raw-feature sensitivity or a native fallback."""

    if validation_frame is not None and feature_names:
        # Use probability sensitivity rather than only label changes. A feature
        # can change model confidence without changing the winning class, and
        # that behavior is still meaningful for explanation and monitoring.
        raw_features = validation_frame[feature_names].copy()
        baseline_probabilities = model.predict_proba(raw_features)
        values: list[float] = []
        for feature in feature_names:
            perturbed = raw_features.copy()
            series = raw_features[feature]
            if pd.api.types.is_numeric_dtype(series):
                replacement: Any = float(
                    pd.to_numeric(series, errors="coerce").median()
                )
                if not np.isfinite(replacement):
                    raise MovementTrainingError(
                        f"Validation reference is unavailable: {feature}"
                    )
            elif feature == "event_text":
                replacement = ""
            else:
                modes = series.dropna().astype(str).mode()
                if modes.empty:
                    raise MovementTrainingError(
                        f"Validation reference is unavailable: {feature}"
                    )
                replacement = str(sorted(modes.tolist())[0])
            perturbed[feature] = replacement
            changed_probabilities = model.predict_proba(perturbed)
            values.append(
                float(
                    np.mean(
                        np.abs(
                            baseline_probabilities - changed_probabilities
                        )
                    )
                )
            )
        importance_values = np.asarray(values, dtype=float)
        method = "validation_reference_probability_sensitivity"
        names = list(feature_names)
    else:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        names = [str(value) for value in preprocessor.get_feature_names_out()]
        if hasattr(classifier, "feature_importances_"):
            importance_values = np.asarray(
                classifier.feature_importances_,
                dtype=float,
            )
            method = "model_native_impurity_importance_fallback"
        elif hasattr(classifier, "coef_"):
            importance_values = np.abs(
                np.asarray(classifier.coef_, dtype=float)
            ).mean(axis=0)
            method = "model_native_absolute_coefficient_fallback"
        else:
            raise MovementTrainingError(
                "Champion has no supported importance method."
            )

    if len(names) != len(importance_values):
        raise MovementTrainingError("Importance and feature counts differ.")
    if not np.isfinite(importance_values).all() or (
        importance_values < 0
    ).any():
        raise MovementTrainingError("Global importance values are invalid.")
    if float(importance_values.max(initial=0.0)) <= 0.0:
        raise MovementTrainingError("Champion produced only zero importance.")
    return pd.DataFrame(
        {
            "feature": names,
            "importance": importance_values,
            "method": method,
        }
    ).sort_values(["importance", "feature"], ascending=[False, True])


def runtime_versions() -> dict[str, str]:
    """Return exact versions needed to reproduce and reload the model."""

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "executable": sys.executable,
    }



def _candidate_result(
    definition: CandidateDefinition,
    *,
    status: str,
    training_seconds: float | None = None,
    latency_ms_per_record: float | None = None,
    metrics: dict[str, Any] | None = None,
    decision_policy: dict[str, Any] | None = None,
    prediction_diagnostics: dict[str, Any] | None = None,
    convergence_status: str | None = None,
    convergence_diagnostics: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Create one JSON-safe candidate diagnostic row."""

    return {
        "model_name": definition.model_name,
        "model_family": definition.model_family,
        "parameters": json.loads(json.dumps(dict(definition.parameters))),
        "status": status,
        "training_seconds": training_seconds,
        "latency_ms_per_record": latency_ms_per_record,
        "metrics": metrics,
        "decision_policy": decision_policy,
        "prediction_diagnostics": prediction_diagnostics,
        "convergence_status": convergence_status,
        "convergence_diagnostics": convergence_diagnostics or [],
        "error": error,
    }


def _iteration_diagnostics(
    model: Pipeline,
    candidate_name: str,
    fit_stage: str,
) -> list[dict[str, Any]]:
    """Return estimator iteration evidence after one successful fit.

    Iteration counts are diagnostic only. Candidate selection never ranks on
    these values, but they prove whether iterative estimators reached a stable
    result without a convergence warning.
    """

    classifier = model.named_steps["classifier"]
    estimators: list[tuple[str, Any]] = [("classifier", classifier)]
    if isinstance(classifier, VotingClassifier):
        estimators = [
            (str(name), estimator)
            for name, estimator in classifier.named_estimators_.items()
        ]

    rows: list[dict[str, Any]] = []
    for estimator_name, estimator in estimators:
        raw_iterations = getattr(estimator, "n_iter_", None)
        if raw_iterations is None:
            iteration_values: list[int] = []
        else:
            iteration_values = [
                int(value)
                for value in np.asarray(raw_iterations).reshape(-1).tolist()
            ]
        maximum = getattr(estimator, "max_iter", None)
        rows.append(
            {
                "candidate_name": candidate_name,
                "fit_stage": fit_stage,
                "estimator_name": estimator_name,
                "estimator_class": type(estimator).__name__,
                "configured_max_iter": (
                    int(maximum) if maximum is not None else None
                ),
                "observed_iterations": iteration_values,
                "status": "converged",
            }
        )
    return rows


def _fit_candidate(
    model: Pipeline,
    train_frame: pd.DataFrame,
    feature_names: list[str],
    fit_arguments: Mapping[str, Any],
    definition: CandidateDefinition,
    fit_stage: str,
    config: TrainingConfig,
) -> tuple[float, list[dict[str, Any]]]:
    """Fit one candidate while treating convergence warnings as failures.

    Inputs are one training-only frame, the approved feature list, and optional
    sample weights. Other warnings remain visible through Python's normal
    warning machinery; only ``ConvergenceWarning`` becomes a fail-closed error.
    """

    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            if config.convergence_warnings_are_errors:
                warnings.filterwarnings(
                    "error",
                    category=ConvergenceWarning,
                )
            model.fit(
                train_frame[feature_names],
                train_frame["movement_label"],
                **dict(fit_arguments),
            )
    except ConvergenceWarning as exc:
        raise MovementTrainingError(
            f"Candidate {definition.model_name} failed to converge "
            f"during {fit_stage}: {exc}"
        ) from exc
    elapsed = time.perf_counter() - started
    diagnostics = _iteration_diagnostics(
        model,
        definition.model_name,
        fit_stage,
    )
    return float(elapsed), diagnostics


def _split_development_confirmation(
    development_frame: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reserve one terminal development block for a fixed shortlist.

    The input grain is one ticker-target session row. Unique session dates are
    sorted, a final tournament block is reserved, and configured purge dates
    are removed between rolling selection and the tournament. The shortlist is
    frozen before this block is scored, and the historical audit remains
    completely outside this operation.
    """

    frame = development_frame.copy()
    frame["target_session_date"] = pd.to_datetime(
        frame["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    if frame["target_session_date"].isna().any():
        raise MovementTrainingError(
            "Development confirmation received invalid target dates."
        )

    unique_dates = np.asarray(
        sorted(frame["target_session_date"].unique()),
        dtype="datetime64[ns]",
    )
    requested = int(
        math.ceil(len(unique_dates) * config.development_confirmation_ratio)
    )
    confirmation_date_count = max(
        config.minimum_confirmation_dates,
        requested,
    )
    purge_count = int(config.rolling_purge_dates)
    selection_count = len(unique_dates) - confirmation_date_count - purge_count
    if selection_count < 120:
        raise MovementTrainingError(
            "Not enough development dates remain before confirmation."
        )

    selection_dates = unique_dates[:selection_count]
    purged_dates = unique_dates[
        selection_count : selection_count + purge_count
    ]
    confirmation_dates = unique_dates[selection_count + purge_count :]
    selection_frame = frame[
        frame["target_session_date"].isin(selection_dates)
    ].copy()
    confirmation_frame = frame[
        frame["target_session_date"].isin(confirmation_dates)
    ].copy()
    if selection_frame.empty or confirmation_frame.empty:
        raise MovementTrainingError(
            "Selection or development-confirmation evidence is empty."
        )
    if (
        selection_frame["target_session_date"].max()
        >= confirmation_frame["target_session_date"].min()
    ):
        raise MovementTrainingError(
            "Selection and development-confirmation dates overlap."
        )

    report = {
        "selection_start_date": str(
            selection_frame["target_session_date"].min().date()
        ),
        "selection_end_date": str(
            selection_frame["target_session_date"].max().date()
        ),
        "selection_rows": int(len(selection_frame)),
        "selection_unique_dates": int(len(selection_dates)),
        "confirmation_start_date": str(
            confirmation_frame["target_session_date"].min().date()
        ),
        "confirmation_end_date": str(
            confirmation_frame["target_session_date"].max().date()
        ),
        "confirmation_rows": int(len(confirmation_frame)),
        "confirmation_unique_dates": int(len(confirmation_dates)),
        "purged_dates": [
            str(pd.Timestamp(value).date()) for value in purged_dates
        ],
        "used_for_candidate_selection": True,
        "pristine": False,
        "known_from_prior_v6_evidence": True,
    }
    return selection_frame, confirmation_frame, report


def _confirmation_gate_report(
    champion_name: str,
    champion_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    config: TrainingConfig,
    *,
    used_for_candidate_selection: bool = False,
) -> dict[str, Any]:
    """Gate one pre-registered candidate on terminal development evidence.

    The caller explicitly records whether the terminal block only rejects a
    frozen champion or selects among a shortlist fixed by earlier folds.
    Historical-audit labels are never accepted by this function.
    """

    improvement = float(
        champion_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    )
    failures: list[str] = []
    if champion_metrics["macro_f1"] < config.minimum_validation_macro_f1:
        failures.append(
            "Development-confirmation macro F1 is below the validation gate."
        )
    if improvement < config.minimum_validation_improvement:
        failures.append(
            "Development-confirmation improvement over baseline is too small."
        )
    if (
        champion_metrics["weighted_f1"]
        < config.minimum_fold_weighted_f1
    ):
        failures.append(
            "Development-confirmation weighted F1 is below the fold gate."
        )
    if (
        champion_metrics["predicted_class_count"]
        < config.minimum_predicted_classes
    ):
        failures.append(
            "Development-confirmation predictions omit a movement class."
        )
    return {
        "status": "passed" if not failures else "failed",
        "champion_name": champion_name,
        "champion_macro_f1": float(champion_metrics["macro_f1"]),
        "champion_weighted_f1": float(champion_metrics["weighted_f1"]),
        "baseline_macro_f1": float(baseline_metrics["macro_f1"]),
        "baseline_weighted_f1": float(baseline_metrics["weighted_f1"]),
        "macro_f1_improvement": improvement,
        "used_for_candidate_selection": bool(
            used_for_candidate_selection
        ),
        "failures": failures,
    }


def select_terminal_shortlist(
    validation_ranking: list[dict[str, Any]],
    config: TrainingConfig,
) -> list[dict[str, Any]]:
    """Freeze a diverse shortlist before terminal labels are inspected.

    Candidates must pass the unchanged rolling validation gates. At most the
    configured number from one model family can enter, which prevents a single
    highly correlated family from occupying the complete terminal tournament.
    The prior baseline is never eligible.
    """

    if config.terminal_shortlist_size < 1:
        raise MovementTrainingError(
            "Terminal shortlist size must be positive."
        )
    if config.terminal_shortlist_max_per_family < 1:
        raise MovementTrainingError(
            "Terminal family limit must be positive."
        )

    shortlist: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for row in validation_ranking:
        model_name = str(row["model_name"])
        if model_name == "prior_baseline":
            continue
        gate = _validation_gate_report(
            model_name,
            validation_ranking,
            config,
        )
        if gate["status"] != "passed":
            continue
        family = str(row.get("model_family") or "unknown")
        if family_counts.get(family, 0) >= (
            config.terminal_shortlist_max_per_family
        ):
            continue
        shortlist.append(row)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(shortlist) >= config.terminal_shortlist_size:
            break

    if not shortlist:
        raise MovementTrainingError(
            "No learned candidate passed rolling gates for confirmation."
        )
    return shortlist


def rank_confirmation_tournament(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank terminal-development candidates without audit evidence.

    Only candidates whose terminal gate passed are eligible. Weighted F1 is
    primary because the unchanged historical gate is weighted F1; macro F1 and
    rolling recency evidence protect class balance and temporal stability.
    """

    eligible = [
        row
        for row in rows
        if row.get("status") == "passed"
        and (row.get("confirmation_gates") or {}).get("status") == "passed"
    ]
    return sorted(
        eligible,
        key=lambda row: (
            -float(row["metrics"]["weighted_f1"]),
            -float(row["metrics"]["macro_f1"]),
            -float(row.get("recency_weighted_fold_weighted_f1", -1.0)),
            -float(row.get("latest_fold_weighted_f1", -1.0)),
            -float(row.get("minimum_fold_weighted_f1", -1.0)),
            str(row["model_name"]),
        ),
    )


def _candidate_sample_weight(
    frame: pd.DataFrame,
    definition: CandidateDefinition,
) -> np.ndarray | None:
    """Return deterministic candidate-specific row weights when requested.

    Ticker-balanced weights give every issuer approximately equal total
    influence. Recency weights use only dates already inside the candidate's
    training fold, so later validation or audit dates cannot influence them.
    """

    mode = definition.parameters.get("sample_weight_mode")
    if mode is None:
        return None

    if mode == "ticker_balanced":
        ticker_counts = frame["ticker"].astype(str).value_counts()
        if ticker_counts.empty or (ticker_counts <= 0).any():
            raise MovementTrainingError(
                "Ticker counts are unavailable for weighting."
            )
        raw = frame["ticker"].astype(str).map(
            1.0 / ticker_counts
        ).to_numpy(float)
    elif str(mode).startswith("recency_"):
        try:
            half_life_days = float(str(mode).split("_", maxsplit=1)[1])
        except (IndexError, ValueError) as exc:
            raise MovementTrainingError(
                f"Invalid recency sample-weight mode: {mode}"
            ) from exc
        if half_life_days <= 0:
            raise MovementTrainingError(
                "Recency half-life must be positive."
            )
        dates = pd.to_datetime(
            frame["target_session_date"],
            errors="coerce",
        ).dt.normalize()
        if dates.isna().any():
            raise MovementTrainingError(
                "Recency weighting received invalid target dates."
            )
        age_days = (dates.max() - dates).dt.days.to_numpy(float)
        raw = np.power(0.5, age_days / half_life_days)
    else:
        raise MovementTrainingError(f"Unknown sample-weight mode: {mode}")

    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise MovementTrainingError("Candidate sample weights are invalid.")
    return raw / raw.mean()


def _rolling_validation_frames(
    development_frame: pd.DataFrame,
    config: TrainingConfig,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """Create expanding chronological folds inside development evidence.

    The historical audit has been inspected repeatedly, so one earlier
    validation block is not sufficient development evidence.
    Candidate selection uses several purged expanding folds that all end before
    the historical audit begins. No fold may overlap or look forward.
    """

    if config.rolling_validation_folds < 2:
        raise MovementTrainingError(
            "At least two rolling validation folds are required."
        )
    if not 0.40 <= config.rolling_initial_train_ratio <= 0.75:
        raise MovementTrainingError(
            "Rolling initial-train ratio must be between 0.40 and 0.75."
        )
    if config.rolling_purge_dates < 1:
        raise MovementTrainingError(
            "Rolling validation requires at least one purged date."
        )

    frame = development_frame.copy()
    frame["target_session_date"] = pd.to_datetime(
        frame["target_session_date"],
        errors="coerce",
    ).dt.normalize()
    if frame["target_session_date"].isna().any():
        raise MovementTrainingError(
            "Rolling validation received invalid target dates."
        )
    unique_dates = np.asarray(
        sorted(frame["target_session_date"].unique()),
        dtype="datetime64[ns]",
    )
    initial_train_dates = int(
        len(unique_dates) * config.rolling_initial_train_ratio
    )
    remaining_dates = (
        len(unique_dates)
        - initial_train_dates
        - config.rolling_purge_dates * config.rolling_validation_folds
    )
    fold_size = remaining_dates // config.rolling_validation_folds
    if initial_train_dates < 60 or fold_size < 20:
        raise MovementTrainingError(
            "Not enough dates for stable rolling validation folds."
        )

    folds: list[tuple[str, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    for fold_index in range(config.rolling_validation_folds):
        train_end = initial_train_dates + fold_index * (
            fold_size + config.rolling_purge_dates
        )
        validation_start = train_end + config.rolling_purge_dates
        if fold_index == config.rolling_validation_folds - 1:
            validation_end = len(unique_dates)
        else:
            validation_end = validation_start + fold_size
        train_dates = unique_dates[:train_end]
        validation_dates = unique_dates[validation_start:validation_end]
        if len(validation_dates) < 20:
            raise MovementTrainingError(
                "A rolling validation fold has too few dates."
            )

        train_fold = frame[
            frame["target_session_date"].isin(train_dates)
        ].copy()
        validation_fold = frame[
            frame["target_session_date"].isin(validation_dates)
        ].copy()
        if train_fold.empty or validation_fold.empty:
            raise MovementTrainingError(
                "A rolling validation fold is empty."
            )
        if train_fold["movement_label"].nunique() != len(LABEL_ORDER):
            raise MovementTrainingError(
                "A rolling training fold lacks a movement class."
            )
        if validation_fold["movement_label"].nunique() != len(LABEL_ORDER):
            raise MovementTrainingError(
                "A rolling validation fold lacks a movement class."
            )
        train_end_date = pd.Timestamp(train_dates[-1])
        validation_start_date = pd.Timestamp(validation_dates[0])
        if train_end_date >= validation_start_date:
            raise MovementTrainingError(
                "Rolling validation fold dates overlap."
            )

        fold_name = f"rolling_fold_{fold_index + 1}"
        report = {
            "fold_name": fold_name,
            "train_start_date": str(pd.Timestamp(train_dates[0]).date()),
            "train_end_date": str(train_end_date.date()),
            "validation_start_date": str(validation_start_date.date()),
            "validation_end_date": str(
                pd.Timestamp(validation_dates[-1]).date()
            ),
            "train_rows": int(len(train_fold)),
            "validation_rows": int(len(validation_fold)),
            "train_unique_dates": int(len(train_dates)),
            "validation_unique_dates": int(len(validation_dates)),
            "purged_dates": int(config.rolling_purge_dates),
        }
        folds.append((fold_name, train_fold, validation_fold, report))
    return folds


def _candidate_fold_summary(
    fold_metrics: list[dict[str, Any]],
    config: TrainingConfig,
) -> dict[str, Any]:
    """Aggregate one candidate's chronological stability evidence.

    Fold rows arrive in chronological order. Linear recency weights are raised
    to the configured power, normalized by ``numpy.average``, and applied only
    to development-fold metrics. The historical audit is never an input.
    """

    if not fold_metrics:
        raise MovementTrainingError(
            "Candidate rolling-fold metrics are missing."
        )
    if config.rolling_recency_weight_power <= 0:
        raise MovementTrainingError(
            "Rolling recency-weight power must be positive."
        )

    macro_values = np.asarray(
        [float(row["metrics"]["macro_f1"]) for row in fold_metrics],
        dtype=float,
    )
    weighted_values = np.asarray(
        [float(row["metrics"]["weighted_f1"]) for row in fold_metrics],
        dtype=float,
    )
    class_counts = [
        int(row["metrics"]["predicted_class_count"])
        for row in fold_metrics
    ]
    recency_weights = np.power(
        np.arange(1, len(fold_metrics) + 1, dtype=float),
        config.rolling_recency_weight_power,
    )
    return {
        "fold_count": int(len(fold_metrics)),
        "mean_fold_macro_f1": float(macro_values.mean()),
        "median_fold_macro_f1": float(np.median(macro_values)),
        "minimum_fold_macro_f1": float(macro_values.min()),
        "latest_fold_macro_f1": float(macro_values[-1]),
        "recency_weighted_fold_macro_f1": float(
            np.average(macro_values, weights=recency_weights)
        ),
        "macro_f1_std": float(macro_values.std(ddof=0)),
        "mean_fold_weighted_f1": float(weighted_values.mean()),
        "median_fold_weighted_f1": float(np.median(weighted_values)),
        "minimum_fold_weighted_f1": float(weighted_values.min()),
        "latest_fold_weighted_f1": float(weighted_values[-1]),
        "recency_weighted_fold_weighted_f1": float(
            np.average(weighted_values, weights=recency_weights)
        ),
        "weighted_f1_std": float(weighted_values.std(ddof=0)),
        "minimum_fold_predicted_class_count": int(min(class_counts)),
    }


def train_and_evaluate(
    table: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    text_features: list[str] | None = None,
    config: TrainingConfig | None = None,
    *,
    allow_failed_result: bool = False,
) -> dict[str, Any]:
    """Select, confirm, and audit one movement model without leakage.

    Purged expanding folds rank candidates inside an early development
    block and freeze a diverse shortlist. Each shortlisted candidate receives
    an OOF-only policy, then the purged terminal development block selects the
    strongest passing candidate. The known historical audit remains excluded
    from every candidate, solver, policy, and feature decision.
    """

    active = config or TrainingConfig()
    approved_text = list(text_features or [])
    feature_names = numeric_features + categorical_features + approved_text
    if set(table["split"].unique()) != {"train", "validation", "test"}:
        raise MovementTrainingError(
            "Model table must contain exactly three chronological blocks."
        )
    for feature in feature_names:
        if feature not in table.columns:
            raise MovementTrainingError(f"Missing model feature: {feature}")

    development_frame = table[
        table["split"].isin({"train", "validation"})
    ].copy()
    historical_audit_frame = table[table["split"] == "test"].copy()
    if development_frame.empty or historical_audit_frame.empty:
        raise MovementTrainingError(
            "Development or historical-audit evidence is empty."
        )

    latest_development_date = pd.to_datetime(
        development_frame["target_session_date"]
    ).max()
    first_audit_date = pd.to_datetime(
        historical_audit_frame["target_session_date"]
    ).min()
    if latest_development_date >= first_audit_date:
        raise MovementTrainingError(
            "Development and historical-audit dates overlap."
        )

    (
        selection_frame,
        confirmation_frame,
        confirmation_split_report,
    ) = _split_development_confirmation(development_frame, active)
    rolling_folds = _rolling_validation_frames(selection_frame, active)
    fold_reports = [report for _, _, _, report in rolling_folds]
    definitions = _definition_lookup(active)
    candidate_results: list[dict[str, Any]] = []
    candidate_oof_predictions: dict[str, pd.DataFrame] = {}

    # One native thread limits oversubscription. The separate runtime preflight
    # proves that Intel and LLVM OpenMP are not loaded together before training.
    with threadpool_limits(limits=1):
        for definition in candidate_definitions(active):
            fold_metric_rows: list[dict[str, Any]] = []
            convergence_rows: list[dict[str, Any]] = []
            pooled_actual: list[str] = []
            pooled_predicted: list[str] = []
            pooled_probabilities: list[np.ndarray] = []
            pooled_tickers: list[str] = []
            pooled_fold_names: list[str] = []
            pooled_target_dates: list[str] = []
            total_training_seconds = 0.0
            total_inference_seconds = 0.0
            total_validation_rows = 0
            final_policy = _zero_decision_policy()
            final_diagnostics: dict[str, Any] | None = None
            try:
                for (
                    fold_name,
                    train_fold,
                    validation_fold,
                    fold_report,
                ) in rolling_folds:
                    model = build_candidates(
                        numeric_features,
                        categorical_features,
                        approved_text,
                        active,
                    )[definition.model_name]
                    fit_arguments: dict[str, Any] = {}
                    sample_weight = _candidate_sample_weight(
                        train_fold,
                        definition,
                    )
                    if sample_weight is not None:
                        fit_arguments["classifier__sample_weight"] = (
                            sample_weight
                        )

                    training_seconds, fit_diagnostics = _fit_candidate(
                        model,
                        train_fold,
                        feature_names,
                        fit_arguments,
                        definition,
                        fold_name,
                        active,
                    )
                    convergence_rows.extend(fit_diagnostics)

                    inference_started = time.perf_counter()
                    raw_probabilities = _aligned_probabilities(
                        model,
                        validation_fold,
                        feature_names,
                    )
                    # Candidate ranking uses identity decisions only. This
                    # prevents fold labels from tuning a policy inside the same
                    # fold that is later used to score that candidate.
                    policy = _zero_decision_policy()
                    adjusted, predicted = _apply_decision_policy(
                        raw_probabilities,
                        validation_fold["ticker"],
                        policy,
                    )
                    sorted_probabilities = np.sort(adjusted, axis=1)
                    diagnostics = {
                        "mean_max_probability": float(
                            adjusted.max(axis=1).mean()
                        ),
                        "mean_top_two_margin": float(
                            np.mean(
                                sorted_probabilities[:, -1]
                                - sorted_probabilities[:, -2]
                            )
                        ),
                        "ticker_policy_count": 0,
                        "prediction_probability_mismatch_count": 0,
                    }
                    inference_seconds = (
                        time.perf_counter() - inference_started
                    )
                    metrics = classification_metrics(
                        validation_fold["movement_label"],
                        predicted,
                    )
                    fold_metric_rows.append(
                        {
                            "fold_name": fold_name,
                            "metrics": metrics,
                            "training_seconds": float(training_seconds),
                            "latency_ms_per_record": float(
                                inference_seconds
                                * 1000.0
                                / len(validation_fold)
                            ),
                            "train_start_date": fold_report[
                                "train_start_date"
                            ],
                            "train_end_date": fold_report["train_end_date"],
                            "validation_start_date": fold_report[
                                "validation_start_date"
                            ],
                            "validation_end_date": fold_report[
                                "validation_end_date"
                            ],
                            "convergence_status": "converged",
                        }
                    )
                    pooled_actual.extend(
                        validation_fold["movement_label"].astype(str).tolist()
                    )
                    pooled_predicted.extend(
                        np.asarray(predicted, dtype=object).astype(str).tolist()
                    )
                    pooled_probabilities.append(
                        np.asarray(raw_probabilities, dtype=float)
                    )
                    pooled_tickers.extend(
                        validation_fold["ticker"].astype(str).tolist()
                    )
                    pooled_fold_names.extend(
                        [fold_name] * len(validation_fold)
                    )
                    pooled_target_dates.extend(
                        pd.to_datetime(
                            validation_fold["target_session_date"]
                        ).dt.date.astype(str).tolist()
                    )
                    total_training_seconds += training_seconds
                    total_inference_seconds += inference_seconds
                    total_validation_rows += len(validation_fold)
                    final_policy = policy
                    final_diagnostics = diagnostics

                pooled_metrics = classification_metrics(
                    pd.Series(pooled_actual),
                    np.asarray(pooled_predicted, dtype=object),
                )
                candidate = _candidate_result(
                    definition,
                    status="passed",
                    training_seconds=float(total_training_seconds),
                    latency_ms_per_record=float(
                        total_inference_seconds
                        * 1000.0
                        / total_validation_rows
                    ),
                    metrics=pooled_metrics,
                    decision_policy=final_policy,
                    prediction_diagnostics=final_diagnostics,
                    convergence_status="converged",
                    convergence_diagnostics=convergence_rows,
                )
                candidate.update(_candidate_fold_summary(fold_metric_rows, active))
                candidate["fold_metrics"] = fold_metric_rows
                candidate_results.append(candidate)
                probability_matrix = np.vstack(pooled_probabilities)
                candidate_oof_predictions[definition.model_name] = pd.DataFrame(
                    {
                        "fold_name": pooled_fold_names,
                        "ticker": pooled_tickers,
                        "target_session_date": pooled_target_dates,
                        "actual_movement": pooled_actual,
                        "prob_down": probability_matrix[:, 0],
                        "prob_flat": probability_matrix[:, 1],
                        "prob_up": probability_matrix[:, 2],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - failure is evidence.
                error_text = f"{type(exc).__name__}: {exc}"
                convergence_status = (
                    "failed"
                    if "failed to converge" in str(exc).lower()
                    else "not_completed"
                )
                candidate_results.append(
                    _candidate_result(
                        definition,
                        status="failed",
                        training_seconds=float(total_training_seconds),
                        convergence_status=convergence_status,
                        convergence_diagnostics=convergence_rows,
                        error=error_text,
                    )
                )

    ranking = rank_validation_results(candidate_results, active)
    learned_ranking = [
        row for row in ranking if row["model_name"] != "prior_baseline"
    ]
    if not learned_ranking:
        raise MovementTrainingError(
            "No learned movement candidate completed rolling validation."
        )

    # Check the strongest rolling candidate before terminal labels are touched.
    # A deliberately strict or genuinely failed development gate must return
    # diagnostics without fitting on the terminal block or historical audit.
    provisional_name = str(learned_ranking[0]["model_name"])
    provisional_gates = _validation_gate_report(
        provisional_name,
        ranking,
        active,
    )
    if provisional_gates["status"] != "passed":
        failed_result: dict[str, Any] = {
            "status": "validation_failed",
            "champion_name": provisional_name,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "text_features": approved_text,
            "feature_names": feature_names,
            "candidate_results": candidate_results,
            "validation_ranking": ranking,
            "validation_gates": provisional_gates,
            "rolling_fold_reports": fold_reports,
            "development_confirmation_split": confirmation_split_report,
            "development_confirmation_tournament": [],
            "development_confirmation_evaluation_count": 0,
            "development_confirmation_pristine": False,
            "development_confirmation_known_from_prior_run": True,
            "development_confirmation_used_for_candidate_selection": True,
            "evaluation_protocol": (
                "purged_four_fold_recency_ranking_plus_oof_policy_"
                "calibration_plus_terminal_development_tournament_"
                "plus_known_historical_audit"
            ),
            "historical_audit_pristine": False,
            "historical_audit_used_for_selection": False,
            "test_used_for_selection": False,
            "test_evaluation_count": 0,
            "historical_audit_evaluation_count": 0,
            "random_seed": active.random_seed,
            "runtime_versions": runtime_versions(),
            "training_config": asdict(active),
            "decision_policy": _zero_decision_policy(),
            "decision_policy_calibration": {
                "status": "not_run_validation_failed",
                "historical_audit_used_for_selection": False,
            },
            "decision_policy_candidates": [],
            "decision_policy_oof_predictions": None,
        }
        if allow_failed_result:
            return failed_result
        raise MovementTrainingError(
            "Rolling validation quality gates failed: "
            + " | ".join(provisional_gates["failures"])
        )

    shortlist = select_terminal_shortlist(ranking, active)
    baseline_definition = definitions["prior_baseline"]
    confirmation_baseline = build_candidates(
        numeric_features,
        categorical_features,
        approved_text,
        active,
    )["prior_baseline"]

    # The shortlist is frozen from rolling folds before any terminal label is
    # scored. The baseline is fitted once, while every shortlisted candidate
    # carries its own OOF-frozen decision policy into the same terminal block.
    with threadpool_limits(limits=1):
        _fit_candidate(
            confirmation_baseline,
            selection_frame,
            feature_names,
            {},
            baseline_definition,
            "development_confirmation_baseline",
            active,
        )
        confirmation_baseline_output = _predict_output(
            confirmation_baseline,
            confirmation_frame,
            feature_names,
            _zero_decision_policy(),
        )
    confirmation_baseline_metrics = classification_metrics(
        confirmation_baseline_output["actual_movement"],
        confirmation_baseline_output["predicted_movement"].to_numpy(),
    )

    tournament_rows: list[dict[str, Any]] = []
    tournament_models: dict[str, Pipeline] = {}
    tournament_outputs: dict[str, pd.DataFrame] = {}
    tournament_policy_candidates: dict[str, list[dict[str, Any]]] = {}
    tournament_policy_oof: dict[str, pd.DataFrame] = {}
    all_confirmation_convergence: list[dict[str, Any]] = []

    for shortlist_row in shortlist:
        candidate_name = str(shortlist_row["model_name"])
        definition = definitions[candidate_name]
        candidate_oof = candidate_oof_predictions.get(candidate_name)
        if candidate_oof is None:
            raise MovementTrainingError(
                "Shortlisted candidate OOF predictions are unavailable: "
                f"{candidate_name}"
            )
        (
            candidate_policy,
            candidate_policy_report,
            candidate_policy_rows,
            candidate_policy_oof,
        ) = fit_stable_global_policy(candidate_oof, active)
        confirmation_model = build_candidates(
            numeric_features,
            categorical_features,
            approved_text,
            active,
        )[candidate_name]

        try:
            fit_arguments: dict[str, Any] = {}
            candidate_weight = _candidate_sample_weight(
                selection_frame,
                definition,
            )
            if candidate_weight is not None:
                fit_arguments["classifier__sample_weight"] = candidate_weight

            with threadpool_limits(limits=1):
                fit_seconds, convergence_rows = _fit_candidate(
                    confirmation_model,
                    selection_frame,
                    feature_names,
                    fit_arguments,
                    definition,
                    "development_confirmation_tournament",
                    active,
                )
                prediction_started = time.perf_counter()
                candidate_output = _predict_output(
                    confirmation_model,
                    confirmation_frame,
                    feature_names,
                    candidate_policy,
                )
                prediction_seconds = (
                    time.perf_counter() - prediction_started
                )

            candidate_metrics = classification_metrics(
                candidate_output["actual_movement"],
                candidate_output["predicted_movement"].to_numpy(),
            )
            candidate_gates = _confirmation_gate_report(
                candidate_name,
                candidate_metrics,
                confirmation_baseline_metrics,
                active,
                used_for_candidate_selection=True,
            )
            tournament_row = {
                "model_name": candidate_name,
                "model_family": shortlist_row.get("model_family"),
                "status": (
                    "passed"
                    if candidate_gates["status"] == "passed"
                    else "gate_failed"
                ),
                "metrics": candidate_metrics,
                "confirmation_gates": candidate_gates,
                "decision_policy": candidate_policy,
                "decision_policy_calibration": candidate_policy_report,
                "fit_seconds": float(fit_seconds),
                "latency_ms_per_record": float(
                    prediction_seconds
                    * 1000.0
                    / len(confirmation_frame)
                ),
                "convergence_diagnostics": convergence_rows,
                "minimum_fold_macro_f1": shortlist_row.get(
                    "minimum_fold_macro_f1"
                ),
                "minimum_fold_weighted_f1": shortlist_row.get(
                    "minimum_fold_weighted_f1"
                ),
                "latest_fold_macro_f1": shortlist_row.get(
                    "latest_fold_macro_f1"
                ),
                "latest_fold_weighted_f1": shortlist_row.get(
                    "latest_fold_weighted_f1"
                ),
                "recency_weighted_fold_macro_f1": shortlist_row.get(
                    "recency_weighted_fold_macro_f1"
                ),
                "recency_weighted_fold_weighted_f1": shortlist_row.get(
                    "recency_weighted_fold_weighted_f1"
                ),
                "error": None,
            }
            tournament_models[candidate_name] = confirmation_model
            tournament_outputs[candidate_name] = candidate_output
            tournament_policy_candidates[candidate_name] = (
                candidate_policy_rows
            )
            tournament_policy_oof[candidate_name] = candidate_policy_oof
            all_confirmation_convergence.extend(convergence_rows)
        except Exception as exc:  # noqa: BLE001 - failure is evidence.
            tournament_row = {
                "model_name": candidate_name,
                "model_family": shortlist_row.get("model_family"),
                "status": "failed",
                "metrics": None,
                "confirmation_gates": {
                    "status": "failed",
                    "used_for_candidate_selection": True,
                    "failures": [f"{type(exc).__name__}: {exc}"],
                },
                "decision_policy": candidate_policy,
                "decision_policy_calibration": candidate_policy_report,
                "fit_seconds": None,
                "latency_ms_per_record": None,
                "convergence_diagnostics": [],
                "minimum_fold_macro_f1": shortlist_row.get(
                    "minimum_fold_macro_f1"
                ),
                "minimum_fold_weighted_f1": shortlist_row.get(
                    "minimum_fold_weighted_f1"
                ),
                "latest_fold_macro_f1": shortlist_row.get(
                    "latest_fold_macro_f1"
                ),
                "latest_fold_weighted_f1": shortlist_row.get(
                    "latest_fold_weighted_f1"
                ),
                "recency_weighted_fold_macro_f1": shortlist_row.get(
                    "recency_weighted_fold_macro_f1"
                ),
                "recency_weighted_fold_weighted_f1": shortlist_row.get(
                    "recency_weighted_fold_weighted_f1"
                ),
                "error": f"{type(exc).__name__}: {exc}",
            }
        tournament_rows.append(tournament_row)

    confirmation_ranking = rank_confirmation_tournament(tournament_rows)
    fallback_name = str(shortlist[0]["model_name"])
    champion_name = (
        str(confirmation_ranking[0]["model_name"])
        if confirmation_ranking
        else fallback_name
    )
    champion_definition = definitions[champion_name]
    validation_gates = _validation_gate_report(
        champion_name,
        ranking,
        active,
    )
    selected_tournament_row = next(
        row
        for row in tournament_rows
        if row["model_name"] == champion_name
    )
    champion_policy = dict(
        selected_tournament_row.get("decision_policy") or {}
    )
    policy_calibration = dict(
        selected_tournament_row.get("decision_policy_calibration") or {}
    )
    policy_candidates = tournament_policy_candidates.get(
        champion_name,
        [],
    )
    policy_oof_predictions = tournament_policy_oof.get(champion_name)
    confirmation_metrics = selected_tournament_row.get("metrics")
    confirmation_gates = selected_tournament_row.get(
        "confirmation_gates"
    )
    confirmation_output = tournament_outputs.get(champion_name)
    confirmation_champion = tournament_models.get(champion_name)

    base_result: dict[str, Any] = {
        "status": (
            "confirmation_passed"
            if confirmation_ranking
            else "confirmation_failed"
        ),
        "champion_name": champion_name,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "text_features": approved_text,
        "feature_names": feature_names,
        "candidate_results": candidate_results,
        "validation_ranking": ranking,
        "validation_gates": validation_gates,
        "rolling_fold_reports": fold_reports,
        "development_confirmation_split": confirmation_split_report,
        "development_confirmation_metrics": confirmation_metrics,
        "baseline_development_confirmation_metrics": (
            confirmation_baseline_metrics
        ),
        "development_confirmation_gates": confirmation_gates,
        "development_confirmation_predictions": confirmation_output,
        "development_confirmation_tournament": tournament_rows,
        "development_confirmation_evaluation_count": int(
            len(tournament_rows)
        ),
        "development_confirmation_pristine": False,
        "development_confirmation_known_from_prior_run": True,
        "development_confirmation_used_for_candidate_selection": True,
        "development_confirmation_convergence": (
            all_confirmation_convergence
        ),
        "evaluation_protocol": (
            "purged_four_fold_recency_ranking_plus_oof_policy_"
            "calibration_plus_terminal_development_tournament_"
            "plus_known_historical_audit"
        ),
        "historical_audit_pristine": False,
        "historical_audit_used_for_selection": False,
        "test_used_for_selection": False,
        "test_evaluation_count": 0,
        "historical_audit_evaluation_count": 0,
        "random_seed": active.random_seed,
        "runtime_versions": runtime_versions(),
        "training_config": asdict(active),
        "decision_policy": champion_policy,
        "decision_policy_calibration": policy_calibration,
        "decision_policy_candidates": policy_candidates,
        "decision_policy_oof_predictions": policy_oof_predictions,
    }
    if not confirmation_ranking:
        if allow_failed_result:
            return base_result
        failure_messages = [
            message
            for row in tournament_rows
            for message in (
                (row.get("confirmation_gates") or {}).get("failures") or []
            )
        ]
        raise MovementTrainingError(
            "No terminal-development candidate passed: "
            + " | ".join(failure_messages)
        )
    if (
        confirmation_champion is None
        or confirmation_output is None
        or not isinstance(confirmation_metrics, Mapping)
        or not isinstance(confirmation_gates, Mapping)
    ):
        raise MovementTrainingError(
            "Selected terminal-development evidence is incomplete."
        )

    # Importance uses only the unseen development-confirmation block. It cannot
    # inspect the historical audit and is frozen before the final refit.
    importance = global_importance(
        confirmation_champion,
        confirmation_frame,
        feature_names,
        active,
    )

    champion = build_candidates(
        numeric_features,
        categorical_features,
        approved_text,
        active,
    )[champion_name]
    baseline = build_candidates(
        numeric_features,
        categorical_features,
        approved_text,
        active,
    )["prior_baseline"]

    with threadpool_limits(limits=1):
        refit_arguments: dict[str, Any] = {}
        refit_weight = _candidate_sample_weight(
            development_frame,
            champion_definition,
        )
        if refit_weight is not None:
            refit_arguments["classifier__sample_weight"] = refit_weight
        refit_seconds, refit_convergence = _fit_candidate(
            champion,
            development_frame,
            feature_names,
            refit_arguments,
            champion_definition,
            "full_development_refit",
            active,
        )
        _fit_candidate(
            baseline,
            development_frame,
            feature_names,
            {},
            baseline_definition,
            "full_development_baseline",
            active,
        )

        # The known historical audit is evaluated once only after every
        # development selection and confirmation decision is frozen.
        audit_started = time.perf_counter()
        audit_output = _predict_output(
            champion,
            historical_audit_frame,
            feature_names,
            champion_policy,
        )
        baseline_output = _predict_output(
            baseline,
            historical_audit_frame,
            feature_names,
            _zero_decision_policy(),
        )
        audit_seconds = time.perf_counter() - audit_started

    audit_metrics = classification_metrics(
        audit_output["actual_movement"],
        audit_output["predicted_movement"].to_numpy(),
    )
    baseline_audit_metrics = classification_metrics(
        baseline_output["actual_movement"],
        baseline_output["predicted_movement"].to_numpy(),
    )
    ticker_metrics = per_ticker_metrics(audit_output)
    quality_gates = evaluate_quality_gates(
        champion_name,
        ranking,
        audit_metrics,
        baseline_audit_metrics,
        ticker_metrics,
        active,
        raise_on_failure=False,
    )

    base_result.update(
        {
            "status": (
                "passed"
                if quality_gates["status"] == "passed"
                else "quality_failed"
            ),
            "champion_pipeline": champion,
            "test_metrics": audit_metrics,
            "historical_audit_metrics": audit_metrics,
            "baseline_test_metrics": baseline_audit_metrics,
            "baseline_historical_audit_metrics": baseline_audit_metrics,
            "per_ticker_test_metrics": ticker_metrics,
            "per_ticker_historical_audit_metrics": ticker_metrics,
            "quality_gates": quality_gates,
            "test_latency_ms_per_record": float(
                audit_seconds * 1000.0 / len(historical_audit_frame)
            ),
            "historical_audit_latency_ms_per_record": float(
                audit_seconds * 1000.0 / len(historical_audit_frame)
            ),
            "refit_seconds": float(refit_seconds),
            "champion_refit_convergence": refit_convergence,
            "test_predictions": audit_output,
            "historical_audit_predictions": audit_output,
            "global_importance": importance,
            "decision_policy": champion_policy,
            "test_evaluation_count": 1,
            "historical_audit_evaluation_count": 1,
        }
    )
    if quality_gates["status"] != "passed" and not allow_failed_result:
        raise MovementTrainingError(
            "Movement quality gates failed: "
            + " | ".join(quality_gates["failures"])
        )
    return base_result
