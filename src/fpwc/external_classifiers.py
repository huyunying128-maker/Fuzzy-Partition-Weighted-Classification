"""External classifiers for fuzzy-incorporated representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .metrics import ClassificationReport, classification_report

ExternalClassifierName = Literal[
    "ann",
    "adam_ann",
    "svm",
    "random_forest",
    "xgboost",
    "logistic_regression",
]


@dataclass(frozen=True)
class ExternalClassifierConfig:
    """Configuration for an external classifier.

    Parameters
    ----------
    name:
        Classifier name. Supported values are ``ann``, ``adam_ann``, ``svm``,
        ``random_forest``, ``xgboost``, and ``logistic_regression``.
    random_state:
        Random seed used by classifiers that expose a random state.
    params:
        Additional keyword arguments passed to the estimator constructor.
    """

    name: str
    random_state: int = 0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalClassifierResult:
    """Prediction result for an external classifier."""

    name: str
    report: ClassificationReport
    predictions: np.ndarray
    probabilities: np.ndarray | None = None


def _normalize_name(name: str) -> str:
    """Normalize a classifier name for lookup."""
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mlp": "ann",
        "neural_network": "ann",
        "adam": "adam_ann",
        "adam_mlp": "adam_ann",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "xgb": "xgboost",
        "logit": "logistic_regression",
        "logistic": "logistic_regression",
    }
    return aliases.get(normalized, normalized)


def _merge_params(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge default estimator parameters with user-provided overrides."""
    params = dict(defaults)
    if overrides:
        params.update(overrides)
    return params


def build_external_classifier(
    name: str,
    random_state: int = 0,
    **params: Any,
) -> Any:
    """Create an external classifier estimator.

    The returned estimator follows the scikit-learn estimator interface. The
    function constructs classifiers used in the fuzzy-incorporation comparison:
    ANN, Adam ANN, SVM, random forest, XGBoost, and logistic regression.

    Parameters
    ----------
    name:
        Name of the classifier.
    random_state:
        Random seed for estimators that support reproducibility.
    **params:
        Optional estimator-specific parameter overrides.

    Returns
    -------
    object
        A classifier with ``fit`` and ``predict`` methods.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    classifier_name = _normalize_name(name)

    if classifier_name == "ann":
        defaults = {
            "hidden_layer_sizes": (256, 128),
            "activation": "relu",
            "solver": "sgd",
            "learning_rate_init": 0.01,
            "batch_size": 256,
            "max_iter": 100,
            "early_stopping": True,
            "n_iter_no_change": 10,
            "random_state": random_state,
            "verbose": False,
        }
        return MLPClassifier(**_merge_params(defaults, params))

    if classifier_name == "adam_ann":
        defaults = {
            "hidden_layer_sizes": (256, 128),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "batch_size": 256,
            "max_iter": 100,
            "early_stopping": True,
            "n_iter_no_change": 10,
            "random_state": random_state,
            "verbose": False,
        }
        return MLPClassifier(**_merge_params(defaults, params))

    if classifier_name == "svm":
        defaults = {
            "C": 10.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "random_state": random_state,
        }
        return SVC(**_merge_params(defaults, params))

    if classifier_name == "random_forest":
        defaults = {
            "n_estimators": 300,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": random_state,
        }
        return RandomForestClassifier(**_merge_params(defaults, params))

    if classifier_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "xgboost is required to build the XGBoost classifier. "
                "Install it with `pip install xgboost`."
            ) from exc
        defaults = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        return XGBClassifier(**_merge_params(defaults, params))

    if classifier_name == "logistic_regression":
        defaults = {
            "penalty": "l2",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "multi_class": "auto",
            "random_state": random_state,
        }
        return LogisticRegression(**_merge_params(defaults, params))

    supported = [
        "ann",
        "adam_ann",
        "svm",
        "random_forest",
        "xgboost",
        "logistic_regression",
    ]
    raise ValueError(f"Unknown classifier '{name}'. Supported values are {supported}.")


def _validate_feature_matrix(x: np.ndarray, name: str) -> np.ndarray:
    """Validate a two-dimensional feature matrix."""
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one feature.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_labels(y: np.ndarray, name: str) -> np.ndarray:
    """Validate a one-dimensional label vector."""
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if labels.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one label.")
    return labels


def fit_external_classifier(
    classifier: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Fit an external classifier on a feature matrix."""
    x_train_array = _validate_feature_matrix(x_train, "x_train")
    y_train_array = _validate_labels(y_train, "y_train")
    if x_train_array.shape[0] != y_train_array.shape[0]:
        raise ValueError("x_train and y_train must contain the same number of samples.")
    classifier.fit(x_train_array, y_train_array)
    return classifier


def predict_probabilities(classifier: Any, x: np.ndarray) -> np.ndarray | None:
    """Predict class probabilities when the estimator supports them."""
    x_array = _validate_feature_matrix(x, "x")
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(x_array)
        return np.asarray(probabilities, dtype=np.float64)
    return None


def evaluate_external_classifier(
    classifier: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    name: str | None = None,
) -> ExternalClassifierResult:
    """Evaluate a fitted external classifier."""
    x_test_array = _validate_feature_matrix(x_test, "x_test")
    y_test_array = _validate_labels(y_test, "y_test")
    if x_test_array.shape[0] != y_test_array.shape[0]:
        raise ValueError("x_test and y_test must contain the same number of samples.")

    predictions = np.asarray(classifier.predict(x_test_array))
    probabilities = predict_probabilities(classifier, x_test_array)
    report = classification_report(y_true=y_test_array, y_pred=predictions, probabilities=probabilities)
    result_name = name if name is not None else classifier.__class__.__name__
    return ExternalClassifierResult(
        name=result_name,
        report=report,
        predictions=predictions,
        probabilities=probabilities,
    )


def fit_and_evaluate_external_classifier(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 0,
    **params: Any,
) -> tuple[Any, ExternalClassifierResult]:
    """Build, fit, and evaluate an external classifier."""
    classifier = build_external_classifier(name=name, random_state=random_state, **params)
    fitted = fit_external_classifier(classifier, x_train=x_train, y_train=y_train)
    result = evaluate_external_classifier(fitted, x_test=x_test, y_test=y_test, name=_normalize_name(name))
    return fitted, result


def compare_original_and_incorporated(
    name: str,
    x_train_original: np.ndarray,
    x_test_original: np.ndarray,
    x_train_incorporated: np.ndarray,
    x_test_incorporated: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 0,
    original_params: dict[str, Any] | None = None,
    incorporated_params: dict[str, Any] | None = None,
) -> dict[str, ExternalClassifierResult]:
    """Run a matched original-versus-incorporated classifier comparison."""
    _, original_result = fit_and_evaluate_external_classifier(
        name=name,
        x_train=x_train_original,
        y_train=y_train,
        x_test=x_test_original,
        y_test=y_test,
        random_state=random_state,
        **(original_params or {}),
    )
    _, incorporated_result = fit_and_evaluate_external_classifier(
        name=name,
        x_train=x_train_incorporated,
        y_train=y_train,
        x_test=x_test_incorporated,
        y_test=y_test,
        random_state=random_state,
        **(incorporated_params or {}),
    )
    return {
        "original": original_result,
        "incorporated": incorporated_result,
    }
