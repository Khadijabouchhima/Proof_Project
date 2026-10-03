"""Compare BioVance Risk Engine candidate models on Framingham.

Primary: L2-regularized logistic regression
Challenger: LightGBM

This experiment is for BACKGROUND 10-year CHD risk only.
It does not validate BioVance short-term early warning.

Run from project root:
    python experiments/risk/compare_risk_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "framingham" / "FraminghamHeartStudy.csv"
RESULTS_DIR = ROOT / "results" / "risk"
RESULTS_PATH = RESULTS_DIR / "framingham_model_comparison.csv"

TARGET = "TenYearCHD"

# education intentionally excluded from Risk v1.
FEATURES = [
    "male",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]

BINARY_FEATURES = [
    "male",
    "currentSmoker",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
]

CONTINUOUS_FEATURES = [
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Simple equal-width-bin ECE."""

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Include p=1.0 in final bin.
    bin_ids = np.digitize(
        y_prob,
        edges[1:-1],
        right=False,
    )

    ece = 0.0
    n = len(y_true)

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id

        if not mask.any():
            continue

        observed = y_true[mask].mean()
        predicted = y_prob[mask].mean()
        weight = mask.sum() / n

        ece += weight * abs(observed - predicted)

    return float(ece)


def build_logistic_pipeline() -> Pipeline:
    """Preprocessing is fitted inside every CV fold."""

    binary_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
        ]
    )

    continuous_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            (
                "binary",
                binary_pipe,
                BINARY_FEATURES,
            ),
            (
                "continuous",
                continuous_pipe,
                CONTINUOUS_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=2000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def build_lightgbm_pipeline() -> Pipeline:
    """Conservative fixed challenger; not hyperparameter tuned."""

    # Tree model does not need standardization.
    preprocess = ColumnTransformer(
        transformers=[
            (
                "binary",
                SimpleImputer(strategy="most_frequent"),
                BINARY_FEATURES,
            ),
            (
                "continuous",
                SimpleImputer(strategy="median"),
                CONTINUOUS_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def evaluate_model(
    name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: RepeatedStratifiedKFold,
) -> pd.DataFrame:

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model.fit(
            X_train,
            y_train,
        )

        prob = model.predict_proba(
            X_test
        )[:, 1]

        rows.append(
            {
                "model": name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_prevalence": float(y_test.mean()),
                "auroc": roc_auc_score(
                    y_test,
                    prob,
                ),
                "auprc": average_precision_score(
                    y_test,
                    prob,
                ),
                "brier": brier_score_loss(
                    y_test,
                    prob,
                ),
                "ece_10bin": expected_calibration_error(
                    y_test.to_numpy(),
                    prob,
                    n_bins=10,
                ),
            }
        )

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:

    summary = (
        results
        .groupby("model")
        .agg(
            folds=("fold", "count"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            brier_mean=("brier", "mean"),
            brier_sd=("brier", "std"),
            ece_mean=("ece_10bin", "mean"),
            ece_sd=("ece_10bin", "std"),
        )
        .reset_index()
    )

    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    print(
        summary.round(4).to_string(index=False)
    )

    logistic = (
        results[
            results["model"] == "logistic"
        ]
        .sort_values("fold")
        .reset_index(drop=True)
    )

    lightgbm = (
        results[
            results["model"] == "lightgbm"
        ]
        .sort_values("fold")
        .reset_index(drop=True)
    )

    if len(logistic) == len(lightgbm):

        diff = (
            logistic["auroc"].to_numpy()
            - lightgbm["auroc"].to_numpy()
        )

        print("\nPaired AUROC difference:")
        print(
            "logistic - lightgbm "
            f"mean={diff.mean():.4f}, "
            f"sd={diff.std(ddof=1):.4f}"
        )

        print(
            "Logistic wins folds:",
            int((diff > 0).sum()),
            "/",
            len(diff),
        )


def main() -> None:

    print("Reading:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing Framingham file:\n{DATA_PATH}\n\n"
            "Place FraminghamHeartStudy.csv in data/raw/."
        )

    df = pd.read_csv(DATA_PATH)

    print("\nShape:", df.shape)

    missing_columns = (
        set(FEATURES + [TARGET])
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    print(
        "\nTarget counts:"
    )
    print(
        y.value_counts().sort_index()
    )

    prevalence = float(y.mean())

    print(
        f"\nTenYearCHD prevalence: {prevalence:.4f}"
    )

    prevalence_brier = float(
        np.mean(
            (y.to_numpy() - prevalence) ** 2
        )
    )

    print(
        "Constant-prevalence Brier:",
        round(prevalence_brier, 4),
    )

    print(
        "\nMissing values in selected predictors:"
    )
    print(
        X.isna().sum()[
            X.isna().sum() > 0
        ]
    )

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42,
    )

    print(
        "\nEvaluating logistic regression..."
    )

    logistic_results = evaluate_model(
        name="logistic",
        model=build_logistic_pipeline(),
        X=X,
        y=y,
        cv=cv,
    )

    # Recreate CV object so both models receive exactly the
    # same deterministic sequence of splits.
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42,
    )

    print(
        "Evaluating LightGBM challenger..."
    )

    lightgbm_results = evaluate_model(
        name="lightgbm",
        model=build_lightgbm_pipeline(),
        X=X,
        y=y,
        cv=cv,
    )

    results = pd.concat(
        [
            logistic_results,
            lightgbm_results,
        ],
        ignore_index=True,
    )

    print_summary(
        results
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        "\nSaved:"
    )
    print(
        RESULTS_PATH
    )

    print(
        "\nIMPORTANT:"
    )
    print(
        "This evaluates background 10-year CHD risk."
    )
    print(
        "It does not validate short-term BioVance early warning."
    )
    print(
        "LightGBM is a challenger only; model selection comes "
        "after comparing discrimination and calibration."
    )


if __name__ == "__main__":
    main()
