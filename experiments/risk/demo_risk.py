from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from biovance.risk import RiskEngine, load_risk_config


DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "framingham"
    / "FraminghamHeartStudy.csv"
)

RESULTS_DIR = ROOT / "results" / "risk"


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Equal-width expected calibration error.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)

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

        ece += (
            mask.sum() / n
        ) * abs(
            observed - predicted
        )

    return float(ece)


def evaluate(
    y_true: pd.Series,
    y_prob: np.ndarray,
) -> dict:
    return {
        "n": len(y_true),
        "prevalence": float(y_true.mean()),
        "auroc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),
        "auprc": float(
            average_precision_score(
                y_true,
                y_prob,
            )
        ),
        "brier": float(
            brier_score_loss(
                y_true,
                y_prob,
            )
        ),
        "ece_10bin": float(
            expected_calibration_error(
                y_true.to_numpy(),
                y_prob,
                n_bins=10,
            )
        ),
    }


def print_metrics(
    label: str,
    metrics: dict,
) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    for key, value in metrics.items():
        if key == "n":
            print(f"{key:12s}: {value}")
        else:
            print(f"{key:12s}: {value:.4f}")


def main() -> None:
    print("Reading:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Framingham file not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nShape:")
    print(df.shape)

    config = load_risk_config()

    target = config.target

    print("\nTarget distribution:")
    print(
        df[target]
        .value_counts()
        .sort_index()
    )

    print(
        "\nOverall prevalence:",
        round(
            float(df[target].mean()),
            4,
        ),
    )

    # ---------------------------------------------------------
    # 70% train
    # 30% temporary set
    # ---------------------------------------------------------
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df[target],
    )

    # ---------------------------------------------------------
    # Split temporary set equally:
    # 15% calibration
    # 15% final test
    # ---------------------------------------------------------
    calibration_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[target],
    )

    print("\nSplit sizes:")
    print(
        "Train:",
        len(train_df),
        f"({len(train_df) / len(df):.3f})",
    )
    print(
        "Calibration:",
        len(calibration_df),
        f"({len(calibration_df) / len(df):.3f})",
    )
    print(
        "Test:",
        len(test_df),
        f"({len(test_df) / len(df):.3f})",
    )

    print("\nSplit prevalence:")
    print(
        "Train:",
        round(
            float(train_df[target].mean()),
            4,
        ),
    )
    print(
        "Calibration:",
        round(
            float(calibration_df[target].mean()),
            4,
        ),
    )
    print(
        "Test:",
        round(
            float(test_df[target].mean()),
            4,
        ),
    )

    # ---------------------------------------------------------
    # Fit Risk Engine on TRAIN ONLY.
    # ---------------------------------------------------------
    print("\nFitting Risk Engine on training set...")

    engine = RiskEngine(
        config
    ).fit(
        train_df
    )

    # ---------------------------------------------------------
    # Calibration split evaluation.
    # We are NOT recalibrating yet.
    # This tells us whether calibration is necessary.
    # ---------------------------------------------------------
    calibration_output = engine.predict(
        calibration_df
    )

    calibration_prob = (
        calibration_output[
            "risk_probability"
        ].to_numpy()
    )

    calibration_metrics = evaluate(
        calibration_df[target],
        calibration_prob,
    )

    print_metrics(
        "CALIBRATION SET — UNCALIBRATED LOGISTIC",
        calibration_metrics,
    )

    # ---------------------------------------------------------
    # Final untouched TEST evaluation.
    #
    # Important:
    # We currently use the original logistic model because
    # no recalibration method has been selected from the
    # calibration set.
    # ---------------------------------------------------------
    test_output = engine.predict(
        test_df
    )

    test_prob = (
        test_output[
            "risk_probability"
        ].to_numpy()
    )

    test_metrics = evaluate(
        test_df[target],
        test_prob,
    )

    print_metrics(
        "FINAL TEST SET — UNCALIBRATED LOGISTIC",
        test_metrics,
    )

    # ---------------------------------------------------------
    # Compare Brier against simple prevalence-only prediction.
    # Use TRAIN prevalence, not full-dataset prevalence.
    # ---------------------------------------------------------
    train_prevalence = float(
        train_df[target].mean()
    )

    baseline_prob = np.full(
        len(test_df),
        train_prevalence,
    )

    baseline_brier = brier_score_loss(
        test_df[target],
        baseline_prob,
    )

    print("\nTest-set prevalence baseline:")
    print(
        "train prevalence:",
        round(
            train_prevalence,
            4,
        ),
    )
    print(
        "constant-probability Brier:",
        round(
            float(baseline_brier),
            4,
        ),
    )
    print(
        "risk model Brier:",
        round(
            test_metrics["brier"],
            4,
        ),
    )

    # ---------------------------------------------------------
    # Attach original Framingham row index so predictions
    # remain traceable without inventing patient IDs.
    # ---------------------------------------------------------
    calibration_predictions = (
        calibration_df[
            [target]
        ]
        .copy()
    )

    calibration_predictions[
        "source_row_index"
    ] = calibration_predictions.index

    calibration_predictions = (
        calibration_predictions
        .reset_index(
            drop=True
        )
    )

    calibration_predictions = pd.concat(
        [
            calibration_predictions,
            calibration_output
            .reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    calibration_predictions[
        "dataset_split"
    ] = "calibration"

    test_predictions = (
        test_df[
            [target]
        ]
        .copy()
    )

    test_predictions[
        "source_row_index"
    ] = test_predictions.index

    test_predictions = (
        test_predictions
        .reset_index(
            drop=True
        )
    )

    test_predictions = pd.concat(
        [
            test_predictions,
            test_output.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    test_predictions[
        "dataset_split"
    ] = "test"

    all_predictions = pd.concat(
        [
            calibration_predictions,
            test_predictions,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Metrics table
    # ---------------------------------------------------------
    metrics_df = pd.DataFrame(
        [
            {
                "split": "calibration",
                **calibration_metrics,
            },
            {
                "split": "test",
                **test_metrics,
            },
            {
                "split": "test_prevalence_baseline",
                "n": len(test_df),
                "prevalence": float(
                    test_df[target].mean()
                ),
                "auroc": np.nan,
                "auprc": np.nan,
                "brier": float(
                    baseline_brier
                ),
                "ece_10bin": np.nan,
            },
        ]
    )

    # ---------------------------------------------------------
    # Input completeness summary
    # ---------------------------------------------------------
    summary_df = pd.DataFrame(
        [
            {
                "dataset_rows": len(df),
                "train_rows": len(train_df),
                "calibration_rows": len(
                    calibration_df
                ),
                "test_rows": len(test_df),
                "train_prevalence": float(
                    train_df[target].mean()
                ),
                "calibration_prevalence": float(
                    calibration_df[target].mean()
                ),
                "test_prevalence": float(
                    test_df[target].mean()
                ),
                "test_mean_input_completeness": float(
                    test_output[
                        "risk_input_completeness"
                    ].mean()
                ),
                "test_low_completeness_count": int(
                    (
                        test_output[
                            "risk_code"
                        ]
                        == "LOW_INPUT_COMPLETENESS"
                    ).sum()
                ),
                "model_version": config.model_version,
            }
        ]
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        RESULTS_DIR
        / "framingham_risk_predictions.parquet"
    )

    metrics_path = (
        RESULTS_DIR
        / "framingham_risk_metrics.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "framingham_risk_summary.csv"
    )

    all_predictions.to_parquet(
        predictions_path,
        index=False,
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\nSaved:")
    print(predictions_path)
    print(metrics_path)
    print(summary_path)

    print("\nIMPORTANT")
    print(
        "This Risk Engine estimates background "
        "10-year CHD risk."
    )
    print(
        "It does not validate BioVance "
        "short-term deterioration detection."
    )
    print(
        "The test split was not used for fitting."
    )
    print(
        "No post-hoc probability calibration "
        "has been applied yet."
    )


if __name__ == "__main__":
    main()