"""Diagnose participants with unusual Context Engine behavior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "baigutanova_context_input.parquet"
)

THRESHOLD_PATH = (
    ROOT
    / "results"
    / "context"
    / "baigutanova_context_thresholds.csv"
)


REFERENCE_FRACTION = 0.60

PATIENTS = [
    "hp91",
    "us73",
]


def chronological_split_patient(
    group: pd.DataFrame,
    fraction: float,
):

    group = (
        group
        .sort_values("timestamp_start")
        .reset_index(drop=True)
    )

    split = int(
        len(group)
        * fraction
    )

    split = max(
        1,
        min(
            split,
            len(group) - 1,
        ),
    )

    return (
        group.iloc[:split].copy(),
        group.iloc[split:].copy(),
    )


def describe_gyro(
    values: pd.Series,
) -> dict:

    x = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if x.empty:

        return {
            "n": 0,
            "mean": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
        }

    return {
        "n": len(x),
        "mean": x.mean(),
        "p10": x.quantile(0.10),
        "p25": x.quantile(0.25),
        "p50": x.quantile(0.50),
        "p75": x.quantile(0.75),
        "p90": x.quantile(0.90),
        "p95": x.quantile(0.95),
    }


def main():

    print("Reading:")
    print(DATA_PATH)

    df = pd.read_parquet(
        DATA_PATH
    )

    df["timestamp_start"] = pd.to_datetime(
        df["timestamp_start"],
        utc=True,
    )

    thresholds = pd.read_csv(
        THRESHOLD_PATH
    )

    rows = []

    for patient_id in PATIENTS:

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"PATIENT: {patient_id}"
        )

        print(
            "=" * 70
        )

        patient = df[
            df["patient_id"].astype(str)
            == patient_id
        ].copy()

        if patient.empty:

            print(
                "Participant not found."
            )

            continue

        (
            reference,
            holdout,
        ) = chronological_split_patient(
            patient,
            REFERENCE_FRACTION,
        )

        # Only awake windows should be used when
        # examining movement thresholds.
        ref_awake = reference[
            (
                reference[
                    "sleep_overlap_fraction"
                ]
                == 0
            )
            &
            reference[
                "gyro_magnitude"
            ].notna()
        ]

        hold_awake = holdout[
            (
                holdout[
                    "sleep_overlap_fraction"
                ]
                == 0
            )
            &
            holdout[
                "gyro_magnitude"
            ].notna()
        ]

        threshold_row = thresholds[
            thresholds[
                "patient_id"
            ].astype(str)
            == patient_id
        ]

        if threshold_row.empty:

            print(
                "No threshold record found."
            )

            continue

        threshold_row = (
            threshold_row.iloc[0]
        )

        rest_threshold = float(
            threshold_row[
                "gyro_rest_threshold"
            ]
        )

        active_threshold = float(
            threshold_row[
                "gyro_active_threshold"
            ]
        )

        print(
            "\nThreshold source:"
        )

        print(
            threshold_row[
                "gyro_threshold_source"
            ]
        )

        print(
            "\nReference awake windows:"
        )

        print(
            len(ref_awake)
        )

        print(
            "\nREST threshold:"
        )

        print(
            rest_threshold
        )

        print(
            "\nACTIVE threshold:"
        )

        print(
            active_threshold
        )

        ref_stats = describe_gyro(
            ref_awake[
                "gyro_magnitude"
            ]
        )

        hold_stats = describe_gyro(
            hold_awake[
                "gyro_magnitude"
            ]
        )

        print(
            "\n=== REFERENCE AWAKE GYRO ==="
        )

        for key, value in ref_stats.items():

            print(
                f"{key:>6}: "
                f"{value:.4f}"
                if isinstance(
                    value,
                    float,
                )
                else f"{key:>6}: {value}"
            )

        print(
            "\n=== HOLDOUT AWAKE GYRO ==="
        )

        for key, value in hold_stats.items():

            print(
                f"{key:>6}: "
                f"{value:.4f}"
                if isinstance(
                    value,
                    float,
                )
                else f"{key:>6}: {value}"
            )

        # ----------------------------------------------------
        # Where holdout gyro sits relative to fitted thresholds
        # ----------------------------------------------------

        hold_gyro = hold_awake[
            "gyro_magnitude"
        ]

        rest_fraction = (
            hold_gyro
            <= rest_threshold
        ).mean()

        unknown_fraction = (
            (
                hold_gyro
                > rest_threshold
            )
            &
            (
                hold_gyro
                < active_threshold
            )
        ).mean()

        active_fraction = (
            hold_gyro
            >= active_threshold
        ).mean()

        print(
            "\n=== HOLDOUT POSITION RELATIVE TO THRESHOLDS ==="
        )

        print(
            "Below REST threshold:",
            round(
                rest_fraction,
                4,
            ),
        )

        print(
            "Between thresholds:",
            round(
                unknown_fraction,
                4,
            ),
        )

        print(
            "Above ACTIVE threshold:",
            round(
                active_fraction,
                4,
            ),
        )

        # ----------------------------------------------------
        # Distribution shift
        # ----------------------------------------------------

        median_shift = (
            hold_stats["p50"]
            - ref_stats["p50"]
        )

        p90_shift = (
            hold_stats["p90"]
            - ref_stats["p90"]
        )

        print(
            "\n=== REFERENCE -> HOLDOUT SHIFT ==="
        )

        print(
            "Median gyro shift:",
            round(
                median_shift,
                4,
            ),
        )

        print(
            "P90 gyro shift:",
            round(
                p90_shift,
                4,
            ),
        )

        rows.append({
            "patient_id":
                patient_id,

            "threshold_source":
                threshold_row[
                    "gyro_threshold_source"
                ],

            "reference_awake_n":
                len(ref_awake),

            "holdout_awake_n":
                len(hold_awake),

            "rest_threshold":
                rest_threshold,

            "active_threshold":
                active_threshold,

            "reference_p50":
                ref_stats["p50"],

            "reference_p90":
                ref_stats["p90"],

            "holdout_p50":
                hold_stats["p50"],

            "holdout_p90":
                hold_stats["p90"],

            "median_shift":
                median_shift,

            "p90_shift":
                p90_shift,

            "holdout_below_rest":
                rest_fraction,

            "holdout_between_thresholds":
                unknown_fraction,

            "holdout_above_active":
                active_fraction,
        })

    summary = pd.DataFrame(
        rows
    )

    print(
        "\n\n=== SUMMARY ==="
    )

    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )

    output_path = (
        ROOT
        / "results"
        / "context"
        / "context_outlier_diagnostics.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()