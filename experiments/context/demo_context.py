"""Evaluate personalized BioVance Context Engine on BAIGUTANOVA.

Important:
    Thresholds are fitted using the FIRST 60% of each participant's
    chronological data.

    Evaluation is performed using the LAST 40%.

This prevents future observations from influencing the personalized
movement thresholds used to classify the holdout period.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from biovance.context import (
    ContextEngine,
    load_context_config,
)


DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "baigutanova_context_input.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "context"
)

HOLDOUT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "baigutanova_context_personalized_holdout.parquet"
)

THRESHOLD_PATH = (
    OUTPUT_DIR
    / "baigutanova_context_thresholds.csv"
)

PARTICIPANT_PATH = (
    OUTPUT_DIR
    / "baigutanova_context_personalized_by_participant.csv"
)


REFERENCE_FRACTION = 0.60


def chronological_split(
    df: pd.DataFrame,
    fraction: float = 0.60,
):

    reference_parts = []
    holdout_parts = []

    for patient_id, group in df.groupby(
        "patient_id"
    ):

        group = (
            group
            .sort_values(
                "timestamp_start"
            )
            .reset_index(
                drop=True
            )
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

        reference_parts.append(
            group.iloc[
                :split
            ].copy()
        )

        holdout_parts.append(
            group.iloc[
                split:
            ].copy()
        )

    reference = pd.concat(
        reference_parts,
        ignore_index=True,
    )

    holdout = pd.concat(
        holdout_parts,
        ignore_index=True,
    )

    return (
        reference,
        holdout,
    )


def context_fraction_table(
    result,
):

    counts = (
        result
        .groupby(
            [
                "patient_id",
                "context_state",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    states = [
        "SLEEP",
        "REST",
        "ACTIVE",
        "UNKNOWN",
    ]

    for state in states:

        if state not in counts.columns:
            counts[
                state
            ] = 0

    counts = counts[
        states
    ]

    fractions = (
        counts.div(
            counts.sum(
                axis=1
            ),
            axis=0,
        )
    )

    return (
        counts,
        fractions,
    )


def main():

    print("Reading:")
    print(DATA_PATH)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Missing context data: {DATA_PATH}"
        )

    df = pd.read_parquet(
        DATA_PATH
    )

    df[
        "timestamp_start"
    ] = pd.to_datetime(
        df[
            "timestamp_start"
        ],
        utc=True,
    )

    print("\nFull dataset:")
    print(df.shape)

    print("\nParticipants:")
    print(
        df[
            "patient_id"
        ].nunique()
    )

    # ========================================================
    # Chronological split
    # ========================================================

    (
        reference,
        holdout,
    ) = chronological_split(
        df,
        REFERENCE_FRACTION,
    )

    print(
        "\n=== CHRONOLOGICAL SPLIT ==="
    )

    print(
        "Reference fraction:",
        REFERENCE_FRACTION,
    )

    print(
        "Reference rows:",
        len(reference),
    )

    print(
        "Holdout rows:",
        len(holdout),
    )

    print(
        "Reference participants:",
        reference[
            "patient_id"
        ].nunique(),
    )

    print(
        "Holdout participants:",
        holdout[
            "patient_id"
        ].nunique(),
    )

    # ========================================================
    # Fit personalized engine
    # ========================================================

    cfg = load_context_config(
        "baigutanova"
    )

    engine = ContextEngine(
        cfg
    )

    engine.fit(
        reference
    )

    print(
        "\n=== FITTED GLOBAL GYRO REFERENCE ==="
    )

    print(
        "Global reference n:",
        engine.global_reference_n,
    )

    print(
        "Global REST threshold:",
        round(
            engine.global_rest_gyro,
            4,
        ),
    )

    print(
        "Global ACTIVE threshold:",
        round(
            engine.global_active_gyro,
            4,
        ),
    )

    # ========================================================
    # Threshold table
    # ========================================================

    threshold_table = (
        engine.threshold_table()
    )

    print(
        "\n=== PERSONALIZED THRESHOLDS ==="
    )

    print(
        threshold_table
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n=== THRESHOLD SOURCE COUNTS ==="
    )

    print(
        threshold_table[
            "gyro_threshold_source"
        ]
        .value_counts()
    )

    # ========================================================
    # Evaluate HOLDOUT only
    # ========================================================

    result = engine.classify(
        holdout
    )

    print(
        "\n=== HOLDOUT SANITY CHECK ==="
    )

    print(
        "Input rows:",
        len(holdout),
    )

    print(
        "Output rows:",
        len(result),
    )

    print(
        "Rows preserved:",
        len(holdout)
        == len(result),
    )

    print(
        "Missing context:",
        result[
            "context_state"
        ].isna().sum(),
    )

    # ========================================================
    # Global holdout distribution
    # ========================================================

    print(
        "\n=== PERSONALIZED HOLDOUT CONTEXT DISTRIBUTION ==="
    )

    print(
        result[
            "context_state"
        ]
        .value_counts()
    )

    print(
        "\n=== PERSONALIZED HOLDOUT CONTEXT FRACTIONS ==="
    )

    print(
        result[
            "context_state"
        ]
        .value_counts(
            normalize=True
        )
        .round(4)
    )

    print(
        "\n=== HOLDOUT ACTIVITY LEVEL ==="
    )

    print(
        result[
            "activity_level"
        ].describe()
    )

    print(
        "\n=== HOLDOUT CONTEXT CONFIDENCE ==="
    )

    print(
        result[
            "context_confidence"
        ].describe()
    )

    # ========================================================
    # Participant distributions
    # ========================================================

    (
        counts,
        fractions,
    ) = context_fraction_table(
        result
    )

    print(
        "\n=== HOLDOUT CONTEXT FRACTION BY PARTICIPANT ==="
    )

    print(
        fractions
        .round(3)
        .to_string()
    )

    print(
        "\n=== HIGHEST ACTIVE FRACTION ==="
    )

    print(
        fractions[
            "ACTIVE"
        ]
        .sort_values(
            ascending=False
        )
        .head(10)
        .round(3)
    )

    print(
        "\n=== HIGHEST UNKNOWN FRACTION ==="
    )

    print(
        fractions[
            "UNKNOWN"
        ]
        .sort_values(
            ascending=False
        )
        .head(10)
        .round(3)
    )

    print(
        "\n=== HIGHEST REST FRACTION ==="
    )

    print(
        fractions[
            "REST"
        ]
        .sort_values(
            ascending=False
        )
        .head(10)
        .round(3)
    )

    # ========================================================
    # Thresholds versus resulting behavior
    # ========================================================

    summary = (
        counts
        .add_suffix(
            "_count"
        )
        .join(
            fractions
            .add_suffix(
                "_fraction"
            )
        )
        .reset_index()
    )

    summary = summary.merge(
        threshold_table,
        on="patient_id",
        how="left",
    )

    print(
        "\n=== PARTICIPANT THRESHOLD / CONTEXT SUMMARY ==="
    )

    print(
        summary[
            [
                "patient_id",
                "gyro_rest_threshold",
                "gyro_active_threshold",
                "gyro_reference_n",
                "gyro_threshold_source",
                "REST_fraction",
                "ACTIVE_fraction",
                "UNKNOWN_fraction",
                "SLEEP_fraction",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Diagnostic review
    # ========================================================

    summary[
        "high_active_review"
    ] = (
        summary[
            "ACTIVE_fraction"
        ]
        > 0.30
    )

    summary[
        "high_unknown_review"
    ] = (
        summary[
            "UNKNOWN_fraction"
        ]
        > 0.50
    )

    flagged = summary[
        summary[
            "high_active_review"
        ]
        |
        summary[
            "high_unknown_review"
        ]
    ]

    print(
        "\n=== PARTICIPANTS FOR MANUAL REVIEW ==="
    )

    if flagged.empty:

        print(
            "None."
        )

    else:

        print(
            flagged[
                [
                    "patient_id",
                    "gyro_rest_threshold",
                    "gyro_active_threshold",
                    "gyro_reference_n",
                    "gyro_threshold_source",
                    "ACTIVE_fraction",
                    "UNKNOWN_fraction",
                ]
            ]
            .round(3)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        HOLDOUT_OUTPUT_PATH,
        index=False,
    )

    threshold_table.to_csv(
        THRESHOLD_PATH,
        index=False,
    )

    summary.to_csv(
        PARTICIPANT_PATH,
        index=False,
    )

    print(
        "\n=== SAVED ==="
    )

    print(
        HOLDOUT_OUTPUT_PATH
    )

    print(
        THRESHOLD_PATH
    )

    print(
        PARTICIPANT_PATH
    )

    print(
        "\nPersonalized context evaluation complete."
    )


if __name__ == "__main__":
    main()