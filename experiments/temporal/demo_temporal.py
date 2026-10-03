"""Run Temporal Engine v1 on real DRYAD deviation sequences."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from biovance.temporal import TemporalEngine


INPUT_PATH = (
    ROOT
    / "results"
    / "temporal"
    / "dryad_deviation_input.parquet"
)

CONFIG_PATH = (
    ROOT
    / "configs"
    / "temporal.yaml"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "temporal"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "dryad_temporal_output.parquet"
)

STATE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "dryad_temporal_state_summary.csv"
)

PARTICIPANT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "dryad_temporal_participant_summary.csv"
)


def main():

    print(
        "Reading DRYAD deviation sequence:"
    )

    print(
        INPUT_PATH
    )

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "DRYAD deviation input not found:\n"
            f"{INPUT_PATH}\n\n"
            "Run prepare_dryad_deviation_input.py first."
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        "\nInput shape:",
        df.shape,
    )

    print(
        "Participants:",
        df[
            "patient_id"
        ].nunique(),
    )

    print(
        "Signals:",
        sorted(
            df[
                "signal"
            ]
            .dropna()
            .unique()
        ),
    )

    print(
        "\nLoading Temporal Engine:"
    )

    print(
        CONFIG_PATH
    )

    engine = (
        TemporalEngine
        .from_yaml(
            CONFIG_PATH
        )
    )

    print(
        "\nRunning Temporal Engine..."
    )

    result = engine.transform(
        df
    )

    print(
        "\nRows in:",
        len(df),
    )

    print(
        "Rows out:",
        len(result),
    )

    if len(df) != len(result):

        raise RuntimeError(
            "Temporal Engine changed row count."
        )

    # ========================================================
    # Evaluation output only
    # ========================================================

    evaluation = result[
        result[
            "temporal_code"
        ]
        .isin(
            [
                "OK",
                "INSUFFICIENT_TEMPORAL_EVIDENCE",
            ]
        )
    ].copy()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TEMPORAL CODE COUNTS"
    )

    print(
        "=" * 72
    )

    print(
        result[
            "temporal_code"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TEMPORAL STATE COUNTS"
    )

    print(
        "=" * 72
    )

    print(
        evaluation[
            "temporal_state"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nTemporal state fractions:"
    )

    print(
        evaluation[
            "temporal_state"
        ]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .round(4)
    )

    # ========================================================
    # Confidence
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TEMPORAL CONFIDENCE"
    )

    print(
        "=" * 72
    )

    print(
        evaluation[
            "temporal_confidence"
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
    )

    # ========================================================
    # Persistence
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PERSISTENCE"
    )

    print(
        "=" * 72
    )

    print(
        evaluation[
            [
                "persistence_observation_count",
                "persistence_duration_minutes",
                "short_elevated_fraction",
                "persistence_score",
            ]
        ]
        .describe()
        .round(4)
    )

    # ========================================================
    # Recurrence
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RECURRENCE"
    )

    print(
        "=" * 72
    )

    print(
        evaluation[
            "recurrence_count"
        ]
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # Direction
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "DIRECTION CONSISTENCY"
    )

    print(
        "=" * 72
    )

    print(
        evaluation[
            "direction_consistency"
        ].describe()
    )

    print(
        "\nDominant direction:"
    )

    print(
        evaluation[
            "direction_dominant"
        ]
        .value_counts(
            dropna=False
        )
    )

    # ========================================================
    # Trend
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TREND"
    )

    print(
        "=" * 72
    )

    print(
        evaluation[
            "trend_score"
        ].describe(
            percentiles=[
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ]
        )
    )

    print(
        "\nWorsening trend fraction:"
    )

    print(
        evaluation[
            "worsening_trend"
        ].mean()
    )

    # ========================================================
    # By signal
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "BY SIGNAL"
    )

    print(
        "=" * 72
    )

    signal_summary = (
        evaluation
        .groupby(
            "signal"
        )
        .agg(
            n=(
                "temporal_state",
                "size",
            ),
            mean_confidence=(
                "temporal_confidence",
                "mean",
            ),
            mean_persistence=(
                "persistence_score",
                "mean",
            ),
            mean_recurrence=(
                "recurrence_score",
                "mean",
            ),
            mean_direction_consistency=(
                "direction_consistency",
                "mean",
            ),
            worsening_fraction=(
                "worsening_trend",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        signal_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Strong temporal evidence examples
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "STRONGEST TEMPORAL EVIDENCE EXAMPLES"
    )

    print(
        "=" * 72
    )

    examples = (
        evaluation[
            evaluation[
                "current_elevated"
            ]
        ]
        .sort_values(
            [
                "persistence_score",
                "recurrence_score",
                "temporal_confidence",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
    )

    display_columns = [
        "patient_id",
        "timestamp",
        "signal",
        "deviation_score",
        "deviation_magnitude",
        "temporal_state",
        "persistence_observation_count",
        "persistence_duration_minutes",
        "persistence_score",
        "recurrence_count",
        "direction_consistency",
        "trend_score",
        "temporal_confidence",
    ]

    print(
        examples[
            display_columns
        ]
        .head(40)
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Participant summary
    # ========================================================

    participant_summary = (
        evaluation
        .groupby(
            [
                "patient_id",
                "signal",
            ]
        )
        .agg(
            n_evaluation=(
                "temporal_state",
                "size",
            ),

            elevated_fraction=(
                "current_elevated",
                "mean",
            ),

            mean_persistence_score=(
                "persistence_score",
                "mean",
            ),

            max_persistence_score=(
                "persistence_score",
                "max",
            ),

            max_recurrence_count=(
                "recurrence_count",
                "max",
            ),

            mean_direction_consistency=(
                "direction_consistency",
                "mean",
            ),

            worsening_fraction=(
                "worsening_trend",
                "mean",
            ),

            mean_temporal_confidence=(
                "temporal_confidence",
                "mean",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # State summary
    # ========================================================

    state_summary = (
        evaluation[
            "temporal_state"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "temporal_state"
        )
        .reset_index(
            name="count"
        )
    )

    state_summary[
        "fraction"
    ] = (
        state_summary[
            "count"
        ]
        / len(
            evaluation
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
        OUTPUT_PATH,
        index=False,
    )

    state_summary.to_csv(
        STATE_SUMMARY_PATH,
        index=False,
    )

    participant_summary.to_csv(
        PARTICIPANT_SUMMARY_PATH,
        index=False,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SAVED"
    )

    print(
        "=" * 72
    )

    print(
        OUTPUT_PATH
    )

    print(
        STATE_SUMMARY_PATH
    )

    print(
        PARTICIPANT_SUMMARY_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Temporal states are descriptive evidence states."
    )

    print(
        "No WARN / MONITOR / ABSTAIN decision has been made."
    )

    print(
        "No future observation is used to score an earlier row."
    )


if __name__ == "__main__":
    main()