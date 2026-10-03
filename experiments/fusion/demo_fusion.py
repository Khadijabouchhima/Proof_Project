from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(ROOT / "src"),
)

from biovance.fusion import (
    FusionEngine,
    load_fusion_config,
)


INPUT_PATH = (
    ROOT
    / "results"
    / "temporal"
    / "dryad_temporal_output.parquet"
)

CONFIG_PATH = (
    ROOT
    / "configs"
    / "fusion.yaml"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "fusion"
)


def main() -> None:
    print("Reading:")
    print(INPUT_PATH)

    temporal = pd.read_parquet(
        INPUT_PATH
    )

    print("\nTemporal input shape:")
    print(temporal.shape)

    required = [
        "patient_id",
        "timestamp",
        "signal",
        "risk_aligned_score",
        "deviation_magnitude",
        "persistence_score",
        "recurrence_score",
        "direction_consistency",
        "trend_score",
        "temporal_confidence",
        "temporal_code",
    ]

    missing = [
        col
        for col in required
        if col not in temporal.columns
    ]

    if missing:
        raise ValueError(
            "Missing Temporal columns: "
            + ", ".join(missing)
        )

    temporal[
        "timestamp"
    ] = pd.to_datetime(
        temporal[
            "timestamp"
        ]
    )

    input_moments = (
        temporal[
            [
                "patient_id",
                "timestamp",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "\nUnique patient/timestamp moments "
        "in Temporal input:"
    )
    print(input_moments)

    config = load_fusion_config(
        CONFIG_PATH
    )

    engine = FusionEngine(
        config
    )

    fused = engine.transform(
        temporal
    )

    print("\nFusion output shape:")
    print(fused.shape)

    # ---------------------------------------------------------
    # Contract validation
    # ---------------------------------------------------------
    duplicated = fused.duplicated(
        subset=[
            "patient_id",
            "timestamp",
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicated.sum()
    )

    print(
        "\nDuplicate patient/timestamp "
        "Fusion rows:"
    )
    print(duplicate_count)

    assert duplicate_count == 0, (
        "Fusion must emit exactly one row per "
        "patient_id + timestamp."
    )

    assert len(fused) == input_moments, (
        "Fusion output count should equal the "
        "number of unique patient/timestamp moments."
    )

    usable = fused[
        fused[
            "fusion_code"
        ]
        == "OK"
    ].copy()

    print("\nUsable Fusion moments:")
    print(len(usable))

    # ---------------------------------------------------------
    # Fusion codes
    # ---------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )
    print(
        "FUSION CODES — ALL DRYAD MOMENTS"
    )
    print(
        "=" * 80
    )

    code_summary = (
        fused[
            "fusion_code"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "fusion_code"
        )
        .reset_index(
            name="count"
        )
    )

    code_summary[
        "fraction"
    ] = (
        code_summary[
            "count"
        ]
        / len(fused)
    )

    print(
        code_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Fusion states — usable only
    # ---------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )
    print(
        "FUSION STATES — USABLE DRYAD MOMENTS"
    )
    print(
        "=" * 80
    )

    state_summary = (
        usable[
            "fusion_state"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "fusion_state"
        )
        .reset_index(
            name="count"
        )
    )

    if len(usable) > 0:
        state_summary[
            "fraction"
        ] = (
            state_summary[
                "count"
            ]
            / len(usable)
        )
    else:
        state_summary[
            "fraction"
        ] = 0.0

    print(
        state_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Score distribution
    # ---------------------------------------------------------
    if len(usable) > 0:
        print(
            "\n"
            + "=" * 80
        )
        print(
            "FUSION SCORE DISTRIBUTION"
        )
        print(
            "=" * 80
        )

        print(
            usable[
                "fusion_score"
            ]
            .describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
        )

    # ---------------------------------------------------------
    # State means
    # ---------------------------------------------------------
    if len(usable) > 0:
        print(
            "\n"
            + "=" * 80
        )
        print(
            "MEAN FUSION SCORE BY STATE"
        )
        print(
            "=" * 80
        )

        by_state = (
            usable
            .groupby(
                "fusion_state"
            )
            .agg(
                count=(
                    "fusion_score",
                    "size",
                ),
                mean_fusion_score=(
                    "fusion_score",
                    "mean",
                ),
                mean_fusion_confidence=(
                    "fusion_confidence",
                    "mean",
                ),
                mean_support_fraction=(
                    "support_fraction",
                    "mean",
                ),
                mean_opposition_fraction=(
                    "opposition_fraction",
                    "mean",
                ),
            )
            .reset_index()
            .sort_values(
                "mean_fusion_score",
                ascending=False,
            )
        )

        print(
            by_state.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Highest-scoring examples
    # ---------------------------------------------------------
    if len(usable) > 0:
        example_cols = [
            col
            for col in [
                "patient_id",
                "timestamp",
                "context_state",
                "current_signal_count",
                "current_signals",
                "fusion_state",
                "fusion_score",
                "fusion_confidence",
                "available_signal_count",
                "supporting_signal_count",
                "opposing_signal_count",
                "support_fraction",
                "opposition_fraction",
                "contributing_signals",
                "supporting_signals",
                "opposing_signals",
            ]
            if col in usable.columns
        ]

        print(
            "\n"
            + "=" * 80
        )
        print(
            "HIGHEST FUSION SCORE EXAMPLES"
        )
        print(
            "=" * 80
        )

        print(
            usable
            .sort_values(
                "fusion_score",
                ascending=False,
            )[
                example_cols
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "dryad_fusion_output.parquet"
    )

    state_path = (
        OUTPUT_DIR
        / "dryad_fusion_state_summary.csv"
    )

    code_path = (
        OUTPUT_DIR
        / "dryad_fusion_code_summary.csv"
    )

    fused.to_parquet(
        output_path,
        index=False,
    )

    state_summary.to_csv(
        state_path,
        index=False,
    )

    code_summary.to_csv(
        code_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)
    print(state_path)
    print(code_path)

    print("\nIMPORTANT:")
    print(
        "Fusion now emits exactly one result per "
        "patient_id + timestamp."
    )
    print(
        "All simultaneous physiological signals are "
        "evaluated atomically."
    )


if __name__ == "__main__":
    main()