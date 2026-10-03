from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "results"
    / "decision"
    / "dryad_decision_output.parquet"
)


def main() -> None:
    df = pd.read_parquet(
        INPUT_PATH
    )

    usable = df[
        df["fusion_code"] == "OK"
    ].copy()

    key = [
        "patient_id",
        "timestamp",
    ]

    print("\nUsable rows:")
    print(len(usable))

    print(
        "\nUnique patient/timestamp groups:"
    )
    print(
        usable[key]
        .drop_duplicates()
        .shape[0]
    )

    # ---------------------------------------------------------
    # Build one diagnostic row per patient/timestamp
    # ---------------------------------------------------------
    grouped = (
        usable
        .groupby(
            key,
            dropna=False,
        )
        .agg(
            row_count=(
                "decision_state",
                "size",
            ),

            n_fusion_states=(
                "fusion_state",
                "nunique",
            ),

            n_decision_states=(
                "decision_state",
                "nunique",
            ),

            min_fusion_score=(
                "fusion_score",
                "min",
            ),

            max_fusion_score=(
                "fusion_score",
                "max",
            ),

            min_fusion_confidence=(
                "fusion_confidence",
                "min",
            ),

            max_fusion_confidence=(
                "fusion_confidence",
                "max",
            ),

            min_uncertainty=(
                "uncertainty_score",
                "min",
            ),

            max_uncertainty=(
                "uncertainty_score",
                "max",
            ),
        )
        .reset_index()
    )

    grouped[
        "fusion_score_range"
    ] = (
        grouped[
            "max_fusion_score"
        ]
        - grouped[
            "min_fusion_score"
        ]
    )

    grouped[
        "uncertainty_range"
    ] = (
        grouped[
            "max_uncertainty"
        ]
        - grouped[
            "min_uncertainty"
        ]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ROWS PER PATIENT/TIMESTAMP"
    )

    print(
        "=" * 80
    )

    print(
        grouped[
            "row_count"
        ]
        .describe()
    )

    # ---------------------------------------------------------
    # Fusion-state consistency
    # ---------------------------------------------------------
    inconsistent_fusion = grouped[
        grouped[
            "n_fusion_states"
        ]
        > 1
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FUSION STATE CONSISTENCY"
    )

    print(
        "=" * 80
    )

    print(
        "Groups with >1 fusion state:"
    )
    print(
        len(
            inconsistent_fusion
        )
    )

    # ---------------------------------------------------------
    # Decision-state consistency
    # ---------------------------------------------------------
    inconsistent_decision = grouped[
        grouped[
            "n_decision_states"
        ]
        > 1
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DECISION STATE CONSISTENCY"
    )

    print(
        "=" * 80
    )

    print(
        "Groups with >1 decision state:"
    )
    print(
        len(
            inconsistent_decision
        )
    )

    # ---------------------------------------------------------
    # Show groups with mixed decisions
    # ---------------------------------------------------------
    if len(
        inconsistent_decision
    ) > 0:

        mixed_keys = (
            inconsistent_decision[
                key
            ]
        )

        mixed = usable.merge(
            mixed_keys,
            on=key,
            how="inner",
        )

        cols = [
            col
            for col in [
                "patient_id",
                "timestamp",
                "fusion_state",
                "fusion_score",
                "fusion_confidence",
                "uncertainty_score",
                "decision_state",
                "decision_code",
                "decision_reason_codes",
            ]
            if col in mixed.columns
        ]

        print(
            "\nMixed-decision groups:"
        )

        print(
            mixed[
                cols
            ]
            .sort_values(
                [
                    "patient_id",
                    "timestamp",
                    "fusion_score",
                ]
            )
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # WARN-related groups
    # ---------------------------------------------------------
    warn_groups = (
        usable[
            usable[
                "decision_state"
            ]
            == "WARN"
        ][
            key
        ]
        .drop_duplicates()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "WARN GROUPS"
    )

    print(
        "=" * 80
    )

    print(
        "Unique patient/timestamp groups "
        "containing at least one WARN:"
    )

    print(
        len(
            warn_groups
        )
    )

    # ---------------------------------------------------------
    # Score variation
    # ---------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )

    print(
        "WITHIN-GROUP FUSION SCORE VARIATION"
    )

    print(
        "=" * 80
    )

    print(
        grouped[
            "fusion_score_range"
        ]
        .describe()
    )

    print(
        "\nLargest score ranges:"
    )

    print(
        grouped
        .sort_values(
            "fusion_score_range",
            ascending=False,
        )
        .head(15)
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Save diagnostic
    # ---------------------------------------------------------
    output_path = (
        ROOT
        / "results"
        / "decision"
        / "dryad_decision_group_diagnostic.csv"
    )

    grouped.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()