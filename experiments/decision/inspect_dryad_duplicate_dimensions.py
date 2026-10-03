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
    df = pd.read_parquet(INPUT_PATH)

    usable = df[
        df["fusion_code"] == "OK"
    ].copy()

    key = [
        "patient_id",
        "timestamp",
    ]

    print("\nShape:")
    print(usable.shape)

    print("\nAll columns:")
    for col in usable.columns:
        print(col)

    # ---------------------------------------------------------
    # Find columns that vary inside patient/timestamp groups.
    # ---------------------------------------------------------
    candidate_cols = [
        col
        for col in usable.columns
        if col not in key
    ]

    summary = []

    grouped = usable.groupby(
        key,
        dropna=False,
    )

    for col in candidate_cols:

        try:
            nunique = grouped[
                col
            ].nunique(
                dropna=False
            )

            groups_varying = int(
                (
                    nunique > 1
                ).sum()
            )

            max_unique = int(
                nunique.max()
            )

            summary.append(
                {
                    "column": col,
                    "groups_varying": groups_varying,
                    "max_unique_within_group": max_unique,
                }
            )

        except TypeError:
            # Skip columns containing
            # non-hashable objects.
            continue

    summary_df = pd.DataFrame(
        summary
    ).sort_values(
        [
            "groups_varying",
            "max_unique_within_group",
        ],
        ascending=False,
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "COLUMNS THAT VARY WITHIN "
        "PATIENT/TIMESTAMP"
    )

    print(
        "=" * 90
    )

    print(
        summary_df[
            summary_df[
                "groups_varying"
            ]
            > 0
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Specifically look for likely temporal/window identifiers.
    # ---------------------------------------------------------
    keywords = [
        "window",
        "horizon",
        "lookback",
        "temporal",
        "period",
        "duration",
    ]

    likely_dimension_cols = [
        col
        for col in usable.columns
        if any(
            word in col.lower()
            for word in keywords
        )
    ]

    print(
        "\n"
        + "=" * 90
    )

    print(
        "LIKELY TEMPORAL / WINDOW COLUMNS"
    )

    print(
        "=" * 90
    )

    if likely_dimension_cols:
        for col in likely_dimension_cols:
            print(
                f"\n{col}:"
            )

            try:
                print(
                    usable[
                        col
                    ]
                    .value_counts(
                        dropna=False
                    )
                    .to_string()
                )
            except Exception:
                print(
                    "<could not summarize>"
                )
    else:
        print(
            "No obvious window/horizon "
            "column names found."
        )

    # ---------------------------------------------------------
    # Show complete rows for one mixed-decision timestamp.
    # ---------------------------------------------------------
    state_counts = (
        grouped[
            "decision_state"
        ]
        .nunique()
    )

    mixed_keys = (
        state_counts[
            state_counts > 1
        ]
        .reset_index()[
            key
        ]
    )

    if len(mixed_keys) > 0:

        example_key = (
            mixed_keys
            .iloc[0]
        )

        patient = (
            example_key[
                "patient_id"
            ]
        )

        timestamp = (
            example_key[
                "timestamp"
            ]
        )

        example = usable[
            (
                usable[
                    "patient_id"
                ]
                == patient
            )
            &
            (
                usable[
                    "timestamp"
                ]
                == timestamp
            )
        ]

        print(
            "\n"
            + "=" * 90
        )

        print(
            "FULL MIXED-DECISION EXAMPLE"
        )

        print(
            "=" * 90
        )

        print(
            f"patient_id={patient}"
        )

        print(
            f"timestamp={timestamp}"
        )

        print(
            example
            .to_string(
                index=False
            )
        )

    output_path = (
        ROOT
        / "results"
        / "decision"
        / "dryad_duplicate_dimension_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()