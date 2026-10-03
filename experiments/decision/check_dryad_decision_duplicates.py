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

    print("\nUsable rows:")
    print(len(usable))

    print("\nUnique patient/timestamp pairs:")
    print(
        usable[key]
        .drop_duplicates()
        .shape[0]
    )

    duplicates = usable[
        usable.duplicated(
            subset=key,
            keep=False,
        )
    ].copy()

    print("\nRows belonging to duplicated patient/timestamps:")
    print(len(duplicates))

    print("\nNumber of duplicated patient/timestamp groups:")
    print(
        duplicates[key]
        .drop_duplicates()
        .shape[0]
    )

    warn = usable[
        usable[
            "decision_state"
        ]
        == "WARN"
    ].copy()

    print("\nWARN rows:")
    print(len(warn))

    print("\nUnique WARN patient/timestamp pairs:")
    print(
        warn[key]
        .drop_duplicates()
        .shape[0]
    )

    warn_duplicates = warn[
        warn.duplicated(
            subset=key,
            keep=False,
        )
    ].copy()

    print("\nWARN rows with duplicated patient/timestamps:")
    print(len(warn_duplicates))

    if len(warn_duplicates) > 0:
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
            ]
            if col in warn_duplicates.columns
        ]

        print(
            "\nDuplicated WARN examples:"
        )

        print(
            warn_duplicates[
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

    print("\nWARN count by patient:")

    print(
        warn[
            "patient_id"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nUnique WARN timestamps by patient:"
    )

    unique_warn = (
        warn[
            key
        ]
        .drop_duplicates()
    )

    print(
        unique_warn[
            "patient_id"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()