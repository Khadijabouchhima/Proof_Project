from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(ROOT / "src"),
)

from biovance.uncertainty import (
    UncertaintyEngine,
    load_uncertainty_config,
)


INPUT_PATH = (
    ROOT
    / "results"
    / "fusion"
    / "dryad_fusion_output.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "uncertainty"
)


def main() -> None:
    print("Reading:")
    print(INPUT_PATH)

    fusion = pd.read_parquet(
        INPUT_PATH
    )

    print("\nInput shape:")
    print(fusion.shape)

    required = [
        "patient_id",
        "timestamp",
        "fusion_code",
        "fusion_state",
        "fusion_confidence",
    ]

    missing = [
        col
        for col in required
        if col not in fusion.columns
    ]

    if missing:
        raise ValueError(
            "Missing Fusion columns: "
            + ", ".join(missing)
        )

    fusion[
        "timestamp"
    ] = pd.to_datetime(
        fusion[
            "timestamp"
        ]
    )

    duplicated = fusion.duplicated(
        subset=[
            "patient_id",
            "timestamp",
        ],
        keep=False,
    )

    if duplicated.any():
        raise ValueError(
            "Corrected Fusion output must contain "
            "one row per patient_id + timestamp."
        )

    # ---------------------------------------------------------
    # DRYAD contains physiological evidence only.
    #
    # Framingham participants are unrelated, therefore Risk
    # completeness is intentionally unavailable.
    # ---------------------------------------------------------
    uncertainty_input = pd.DataFrame(
        {
            "fusion_confidence": (
                fusion[
                    "fusion_confidence"
                ]
            ),

            "fusion_state": (
                fusion[
                    "fusion_state"
                ]
            ),

            "risk_input_completeness": (
                pd.Series(
                    [pd.NA] * len(fusion),
                    index=fusion.index,
                    dtype="Float64",
                )
            ),
        },
        index=fusion.index,
    )

    config = load_uncertainty_config()

    engine = UncertaintyEngine(
        config
    )

    scored = engine.score(
        uncertainty_input
    )

    result = pd.concat(
        [
            fusion.reset_index(
                drop=True
            ),

            scored.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # Correct moment-level invariant
    # ---------------------------------------------------------
    duplicates_after = result.duplicated(
        subset=[
            "patient_id",
            "timestamp",
        ],
        keep=False,
    )

    assert not duplicates_after.any(), (
        "Uncertainty output must preserve the "
        "one-row-per-moment Fusion contract."
    )

    usable = result[
        result[
            "fusion_code"
        ]
        == "OK"
    ].copy()

    print("\nUsable Fusion moments:")
    print(len(usable))

    # ---------------------------------------------------------
    # Levels
    # ---------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )
    print(
        "UNCERTAINTY LEVELS — USABLE DRYAD FUSION MOMENTS"
    )
    print(
        "=" * 80
    )

    level_summary = (
        usable[
            "uncertainty_level"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "uncertainty_level"
        )
        .reset_index(
            name="count"
        )
    )

    if len(usable) > 0:
        level_summary[
            "fraction"
        ] = (
            level_summary[
                "count"
            ]
            / len(usable)
        )
    else:
        level_summary[
            "fraction"
        ] = 0.0

    print(
        level_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Codes
    # ---------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )
    print(
        "UNCERTAINTY CODES — USABLE DRYAD FUSION MOMENTS"
    )
    print(
        "=" * 80
    )

    code_summary = (
        usable[
            "uncertainty_code"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "uncertainty_code"
        )
        .reset_index(
            name="count"
        )
    )

    if len(usable) > 0:
        code_summary[
            "fraction"
        ] = (
            code_summary[
                "count"
            ]
            / len(usable)
        )
    else:
        code_summary[
            "fraction"
        ] = 0.0

    print(
        code_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Distribution
    # ---------------------------------------------------------
    if len(usable) > 0:
        print(
            "\n"
            + "=" * 80
        )
        print(
            "UNCERTAINTY SCORE DISTRIBUTION"
        )
        print(
            "=" * 80
        )

        print(
            usable[
                "uncertainty_score"
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
    # Conflict effect
    # ---------------------------------------------------------
    if len(usable) > 0:
        conflict = usable[
            usable[
                "fusion_state"
            ]
            == "CONFLICTING_EVIDENCE"
        ]

        non_conflict = usable[
            usable[
                "fusion_state"
            ]
            != "CONFLICTING_EVIDENCE"
        ]

        print(
            "\n"
            + "=" * 80
        )
        print(
            "CONFLICT EFFECT"
        )
        print(
            "=" * 80
        )

        print(
            "Non-conflict moments:"
        )
        print(
            len(non_conflict)
        )

        if len(non_conflict) > 0:
            print(
                "Mean uncertainty:"
            )
            print(
                round(
                    non_conflict[
                        "uncertainty_score"
                    ].mean(),
                    4,
                )
            )

        print(
            "\nConflict moments:"
        )
        print(
            len(conflict)
        )

        if len(conflict) > 0:
            print(
                "Mean uncertainty:"
            )
            print(
                round(
                    conflict[
                        "uncertainty_score"
                    ].mean(),
                    4,
                )
            )

    # ---------------------------------------------------------
    # Correlations
    # ---------------------------------------------------------
    if (
        len(usable) > 1
        and usable[
            "fusion_confidence"
        ].nunique() > 1
    ):
        corr = (
            usable[
                [
                    "fusion_confidence",
                    "uncertainty_score",
                ]
            ]
            .corr()
            .iloc[
                0,
                1,
            ]
        )

        print(
            "\nFusion confidence vs uncertainty correlation:"
        )
        print(
            round(
                corr,
                4,
            )
        )

    # Direction agreement is retained only as a diagnostic.
    # It is NOT part of Uncertainty v1 scoring.
    if (
        "direction_agreement"
        in usable.columns
        and len(usable) > 1
        and usable[
            "direction_agreement"
        ].nunique() > 1
    ):
        corr = (
            usable[
                [
                    "direction_agreement",
                    "uncertainty_score",
                ]
            ]
            .corr()
            .iloc[
                0,
                1,
            ]
        )

        print(
            "\nDirection agreement vs uncertainty correlation "
            "(diagnostic only):"
        )
        print(
            round(
                corr,
                4,
            )
        )

    # ---------------------------------------------------------
    # Examples
    # ---------------------------------------------------------
    example_cols = [
        col
        for col in [
            "patient_id",
            "timestamp",
            "fusion_state",
            "fusion_score",
            "fusion_confidence",
            "direction_agreement",
            "certainty_score",
            "uncertainty_score",
            "uncertainty_level",
            "uncertainty_code",
            "uncertainty_reason_codes",
        ]
        if col in usable.columns
    ]

    if len(usable) > 0:
        print(
            "\n"
            + "=" * 80
        )
        print(
            "LOWEST UNCERTAINTY EXAMPLES"
        )
        print(
            "=" * 80
        )

        print(
            usable
            .sort_values(
                "uncertainty_score",
                ascending=True,
            )[
                example_cols
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        print(
            "\n"
            + "=" * 80
        )
        print(
            "HIGHEST UNCERTAINTY EXAMPLES"
        )
        print(
            "=" * 80
        )

        print(
            usable
            .sort_values(
                "uncertainty_score",
                ascending=False,
            )[
                example_cols
            ]
            .head(10)
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
        / "dryad_uncertainty_output.parquet"
    )

    level_path = (
        OUTPUT_DIR
        / "dryad_uncertainty_level_summary.csv"
    )

    code_path = (
        OUTPUT_DIR
        / "dryad_uncertainty_code_summary.csv"
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    level_summary.to_csv(
        level_path,
        index=False,
    )

    code_summary.to_csv(
        code_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)
    print(level_path)
    print(code_path)

    print("\nIMPORTANT:")
    print(
        "DRYAD validates physiological uncertainty only."
    )
    print(
        "Risk is intentionally unavailable because "
        "Framingham participants are unrelated."
    )
    print(
        "Each uncertainty row now represents exactly one "
        "patient/timestamp physiological moment."
    )


if __name__ == "__main__":
    main()