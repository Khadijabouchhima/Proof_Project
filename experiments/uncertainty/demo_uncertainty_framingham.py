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
    / "risk"
    / "framingham_risk_predictions.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "uncertainty"
)


def main() -> None:
    print(
        "Reading:"
    )

    print(
        INPUT_PATH
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Risk output:\n"
            f"{INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        "\nInput shape:"
    )

    print(
        df.shape
    )

    required = [
        "risk_input_completeness",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required Risk columns: "
            + ", ".join(
                missing
            )
        )

    config = load_uncertainty_config()

    engine = UncertaintyEngine(
        config
    )

    # ---------------------------------------------------------
    # Framingham provides Risk evidence only.
    #
    # There is no matching DRYAD Fusion estimate for these
    # participants, so Fusion is intentionally unavailable.
    # ---------------------------------------------------------
    uncertainty_input = pd.DataFrame(
        {
            "fusion_confidence": pd.NA,
            "direction_agreement": pd.NA,
            "fusion_state": pd.NA,
            "risk_input_completeness": df[
                "risk_input_completeness"
            ],
        },
        index=df.index,
    )

    scored = engine.score(
        uncertainty_input
    )

    result = pd.concat(
        [
            df.reset_index(
                drop=True
            ),
            scored.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "UNCERTAINTY LEVELS — FRAMINGHAM RISK OUTPUT"
    )

    print(
        "=" * 80
    )

    level_summary = (
        result[
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

    level_summary[
        "fraction"
    ] = (
        level_summary[
            "count"
        ]
        / len(result)
    )

    print(
        level_summary.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "UNCERTAINTY CODES — FRAMINGHAM RISK OUTPUT"
    )

    print(
        "=" * 80
    )

    code_summary = (
        result[
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

    code_summary[
        "fraction"
    ] = (
        code_summary[
            "count"
        ]
        / len(result)
    )

    print(
        code_summary.to_string(
            index=False
        )
    )

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
        result[
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

    print(
        "\n"
        + "=" * 80
    )

    print(
        "RISK INPUT COMPLETENESS"
    )

    print(
        "=" * 80
    )

    print(
        result[
            "risk_input_completeness"
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

    valid = result[
        result[
            "risk_input_completeness"
        ].notna()
    ]

    if len(valid) > 1:
        corr = (
            valid[
                [
                    "risk_input_completeness",
                    "uncertainty_score",
                ]
            ]
            .corr()
            .loc[
                "risk_input_completeness",
                "uncertainty_score",
            ]
        )

        print(
            "\nRisk completeness vs uncertainty correlation:"
        )

        print(
            round(
                float(corr),
                4,
            )
        )

    example_cols = [
        col
        for col in [
            "source_row_index",
            "dataset_split",
            "TenYearCHD",
            "risk_probability",
            "risk_input_completeness",
            "risk_code",
            "certainty_score",
            "uncertainty_score",
            "uncertainty_level",
            "uncertainty_code",
            "uncertainty_reason_codes",
        ]
        if col in result.columns
    ]

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
        result
        .sort_values(
            "uncertainty_score"
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
        result
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "framingham_uncertainty_output.parquet"
    )

    level_path = (
        OUTPUT_DIR
        / "framingham_uncertainty_level_summary.csv"
    )

    code_path = (
        OUTPUT_DIR
        / "framingham_uncertainty_code_summary.csv"
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

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        level_path
    )

    print(
        code_path
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Framingham validates Risk-related uncertainty only."
    )

    print(
        "Fusion is intentionally unavailable because "
        "DRYAD participants are unrelated."
    )


if __name__ == "__main__":
    main()