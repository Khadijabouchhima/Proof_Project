from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from biovance.decision import (
    DecisionEngine,
    load_decision_config,
)


RISK_PATH = (
    ROOT
    / "results"
    / "risk"
    / "framingham_risk_predictions.parquet"
)

UNCERTAINTY_PATH = (
    ROOT
    / "results"
    / "uncertainty"
    / "framingham_uncertainty_output.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "decision"
)


# ============================================================
# Helpers
# ============================================================

def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n"
            f"{path}\n\n"
            "Generate the required upstream result first."
        )


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(
                missing
            )
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 100
    )

    print(
        "FRAMINGHAM DECISION SAFETY CHECK"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Validate exact upstream files
    # --------------------------------------------------------

    require_file(
        RISK_PATH,
        "Framingham Risk output",
    )

    require_file(
        UNCERTAINTY_PATH,
        "Framingham Uncertainty output",
    )

    # --------------------------------------------------------
    # Read Framingham Risk
    # --------------------------------------------------------

    print(
        "\nRisk input:"
    )

    print(
        RISK_PATH
    )

    risk = pd.read_parquet(
        RISK_PATH
    )

    print(
        "\nRisk shape:"
    )

    print(
        risk.shape
    )

    require_columns(
        risk,
        [
            "risk_probability",
        ],
        "Framingham Risk output",
    )

    # --------------------------------------------------------
    # Read Framingham Uncertainty
    # --------------------------------------------------------

    print(
        "\nUncertainty input:"
    )

    print(
        UNCERTAINTY_PATH
    )

    uncertainty = pd.read_parquet(
        UNCERTAINTY_PATH
    )

    print(
        "\nUncertainty shape:"
    )

    print(
        uncertainty.shape
    )

    require_columns(
        uncertainty,
        [
            "uncertainty_score",
            "uncertainty_code",
        ],
        "Framingham Uncertainty output",
    )

    # --------------------------------------------------------
    # Alignment validation
    # --------------------------------------------------------

    if len(
        risk
    ) != len(
        uncertainty
    ):
        raise ValueError(
            "\nFramingham Risk and Framingham Uncertainty "
            "must describe the same rows.\n"
            f"Risk rows: {len(risk)}\n"
            f"Uncertainty rows: {len(uncertainty)}\n\n"
            "Do not combine DRYAD uncertainty with "
            "Framingham risk."
        )

    print(
        "\nAligned Framingham rows:"
    )

    print(
        len(risk)
    )

    # --------------------------------------------------------
    # Build Decision input
    #
    # Framingham provides long-term background cardiovascular
    # risk.
    #
    # It does NOT provide matching current wearable Fusion
    # evidence.
    #
    # Therefore Fusion is deliberately unavailable.
    # --------------------------------------------------------

    decision_input = pd.DataFrame(
        {
            "fusion_score": pd.Series(
                [pd.NA] * len(risk),
                dtype="Float64",
            ),

            "fusion_state": pd.Series(
                [pd.NA] * len(risk),
                dtype="string",
            ),

            "risk_probability": pd.to_numeric(
                risk[
                    "risk_probability"
                ],
                errors="coerce",
            )
            .reset_index(
                drop=True
            ),

            "uncertainty_score": pd.to_numeric(
                uncertainty[
                    "uncertainty_score"
                ],
                errors="coerce",
            )
            .reset_index(
                drop=True
            ),

            "uncertainty_code": (
                uncertainty[
                    "uncertainty_code"
                ]
                .astype(
                    "string"
                )
                .reset_index(
                    drop=True
                )
            ),
        }
    )

    print(
        "\nDecision input shape:"
    )

    print(
        decision_input.shape
    )

    # --------------------------------------------------------
    # Background-risk summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FRAMINGHAM BACKGROUND RISK DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    print(
        decision_input[
            "risk_probability"
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

    # --------------------------------------------------------
    # Uncertainty summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FRAMINGHAM UNCERTAINTY DISTRIBUTION"
    )

    print(
        "=" * 100
    )

    print(
        decision_input[
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

    # --------------------------------------------------------
    # Run Decision
    # --------------------------------------------------------

    config = load_decision_config()

    engine = DecisionEngine(
        config
    )

    decisions = engine.decide(
        decision_input
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    result = pd.concat(
        [
            decision_input.reset_index(
                drop=True
            ),

            decisions.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Decision state summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "DECISION STATES"
    )

    print(
        "=" * 100
    )

    state_summary = (
        result[
            "decision_state"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "decision_state"
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
        / len(result)
    )

    print(
        state_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Decision code summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "DECISION CODES"
    )

    print(
        "=" * 100
    )

    code_summary = (
        result[
            "decision_code"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "decision_code"
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

    # --------------------------------------------------------
    # Safety counts
    # --------------------------------------------------------

    warn_count = int(
        (
            result[
                "decision_state"
            ]
            == "WARN"
        ).sum()
    )

    monitor_count = int(
        (
            result[
                "decision_state"
            ]
            == "MONITOR"
        ).sum()
    )

    abstain_count = int(
        (
            result[
                "decision_state"
            ]
            == "ABSTAIN"
        ).sum()
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SAFETY CHECKS"
    )

    print(
        "=" * 100
    )

    print(
        f"Total Framingham rows: {len(result)}"
    )

    print(
        f"WARN count: {warn_count}"
    )

    print(
        f"MONITOR count: {monitor_count}"
    )

    print(
        f"ABSTAIN count: {abstain_count}"
    )

    # ========================================================
    # SAFETY INVARIANT 1
    #
    # Background risk alone must NEVER produce WARN.
    # ========================================================

    assert warn_count == 0, (
        "\nSAFETY FAILURE:\n"
        "Framingham background risk generated WARN "
        "without current physiological Fusion evidence."
    )

    # ========================================================
    # SAFETY INVARIANT 2
    #
    # Fusion is genuinely unavailable in Framingham.
    # Decision should therefore ABSTAIN.
    # ========================================================

    assert abstain_count == len(
        result
    ), (
        "\nSAFETY FAILURE:\n"
        "Framingham-only cases should ABSTAIN because "
        "current physiological Fusion evidence is unavailable."
    )

    # ========================================================
    # SAFETY INVARIANT 3
    #
    # Check highest-risk people specifically.
    # ========================================================

    highest_risk = (
        result
        .sort_values(
            "risk_probability",
            ascending=False,
        )
        .head(
            20
        )
        .copy()
    )

    highest_risk_warns = int(
        (
            highest_risk[
                "decision_state"
            ]
            == "WARN"
        ).sum()
    )

    assert highest_risk_warns == 0, (
        "\nSAFETY FAILURE:\n"
        "At least one of the highest-background-risk "
        "Framingham cases generated WARN without Fusion."
    )

    # --------------------------------------------------------
    # Confirm expected abstention reason
    # --------------------------------------------------------

    unexpected_codes = result[
        ~result[
            "decision_code"
        ].isin(
            [
                "ABSTAIN_FUSION_UNAVAILABLE",
                "ABSTAIN_HIGH_UNCERTAINTY",
                "ABSTAIN_INSUFFICIENT_EVIDENCE",
            ]
        )
    ]

    if len(
        unexpected_codes
    ) > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Some Framingham rows used an unexpected "
            "abstention pathway:"
        )

        print(
            unexpected_codes[
                [
                    "risk_probability",
                    "uncertainty_score",
                    "uncertainty_code",
                    "decision_state",
                    "decision_code",
                ]
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Highest-risk examples
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "HIGHEST BACKGROUND-RISK FRAMINGHAM EXAMPLES"
    )

    print(
        "=" * 100
    )

    example_columns = [
        "risk_probability",
        "uncertainty_score",
        "uncertainty_code",
        "decision_state",
        "decision_code",
        "decision_reason_codes",
    ]

    print(
        highest_risk[
            example_columns
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "framingham_decision_safety_validation.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "framingham_decision_safety_summary.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    safety_summary = pd.DataFrame(
        [
            {
                "n_rows": len(
                    result
                ),
                "warn_count": (
                    warn_count
                ),
                "monitor_count": (
                    monitor_count
                ),
                "abstain_count": (
                    abstain_count
                ),
                "max_risk_probability": (
                    result[
                        "risk_probability"
                    ].max()
                ),
                "mean_uncertainty": (
                    result[
                        "uncertainty_score"
                    ].mean()
                ),
                "background_risk_alone_warned": (
                    warn_count > 0
                ),
                "safety_check_passed": (
                    warn_count == 0
                    and
                    abstain_count
                    == len(result)
                ),
            }
        ]
    )

    safety_summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FRAMINGHAM SAFETY CHECK: PASSED"
    )

    print(
        "=" * 100
    )

    print(
        "\nBackground Framingham risk alone produced "
        "zero WARN decisions."
    )

    print(
        "Without current physiological Fusion evidence, "
        "Decision abstained."
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()