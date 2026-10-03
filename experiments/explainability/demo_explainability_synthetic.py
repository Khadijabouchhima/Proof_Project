from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from biovance.explainability import (
    ExplainabilityEngine,
    load_explainability_config,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "explainability"
)


def main() -> None:

    cases = pd.DataFrame(
        [
            {
                "case": "strong_multimodal_warn",
                "decision_state": "WARN",
                "decision_code": "WARN_STRONG_FUSION",
                "decision_reason_codes": (
                    "STRONG_MULTIMODAL_FUSION"
                ),

                "fusion_state": "MULTIMODAL_CONSENSUS",
                "fusion_score": 0.88,
                "fusion_confidence": 0.96,
                "supporting_signals": '["sbp", "dbp", "hr"]',
                "opposing_signals": "[]",
                "available_signal_count": 3,
                "supporting_signal_count": 3,
                "opposing_signal_count": 0,

                "risk_probability": pd.NA,
                "risk_input_completeness": pd.NA,

                "uncertainty_score": 0.36,
                "uncertainty_level": "MODERATE",
                "uncertainty_code": "PARTIAL_EVIDENCE",
                "uncertainty_reason_codes": "RISK_UNAVAILABLE",
            },

            {
                "case": "fusion_plus_risk_warn",
                "decision_state": "WARN",
                "decision_code": "WARN_FUSION_WITH_RISK",
                "decision_reason_codes": (
                    "MULTIMODAL_FUSION|BACKGROUND_RISK_CONTEXT"
                ),

                "fusion_state": "MULTIMODAL_SUPPORT",
                "fusion_score": 0.67,
                "fusion_confidence": 0.95,
                "supporting_signals": '["sbp", "hr"]',
                "opposing_signals": "[]",
                "available_signal_count": 3,
                "supporting_signal_count": 2,
                "opposing_signal_count": 0,

                "risk_probability": 0.32,
                "risk_input_completeness": 1.0,

                "uncertainty_score": 0.24,
                "uncertainty_level": "LOW",
                "uncertainty_code": "FULL_EVIDENCE",
                "uncertainty_reason_codes": (
                    "EVIDENCE_WELL_SUPPORTED"
                ),
            },

            {
                "case": "single_signal_monitor",
                "decision_state": "MONITOR",
                "decision_code": "MONITOR_NO_ESCALATION",
                "decision_reason_codes": (
                    "NO_WARNING_CRITERIA_MET"
                ),

                "fusion_state": "SINGLE_SIGNAL_SUPPORT",
                "fusion_score": 0.47,
                "fusion_confidence": 0.94,
                "supporting_signals": '["hr"]',
                "opposing_signals": "[]",
                "available_signal_count": 3,
                "supporting_signal_count": 1,
                "opposing_signal_count": 0,

                "risk_probability": 0.28,
                "risk_input_completeness": 1.0,

                "uncertainty_score": 0.25,
                "uncertainty_level": "LOW",
                "uncertainty_code": "FULL_EVIDENCE",
                "uncertainty_reason_codes": (
                    "EVIDENCE_WELL_SUPPORTED"
                ),
            },

            {
                "case": "conflict_monitor",
                "decision_state": "MONITOR",
                "decision_code": "MONITOR_NO_ESCALATION",
                "decision_reason_codes": (
                    "NO_WARNING_CRITERIA_MET|"
                    "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
                ),

                "fusion_state": "CONFLICTING_EVIDENCE",
                "fusion_score": 0.31,
                "fusion_confidence": 0.95,
                "supporting_signals": '["sbp"]',
                "opposing_signals": '["hr"]',
                "available_signal_count": 3,
                "supporting_signal_count": 1,
                "opposing_signal_count": 1,

                "risk_probability": pd.NA,
                "risk_input_completeness": pd.NA,

                "uncertainty_score": 0.52,
                "uncertainty_level": "MODERATE",
                "uncertainty_code": "PARTIAL_EVIDENCE",
                "uncertainty_reason_codes": (
                    "RISK_UNAVAILABLE|"
                    "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
                ),
            },

            {
                "case": "uncertainty_blocks_warn",
                "decision_state": "MONITOR",
                "decision_code": "MONITOR_NO_ESCALATION",
                "decision_reason_codes": (
                    "NO_WARNING_CRITERIA_MET"
                ),

                "fusion_state": "MULTIMODAL_CONSENSUS",
                "fusion_score": 0.82,
                "fusion_confidence": 0.87,
                "supporting_signals": '["sbp", "dbp", "hr"]',
                "opposing_signals": "[]",
                "available_signal_count": 3,
                "supporting_signal_count": 3,
                "opposing_signal_count": 0,

                "risk_probability": pd.NA,
                "risk_input_completeness": pd.NA,

                "uncertainty_score": 0.403,
                "uncertainty_level": "MODERATE",
                "uncertainty_code": "PARTIAL_EVIDENCE",
                "uncertainty_reason_codes": "RISK_UNAVAILABLE",
            },

            {
                "case": "high_uncertainty_abstain",
                "decision_state": "ABSTAIN",
                "decision_code": "ABSTAIN_HIGH_UNCERTAINTY",
                "decision_reason_codes": "HIGH_UNCERTAINTY",

                "fusion_state": "MULTIMODAL_SUPPORT",
                "fusion_score": 0.67,
                "fusion_confidence": 0.50,
                "supporting_signals": '["sbp", "hr"]',
                "opposing_signals": "[]",
                "available_signal_count": 3,
                "supporting_signal_count": 2,
                "opposing_signal_count": 0,

                "risk_probability": pd.NA,
                "risk_input_completeness": pd.NA,

                "uncertainty_score": 0.70,
                "uncertainty_level": "HIGH",
                "uncertainty_code": "PARTIAL_EVIDENCE",
                "uncertainty_reason_codes": (
                    "RISK_UNAVAILABLE|LOW_FUSION_CONFIDENCE"
                ),
            },

            {
                "case": "risk_only_abstain",
                "decision_state": "ABSTAIN",
                "decision_code": "ABSTAIN_FUSION_UNAVAILABLE",
                "decision_reason_codes": "FUSION_UNAVAILABLE",

                "fusion_state": pd.NA,
                "fusion_score": pd.NA,
                "fusion_confidence": pd.NA,
                "supporting_signals": pd.NA,
                "opposing_signals": pd.NA,
                "available_signal_count": pd.NA,
                "supporting_signal_count": pd.NA,
                "opposing_signal_count": pd.NA,

                "risk_probability": 0.81,
                "risk_input_completeness": 1.0,

                "uncertainty_score": 0.65,
                "uncertainty_level": "HIGH",
                "uncertainty_code": "PARTIAL_EVIDENCE",
                "uncertainty_reason_codes": "FUSION_UNAVAILABLE",
            },
        ]
    )

    config = load_explainability_config()

    engine = ExplainabilityEngine(
        config
    )

    explained = engine.explain(
        cases
    )

    result = pd.concat(
        [
            cases[
                [
                    "case",
                    "decision_state",
                    "decision_code",
                ]
            ].reset_index(
                drop=True
            ),

            explained.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SYNTHETIC EXPLAINABILITY VALIDATION"
    )

    print(
        "=" * 100
    )

    for _, row in result.iterrows():

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"CASE: {row['case']}"
        )

        print(
            f"DECISION: {row['decision_state']}"
        )

        print(
            f"PRIMARY REASON: "
            f"{row['primary_reason']}"
        )

        print(
            f"SUMMARY: "
            f"{row['explanation_summary']}"
        )

        print(
            f"SUPPORTING EVIDENCE: "
            f"{row['supporting_evidence']}"
        )

        print(
            f"LIMITING FACTORS: "
            f"{row['limiting_factors']}"
        )

        print(
            f"RISK CONTEXT: "
            f"{row['risk_context']}"
        )

        print(
            f"CONFIDENCE: "
            f"{row['confidence_statement']}"
        )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    assert len(
        explained
    ) == len(
        cases
    )

    assert (
        explained[
            "explanation_summary"
        ]
        .str.len()
        .gt(0)
        .all()
    )

    risk_only = result[
        result[
            "case"
        ]
        == "risk_only_abstain"
    ].iloc[0]

    assert (
        "risk alone cannot"
        in risk_only[
            "explanation_summary"
        ].lower()
    )

    print(
        "\nSanity checks: PASSED"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "synthetic_explainability_validation.csv"
    )

    result.to_csv(
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