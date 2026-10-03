from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(ROOT / "src"),
)

from biovance.decision import (
    DecisionEngine,
    load_decision_config,
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "decision"
)


def main() -> None:
    config = load_decision_config()
    engine = DecisionEngine(config)

    cases = pd.DataFrame(
        {
            "case": [
                "strong_fusion_warn",
                "fusion_plus_risk_warn",
                "high_risk_only_monitor",
                "single_signal_monitor",
                "conflict_monitor",
                "uncertainty_blocks_warn",
                "high_uncertainty_abstain",
                "fusion_missing_abstain",
                "insufficient_evidence_abstain",
            ],

            "fusion_score": [
                0.85,
                0.65,
                0.20,
                0.90,
                0.90,
                0.90,
                0.90,
                None,
                0.90,
            ],

            "fusion_state": [
                "MULTIMODAL_CONSENSUS",
                "MULTIMODAL_SUPPORT",
                "NO_SUPPORT",
                "SINGLE_SIGNAL_SUPPORT",
                "CONFLICTING_EVIDENCE",
                "MULTIMODAL_CONSENSUS",
                "MULTIMODAL_CONSENSUS",
                None,
                "MULTIMODAL_CONSENSUS",
            ],

            "risk_probability": [
                0.10,
                0.30,
                0.80,
                0.50,
                0.50,
                0.50,
                0.30,
                0.80,
                0.30,
            ],

            "uncertainty_score": [
                0.20,
                0.20,
                0.20,
                0.20,
                0.30,
                0.50,
                0.70,
                0.30,
                0.90,
            ],

            "uncertainty_code": [
                "OK",
                "OK",
                "OK",
                "OK",
                "OK",
                "OK",
                "OK",
                "PARTIAL_EVIDENCE",
                "INSUFFICIENT_EVIDENCE",
            ],
        }
    )

    scored = engine.decide(
        cases[
            [
                "fusion_score",
                "fusion_state",
                "risk_probability",
                "uncertainty_score",
                "uncertainty_code",
            ]
        ]
    )

    result = pd.concat(
        [
            cases[["case"]].reset_index(drop=True),
            scored.reset_index(drop=True),
        ],
        axis=1,
    )

    print("\n" + "=" * 110)
    print("SYNTHETIC DECISION VALIDATION")
    print("=" * 110)

    print(
        result[
            [
                "case",
                "decision_state",
                "decision_code",
                "decision_reason_codes",
            ]
        ].to_string(index=False)
    )

    by_case = result.set_index("case")

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    assert (
        by_case.loc[
            "strong_fusion_warn",
            "decision_state",
        ]
        == "WARN"
    )

    assert (
        by_case.loc[
            "fusion_plus_risk_warn",
            "decision_state",
        ]
        == "WARN"
    )

    assert (
        by_case.loc[
            "high_risk_only_monitor",
            "decision_state",
        ]
        == "MONITOR"
    )

    assert (
        by_case.loc[
            "single_signal_monitor",
            "decision_state",
        ]
        == "MONITOR"
    )

    assert (
        by_case.loc[
            "conflict_monitor",
            "decision_state",
        ]
        == "MONITOR"
    )

    assert (
        by_case.loc[
            "uncertainty_blocks_warn",
            "decision_state",
        ]
        == "MONITOR"
    )

    assert (
        by_case.loc[
            "high_uncertainty_abstain",
            "decision_state",
        ]
        == "ABSTAIN"
    )

    assert (
        by_case.loc[
            "fusion_missing_abstain",
            "decision_state",
        ]
        == "ABSTAIN"
    )

    assert (
        by_case.loc[
            "insufficient_evidence_abstain",
            "decision_state",
        ]
        == "ABSTAIN"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "synthetic_decision_validation.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\nSanity checks: PASSED")

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()