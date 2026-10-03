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


OUTPUT_DIR = (
    ROOT
    / "results"
    / "uncertainty"
)


def main() -> None:
    config = load_uncertainty_config()

    engine = UncertaintyEngine(
        config
    )

    cases = pd.DataFrame(
        {
            "case": [
                "strong_complete",
                "low_fusion_confidence",
                "conflicting_evidence",
                "low_risk_completeness",
                "fusion_missing",
                "risk_missing",
                "both_missing",
            ],

            "fusion_confidence": [
                0.95,
                0.25,
                0.90,
                0.90,
                None,
                0.90,
                None,
            ],

            "fusion_state": [
                "MULTIMODAL_CONSENSUS",
                "NO_SUPPORT",
                "CONFLICTING_EVIDENCE",
                "NO_SUPPORT",
                None,
                "NO_SUPPORT",
                None,
            ],

            "risk_input_completeness": [
                1.00,
                1.00,
                1.00,
                0.50,
                1.00,
                None,
                None,
            ],
        }
    )

    uncertainty_input = cases[
        [
            "fusion_confidence",
            "fusion_state",
            "risk_input_completeness",
        ]
    ]

    scored = engine.score(
        uncertainty_input
    )

    result = pd.concat(
        [
            cases[
                [
                    "case",
                    "fusion_confidence",
                    "fusion_state",
                    "risk_input_completeness",
                ]
            ].reset_index(
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
        + "=" * 120
    )

    print(
        "SYNTHETIC UNCERTAINTY VALIDATION"
    )

    print(
        "=" * 120
    )

    display_cols = [
        "case",
        "fusion_confidence",
        "fusion_state",
        "risk_input_completeness",
        "certainty_score",
        "uncertainty_score",
        "uncertainty_level",
        "uncertainty_code",
        "source_coverage",
        "coherence_score",
        "uncertainty_reason_codes",
    ]

    print(
        result[
            display_cols
        ].to_string(
            index=False
        )
    )

    by_case = result.set_index(
        "case"
    )

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    assert (
        by_case.loc[
            "strong_complete",
            "uncertainty_score",
        ]
        <
        by_case.loc[
            "low_fusion_confidence",
            "uncertainty_score",
        ]
    )

    assert (
        by_case.loc[
            "conflicting_evidence",
            "uncertainty_score",
        ]
        >
        by_case.loc[
            "strong_complete",
            "uncertainty_score",
        ]
    )

    assert (
        by_case.loc[
            "strong_complete",
            "uncertainty_score",
        ]
        <
        by_case.loc[
            "low_risk_completeness",
            "uncertainty_score",
        ]
    )

    assert (
        by_case.loc[
            "fusion_missing",
            "source_coverage",
        ]
        == 0.5
    )

    assert (
        by_case.loc[
            "risk_missing",
            "source_coverage",
        ]
        == 0.5
    )

    assert (
        by_case.loc[
            "both_missing",
            "source_coverage",
        ]
        == 0.0
    )

    assert (
        by_case.loc[
            "both_missing",
            "uncertainty_code",
        ]
        == "INSUFFICIENT_EVIDENCE"
    )

    assert (
        by_case.loc[
            "low_fusion_confidence",
            "uncertainty_level",
        ]
        == "MODERATE"
    )

    assert (
        "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
        in by_case.loc[
            "conflicting_evidence",
            "uncertainty_reason_codes",
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "synthetic_uncertainty_validation.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nSanity checks: PASSED"
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()