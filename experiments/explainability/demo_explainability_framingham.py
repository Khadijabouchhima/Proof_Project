from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project setup
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from biovance.explainability import (
    ExplainabilityEngine,
    load_explainability_config,
)


DECISION_PATH = (
    ROOT
    / "results"
    / "decision"
    / "framingham_decision_safety_validation.csv"
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
    / "explainability"
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
            f"{path}"
        )


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(missing)
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 100
    )

    print(
        "FRAMINGHAM EXPLAINABILITY VALIDATION"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    require_file(
        DECISION_PATH,
        "Framingham Decision safety output",
    )

    require_file(
        UNCERTAINTY_PATH,
        "Framingham Uncertainty output",
    )

    # --------------------------------------------------------
    # Decision output
    # --------------------------------------------------------

    print(
        "\nDecision input:"
    )

    print(
        DECISION_PATH
    )

    decision = pd.read_csv(
        DECISION_PATH
    )

    print(
        "\nDecision shape:"
    )

    print(
        decision.shape
    )

    require_columns(
        decision,
        [
            "risk_probability",
            "uncertainty_score",
            "uncertainty_code",
            "decision_state",
            "decision_code",
            "decision_reason_codes",
        ],
        "Framingham Decision output",
    )

    # --------------------------------------------------------
    # Upstream Uncertainty output
    #
    # The safety-validation CSV deliberately contained only
    # the fields needed for the Decision safety experiment.
    #
    # Explainability also needs uncertainty_level, so retrieve
    # that existing upstream value here.
    #
    # We do NOT recompute uncertainty.
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
            "risk_probability",
            "uncertainty_score",
            "uncertainty_level",
            "uncertainty_code",
            "uncertainty_reason_codes",
        ],
        "Framingham Uncertainty output",
    )

    # --------------------------------------------------------
    # Alignment checks
    # --------------------------------------------------------

    if len(
        decision
    ) != len(
        uncertainty
    ):
        raise ValueError(
            "Framingham Decision and Uncertainty outputs "
            "must describe the same rows.\n"
            f"Decision rows: {len(decision)}\n"
            f"Uncertainty rows: {len(uncertainty)}"
        )

    decision_risk = pd.to_numeric(
        decision[
            "risk_probability"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    uncertainty_risk = pd.to_numeric(
        uncertainty[
            "risk_probability"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    aligned = np.allclose(
        decision_risk,
        uncertainty_risk,
        equal_nan=True,
        rtol=1e-10,
        atol=1e-12,
    )

    if not aligned:
        raise ValueError(
            "Framingham Decision and Uncertainty outputs "
            "are not row-aligned. Risk probabilities differ."
        )

    decision_uncertainty = pd.to_numeric(
        decision[
            "uncertainty_score"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    upstream_uncertainty = pd.to_numeric(
        uncertainty[
            "uncertainty_score"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    uncertainty_aligned = np.allclose(
        decision_uncertainty,
        upstream_uncertainty,
        equal_nan=True,
        rtol=1e-10,
        atol=1e-12,
    )

    if not uncertainty_aligned:
        raise ValueError(
            "Framingham Decision and Uncertainty outputs "
            "are not row-aligned. Uncertainty scores differ."
        )

    print(
        "\nAlignment checks: PASSED"
    )

    # --------------------------------------------------------
    # Restore upstream Explainability fields.
    #
    # These are COPIED from Uncertainty output.
    # Nothing is recalculated here.
    # --------------------------------------------------------

    decision[
        "uncertainty_level"
    ] = (
        uncertainty[
            "uncertainty_level"
        ]
        .reset_index(
            drop=True
        )
    )

    decision[
        "uncertainty_reason_codes"
    ] = (
        uncertainty[
            "uncertainty_reason_codes"
        ]
        .reset_index(
            drop=True
        )
    )

    # Risk completeness is optional for Explainability, but
    # include it when available because it makes risk context
    # more informative.
    if (
        "risk_input_completeness"
        in uncertainty.columns
    ):
        decision[
            "risk_input_completeness"
        ] = (
            uncertainty[
                "risk_input_completeness"
            ]
            .reset_index(
                drop=True
            )
        )

    # --------------------------------------------------------
    # Framingham has no matching current physiological Fusion.
    #
    # Explicitly retain that fact.
    # --------------------------------------------------------

    decision[
        "fusion_state"
    ] = pd.Series(
        [pd.NA] * len(decision),
        dtype="string",
    )

    decision[
        "fusion_score"
    ] = pd.Series(
        [pd.NA] * len(decision),
        dtype="Float64",
    )

    decision[
        "fusion_confidence"
    ] = pd.Series(
        [pd.NA] * len(decision),
        dtype="Float64",
    )

    decision[
        "supporting_signals"
    ] = pd.Series(
        [pd.NA] * len(decision),
        dtype="string",
    )

    decision[
        "opposing_signals"
    ] = pd.Series(
        [pd.NA] * len(decision),
        dtype="string",
    )

    decision[
        "available_signal_count"
    ] = pd.Series(
        [pd.NA] * len(decision),
        dtype="Float64",
    )

    decision[
        "supporting_signal_count"
    ] = pd.Series(
        [pd.NA] * len(decision),
        dtype="Float64",
    )

    decision[
        "opposing_signal_count"
    ] = pd.Series(
        [pd.NA] * len(decision),
        dtype="Float64",
    )

    # --------------------------------------------------------
    # Confirm frozen Decision safety behavior BEFORE
    # Explainability runs.
    # --------------------------------------------------------

    warn_count_before = int(
        (
            decision[
                "decision_state"
            ]
            == "WARN"
        ).sum()
    )

    monitor_count_before = int(
        (
            decision[
                "decision_state"
            ]
            == "MONITOR"
        ).sum()
    )

    abstain_count_before = int(
        (
            decision[
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
        "UPSTREAM DECISION STATE"
    )

    print(
        "=" * 100
    )

    print(
        f"WARN: {warn_count_before}"
    )

    print(
        f"MONITOR: {monitor_count_before}"
    )

    print(
        f"ABSTAIN: {abstain_count_before}"
    )

    assert warn_count_before == 0

    assert monitor_count_before == 0

    assert abstain_count_before == len(
        decision
    )

    # --------------------------------------------------------
    # Explainability
    # --------------------------------------------------------

    config = load_explainability_config()

    engine = ExplainabilityEngine(
        config
    )

    explanations = engine.explain(
        decision
    )

    assert len(
        explanations
    ) == len(
        decision
    )

    result = pd.concat(
        [
            decision.reset_index(
                drop=True
            ),

            explanations.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Explainability must NOT change decisions.
    # --------------------------------------------------------

    assert (
        result[
            "decision_state"
        ]
        == decision[
            "decision_state"
        ]
        .reset_index(
            drop=True
        )
    ).all()

    assert (
        result[
            "decision_code"
        ]
        == decision[
            "decision_code"
        ]
        .reset_index(
            drop=True
        )
    ).all()

    assert (
        result[
            "decision_state"
        ]
        == "ABSTAIN"
    ).all()

    assert (
        result[
            "decision_code"
        ]
        == "ABSTAIN_FUSION_UNAVAILABLE"
    ).all()

    # --------------------------------------------------------
    # Explanation safety checks
    # --------------------------------------------------------

    summaries = (
        result[
            "explanation_summary"
        ]
        .astype(str)
        .str.lower()
    )

    fusion_unavailable_explained = (
        summaries
        .str.contains(
            "fusion evidence was unavailable",
            regex=False,
        )
        .all()
    )

    risk_alone_explained = (
        summaries
        .str.contains(
            "risk alone cannot",
            regex=False,
        )
        .all()
    )

    assert fusion_unavailable_explained, (
        "Every Framingham explanation should state that "
        "current Fusion evidence was unavailable."
    )

    assert risk_alone_explained, (
        "Every Framingham explanation should make clear that "
        "background risk alone cannot create a warning."
    )

    # --------------------------------------------------------
    # Highest background-risk cases
    # --------------------------------------------------------

    highest = (
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

    columns = [
        "risk_probability",
        "risk_input_completeness",
        "uncertainty_score",
        "uncertainty_level",
        "decision_state",
        "decision_code",
        "primary_reason",
        "explanation_summary",
        "risk_context",
    ]

    columns = [
        column
        for column in columns
        if column in highest.columns
    ]

    print(
        highest[
            columns
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "n_rows": len(
                    result
                ),
                "warn_count": int(
                    (
                        result[
                            "decision_state"
                        ]
                        == "WARN"
                    ).sum()
                ),
                "monitor_count": int(
                    (
                        result[
                            "decision_state"
                        ]
                        == "MONITOR"
                    ).sum()
                ),
                "abstain_count": int(
                    (
                        result[
                            "decision_state"
                        ]
                        == "ABSTAIN"
                    ).sum()
                ),
                "fusion_unavailable_explained": (
                    fusion_unavailable_explained
                ),
                "risk_alone_safety_explained": (
                    risk_alone_explained
                ),
                "explainability_version": (
                    config.version
                ),
            }
        ]
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
        / "framingham_explainability_output.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "framingham_explainability_summary.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FRAMINGHAM EXPLAINABILITY CHECK: PASSED"
    )

    print(
        "=" * 100
    )

    print(
        "\nExplainability preserved all 1,272 "
        "Framingham ABSTAIN decisions."
    )

    print(
        "Background risk alone was explicitly described "
        "as insufficient for a current warning."
    )

    print(
        "No uncertainty, risk, Fusion, or Decision value "
        "was recomputed."
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