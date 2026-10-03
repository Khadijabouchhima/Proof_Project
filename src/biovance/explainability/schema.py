from __future__ import annotations

import pandas as pd


REQUIRED_EXPLAINABILITY_COLUMNS = (
    "decision_state",
    "decision_code",
    "uncertainty_score",
    "uncertainty_level",
)


OPTIONAL_EVIDENCE_COLUMNS = (
    # Fusion
    "fusion_state",
    "fusion_score",
    "fusion_confidence",
    "supporting_signals",
    "opposing_signals",
    "supporting_signal_count",
    "opposing_signal_count",
    "available_signal_count",

    # Risk
    "risk_probability",
    "risk_input_completeness",

    # Uncertainty reasons
    "uncertainty_code",
    "uncertainty_reason_codes",

    # Decision reasons
    "decision_reason_codes",
)


EXPLAINABILITY_OUTPUT_COLUMNS = (
    "explanation_summary",
    "primary_reason",
    "supporting_evidence",
    "limiting_factors",
    "risk_context",
    "confidence_statement",
    "explainability_version",
)


def validate_explainability_input(
    df: pd.DataFrame,
) -> None:
    """
    Validate the minimum contract required to explain an already-made
    BioVance decision.

    Evidence fields from Fusion and Risk are optional because real
    validation datasets intentionally exercise partial pipelines:
      - DRYAD has Fusion but no matching Framingham risk.
      - Framingham has Risk but no matching current Fusion evidence.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "Explainability input must be a pandas DataFrame."
        )

    missing = [
        column
        for column in REQUIRED_EXPLAINABILITY_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Explainability input is missing required columns: "
            + ", ".join(missing)
        )

    valid_states = {
        "WARN",
        "MONITOR",
        "ABSTAIN",
    }

    observed = set(
        df[
            "decision_state"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    invalid = (
        observed
        - valid_states
    )

    if invalid:
        raise ValueError(
            "Unknown decision_state values: "
            + ", ".join(
                sorted(invalid)
            )
        )