"""Schemas and machine-readable states for Temporal Engine v1."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class TemporalCode(str, Enum):
    OK = "OK"

    NOT_EVALUATION = "NOT_EVALUATION"

    CURRENT_DEVIATION_UNAVAILABLE = (
        "CURRENT_DEVIATION_UNAVAILABLE"
    )

    INSUFFICIENT_TEMPORAL_EVIDENCE = (
        "INSUFFICIENT_TEMPORAL_EVIDENCE"
    )


class TemporalState(str, Enum):
    UNKNOWN = "UNKNOWN"

    INSUFFICIENT_EVIDENCE = (
        "INSUFFICIENT_EVIDENCE"
    )

    NO_CURRENT_ELEVATION = (
        "NO_CURRENT_ELEVATION"
    )

    ISOLATED_ELEVATION = (
        "ISOLATED_ELEVATION"
    )

    PERSISTENT = "PERSISTENT"

    RECURRENT = "RECURRENT"

    PERSISTENT_RECURRENT = (
        "PERSISTENT_RECURRENT"
    )

    WORSENING = "WORSENING"

    PERSISTENT_WORSENING = (
        "PERSISTENT_WORSENING"
    )

    RECURRENT_WORSENING = (
        "RECURRENT_WORSENING"
    )

    PERSISTENT_RECURRENT_WORSENING = (
        "PERSISTENT_RECURRENT_WORSENING"
    )


REQUIRED_INPUT_COLUMNS = {
    "patient_id",
    "timestamp",
    "signal",
    "deviation_score",
    "deviation_magnitude",
    "risk_aligned_score",
    "deviation_code",
}


TEMPORAL_OUTPUT_COLUMNS = (
    "temporal_state",
    "temporal_code",

    "current_elevated",

    "persistence_observation_count",
    "persistence_duration_minutes",
    "short_elevated_fraction",
    "persistence_score",

    "recurrence_count",
    "recurrence_score",

    "direction_consistency",
    "direction_dominant",

    "trend_slope_z_per_hour",
    "trend_score",
    "worsening_trend",

    "evidence_count",
    "opportunity_count",
    "evidence_fraction",
    "gap_continuity",

    "temporal_confidence",

    "temporal_reason_codes",
    "temporal_explanation",
)


def validate_temporal_input(
    df: pd.DataFrame,
) -> None:
    """Validate required Temporal Engine columns."""

    missing = (
        REQUIRED_INPUT_COLUMNS
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Temporal input missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )