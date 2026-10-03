"""Schemas and states for BioVance Fusion Engine v1."""

from __future__ import annotations

from enum import Enum

import pandas as pd


class FusionCode(str, Enum):
    OK = "OK"

    NOT_EVALUATION = "NOT_EVALUATION"

    CURRENT_TEMPORAL_UNAVAILABLE = (
        "CURRENT_TEMPORAL_UNAVAILABLE"
    )

    INSUFFICIENT_MULTISIGNAL_EVIDENCE = (
        "INSUFFICIENT_MULTISIGNAL_EVIDENCE"
    )


class FusionState(str, Enum):
    UNKNOWN = "UNKNOWN"

    INSUFFICIENT_EVIDENCE = (
        "INSUFFICIENT_EVIDENCE"
    )

    NO_SUPPORT = "NO_SUPPORT"

    SINGLE_SIGNAL_SUPPORT = (
        "SINGLE_SIGNAL_SUPPORT"
    )

    MULTISIGNAL_SAME_FAMILY = (
        "MULTISIGNAL_SAME_FAMILY"
    )

    MULTIMODAL_SUPPORT = (
        "MULTIMODAL_SUPPORT"
    )

    MULTIMODAL_CONSENSUS = (
        "MULTIMODAL_CONSENSUS"
    )

    CONFLICTING_EVIDENCE = (
        "CONFLICTING_EVIDENCE"
    )


REQUIRED_INPUT_COLUMNS = {
    "patient_id",
    "timestamp",
    "signal",

    "risk_aligned_score",
    "deviation_magnitude",

    "persistence_score",
    "recurrence_score",

    "direction_consistency",

    "trend_score",
    "temporal_confidence",

    "temporal_code",
}


def validate_fusion_input(
    df: pd.DataFrame,
) -> None:

    missing = (
        REQUIRED_INPUT_COLUMNS
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            "Fusion input missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )