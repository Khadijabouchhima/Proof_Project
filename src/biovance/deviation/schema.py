"""Canonical input schema for the BioVance Deviation Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "patient_id",
    "timestamp",
    "signal",
    "value",
    "baseline_center",
    "baseline_scale",
)


OPTIONAL_DEFAULTS = {

    "baseline_n":
        np.nan,

    "baseline_status":
        "UNKNOWN",

    "baseline_personal_weight":
        np.nan,

    "baseline_population_weight":
        np.nan,

    "quality_score":
        np.nan,

    "quality_status":
        "UNKNOWN",

    "context_state":
        "UNKNOWN",

    "context_confidence":
        np.nan,
}


def normalize_deviation_input(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize data for the Deviation Engine.

    Required columns
    ----------------
    patient_id
    timestamp
    signal
    value
    baseline_center
    baseline_scale

    Optional metadata are preserved when available.
    """

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Deviation input missing required columns: "
            f"{missing}"
        )

    out = df.copy()

    # --------------------------------------------------------
    # IDs / timestamps / signal names
    # --------------------------------------------------------

    out["patient_id"] = (
        out["patient_id"]
        .astype(str)
        .str.strip()
    )

    out["timestamp"] = pd.to_datetime(
        out["timestamp"],
        errors="coerce",
        utc=True,
    )

    if out[
        "timestamp"
    ].isna().any():

        raise ValueError(
            "Deviation input contains invalid timestamps"
        )

    out["signal"] = (
        out["signal"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    numeric_columns = (
        "value",
        "baseline_center",
        "baseline_scale",
        "baseline_n",
        "baseline_personal_weight",
        "baseline_population_weight",
        "quality_score",
        "context_confidence",
    )

    for column in numeric_columns:

        if column not in out.columns:

            out[column] = (
                OPTIONAL_DEFAULTS[
                    column
                ]
            )

        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # String metadata
    # --------------------------------------------------------

    for column in (
        "baseline_status",
        "quality_status",
        "context_state",
    ):

        if column not in out.columns:

            out[column] = (
                OPTIONAL_DEFAULTS[
                    column
                ]
            )

        out[column] = (
            out[column]
            .fillna(
                OPTIONAL_DEFAULTS[
                    column
                ]
            )
            .astype(str)
            .str.upper()
        )

    return (
        out
        .sort_values(
            [
                "patient_id",
                "timestamp",
                "signal",
            ]
        )
        .reset_index(
            drop=True
        )
    )