"""Dataset-specific adapters for the BioVance Data Quality Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd


def from_pmdata_daily(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert PMData daily model-ready rows to canonical quality input.

    Initial implementation assesses daily resting HR.

    The original PMData row is not deleted or modified.
    """

    required = {
        "participant",
        "date",
        "hr_rest_median",
        "hr_rest_n",
        "hr_coverage",
        "hr_valid_day",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "PMData quality adapter missing required columns: "
            f"{sorted(missing)}"
        )

    out = pd.DataFrame({
        "patient_id": df["participant"].astype(str),
        "timestamp": pd.to_datetime(
            df["date"],
            errors="coerce",
        ),
        "signal": "hr",
        "value": pd.to_numeric(
            df["hr_rest_median"],
            errors="coerce",
        ),
        "coverage": pd.to_numeric(
            df["hr_coverage"],
            errors="coerce",
        ),
        "sample_count": pd.to_numeric(
            df["hr_rest_n"],
            errors="coerce",
        ),
        "valid": df["hr_valid_day"],
    })

    # Preserve confidence as raw metadata if available.
    # The engine will ignore it until its normalization
    # is explicitly configured.
    if "hr_conf_mean" in df.columns:
        out["sensor_confidence"] = pd.to_numeric(
            df["hr_conf_mean"],
            errors="coerce",
        )
    else:
        out["sensor_confidence"] = np.nan

    return out


def from_canonical_observations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Pass-through adapter for already-canonical quality data."""
    return df.copy()