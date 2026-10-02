"""Adapters from upstream tables to the engine's input.

`from_simulator_tables` follows the planned data contract (hourly / daily / bp). If your friend's
pipeline delivers one wide table, skip adapters and pass it straight to the engine: column aliases in
configs/personalization.yaml handle renames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def derive_context_from_activity(df: pd.DataFrame, activity_col: str, rest_max: float,
                                 sleep_col: str | None = None) -> pd.DataFrame:
    """Fallback when no context label exists: activity <= rest_max -> 'rest', else 'active';
    rows flagged asleep (truthy `sleep_col`) -> 'sleep'. Thresholds depend on the device: set them per dataset."""
    out = df.copy()
    out["context"] = np.where(out[activity_col] <= rest_max, "rest", "active")
    if sleep_col is not None:
        out.loc[out[sleep_col].astype(bool), "context"] = "sleep"
    return out


def from_simulator_tables(hourly: pd.DataFrame, daily: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    """Stack the three model-input tables into one wide, sparse observation table."""
    h = pd.DataFrame({"patient_id": hourly["subject_id"], "timestamp": hourly["timestamp"],
                      "hr": hourly["hr"], "hr_quality": hourly["hr_quality"],
                      "context": hourly["activity_state"]})
    d = pd.DataFrame({"patient_id": daily["subject_id"],
                      "timestamp": pd.to_datetime(daily["date"]) + pd.Timedelta(hours=4),
                      "hrv": daily["hrv"], "hrv_quality": daily["hrv_quality"], "context": "sleep"})
    b = pd.DataFrame({"patient_id": bp["subject_id"], "timestamp": bp["timestamp"],
                      "sbp": bp["sbp"], "dbp": bp["dbp"], "bp_quality": bp["bp_quality"], "context": "rest"})
    return pd.concat([h, d, b], ignore_index=True)


def from_pmdata_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt the daily PMData model-ready table to personalization input.

    PMData has already been aggregated to one row per participant/day.
    We use resting HR as the personalization signal because it is less
    confounded by activity than all-day HR.

    The adapter preserves the upstream number of HR samples so the
    personalization engine can distinguish a valid pre-aggregated day
    from a single raw observation.
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
        raise ValueError(f"PMData daily table missing required columns: {sorted(missing)}")

    out = pd.DataFrame({
        "patient_id": df["participant"].astype(str),
        "timestamp": pd.to_datetime(df["date"]),

        # Daily resting-HR summary
        "hr": pd.to_numeric(df["hr_rest_median"], errors="coerce"),

        # Important: original number of observations contributing
        # to this already-aggregated daily value
        "hr__n_source": pd.to_numeric(df["hr_rest_n"], errors="coerce"),

        # Dataset-level coverage is already on [0, 1]
        "hr_quality": pd.to_numeric(df["hr_coverage"], errors="coerce"),

        # Personalization baseline is resting HR
        "context": "rest",

        # Preserve useful provenance / validity information
        "hr_valid_day": df["hr_valid_day"].astype(bool),
        "preaggregated_daily": True,
    })

    # A day that upstream PMData marked invalid should not contribute
    # to the baseline.
    out.loc[~out["hr_valid_day"], "hr_quality"] = 0.0

    return out