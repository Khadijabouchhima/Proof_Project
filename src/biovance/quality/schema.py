"""Canonical input schema for the BioVance Data Quality Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biovance.quality.config import QualityConfig


def _find(
    df_cols: dict,
    aliases,
) -> str | None:
    """Return the original column matching one of the aliases."""
    for alias in aliases:
        if alias.lower() in df_cols:
            return df_cols[alias.lower()]

    return None


def _to_bool(value):
    """Convert common validity representations to bool/NaN."""

    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "valid", "good"}:
        return True

    if text in {"false", "0", "no", "n", "invalid", "bad"}:
        return False

    return np.nan


def normalize_quality_observations(
    df: pd.DataFrame,
    cfg: QualityConfig,
) -> pd.DataFrame:
    """Normalize a table to the canonical quality-engine schema.

    Canonical columns:

        patient_id
        timestamp
        signal
        value
        coverage
        sample_count
        sensor_confidence
        valid

    Optional quality fields are preserved as NaN when unavailable.
    """

    cols = {
        c.lower(): c
        for c in df.columns
    }

    pid_col = _find(
        cols,
        cfg.column_aliases["patient_id"],
    )

    ts_col = _find(
        cols,
        cfg.column_aliases["timestamp"],
    )

    signal_col = _find(
        cols,
        cfg.column_aliases["signal"],
    )

    value_col = _find(
        cols,
        cfg.column_aliases["value"],
    )

    if pid_col is None:
        raise ValueError(
            "quality input requires a patient id column"
        )

    if ts_col is None:
        raise ValueError(
            "quality input requires a timestamp column"
        )

    if signal_col is None:
        raise ValueError(
            "quality input requires a signal column"
        )

    if value_col is None:
        raise ValueError(
            "quality input requires a value column"
        )

    out = pd.DataFrame({
        "patient_id": df[pid_col].astype(str),
        "timestamp": pd.to_datetime(
            df[ts_col],
            errors="coerce",
        ),
        "signal": df[signal_col]
        .astype(str)
        .str.strip()
        .str.lower(),
        "value": pd.to_numeric(
            df[value_col],
            errors="coerce",
        ),
    })

    optional_numeric = {
        "coverage": "coverage",
        "sample_count": "sample_count",
        "sensor_confidence": "sensor_confidence",
    }

    for canonical, alias_key in optional_numeric.items():
        col = _find(
            cols,
            cfg.column_aliases[alias_key],
        )

        if col is None:
            out[canonical] = np.nan
        else:
            out[canonical] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    valid_col = _find(
        cols,
        cfg.column_aliases["valid"],
    )

    if valid_col is None:
        out["valid"] = np.nan
    else:
        out["valid"] = df[valid_col].map(_to_bool)

    if out["timestamp"].isna().any():
        raise ValueError(
            "quality input contains invalid timestamps"
        )

    known = {
        spec.name
        for spec in cfg.signals
    }

    unknown = sorted(
        set(out["signal"]) - known
    )

    if unknown:
        raise ValueError(
            f"unknown signals for quality engine: {unknown}"
        )

    return (
        out
        .sort_values(
            ["patient_id", "timestamp", "signal"]
        )
        .reset_index(drop=True)
    )