"""Canonical schema for the BioVance Context Engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biovance.context.config import (
    ContextConfig,
)


def _find(
    df_cols: dict,
    aliases,
) -> str | None:
    """Return the first column matching an alias."""

    for alias in aliases:

        key = alias.lower()

        if key in df_cols:
            return df_cols[key]

    return None


def normalize_context_observations(
    df: pd.DataFrame,
    cfg: ContextConfig,
) -> pd.DataFrame:
    """Normalize Context Engine input.

    Canonical schema:

        patient_id
        timestamp_start
        timestamp_end

        sleep_overlap_fraction

        gyro_magnitude
        dynamic_acc
        acc_magnitude

        steps
        distance
        calories

        missingness_score

    Only patient_id and timestamp_start are strictly required.

    Movement clues are optional because Context Engine may need
    to return UNKNOWN when evidence is missing.
    """

    cols = {
        c.lower(): c
        for c in df.columns
    }

    pid_col = _find(
        cols,
        cfg.column_aliases[
            "patient_id"
        ],
    )

    start_col = _find(
        cols,
        cfg.column_aliases[
            "timestamp_start"
        ],
    )

    if pid_col is None:
        raise ValueError(
            "context input requires a patient id column"
        )

    if start_col is None:
        raise ValueError(
            "context input requires a timestamp_start column"
        )

    out = pd.DataFrame({

        "patient_id":
            df[pid_col]
            .astype(str)
            .str.strip(),

        "timestamp_start":
            pd.to_datetime(
                df[start_col],
                errors="coerce",
                utc=True,
            ),
    })

    # ---------------------------------------------------------
    # Optional end timestamp
    # ---------------------------------------------------------

    end_col = _find(
        cols,
        cfg.column_aliases[
            "timestamp_end"
        ],
    )

    if end_col is None:
        out["timestamp_end"] = pd.NaT

    else:
        out["timestamp_end"] = pd.to_datetime(
            df[end_col],
            errors="coerce",
            utc=True,
        )

    # ---------------------------------------------------------
    # Optional numeric fields
    # ---------------------------------------------------------

    optional_numeric = (
        "sleep_overlap_fraction",
        "gyro_magnitude",
        "dynamic_acc",
        "acc_magnitude",
        "steps",
        "distance",
        "calories",
        "missingness_score",
    )

    for canonical in optional_numeric:

        source = _find(
            cols,
            cfg.column_aliases[
                canonical
            ],
        )

        if source is None:
            out[canonical] = np.nan

        else:
            out[canonical] = pd.to_numeric(
                df[source],
                errors="coerce",
            )

    # ---------------------------------------------------------
    # Validate timestamps
    # ---------------------------------------------------------

    if out[
        "timestamp_start"
    ].isna().any():

        raise ValueError(
            "context input contains invalid "
            "timestamp_start values"
        )

    # ---------------------------------------------------------
    # Clamp sleep overlap to [0,1]
    #
    # A malformed upstream fraction should not propagate.
    # ---------------------------------------------------------

    out["sleep_overlap_fraction"] = (
        out["sleep_overlap_fraction"]
        .clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # ---------------------------------------------------------
    # Negative movement quantities are physically meaningless.
    #
    # Convert obvious malformed values to NaN instead of using
    # them as context evidence.
    # ---------------------------------------------------------

    nonnegative = (
        "gyro_magnitude",
        "dynamic_acc",
        "acc_magnitude",
        "steps",
        "distance",
        "calories",
    )

    for col in nonnegative:

        bad = (
            out[col].notna()
            & (out[col] < 0)
        )

        out.loc[
            bad,
            col,
        ] = np.nan

    return (
        out
        .sort_values(
            [
                "patient_id",
                "timestamp_start",
            ]
        )
        .reset_index(
            drop=True
        )
    )