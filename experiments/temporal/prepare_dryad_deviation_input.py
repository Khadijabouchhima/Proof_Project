"""Prepare leakage-safe DRYAD deviation sequences for the BioVance Temporal Engine.

Purpose
-------
Convert preprocessed DRYAD repeated measurements into standardized,
per-participant deviation sequences for Temporal Engine development.

This is a VALIDATION BRIDGE, not a replacement for the full
BioVance Personalization Engine.

Important improvements in this version
--------------------------------------
1. Duplicate patient/signal/timestamp rows are collapsed.
2. Only the earliest valid WAKE observations build the reference.
3. The reference is frozen before evaluation.
4. SLEEP observations are preserved but not scored against WAKE baseline.
5. Personal reference scale is estimated robustly.
6. Personal scale is SHRUNK toward a robust signal-level scale prior.
7. A small global-relative scale floor prevents numerical scale collapse.
8. Raw and final scale estimates are both preserved for explainability.
9. Evaluation observations never update the baseline.

No temporal persistence, recurrence, warning, or risk decision is made here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


# ============================================================
# Paths
# ============================================================

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "dryad_temporal_input.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "temporal"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "dryad_deviation_input.parquet"
)

REFERENCE_PATH = (
    OUTPUT_DIR
    / "dryad_reference_table.csv"
)

GLOBAL_SCALE_PATH = (
    OUTPUT_DIR
    / "dryad_global_scale_table.csv"
)

PARTICIPANT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "dryad_deviation_participant_summary.csv"
)


# ============================================================
# Experiment configuration
# ============================================================

SIGNALS = (
    "sbp",
    "dbp",
    "hr",
)

REFERENCE_CONTEXT = "WAKE"

REFERENCE_TARGET_N = 8

REFERENCE_MIN_N = 5


# ------------------------------------------------------------
# Scale stabilization
# ------------------------------------------------------------

# Empirical-Bayes style shrinkage:
#
#     w = n / (n + SHRINKAGE_STRENGTH)
#
# With n=8 and strength=8:
#
#     w = 0.5
#
# So personal scale and global scale each contribute.
#
# This is an engineering choice for the short DRYAD reference,
# not a universal physiological constant.

SCALE_SHRINKAGE_STRENGTH = 8.0


# Final numerical safety floor:
#
#     final_scale >= global_scale * MIN_GLOBAL_SCALE_FRACTION
#
# Normally shrinkage should already prevent collapse.
# This exists only as a final safeguard.

MIN_GLOBAL_SCALE_FRACTION = 0.25


# Need enough participant-level scales before trusting
# a signal-level global prior.

MIN_GLOBAL_SCALE_PARTICIPANTS = 10


SIGNAL_ORIENTATION = {
    "sbp": 1,
    "dbp": 1,
    "hr": 1,
}


# ============================================================
# Robust statistics
# ============================================================

def robust_center(
    values: pd.Series,
) -> float:
    """Median center."""

    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    return float(
        clean.median()
    )


def mad_scale(
    values: pd.Series,
) -> float:
    """Gaussian-consistent MAD scale."""

    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if clean.empty:
        return np.nan

    center = clean.median()

    mad = (
        clean
        .sub(center)
        .abs()
        .median()
    )

    if (
        pd.isna(mad)
        or mad <= 0
    ):
        return np.nan

    return float(
        1.4826 * mad
    )


def iqr_scale(
    values: pd.Series,
) -> float:
    """Gaussian-consistent IQR scale."""

    clean = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(clean) < 2:
        return np.nan

    q25 = clean.quantile(0.25)
    q75 = clean.quantile(0.75)

    iqr = q75 - q25

    if (
        pd.isna(iqr)
        or iqr <= 0
    ):
        return np.nan

    return float(
        iqr / 1.349
    )


def estimate_personal_scale(
    values: pd.Series,
) -> tuple[
    float,
    float,
    str,
]:
    """Estimate personal reference center and RAW scale."""

    center = robust_center(
        values
    )

    scale = mad_scale(
        values
    )

    if (
        not pd.isna(scale)
        and scale > 0
    ):

        return (
            center,
            scale,
            "MAD",
        )

    fallback = iqr_scale(
        values
    )

    if (
        not pd.isna(fallback)
        and fallback > 0
    ):

        return (
            center,
            fallback,
            "IQR_FALLBACK",
        )

    return (
        center,
        np.nan,
        "UNUSABLE_SCALE",
    )


# ============================================================
# Duplicate collapse
# ============================================================

def collapse_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse duplicate patient/signal/timestamp observations."""

    work = df.copy()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        errors="coerce",
    )

    group_columns = [
        "patient_id",
        "timestamp",
        "signal",
        "context_state",
    ]

    metadata_columns = [
        column
        for column in (
            "sex",
            "age",
            "bmi",
            "caffeine_cups_per_day",
            "alcohol_units_per_day",
            "collection_issue",
            "collection_issue_code",
        )
        if column in work.columns
    ]

    rows = []

    for keys, group in work.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):

        (
            patient_id,
            timestamp,
            signal,
            context_state,
        ) = keys

        valid_mask = (
            group[
                "value_valid"
            ]
            .fillna(False)
            .astype(bool)
        )

        valid_values = pd.to_numeric(
            group.loc[
                valid_mask,
                "value",
            ],
            errors="coerce",
        ).dropna()

        if len(valid_values):

            value = float(
                valid_values.median()
            )

            value_valid = True

            quality_status = "GOOD"

            quality_reason = (
                "VALID_AFTER_DUPLICATE_COLLAPSE"
            )

        else:

            value = np.nan

            value_valid = False

            quality_status = "POOR"

            quality_reason = (
                "NO_VALID_VALUE_AFTER_DUPLICATE_COLLAPSE"
            )

        row = {

            "patient_id":
                str(patient_id),

            "timestamp":
                timestamp,

            "signal":
                str(signal).lower(),

            "context_state":
                str(context_state).upper(),

            "value":
                value,

            "value_valid":
                value_valid,

            "quality_status":
                quality_status,

            "quality_reason":
                quality_reason,

            "duplicate_count":
                int(
                    len(group)
                ),

            "had_duplicate":
                bool(
                    len(group) > 1
                ),
        }

        for column in metadata_columns:

            non_missing = (
                group[
                    column
                ]
                .dropna()
            )

            row[column] = (
                non_missing.iloc[0]
                if len(non_missing)
                else np.nan
            )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "patient_id",
                "signal",
                "timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Step 1: raw personal references
# ============================================================

def build_raw_reference_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate raw early-WAKE personal references."""

    rows = []

    participants = sorted(
        df[
            "patient_id"
        ]
        .astype(str)
        .unique()
    )

    for patient_id in participants:

        for signal in SIGNALS:

            group = df[
                (
                    df[
                        "patient_id"
                    ].astype(str)
                    == str(patient_id)
                )
                &
                (
                    df[
                        "signal"
                    ]
                    == signal
                )
            ].copy()

            group = (
                group
                .sort_values(
                    "timestamp"
                )
                .reset_index(
                    drop=True
                )
            )

            eligible = group[
                (
                    group[
                        "context_state"
                    ]
                    == REFERENCE_CONTEXT
                )
                &
                (
                    group[
                        "value_valid"
                    ]
                )
                &
                (
                    group[
                        "quality_status"
                    ]
                    != "POOR"
                )
            ].copy()

            reference = (
                eligible
                .head(
                    REFERENCE_TARGET_N
                )
                .copy()
            )

            reference_n = len(
                reference
            )

            if reference_n:

                reference_start = (
                    reference[
                        "timestamp"
                    ].min()
                )

                reference_end = (
                    reference[
                        "timestamp"
                    ].max()
                )

            else:

                reference_start = pd.NaT
                reference_end = pd.NaT

            if (
                reference_n
                < REFERENCE_MIN_N
            ):

                center = np.nan
                raw_scale = np.nan

                scale_method = (
                    "NOT_ESTIMATED"
                )

                raw_status = (
                    "INSUFFICIENT_REFERENCE"
                )

            else:

                (
                    center,
                    raw_scale,
                    scale_method,
                ) = estimate_personal_scale(
                    reference[
                        "value"
                    ]
                )

                if (
                    pd.isna(center)
                    or pd.isna(raw_scale)
                    or raw_scale <= 0
                ):

                    raw_status = (
                        "INVALID_REFERENCE_SCALE"
                    )

                else:

                    raw_status = (
                        "RAW_ESTABLISHED"
                    )

            rows.append({

                "patient_id":
                    patient_id,

                "signal":
                    signal,

                "reference_context":
                    REFERENCE_CONTEXT,

                "reference_target_n":
                    REFERENCE_TARGET_N,

                "reference_min_n":
                    REFERENCE_MIN_N,

                "reference_n":
                    reference_n,

                "reference_start":
                    reference_start,

                "reference_end":
                    reference_end,

                "reference_center":
                    center,

                "reference_scale_raw":
                    raw_scale,

                "reference_scale_raw_method":
                    scale_method,

                "reference_raw_status":
                    raw_status,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Step 2: signal-level scale priors
# ============================================================

def build_global_scale_table(
    raw_references: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate robust signal-level scale priors.

    Important
    ---------
    We do NOT pool raw physiological values across participants.

    That would mix:

        between-person baseline differences

    with:

        within-person variability.

    Instead, we first estimate each person's within-person reference
    scale and then take the MEDIAN of those scales for each signal.
    """

    rows = []

    for signal in SIGNALS:

        group = raw_references[
            (
                raw_references[
                    "signal"
                ]
                == signal
            )
            &
            (
                raw_references[
                    "reference_raw_status"
                ]
                == "RAW_ESTABLISHED"
            )
            &
            (
                raw_references[
                    "reference_scale_raw"
                ]
                .notna()
            )
            &
            (
                raw_references[
                    "reference_scale_raw"
                ]
                > 0
            )
        ].copy()

        n = len(
            group
        )

        if (
            n
            < MIN_GLOBAL_SCALE_PARTICIPANTS
        ):

            global_scale = np.nan

            status = (
                "INSUFFICIENT_GLOBAL_SCALE"
            )

        else:

            global_scale = float(
                group[
                    "reference_scale_raw"
                ].median()
            )

            status = (
                "ESTABLISHED"
            )

        rows.append({

            "signal":
                signal,

            "global_scale_n":
                n,

            "global_scale":
                global_scale,

            "global_scale_status":
                status,

            "global_scale_method":
                "MEDIAN_OF_PERSONAL_ROBUST_SCALES",
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Step 3: shrink personal scales
# ============================================================

def stabilize_reference_scales(
    raw_references: pd.DataFrame,
    global_scales: pd.DataFrame,
) -> pd.DataFrame:
    """Shrink personal scales toward signal-level global scale."""

    merged = raw_references.merge(
        global_scales,
        on="signal",
        how="left",
        validate="many_to_one",
    )

    rows = []

    for row in merged.itertuples(
        index=False
    ):

        output = row._asdict()

        personal_scale = (
            row.reference_scale_raw
        )

        global_scale = (
            row.global_scale
        )

        n = (
            row.reference_n
        )

        final_scale = np.nan
        personal_weight = np.nan
        global_weight = np.nan
        floor_value = np.nan
        floor_applied = False

        if (
            row.reference_raw_status
            != "RAW_ESTABLISHED"
        ):

            final_status = (
                row.reference_raw_status
            )

            final_method = (
                "NO_FINAL_SCALE"
            )

        elif (
            row.global_scale_status
            != "ESTABLISHED"
            or pd.isna(
                global_scale
            )
            or global_scale <= 0
        ):

            # If the global prior cannot be estimated,
            # preserve the valid personal robust scale.

            final_scale = float(
                personal_scale
            )

            personal_weight = 1.0
            global_weight = 0.0

            final_status = (
                "ESTABLISHED"
            )

            final_method = (
                "PERSONAL_ONLY_GLOBAL_UNAVAILABLE"
            )

        else:

            # ------------------------------------------------
            # Empirical-Bayes-style reliability weight
            # ------------------------------------------------

            personal_weight = float(
                n
                /
                (
                    n
                    +
                    SCALE_SHRINKAGE_STRENGTH
                )
            )

            global_weight = (
                1.0
                - personal_weight
            )

            # ------------------------------------------------
            # Shrink VARIANCE rather than raw SD.
            #
            # This prevents a tiny personal SD from dominating
            # while preserving scale positivity.
            # ------------------------------------------------

            shrunk_variance = (
                personal_weight
                * (
                    float(
                        personal_scale
                    )
                    ** 2
                )
                +
                global_weight
                * (
                    float(
                        global_scale
                    )
                    ** 2
                )
            )

            shrunk_scale = float(
                np.sqrt(
                    shrunk_variance
                )
            )

            # ------------------------------------------------
            # Final numerical floor
            # ------------------------------------------------

            floor_value = float(
                global_scale
                * MIN_GLOBAL_SCALE_FRACTION
            )

            if (
                shrunk_scale
                < floor_value
            ):

                final_scale = (
                    floor_value
                )

                floor_applied = True

            else:

                final_scale = (
                    shrunk_scale
                )

            final_status = (
                "ESTABLISHED"
            )

            final_method = (
                "SHRUNK_PERSONAL_GLOBAL"
            )

        output.update({

            "reference_scale":
                final_scale,

            "reference_personal_weight":
                personal_weight,

            "reference_global_weight":
                global_weight,

            "reference_scale_floor":
                floor_value,

            "reference_scale_floor_applied":
                floor_applied,

            "reference_scale_method":
                final_method,

            "reference_status":
                final_status,
        })

        rows.append(
            output
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Score observations
# ============================================================

def assign_phase_and_score(
    observations: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    """Assign phase and compute standardized deviations."""

    merged = observations.merge(
        references,
        on=[
            "patient_id",
            "signal",
        ],
        how="left",
        validate="many_to_one",
    )

    rows = []

    for row in merged.itertuples(
        index=False
    ):

        phase = "INELIGIBLE"

        deviation_code = (
            "UNKNOWN"
        )

        deviation_score = np.nan
        deviation_magnitude = np.nan
        deviation_direction = "NONE"

        risk_aligned_score = np.nan

        toward_risk = None

        delta_units = np.nan

        # ----------------------------------------------------
        # Failed / missing observation
        # ----------------------------------------------------

        if (
            not bool(
                row.value_valid
            )
            or pd.isna(
                row.value
            )
        ):

            deviation_code = (
                "MISSING_OR_FAILED_MEASUREMENT"
            )

        # ----------------------------------------------------
        # Reference unavailable
        # ----------------------------------------------------

        elif (
            row.reference_status
            != "ESTABLISHED"
        ):

            deviation_code = (
                row.reference_status
                if pd.notna(
                    row.reference_status
                )
                else "NO_REFERENCE"
            )

        # ----------------------------------------------------
        # Reference-period observation
        # ----------------------------------------------------

        elif (
            row.context_state
            == REFERENCE_CONTEXT
            and
            pd.notna(
                row.reference_start
            )
            and
            pd.notna(
                row.reference_end
            )
            and
            row.timestamp
            >= row.reference_start
            and
            row.timestamp
            <= row.reference_end
        ):

            phase = "REFERENCE"

            deviation_code = (
                "REFERENCE_OBSERVATION"
            )

        # ----------------------------------------------------
        # Context mismatch
        # ----------------------------------------------------

        elif (
            row.context_state
            != REFERENCE_CONTEXT
        ):

            phase = "INELIGIBLE"

            deviation_code = (
                "CONTEXT_INELIGIBLE"
            )

        # ----------------------------------------------------
        # Leakage guard
        # ----------------------------------------------------

        elif (
            pd.notna(
                row.reference_end
            )
            and
            row.timestamp
            <= row.reference_end
        ):

            phase = "REFERENCE"

            deviation_code = (
                "REFERENCE_OBSERVATION"
            )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        else:

            phase = "EVALUATION"

            delta_units = (
                float(
                    row.value
                )
                -
                float(
                    row.reference_center
                )
            )

            deviation_score = (
                delta_units
                /
                float(
                    row.reference_scale
                )
            )

            deviation_magnitude = abs(
                deviation_score
            )

            if deviation_score > 0:

                deviation_direction = (
                    "HIGH"
                )

            elif deviation_score < 0:

                deviation_direction = (
                    "LOW"
                )

            else:

                deviation_direction = (
                    "NONE"
                )

            orientation = (
                SIGNAL_ORIENTATION[
                    row.signal
                ]
            )

            risk_aligned_score = (
                deviation_score
                * orientation
            )

            toward_risk = bool(
                risk_aligned_score
                > 0
            )

            deviation_code = (
                "OK"
            )

        # ----------------------------------------------------
        # Reference reliability
        # ----------------------------------------------------

        if pd.isna(
            row.reference_n
        ):

            reference_reliability = (
                np.nan
            )

        else:

            reference_reliability = float(
                np.clip(
                    row.reference_n
                    / REFERENCE_TARGET_N,
                    0.0,
                    1.0,
                )
            )

        output = row._asdict()

        output.update({

            "phase":
                phase,

            "deviation_code":
                deviation_code,

            "delta_units":
                delta_units,

            "deviation_score":
                deviation_score,

            "deviation_magnitude":
                deviation_magnitude,

            "deviation_direction":
                deviation_direction,

            "risk_aligned_score":
                risk_aligned_score,

            "toward_risk":
                toward_risk,

            "reference_reliability":
                reference_reliability,
        })

        rows.append(
            output
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "patient_id",
                "signal",
                "timestamp",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Temporal spacing
# ============================================================

def add_temporal_spacing(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add temporal-gap metadata for the future Temporal Engine."""

    out = df.copy()

    out[
        "minutes_since_previous_observation"
    ] = (
        out
        .groupby(
            [
                "patient_id",
                "signal",
            ]
        )[
            "timestamp"
        ]
        .diff()
        .dt.total_seconds()
        / 60.0
    )

    out[
        "minutes_since_previous_evaluation"
    ] = np.nan

    for (
        patient_id,
        signal,
    ), group in out.groupby(
        [
            "patient_id",
            "signal",
        ]
    ):

        evaluation_index = group[
            (
                group[
                    "phase"
                ]
                == "EVALUATION"
            )
            &
            (
                group[
                    "deviation_code"
                ]
                == "OK"
            )
        ].index

        if not len(
            evaluation_index
        ):
            continue

        timestamps = out.loc[
            evaluation_index,
            "timestamp",
        ]

        gaps = (
            timestamps
            .diff()
            .dt.total_seconds()
            / 60.0
        )

        out.loc[
            evaluation_index,
            "minutes_since_previous_evaluation",
        ] = gaps.values

    return out


# ============================================================
# Participant diagnostics
# ============================================================

def build_participant_summary(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Participant/signal-level diagnostics."""

    rows = []

    for (
        patient_id,
        signal,
    ), group in scored.groupby(
        [
            "patient_id",
            "signal",
        ]
    ):

        evaluation = group[
            (
                group[
                    "phase"
                ]
                == "EVALUATION"
            )
            &
            (
                group[
                    "deviation_code"
                ]
                == "OK"
            )
        ]

        if len(
            evaluation
        ):

            mean_z = (
                evaluation[
                    "deviation_score"
                ].mean()
            )

            median_z = (
                evaluation[
                    "deviation_score"
                ].median()
            )

            mean_magnitude = (
                evaluation[
                    "deviation_magnitude"
                ].mean()
            )

            max_magnitude = (
                evaluation[
                    "deviation_magnitude"
                ].max()
            )

            risk_fraction = (
                evaluation[
                    "toward_risk"
                ]
                .fillna(False)
                .astype(bool)
                .mean()
            )

            above_2_fraction = (
                evaluation[
                    "deviation_magnitude"
                ]
                .ge(2.0)
                .mean()
            )

            above_3_fraction = (
                evaluation[
                    "deviation_magnitude"
                ]
                .ge(3.0)
                .mean()
            )

        else:

            mean_z = np.nan
            median_z = np.nan
            mean_magnitude = np.nan
            max_magnitude = np.nan
            risk_fraction = np.nan
            above_2_fraction = np.nan
            above_3_fraction = np.nan

        ref = group.iloc[0]

        rows.append({

            "patient_id":
                patient_id,

            "signal":
                signal,

            "reference_status":
                ref[
                    "reference_status"
                ],

            "reference_n":
                ref[
                    "reference_n"
                ],

            "reference_center":
                ref[
                    "reference_center"
                ],

            "reference_scale_raw":
                ref[
                    "reference_scale_raw"
                ],

            "global_scale":
                ref[
                    "global_scale"
                ],

            "reference_scale":
                ref[
                    "reference_scale"
                ],

            "reference_personal_weight":
                ref[
                    "reference_personal_weight"
                ],

            "reference_global_weight":
                ref[
                    "reference_global_weight"
                ],

            "reference_scale_floor_applied":
                ref[
                    "reference_scale_floor_applied"
                ],

            "reference_scale_method":
                ref[
                    "reference_scale_method"
                ],

            "n_total_rows":
                len(group),

            "n_reference_rows":
                (
                    group[
                        "phase"
                    ]
                    == "REFERENCE"
                ).sum(),

            "n_evaluation_rows":
                len(
                    evaluation
                ),

            "n_context_ineligible":
                (
                    group[
                        "deviation_code"
                    ]
                    == "CONTEXT_INELIGIBLE"
                ).sum(),

            "n_missing_failed":
                (
                    group[
                        "deviation_code"
                    ]
                    == "MISSING_OR_FAILED_MEASUREMENT"
                ).sum(),

            "mean_deviation_score":
                mean_z,

            "median_deviation_score":
                median_z,

            "mean_deviation_magnitude":
                mean_magnitude,

            "max_deviation_magnitude":
                max_magnitude,

            "toward_risk_fraction":
                risk_fraction,

            "magnitude_ge_2_fraction":
                above_2_fraction,

            "magnitude_ge_3_fraction":
                above_3_fraction,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Reading processed DRYAD:"
    )

    print(
        INPUT_PATH
    )

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "Processed DRYAD input not found:\n"
            f"{INPUT_PATH}"
        )

    raw = pd.read_parquet(
        INPUT_PATH
    )

    print(
        "\nInput shape:",
        raw.shape,
    )

    print(
        "Participants:",
        raw[
            "patient_id"
        ].nunique(),
    )

    # ========================================================
    # Filter signals
    # ========================================================

    work = raw[
        raw[
            "signal"
        ]
        .astype(str)
        .str.lower()
        .isin(
            SIGNALS
        )
    ].copy()

    work[
        "signal"
    ] = (
        work[
            "signal"
        ]
        .astype(str)
        .str.lower()
    )

    work[
        "context_state"
    ] = (
        work[
            "context_state"
        ]
        .astype(str)
        .str.upper()
    )

    work[
        "timestamp"
    ] = pd.to_datetime(
        work[
            "timestamp"
        ],
        errors="coerce",
    )

    # ========================================================
    # Collapse duplicates
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "COLLAPSING DUPLICATES"
    )

    print(
        "=" * 72
    )

    print(
        "\nRows before collapse:",
        len(
            work
        ),
    )

    observations = collapse_duplicates(
        work
    )

    print(
        "Rows after collapse:",
        len(
            observations
        ),
    )

    print(
        "Collapsed timestamp groups with duplicates:",
        int(
            observations[
                "had_duplicate"
            ].sum()
        ),
    )

    # ========================================================
    # Raw personal reference
    # ========================================================

    raw_references = (
        build_raw_reference_table(
            observations
        )
    )

    # ========================================================
    # Global scale prior
    # ========================================================

    global_scales = (
        build_global_scale_table(
            raw_references
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GLOBAL SIGNAL SCALE PRIORS"
    )

    print(
        "=" * 72
    )

    print(
        global_scales
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Stabilize personal scale
    # ========================================================

    references = (
        stabilize_reference_scales(
            raw_references,
            global_scales,
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "REFERENCE STATUS"
    )

    print(
        "=" * 72
    )

    print(
        "\nFinal reference status:"
    )

    print(
        references[
            "reference_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nFinal scale methods:"
    )

    print(
        references[
            "reference_scale_method"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nScale floor applied:"
    )

    print(
        references[
            "reference_scale_floor_applied"
        ]
        .value_counts(
            dropna=False
        )
    )

    # ========================================================
    # Scale comparison
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RAW VS STABILIZED SCALE"
    )

    print(
        "=" * 72
    )

    scale_columns = [
        "patient_id",
        "signal",
        "reference_n",
        "reference_scale_raw",
        "global_scale",
        "reference_personal_weight",
        "reference_global_weight",
        "reference_scale",
        "reference_scale_floor_applied",
    ]

    print(
        references[
            scale_columns
        ]
        .sort_values(
            [
                "signal",
                "reference_scale_raw",
            ]
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Score
    # ========================================================

    scored = (
        assign_phase_and_score(
            observations,
            references,
        )
    )

    scored = (
        add_temporal_spacing(
            scored
        )
    )

    # ========================================================
    # Leakage
    # ========================================================

    leakage = scored[
        (
            scored[
                "phase"
            ]
            == "EVALUATION"
        )
        &
        (
            scored[
                "timestamp"
            ]
            <= scored[
                "reference_end"
            ]
        )
    ]

    print(
        "\n"
        + "=" * 72
    )

    print(
        "LEAKAGE CHECK"
    )

    print(
        "=" * 72
    )

    print(
        "\nEvaluation rows at/before reference end:",
        len(
            leakage
        ),
    )

    if len(
        leakage
    ):

        raise RuntimeError(
            "LEAKAGE CHECK FAILED."
        )

    print(
        "Leakage check: PASSED"
    )

    # ========================================================
    # Evaluation rows
    # ========================================================

    evaluation = scored[
        (
            scored[
                "phase"
            ]
            == "EVALUATION"
        )
        &
        (
            scored[
                "deviation_code"
            ]
            == "OK"
        )
    ].copy()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "STABILIZED EVALUATION DISTRIBUTION"
    )

    print(
        "=" * 72
    )

    print(
        "\nUsable evaluation rows:",
        len(
            evaluation
        ),
    )

    print(
        "\nDeviation score:"
    )

    print(
        evaluation[
            "deviation_score"
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print(
        "\nDeviation magnitude:"
    )

    print(
        evaluation[
            "deviation_magnitude"
        ].describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print(
        "\nMagnitude >= 2:"
    )

    print(
        round(
            evaluation[
                "deviation_magnitude"
            ]
            .ge(2.0)
            .mean(),
            4,
        )
    )

    print(
        "\nMagnitude >= 3:"
    )

    print(
        round(
            evaluation[
                "deviation_magnitude"
            ]
            .ge(3.0)
            .mean(),
            4,
        )
    )

    print(
        "\nToward-risk fraction:"
    )

    print(
        round(
            evaluation[
                "toward_risk"
            ]
            .fillna(False)
            .astype(bool)
            .mean(),
            4,
        )
    )

    # ========================================================
    # By signal
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "BY SIGNAL"
    )

    print(
        "=" * 72
    )

    signal_rows = []

    for signal, group in evaluation.groupby(
        "signal"
    ):

        signal_rows.append({

            "signal":
                signal,

            "n":
                len(group),

            "mean_z":
                group[
                    "deviation_score"
                ].mean(),

            "median_z":
                group[
                    "deviation_score"
                ].median(),

            "mean_magnitude":
                group[
                    "deviation_magnitude"
                ].mean(),

            "p90_magnitude":
                group[
                    "deviation_magnitude"
                ].quantile(
                    0.90
                ),

            "p95_magnitude":
                group[
                    "deviation_magnitude"
                ].quantile(
                    0.95
                ),

            "p99_magnitude":
                group[
                    "deviation_magnitude"
                ].quantile(
                    0.99
                ),

            "max_magnitude":
                group[
                    "deviation_magnitude"
                ].max(),

            "toward_risk_fraction":
                group[
                    "toward_risk"
                ]
                .fillna(False)
                .astype(bool)
                .mean(),
        })

    signal_summary = pd.DataFrame(
        signal_rows
    )

    print(
        signal_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Time gaps
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "EVALUATION TIME GAPS"
    )

    print(
        "=" * 72
    )

    print(
        evaluation[
            "minutes_since_previous_evaluation"
        ].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
    )

    # ========================================================
    # Participant summary
    # ========================================================

    participant_summary = (
        build_participant_summary(
            scored
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "HIGHEST DEVIATION PARTICIPANT/SIGNAL PAIRS"
    )

    print(
        "=" * 72
    )

    print(
        participant_summary
        .sort_values(
            "magnitude_ge_2_fraction",
            ascending=False,
        )
        .head(25)
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Largest deviations
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "LARGEST STABILIZED DEVIATIONS"
    )

    print(
        "=" * 72
    )

    display_columns = [
        "patient_id",
        "timestamp",
        "signal",
        "value",
        "reference_center",
        "reference_scale_raw",
        "global_scale",
        "reference_scale",
        "delta_units",
        "deviation_score",
        "deviation_magnitude",
        "toward_risk",
        "minutes_since_previous_evaluation",
    ]

    print(
        evaluation
        .sort_values(
            "deviation_magnitude",
            ascending=False,
        )[
            display_columns
        ]
        .head(30)
        .round(
            {
                "value": 2,
                "reference_center": 2,
                "reference_scale_raw": 2,
                "global_scale": 2,
                "reference_scale": 2,
                "delta_units": 2,
                "deviation_score": 2,
                "deviation_magnitude": 2,
                "minutes_since_previous_evaluation": 1,
            }
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    references.to_csv(
        REFERENCE_PATH,
        index=False,
    )

    global_scales.to_csv(
        GLOBAL_SCALE_PATH,
        index=False,
    )

    participant_summary.to_csv(
        PARTICIPANT_SUMMARY_PATH,
        index=False,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SAVED"
    )

    print(
        "=" * 72
    )

    print(
        OUTPUT_PATH
    )

    print(
        REFERENCE_PATH
    )

    print(
        GLOBAL_SCALE_PATH
    )

    print(
        PARTICIPANT_SUMMARY_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Reference centers are still fully personal."
    )

    print(
        "Only short-reference SCALE estimates are shrunk "
        "toward a robust signal-level scale prior."
    )

    print(
        "Evaluation observations never update reference values."
    )

    print(
        "No temporal persistence or warning decision has been calculated."
    )


if __name__ == "__main__":
    main()