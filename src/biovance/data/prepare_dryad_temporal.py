"""Prepare DRYAD for the BioVance Temporal Engine.

Inputs
------
data/raw/dryad/
    Blood_Pressure_Sleep_Info.xlsx
    Data_Collection_Notes.csv
    Participant_Information.csv

Output
------
data/processed/dryad_temporal_input.parquet

Purpose
-------
Create a clean longitudinal dataset while preserving:

- repeated BP / HR measurements
- wake/sleep state
- participant metadata
- data-collection issues
- failed measurements
- duplicate timestamps

IMPORTANT
---------
This script does NOT calculate:

- deviation
- persistence
- recurrence
- warnings
- temporal scores

It only prepares evidence for later engines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "dryad"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

BP_PATH = (
    RAW_DIR
    / "Blood_Pressure_Sleep_Info.xlsx"
)

NOTES_PATH = (
    RAW_DIR
    / "Data_Collection_Notes.csv"
)

PARTICIPANT_PATH = (
    RAW_DIR
    / "Participant_Information.csv"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "dryad_temporal_input.parquet"
)

WIDE_OUTPUT_PATH = (
    PROCESSED_DIR
    / "dryad_temporal_wide.parquet"
)


# ============================================================
# Constants
# ============================================================

PHYSIO_COLUMNS = [
    "Systolic",
    "Diastolic",
    "MAP",
    "PP",
    "HR",
]

SIGNAL_MAP = {
    "Systolic": "sbp",
    "Diastolic": "dbp",
    "MAP": "map",
    "PP": "pp",
    "HR": "hr",
}


# ============================================================
# Helpers
# ============================================================

def normalize_issue(
    value,
) -> str:
    """Normalize free-text collection issue."""

    if pd.isna(value):
        return "NONE_REPORTED"

    text = str(value).strip()

    if text.upper() == "N":
        return "NONE_REPORTED"

    return text


def issue_code(
    text: str,
) -> str:
    """Convert known issue text to a broad machine-readable code.

    These codes describe collection issues only.
    They do NOT automatically invalidate every observation.
    """

    lower = str(text).lower()

    if (
        "no night bp" in lower
        or "no nigt bp" in lower
    ):
        return "NO_NIGHT_BP"

    if "inflation error" in lower:
        return "BP_INFLATION_ERROR"

    if (
        "cuff removed" in lower
    ):
        return "BP_CUFF_REMOVED"

    if (
        "poor ecg contact" in lower
    ):
        return "POOR_ECG_CONTACT"

    if (
        "strap displaced" in lower
    ):
        return "STRAP_DISPLACED"

    if (
        "cgm sensor" in lower
    ):
        return "CGM_ISSUE"

    if text == "NONE_REPORTED":
        return "NONE_REPORTED"

    return "OTHER_REPORTED_ISSUE"


def build_timestamp(
    df: pd.DataFrame,
) -> pd.Series:
    """Combine Day_Date + Time into a timestamp."""

    combined = (
        df["Day_Date"]
        .astype(str)
        .str.strip()
        + " "
        + df["Time"]
        .astype(str)
        .str.strip()
    )

    timestamp = pd.to_datetime(
        combined,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    # Fallback in case some rows use a slightly different representation.
    missing = timestamp.isna()

    if missing.any():

        timestamp.loc[missing] = pd.to_datetime(
            combined.loc[missing],
            dayfirst=True,
            errors="coerce",
        )

    return timestamp


# ============================================================
# Main preparation
# ============================================================

def main():

    print(
        "Reading DRYAD files..."
    )

    for path in (
        BP_PATH,
        NOTES_PATH,
        PARTICIPANT_PATH,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Missing DRYAD file: {path}"
            )

    bp = pd.read_excel(
        BP_PATH
    )

    notes = pd.read_csv(
        NOTES_PATH
    )

    participants = pd.read_csv(
        PARTICIPANT_PATH
    )

    print(
        "\nBlood-pressure file:",
        bp.shape,
    )

    print(
        "Notes file:",
        notes.shape,
    )

    print(
        "Participant file:",
        participants.shape,
    )

    # ========================================================
    # Standardize participant ID
    # ========================================================

    bp = bp.rename(
        columns={
            "ID": "patient_id",
        }
    )

    notes = notes.rename(
        columns={
            "ID": "patient_id",
        }
    )

    participants = participants.rename(
        columns={
            "ID": "patient_id",
        }
    )

    for df in (
        bp,
        notes,
        participants,
    ):

        df["patient_id"] = (
            df["patient_id"]
            .astype(str)
            .str.strip()
        )

    # ========================================================
    # Timestamp
    # ========================================================

    bp["timestamp"] = (
        build_timestamp(
            bp
        )
    )

    invalid_timestamps = (
        bp[
            "timestamp"
        ]
        .isna()
        .sum()
    )

    print(
        "\nInvalid timestamps:",
        invalid_timestamps,
    )

    if invalid_timestamps:

        print(
            bp[
                bp[
                    "timestamp"
                ].isna()
            ][
                [
                    "patient_id",
                    "Day_Date",
                    "Time",
                ]
            ]
        )

    # ========================================================
    # Physiological numeric conversion
    # ========================================================

    for column in PHYSIO_COLUMNS:

        bp[column] = pd.to_numeric(
            bp[column],
            errors="coerce",
        )

    # ========================================================
    # Failed BP measurement rows
    #
    # In this dataset, failed rows use zeros across ALL
    # physiological channels.
    # ========================================================

    bp[
        "all_zero_measurement"
    ] = (
        bp[
            PHYSIO_COLUMNS
        ]
        .eq(0)
        .all(axis=1)
    )

    bp[
        "measurement_valid"
    ] = (
        ~bp[
            "all_zero_measurement"
        ]
    )

    bp[
        "measurement_reason"
    ] = np.where(
        bp[
            "all_zero_measurement"
        ],
        "ALL_PHYSIOLOGY_ZERO",
        "VALID_RECORDED_MEASUREMENT",
    )

    # Replace zero rows with NaN.
    #
    # We preserve the row itself because missingness is evidence.
    bp.loc[
        bp[
            "all_zero_measurement"
        ],
        PHYSIO_COLUMNS,
    ] = np.nan

    # ========================================================
    # Wake / sleep
    # ========================================================

    bp[
        "Wake_Sleep"
    ] = pd.to_numeric(
        bp[
            "Wake_Sleep"
        ],
        errors="coerce",
    )

    bp[
        "context_state"
    ] = bp[
        "Wake_Sleep"
    ].map(
        {
            1: "WAKE",
            0: "SLEEP",
        }
    ).fillna(
        "UNKNOWN"
    )

    # ========================================================
    # Duplicate timestamp detection
    # ========================================================

    bp[
        "duplicate_timestamp"
    ] = bp.duplicated(
        subset=[
            "patient_id",
            "timestamp",
        ],
        keep=False,
    )

    bp[
        "duplicate_rank"
    ] = (
        bp
        .groupby(
            [
                "patient_id",
                "timestamp",
            ],
            dropna=False,
        )
        .cumcount()
        + 1
    )

    # ========================================================
    # Notes
    # ========================================================

    issue_column = (
        "Issue reported/known?"
    )

    notes[
        "collection_issue"
    ] = notes[
        issue_column
    ].apply(
        normalize_issue
    )

    notes[
        "collection_issue_code"
    ] = notes[
        "collection_issue"
    ].apply(
        issue_code
    )

    notes_keep = [
        "patient_id",
        "collection_issue",
        "collection_issue_code",
    ]

    # Preserve useful device metadata too.
    for column in (
        "CGM? Yes/No",
        "Zephyr unit",
        "BP monitor",
        "BP cuff",
        "ECG setup",
    ):

        if column in notes.columns:

            notes_keep.append(
                column
            )

    notes = notes[
        notes_keep
    ].copy()

    # ========================================================
    # Participant metadata
    # ========================================================

    participants = participants.rename(
        columns={
            "Sex":
                "sex",

            "Age":
                "age",

            "BMI":
                "bmi",

            "Caffeine (number of cups per day)":
                "caffeine_cups_per_day",

            "Alcohol (number of units per day)":
                "alcohol_units_per_day",
        }
    )

    # ========================================================
    # Merge metadata
    # ========================================================

    wide = (
        bp
        .merge(
            participants,
            on="patient_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            notes,
            on="patient_id",
            how="left",
            validate="many_to_one",
        )
    )

    # ========================================================
    # Sort
    # ========================================================

    wide = (
        wide
        .sort_values(
            [
                "patient_id",
                "timestamp",
                "duplicate_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Time spacing diagnostics
    # ========================================================

    wide[
        "minutes_since_previous"
    ] = (
        wide
        .groupby(
            "patient_id"
        )[
            "timestamp"
        ]
        .diff()
        .dt.total_seconds()
        / 60.0
    )

    # ========================================================
    # Create long-format signal table
    # ========================================================

    id_vars = [
        "patient_id",
        "timestamp",
        "context_state",
        "Wake_Sleep",
        "measurement_valid",
        "measurement_reason",
        "duplicate_timestamp",
        "duplicate_rank",
        "minutes_since_previous",
        "sex",
        "age",
        "bmi",
        "caffeine_cups_per_day",
        "alcohol_units_per_day",
        "collection_issue",
        "collection_issue_code",
    ]

    for optional in (
        "CGM? Yes/No",
        "Zephyr unit",
        "BP monitor",
        "BP cuff",
        "ECG setup",
    ):

        if optional in wide.columns:
            id_vars.append(
                optional
            )

    long = wide.melt(
        id_vars=id_vars,
        value_vars=PHYSIO_COLUMNS,
        var_name="raw_signal",
        value_name="value",
    )

    long[
        "signal"
    ] = long[
        "raw_signal"
    ].map(
        SIGNAL_MAP
    )

    long = (
        long
        .drop(
            columns=[
                "raw_signal"
            ]
        )
        .sort_values(
            [
                "patient_id",
                "timestamp",
                "signal",
                "duplicate_rank",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Signal-specific value validity
    # ========================================================

    long[
        "value_valid"
    ] = (
        long[
            "measurement_valid"
        ]
        &
        long[
            "value"
        ].notna()
    )

    long[
        "quality_status"
    ] = np.where(
        long[
            "value_valid"
        ],
        "GOOD",
        "POOR",
    )

    long[
        "quality_reason"
    ] = np.where(
        long[
            "value_valid"
        ],
        "RECORDED_MEASUREMENT",
        "FAILED_OR_MISSING_MEASUREMENT",
    )

    # ========================================================
    # Diagnostics
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DRYAD PREPROCESSING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nParticipants:",
        wide[
            "patient_id"
        ].nunique(),
    )

    print(
        "\nWide rows:",
        len(
            wide
        ),
    )

    print(
        "\nLong rows:",
        len(
            long
        ),
    )

    print(
        "\nFailed all-zero measurements:"
    )

    print(
        int(
            wide[
                "all_zero_measurement"
            ].sum()
        )
    )

    print(
        "\nValid measurements:"
    )

    print(
        int(
            wide[
                "measurement_valid"
            ].sum()
        )
    )

    print(
        "\nWake / sleep counts:"
    )

    print(
        wide[
            "context_state"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nDuplicate patient/timestamp rows:"
    )

    print(
        int(
            wide[
                "duplicate_timestamp"
            ].sum()
        )
    )

    print(
        "\nCollection issues:"
    )

    print(
        notes[
            "collection_issue_code"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nMeasurements per participant:"
    )

    print(
        wide
        .groupby(
            "patient_id"
        )
        .size()
        .describe()
    )

    print(
        "\nValid measurement gaps in minutes:"
    )

    valid_gap = wide.loc[
        wide[
            "measurement_valid"
        ],
        "minutes_since_previous",
    ]

    print(
        valid_gap.describe(
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

    print(
        "\nSignal valid counts:"
    )

    print(
        long
        .groupby(
            "signal"
        )[
            "value_valid"
        ]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
    )

    # ========================================================
    # Save
    # ========================================================

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    wide.to_parquet(
        WIDE_OUTPUT_PATH,
        index=False,
    )

    long.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SAVED"
    )

    print(
        "=" * 70
    )

    print(
        "\nWide:"
    )

    print(
        WIDE_OUTPUT_PATH
    )

    print(
        "\nLong:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No temporal score has been calculated."
    )

    print(
        "No zero-valued failed measurement has been interpreted "
        "as real physiology."
    )


if __name__ == "__main__":
    main()