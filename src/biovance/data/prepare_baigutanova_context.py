"""Prepare BAIGUTANOVA inputs for the BioVance Context Engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# Project root:
# BIOVANCE_4/src/biovance/data/file.py
# parents[3] -> BIOVANCE_4
ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from biovance.context.adapters import (
    from_baigutanova,
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "baigutanova"
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

SENSOR_PATH = (
    RAW_DIR
    / "sensor_hrv_filtered.csv"
)

SLEEP_PATH = (
    RAW_DIR
    / "sleep_diary.csv"
)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "baigutanova_context_input.parquet"
)

# ---------------------------------------------------------
# IMPORTANT TIMEZONE CONFIGURATION
# ---------------------------------------------------------
#
# The uploaded CSVs do not themselves document the timezone
# relationship between Unix sensor timestamps and diary clock times.
#
# Do NOT guess this silently.
#
# Keep UTC initially only if that matches the dataset documentation.
# Change this once the BAIGUTANOVA timezone semantics are verified.
#

DIARY_TIMEZONE = "UTC"


def main():
    print("Reading BAIGUTANOVA sensor data:")
    print(SENSOR_PATH)

    if not SENSOR_PATH.exists():
        raise FileNotFoundError(
            f"Missing sensor file: {SENSOR_PATH}"
        )

    if not SLEEP_PATH.exists():
        raise FileNotFoundError(
            f"Missing sleep diary file: {SLEEP_PATH}"
        )

    sensor = pd.read_csv(
        SENSOR_PATH
    )

    sleep = pd.read_csv(
        SLEEP_PATH
    )

    print("\nSensor shape:")
    print(sensor.shape)

    print("\nSleep diary shape:")
    print(sleep.shape)

    # ---------------------------------------------------------
    # Prepare
    # ---------------------------------------------------------

    processed = from_baigutanova(
        sensor,
        sleep,
        diary_timezone=DIARY_TIMEZONE,
    )

    # ---------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n=== PROCESSED BAIGUTANOVA CONTEXT INPUT ===")

    print("\nOutput path:")
    print(OUTPUT_PATH)

    print("\nShape:")
    print(processed.shape)

    print("\nParticipants:")
    print(
        processed["patient_id"].nunique()
    )

    print("\nDate range:")
    print(
        processed["timestamp_start"].min()
    )
    print(
        processed["timestamp_start"].max()
    )

    print("\nColumns:")
    print(
        processed.columns.tolist()
    )

    print("\nSample:")
    print(
        processed.head()
    )

    print("\nSleep evidence:")
    print(
        processed[
            "inside_sleep_interval"
        ].value_counts(
            dropna=False
        )
    )

    print("\nSleep overlap summary:")
    print(
        processed[
            "sleep_overlap_fraction"
        ].describe()
    )

    print("\nAccelerometer magnitude:")
    print(
        processed[
            "acc_magnitude"
        ].describe()
    )

    print("\nGyroscope magnitude:")
    print(
        processed[
            "gyro_magnitude"
        ].describe()
    )

    print("\nMissingness score:")
    print(
        processed[
            "missingness_score"
        ].describe()
    )

    print("\nPreparation complete.")


if __name__ == "__main__":
    main()