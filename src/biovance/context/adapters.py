"""Adapters for BAIGUTANOVA context-engine inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_sensor_windows(
    sensor_df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize BAIGUTANOVA filtered sensor windows.

    This performs only dataset preparation:
    - standardizes patient ID
    - converts Unix millisecond timestamps
    - derives movement magnitudes
    - preserves raw context clues

    It does NOT assign REST / ACTIVE / SLEEP.
    """

    required = {
        "deviceId",
        "ts_start",
        "ts_end",
        "missingness_score",
        "acc_x_avg",
        "acc_y_avg",
        "acc_z_avg",
        "gyr_x_avg",
        "gyr_y_avg",
        "gyr_z_avg",
    }

    missing = required - set(sensor_df.columns)

    if missing:
        raise ValueError(
            "BAIGUTANOVA sensor data missing required columns: "
            f"{sorted(missing)}"
        )

    out = sensor_df.copy()

    # ---------------------------------------------------------
    # Standardize identifiers
    # ---------------------------------------------------------
    out["patient_id"] = (
        out["deviceId"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Convert Unix milliseconds to datetimes
    #
    # We keep UTC explicitly because the raw sensor timestamps
    # are epoch milliseconds. Local-time alignment with the
    # sleep diary must be configured explicitly if required.
    # ---------------------------------------------------------
    out["timestamp_start"] = pd.to_datetime(
        out["ts_start"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    out["timestamp_end"] = pd.to_datetime(
        out["ts_end"],
        unit="ms",
        utc=True,
        errors="coerce",
    )

    if out["timestamp_start"].isna().any():
        raise ValueError(
            "Some BAIGUTANOVA ts_start values could not be converted."
        )

    if out["timestamp_end"].isna().any():
        raise ValueError(
            "Some BAIGUTANOVA ts_end values could not be converted."
        )

    # ---------------------------------------------------------
    # Window duration
    # ---------------------------------------------------------
    out["window_seconds"] = (
        out["timestamp_end"]
        - out["timestamp_start"]
    ).dt.total_seconds()

    # ---------------------------------------------------------
    # Accelerometer magnitude
    # sqrt(x² + y² + z²)
    #
    # This is a context clue, not the final activity level.
    # ---------------------------------------------------------
    out["acc_magnitude"] = np.sqrt(
        out["acc_x_avg"] ** 2
        + out["acc_y_avg"] ** 2
        + out["acc_z_avg"] ** 2
    )

    # ---------------------------------------------------------
    # Gyroscope magnitude
    # ---------------------------------------------------------
    out["gyro_magnitude"] = np.sqrt(
        out["gyr_x_avg"] ** 2
        + out["gyr_y_avg"] ** 2
        + out["gyr_z_avg"] ** 2
    )

    # ---------------------------------------------------------
    # Preserve context-related fields when available
    # ---------------------------------------------------------
    optional_columns = [
        "steps",
        "distance",
        "calories",
        "light_avg",
        "HR",
        "ibi",
        "sdnn",
        "sdsd",
        "rmssd",
        "pnn20",
        "pnn50",
        "lf",
        "hf",
        "lf/hf",
        "missingness_score",
    ]

    keep = [
        "patient_id",
        "timestamp_start",
        "timestamp_end",
        "window_seconds",
        "acc_x_avg",
        "acc_y_avg",
        "acc_z_avg",
        "acc_magnitude",
        "gyr_x_avg",
        "gyr_y_avg",
        "gyr_z_avg",
        "gyro_magnitude",
    ]

    keep += [
        col
        for col in optional_columns
        if col in out.columns
    ]

    out = out[keep].copy()

    return (
        out
        .sort_values(
            ["patient_id", "timestamp_start"]
        )
        .reset_index(drop=True)
    )


def prepare_sleep_intervals(
    sleep_df: pd.DataFrame,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Convert BAIGUTANOVA sleep diary rows into sleep intervals.

    Important:
    The diary contains local-looking date/time fields but the source
    files do not themselves specify timezone semantics.

    `timezone` therefore must be configured explicitly.
    Do not silently guess the timezone.

    The diary `date` is treated as the calendar date associated with
    the reported sleep episode.

    If wakeup clock time is earlier than asleep clock time, wakeup is
    assumed to occur on the following day.
    """

    required = {
        "userId",
        "date",
        "asleep",
        "wakeup",
    }

    missing = required - set(sleep_df.columns)

    if missing:
        raise ValueError(
            "BAIGUTANOVA sleep diary missing required columns: "
            f"{sorted(missing)}"
        )

    out = sleep_df.copy()

    out["patient_id"] = (
        out["userId"]
        .astype(str)
        .str.strip()
    )

    base_date = pd.to_datetime(
        out["date"],
        errors="coerce",
    )

    if base_date.isna().any():
        raise ValueError(
            "Some sleep diary dates could not be parsed."
        )

    asleep_delta = pd.to_timedelta(
        out["asleep"].astype(str)
    )

    wakeup_delta = pd.to_timedelta(
        out["wakeup"].astype(str)
    )

    sleep_start = base_date + asleep_delta
    sleep_end = base_date + wakeup_delta

    # Overnight sleep:
    # e.g. asleep 23:30, wakeup 07:00
    crosses_midnight = (
        wakeup_delta <= asleep_delta
    )

    sleep_end.loc[crosses_midnight] += pd.Timedelta(
        days=1
    )

    # Localize diary times explicitly.
    sleep_start = (
        sleep_start
        .dt.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )
    )

    sleep_end = (
        sleep_end
        .dt.tz_localize(
            timezone,
            ambiguous="NaT",
            nonexistent="NaT",
        )
    )

    out["sleep_start"] = sleep_start
    out["sleep_end"] = sleep_end

    if out["sleep_start"].isna().any():
        raise ValueError(
            "Some sleep_start timestamps could not be constructed."
        )

    if out["sleep_end"].isna().any():
        raise ValueError(
            "Some sleep_end timestamps could not be constructed."
        )

    # Sanity check
    bad = (
        out["sleep_end"]
        <= out["sleep_start"]
    )

    if bad.any():
        raise ValueError(
            "Some sleep diary intervals have "
            "sleep_end <= sleep_start."
        )

    keep = [
        "patient_id",
        "sleep_start",
        "sleep_end",
    ]

    # Preserve diary metadata useful for later analysis/explainability
    optional = [
        "go2bed",
        "wakeup@night",
        "waso",
        "sleep_duration",
        "in_bed_duration",
        "sleep_latency",
        "sleep_efficiency",
    ]

    keep += [
        c
        for c in optional
        if c in out.columns
    ]

    return (
        out[keep]
        .sort_values(
            ["patient_id", "sleep_start"]
        )
        .reset_index(drop=True)
    )


def add_sleep_evidence(
    sensor_windows: pd.DataFrame,
    sleep_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """Annotate sensor windows with diary-based sleep evidence.

    Adds:
        inside_sleep_interval
        sleep_overlap_fraction

    These are context clues, NOT final Context Engine labels.
    """

    required_sensor = {
        "patient_id",
        "timestamp_start",
        "timestamp_end",
    }

    required_sleep = {
        "patient_id",
        "sleep_start",
        "sleep_end",
    }

    if not required_sensor <= set(sensor_windows.columns):
        raise ValueError(
            "sensor_windows missing required canonical columns"
        )

    if not required_sleep <= set(sleep_intervals.columns):
        raise ValueError(
            "sleep_intervals missing required canonical columns"
        )

    out = sensor_windows.copy()

    out["inside_sleep_interval"] = False
    out["sleep_overlap_fraction"] = 0.0

    # Patient-wise interval matching.
    # This is easy to understand and safe for this dataset size.
    for patient_id, intervals in sleep_intervals.groupby(
        "patient_id"
    ):

        sensor_idx = out.index[
            out["patient_id"] == patient_id
        ]

        if len(sensor_idx) == 0:
            continue

        starts = out.loc[
            sensor_idx,
            "timestamp_start",
        ]

        ends = out.loc[
            sensor_idx,
            "timestamp_end",
        ]

        best_overlap = pd.Series(
            0.0,
            index=sensor_idx,
        )

        for interval in intervals.itertuples(
            index=False
        ):

            overlap_start = starts.where(
                starts > interval.sleep_start,
                interval.sleep_start,
            )

            overlap_end = ends.where(
                ends < interval.sleep_end,
                interval.sleep_end,
            )

            overlap_seconds = (
                overlap_end
                - overlap_start
            ).dt.total_seconds().clip(
                lower=0
            )

            window_seconds = (
                ends - starts
            ).dt.total_seconds()

            fraction = (
                overlap_seconds
                / window_seconds.replace(0, np.nan)
            ).fillna(0.0)

            best_overlap = np.maximum(
                best_overlap,
                fraction,
            )

        out.loc[
            sensor_idx,
            "sleep_overlap_fraction",
        ] = best_overlap

    out["inside_sleep_interval"] = (
        out["sleep_overlap_fraction"] > 0
    )

    return out


def from_baigutanova(
    sensor_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
    *,
    diary_timezone: str = "UTC",
) -> pd.DataFrame:
    """Full BAIGUTANOVA preprocessing for Context Engine input."""

    sensor = prepare_sensor_windows(
        sensor_df
    )

    sleep = prepare_sleep_intervals(
        sleep_df,
        timezone=diary_timezone,
    )

    out = add_sleep_evidence(
        sensor,
        sleep,
    )

    return out