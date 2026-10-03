import pandas as pd
import pytest

from biovance.context.adapters import (
    prepare_sensor_windows,
    prepare_sleep_intervals,
    add_sleep_evidence,
    from_baigutanova,
)


def test_prepare_sensor_windows():

    df = pd.DataFrame({
        "deviceId": ["p01"],
        "ts_start": [1617262425031],
        "ts_end": [1617262724833],
        "missingness_score": [0.2],

        "acc_x_avg": [3.0],
        "acc_y_avg": [4.0],
        "acc_z_avg": [0.0],

        "gyr_x_avg": [0.0],
        "gyr_y_avg": [3.0],
        "gyr_z_avg": [4.0],

        "steps": [10],
        "distance": [5.0],
        "calories": [1.0],
    })

    out = prepare_sensor_windows(df)

    assert out.loc[0, "patient_id"] == "p01"

    assert (
        out.loc[0, "acc_magnitude"]
        == pytest.approx(5.0)
    )

    assert (
        out.loc[0, "gyro_magnitude"]
        == pytest.approx(5.0)
    )

    assert pd.notna(
        out.loc[0, "timestamp_start"]
    )

    assert (
        out.loc[0, "window_seconds"]
        > 0
    )


def test_prepare_sleep_interval_crosses_midnight():

    df = pd.DataFrame({
        "userId": ["p01"],
        "date": ["2021-03-09"],
        "asleep": ["23:30:00"],
        "wakeup": ["07:00:00"],
        "go2bed": ["23:00:00"],
    })

    out = prepare_sleep_intervals(
        df,
        timezone="UTC",
    )

    duration = (
        out.loc[0, "sleep_end"]
        - out.loc[0, "sleep_start"]
    ).total_seconds() / 3600

    assert duration == pytest.approx(7.5)


def test_prepare_sleep_same_day():

    df = pd.DataFrame({
        "userId": ["p01"],
        "date": ["2021-03-09"],
        "asleep": ["01:00:00"],
        "wakeup": ["08:00:00"],
    })

    out = prepare_sleep_intervals(
        df,
        timezone="UTC",
    )

    duration = (
        out.loc[0, "sleep_end"]
        - out.loc[0, "sleep_start"]
    ).total_seconds() / 3600

    assert duration == pytest.approx(7.0)


def test_sleep_overlap_is_added():

    sensor = pd.DataFrame({
        "patient_id": ["p01"],
        "timestamp_start": [
            pd.Timestamp(
                "2021-03-09 02:00:00",
                tz="UTC",
            )
        ],
        "timestamp_end": [
            pd.Timestamp(
                "2021-03-09 02:05:00",
                tz="UTC",
            )
        ],
    })

    sleep = pd.DataFrame({
        "patient_id": ["p01"],
        "sleep_start": [
            pd.Timestamp(
                "2021-03-09 01:00:00",
                tz="UTC",
            )
        ],
        "sleep_end": [
            pd.Timestamp(
                "2021-03-09 08:00:00",
                tz="UTC",
            )
        ],
    })

    out = add_sleep_evidence(
        sensor,
        sleep,
    )

    assert bool(
        out.loc[
            0,
            "inside_sleep_interval",
        ]
    )

    assert (
        out.loc[
            0,
            "sleep_overlap_fraction",
        ]
        == pytest.approx(1.0)
    )


def test_no_overlap():

    sensor = pd.DataFrame({
        "patient_id": ["p01"],
        "timestamp_start": [
            pd.Timestamp(
                "2021-03-09 12:00:00",
                tz="UTC",
            )
        ],
        "timestamp_end": [
            pd.Timestamp(
                "2021-03-09 12:05:00",
                tz="UTC",
            )
        ],
    })

    sleep = pd.DataFrame({
        "patient_id": ["p01"],
        "sleep_start": [
            pd.Timestamp(
                "2021-03-09 01:00:00",
                tz="UTC",
            )
        ],
        "sleep_end": [
            pd.Timestamp(
                "2021-03-09 08:00:00",
                tz="UTC",
            )
        ],
    })

    out = add_sleep_evidence(
        sensor,
        sleep,
    )

    assert not bool(
        out.loc[
            0,
            "inside_sleep_interval",
        ]
    )

    assert (
        out.loc[
            0,
            "sleep_overlap_fraction",
        ]
        == pytest.approx(0.0)
    )