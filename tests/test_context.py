import numpy as np
import pandas as pd
import pytest

from biovance.context import (
    ContextEngine,
    load_context_config,
)


def make_rows(
    patient_id="p01",
    n=100,
    gyro_low=0.4,
    gyro_high=2.0,
):

    timestamps = pd.date_range(
        "2021-01-01",
        periods=n,
        freq="5min",
        tz="UTC",
    )

    gyro = np.linspace(
        gyro_low,
        gyro_high,
        n,
    )

    return pd.DataFrame({
        "patient_id":
            patient_id,

        "timestamp_start":
            timestamps,

        "timestamp_end":
            timestamps
            + pd.Timedelta(
                minutes=5
            ),

        "sleep_overlap_fraction":
            np.zeros(
                n
            ),

        "gyro_magnitude":
            gyro,

        "steps":
            np.zeros(
                n
            ),

        "dynamic_acc":
            np.nan,
    })


def fitted_engine(
    df,
    **overrides,
):

    cfg = load_context_config(
        "baigutanova",
        **overrides,
    )

    engine = ContextEngine(
        cfg
    )

    engine.fit(
        df
    )

    return engine


def test_requires_fit_when_personalized():

    cfg = load_context_config(
        "baigutanova"
    )

    engine = ContextEngine(
        cfg
    )

    df = make_rows(
        n=10
    )

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):

        engine.classify(
            df
        )


def test_fit_creates_global_thresholds():

    df = make_rows(
        n=200
    )

    e = fitted_engine(
        df
    )

    assert (
        e.global_active_gyro
        >
        e.global_rest_gyro
    )

    assert (
        e.global_reference_n
        == 200
    )


def test_personal_thresholds_created():

    df = make_rows(
        patient_id="p01",
        n=100,
    )

    e = fitted_engine(
        df
    )

    assert (
        "p01"
        in e.patient_thresholds
    )

    t = e.patient_thresholds[
        "p01"
    ]

    assert (
        t["active"]
        >
        t["rest"]
    )

    assert (
        t["n"]
        == 100
    )


def test_small_sample_uses_global_fallback():

    df = make_rows(
        n=10
    )

    e = fitted_engine(
        df,
        min_personal_awake_windows=30,
        min_global_awake_windows=1000,
    )

    t = e.patient_thresholds[
        "p01"
    ]

    assert (
        t["source"]
        == "global_fallback"
    )


def test_different_people_get_different_thresholds():

    low_motion = make_rows(
        patient_id="low",
        n=100,
        gyro_low=0.1,
        gyro_high=1.0,
    )

    high_motion = make_rows(
        patient_id="high",
        n=100,
        gyro_low=1.0,
        gyro_high=3.0,
    )

    reference = pd.concat(
        [
            low_motion,
            high_motion,
        ],
        ignore_index=True,
    )

    e = fitted_engine(
        reference,
        shrinkage_strength=0,
    )

    low = e.patient_thresholds[
        "low"
    ]

    high = e.patient_thresholds[
        "high"
    ]

    assert (
        high["rest"]
        >
        low["rest"]
    )

    assert (
        high["active"]
        >
        low["active"]
    )


def test_shrinkage_moves_personal_threshold_toward_global():

    p1 = make_rows(
        "p1",
        100,
        0.1,
        1.0,
    )

    p2 = make_rows(
        "p2",
        100,
        1.0,
        3.0,
    )

    reference = pd.concat(
        [
            p1,
            p2,
        ],
        ignore_index=True,
    )

    no_shrink = fitted_engine(
        reference,
        shrinkage_strength=0,
    )

    shrunk = fitted_engine(
        reference,
        shrinkage_strength=100,
    )

    raw_rest = (
        no_shrink
        .patient_thresholds[
            "p1"
        ]["rest"]
    )

    shrunk_rest = (
        shrunk
        .patient_thresholds[
            "p1"
        ]["rest"]
    )

    global_rest = (
        shrunk.global_rest_gyro
    )

    assert (
        abs(
            shrunk_rest
            - global_rest
        )
        <
        abs(
            raw_rest
            - global_rest
        )
    )


def test_sleep_precedence():

    reference = make_rows(
        n=100
    )

    e = fitted_engine(
        reference
    )

    future = pd.DataFrame({
        "patient_id": ["p01"],
        "timestamp_start": [
            "2021-02-01T01:00:00Z"
        ],
        "timestamp_end": [
            "2021-02-01T01:05:00Z"
        ],
        "sleep_overlap_fraction": [
            1.0
        ],
        "gyro_magnitude": [
            10.0
        ],
        "steps": [
            20
        ],
    })

    out = e.classify(
        future
    )

    assert (
        out.iloc[0]
        .context_state
        == "SLEEP"
    )


def test_high_personal_gyro_is_active():

    reference = make_rows(
        n=100
    )

    e = fitted_engine(
        reference
    )

    threshold = (
        e.patient_thresholds[
            "p01"
        ]["active"]
    )

    future = make_rows(
        n=1
    )

    future[
        "gyro_magnitude"
    ] = (
        threshold
        + 1.0
    )

    future[
        "timestamp_start"
    ] = pd.Timestamp(
        "2022-01-01",
        tz="UTC",
    )

    future[
        "timestamp_end"
    ] = (
        future[
            "timestamp_start"
        ]
        + pd.Timedelta(
            minutes=5
        )
    )

    out = e.classify(
        future
    )

    assert (
        out.iloc[0]
        .context_state
        == "ACTIVE"
    )


def test_low_personal_gyro_is_rest():

    reference = make_rows(
        n=100
    )

    e = fitted_engine(
        reference
    )

    threshold = (
        e.patient_thresholds[
            "p01"
        ]["rest"]
    )

    future = make_rows(
        n=1
    )

    future[
        "gyro_magnitude"
    ] = max(
        0,
        threshold
        - 0.1,
    )

    future[
        "steps"
    ] = 0

    future[
        "timestamp_start"
    ] = pd.Timestamp(
        "2022-01-01",
        tz="UTC",
    )

    future[
        "timestamp_end"
    ] = (
        future[
            "timestamp_start"
        ]
        + pd.Timedelta(
            minutes=5
        )
    )

    out = e.classify(
        future
    )

    assert (
        out.iloc[0]
        .context_state
        == "REST"
    )


def test_middle_personal_gyro_is_unknown():

    reference = make_rows(
        n=100
    )

    e = fitted_engine(
        reference
    )

    t = e.patient_thresholds[
        "p01"
    ]

    middle = (
        t["rest"]
        + t["active"]
    ) / 2

    future = make_rows(
        n=1
    )

    future[
        "gyro_magnitude"
    ] = middle

    future[
        "steps"
    ] = np.nan

    future[
        "timestamp_start"
    ] = pd.Timestamp(
        "2022-01-01",
        tz="UTC",
    )

    future[
        "timestamp_end"
    ] = (
        future[
            "timestamp_start"
        ]
        + pd.Timedelta(
            minutes=5
        )
    )

    out = e.classify(
        future
    )

    assert (
        out.iloc[0]
        .context_state
        == "UNKNOWN"
    )


def test_unseen_patient_uses_global_threshold():

    reference = make_rows(
        "known",
        100,
    )

    e = fitted_engine(
        reference
    )

    future = make_rows(
        "new_person",
        1,
    )

    future[
        "timestamp_start"
    ] = pd.Timestamp(
        "2022-01-01",
        tz="UTC",
    )

    future[
        "timestamp_end"
    ] = (
        future[
            "timestamp_start"
        ]
        + pd.Timedelta(
            minutes=5
        )
    )

    out = e.classify(
        future
    )

    assert (
        out.iloc[0]
        .gyro_threshold_source
        == "global_unseen_patient"
    )


def test_thresholds_are_frozen_after_fit():

    reference = make_rows(
        n=100
    )

    e = fitted_engine(
        reference
    )

    before = dict(
        e.patient_thresholds[
            "p01"
        ]
    )

    future = make_rows(
        n=100,
        gyro_low=50,
        gyro_high=100,
    )

    future[
        "timestamp_start"
    ] = pd.date_range(
        "2022-01-01",
        periods=100,
        freq="5min",
        tz="UTC",
    )

    future[
        "timestamp_end"
    ] = (
        future[
            "timestamp_start"
        ]
        + pd.Timedelta(
            minutes=5
        )
    )

    e.classify(
        future
    )

    after = (
        e.patient_thresholds[
            "p01"
        ]
    )

    assert (
        before
        == after
    )


def test_future_data_does_not_change_thresholds():

    reference = make_rows(
        n=100
    )

    e1 = fitted_engine(
        reference
    )

    t1 = dict(
        e1.patient_thresholds[
            "p01"
        ]
    )

    # Future data are classified, not fitted.
    future = make_rows(
        n=100,
        gyro_low=20,
        gyro_high=40,
    )

    future[
        "timestamp_start"
    ] = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="5min",
        tz="UTC",
    )

    future[
        "timestamp_end"
    ] = (
        future[
            "timestamp_start"
        ]
        + pd.Timedelta(
            minutes=5
        )
    )

    e1.classify(
        future
    )

    t2 = (
        e1.patient_thresholds[
            "p01"
        ]
    )

    assert t1 == t2


def test_activity_level_in_range():

    reference = make_rows(
        n=100
    )

    e = fitted_engine(
        reference
    )

    out = e.classify(
        reference
    )

    valid = (
        out[
            "activity_level"
        ]
        .dropna()
    )

    assert (
        valid
        .between(
            0,
            1,
        )
        .all()
    )


def test_threshold_table():

    reference = make_rows(
        n=100
    )

    e = fitted_engine(
        reference
    )

    table = (
        e.threshold_table()
    )

    assert (
        "gyro_rest_threshold"
        in table.columns
    )

    assert (
        "gyro_active_threshold"
        in table.columns
    )


def test_serialization():

    reference = make_rows(
        n=100
    )

    e = fitted_engine(
        reference
    )

    text = e.to_json()

    assert (
        "global_rest_gyro"
        in text
    )

    assert (
        "patient_thresholds"
        in text
    )