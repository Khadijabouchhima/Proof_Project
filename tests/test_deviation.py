import numpy as np
import pandas as pd
import pytest

from biovance.deviation import (
    DeviationEngine,
    load_deviation_config,
)


def engine():

    return DeviationEngine(
        load_deviation_config()
    )


def observation(
    *,
    signal="hr",
    value=70.0,
    center=60.0,
    scale=5.0,
    baseline_n=28,
    baseline_status="ESTABLISHED",
    quality_status="GOOD",
    quality_score=1.0,
    context_state="REST",
):

    return pd.DataFrame({

        "patient_id": [
            "p01"
        ],

        "timestamp": [
            "2025-01-01T12:00:00Z"
        ],

        "signal": [
            signal
        ],

        "value": [
            value
        ],

        "baseline_center": [
            center
        ],

        "baseline_scale": [
            scale
        ],

        "baseline_n": [
            baseline_n
        ],

        "baseline_status": [
            baseline_status
        ],

        "baseline_personal_weight": [
            0.8
        ],

        "baseline_population_weight": [
            0.2
        ],

        "quality_score": [
            quality_score
        ],

        "quality_status": [
            quality_status
        ],

        "context_state": [
            context_state
        ],

        "context_confidence": [
            0.9
        ],
    })


# ============================================================
# Basic deviation
# ============================================================

def test_signed_deviation_score():

    out = engine().score(
        observation(
            value=70,
            center=60,
            scale=5,
        )
    )

    assert (
        out.iloc[0].deviation_score
        == pytest.approx(2.0)
    )


def test_deviation_magnitude_is_absolute():

    out = engine().score(
        observation(
            value=50,
            center=60,
            scale=5,
        )
    )

    assert (
        out.iloc[0].deviation_score
        == pytest.approx(-2.0)
    )

    assert (
        out.iloc[0].deviation_magnitude
        == pytest.approx(2.0)
    )


# ============================================================
# Physical direction
# ============================================================

def test_high_direction():

    out = engine().score(
        observation(
            value=70,
            center=60,
            scale=5,
        )
    )

    assert (
        out.iloc[0].deviation_direction
        == "HIGH"
    )


def test_low_direction():

    out = engine().score(
        observation(
            value=50,
            center=60,
            scale=5,
        )
    )

    assert (
        out.iloc[0].deviation_direction
        == "LOW"
    )


def test_none_direction():

    out = engine().score(
        observation(
            value=60,
            center=60,
            scale=5,
        )
    )

    assert (
        out.iloc[0].deviation_direction
        == "NONE"
    )


# ============================================================
# Risk orientation
# ============================================================

def test_high_hr_is_risk_aligned():

    out = engine().score(
        observation(
            signal="hr",
            value=70,
            center=60,
            scale=5,
        )
    )

    r = out.iloc[0]

    assert (
        r.risk_aligned_score
        == pytest.approx(2.0)
    )

    assert bool(
        r.toward_risk
    )


def test_low_hr_not_risk_aligned():

    out = engine().score(
        observation(
            signal="hr",
            value=50,
            center=60,
            scale=5,
        )
    )

    r = out.iloc[0]

    assert (
        r.risk_aligned_score
        == pytest.approx(-2.0)
    )

    assert not bool(
        r.toward_risk
    )


def test_low_hrv_is_risk_aligned():

    out = engine().score(
        observation(
            signal="hrv",
            value=30,
            center=40,
            scale=5,
            context_state="REST",
        )
    )

    r = out.iloc[0]

    assert (
        r.deviation_score
        == pytest.approx(-2.0)
    )

    assert (
        r.risk_aligned_score
        == pytest.approx(2.0)
    )

    assert bool(
        r.toward_risk
    )


# ============================================================
# Magnitude bands
# ============================================================

@pytest.mark.parametrize(
    "z, expected",
    [
        (0.0, "NORMAL"),
        (1.99, "NORMAL"),
        (2.0, "ELEVATED"),
        (2.99, "ELEVATED"),
        (3.0, "LARGE"),
        (4.49, "LARGE"),
        (4.5, "EXTREME"),
        (6.0, "EXTREME"),
    ],
)
def test_magnitude_bands(
    z,
    expected,
):

    out = engine().score(
        observation(
            value=60 + z * 5,
            center=60,
            scale=5,
        )
    )

    assert (
        out.iloc[0].deviation_status
        == expected
    )


# ============================================================
# Baseline lifecycle
# ============================================================

def test_learning_baseline_blocks_deviation():

    out = engine().score(
        observation(
            baseline_status="LEARNING"
        )
    )

    r = out.iloc[0]

    assert (
        r.deviation_code
        == "BASELINE_LEARNING"
    )

    assert (
        r.deviation_status
        == "UNKNOWN"
    )

    assert pd.isna(
        r.deviation_score
    )


def test_insufficient_baseline_blocks_deviation():

    out = engine().score(
        observation(
            baseline_status="INSUFFICIENT"
        )
    )

    assert (
        out.iloc[0].deviation_code
        == "BASELINE_INSUFFICIENT"
    )


def test_invalid_scale_blocks_deviation():

    out = engine().score(
        observation(
            scale=0
        )
    )

    assert (
        out.iloc[0].deviation_code
        == "INVALID_BASELINE"
    )


# ============================================================
# Quality
# ============================================================

def test_poor_quality_blocks_deviation():

    out = engine().score(
        observation(
            quality_status="POOR"
        )
    )

    r = out.iloc[0]

    assert (
        r.deviation_code
        == "POOR_QUALITY"
    )

    assert pd.isna(
        r.deviation_score
    )


# ============================================================
# Context
# ============================================================

def test_active_hr_is_context_ineligible():

    out = engine().score(
        observation(
            signal="hr",
            context_state="ACTIVE",
        )
    )

    assert (
        out.iloc[0].deviation_code
        == "CONTEXT_INELIGIBLE"
    )


def test_rest_hr_is_eligible():

    out = engine().score(
        observation(
            signal="hr",
            context_state="REST",
        )
    )

    assert (
        out.iloc[0].deviation_code
        == "OK"
    )


# ============================================================
# Missing value
# ============================================================

def test_missing_value_preserved_as_unknown():

    out = engine().score(
        observation(
            value=np.nan
        )
    )

    r = out.iloc[0]

    assert (
        r.deviation_code
        == "MISSING_VALUE"
    )

    assert (
        r.deviation_status
        == "UNKNOWN"
    )


# ============================================================
# Physical change
# ============================================================

def test_delta_units():

    out = engine().score(
        observation(
            value=66,
            center=60,
            scale=2,
        )
    )

    assert (
        out.iloc[0].delta_units
        == pytest.approx(6.0)
    )


def test_percent_change():

    out = engine().score(
        observation(
            value=66,
            center=60,
            scale=2,
        )
    )

    assert (
        out.iloc[0].percent_change
        == pytest.approx(10.0)
    )


# ============================================================
# Baseline reliability
# ============================================================

def test_full_baseline_reliability():

    out = engine().score(
        observation(
            baseline_n=28
        )
    )

    assert (
        out.iloc[0].baseline_reliability
        == pytest.approx(1.0)
    )


def test_partial_baseline_reliability():

    out = engine().score(
        observation(
            baseline_n=14
        )
    )

    assert (
        out.iloc[0].baseline_reliability
        == pytest.approx(0.5)
    )


def test_reliability_capped_at_one():

    out = engine().score(
        observation(
            baseline_n=100
        )
    )

    assert (
        out.iloc[0].baseline_reliability
        == pytest.approx(1.0)
    )


# ============================================================
# Reasons / explanation
# ============================================================

def test_reason_codes_machine_readable():

    out = engine().score(
        observation(
            value=76,
            center=60,
            scale=5,
        )
    )

    reasons = (
        out.iloc[0].reason_codes
    )

    assert (
        "ABOVE_PERSONAL_BASELINE"
        in reasons
    )

    assert (
        "RISK_ALIGNED_DIRECTION"
        in reasons
    )


def test_explanation_contains_real_units():

    out = engine().score(
        observation(
            value=66,
            center=60,
            scale=2,
        )
    )

    explanation = (
        out.iloc[0]
        .deviation_explanation
    )

    assert "6.0 bpm" in explanation

    assert "3.00 SD" in explanation


# ============================================================
# Row preservation
# ============================================================

def test_rows_are_never_deleted():

    a = observation()

    b = observation(
        value=np.nan
    )

    b[
        "timestamp"
    ] = "2025-01-02T12:00:00Z"

    c = observation(
        quality_status="POOR"
    )

    c[
        "timestamp"
    ] = "2025-01-03T12:00:00Z"

    df = pd.concat(
        [
            a,
            b,
            c,
        ],
        ignore_index=True,
    )

    out = engine().score(
        df
    )

    assert len(out) == 3


# ============================================================
# No temporal logic
# ============================================================

def test_future_rows_do_not_change_current_deviation():

    e = engine()

    first = observation(
        value=70,
        center=60,
        scale=5,
    )

    before = e.score(
        first
    )

    future = observation(
        value=200,
        center=60,
        scale=5,
    )

    future[
        "timestamp"
    ] = "2030-01-01T00:00:00Z"

    combined = pd.concat(
        [
            first,
            future,
        ],
        ignore_index=True,
    )

    after = e.score(
        combined
    )

    original = after[
        after[
            "timestamp"
        ]
        == pd.Timestamp(
            "2025-01-01T12:00:00Z"
        )
    ].iloc[0]

    assert (
        original.deviation_score
        == pytest.approx(
            before.iloc[0].deviation_score
        )
    )


# ============================================================
# Serialization
# ============================================================

def test_serialization():

    text = engine().to_json()

    assert (
        "elevated_threshold"
        in text
    )

    assert (
        "orientation"
        in text
    )