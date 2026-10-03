import numpy as np
import pandas as pd
import pytest

from biovance.quality import (
    DataQualityEngine,
    from_pmdata_daily,
    load_quality_config,
    normalize_quality_observations,
)


def cfg(**kw):
    return load_quality_config(
        "pmdata",
        **kw,
    )


def engine(**kw):
    return DataQualityEngine(
        cfg(**kw)
    )


def canonical_row(
    *,
    value=70.0,
    coverage=0.95,
    sample_count=100,
    valid=True,
    confidence=np.nan,
):
    return pd.DataFrame({
        "patient_id": ["p01"],
        "timestamp": ["2025-01-01"],
        "signal": ["hr"],
        "value": [value],
        "coverage": [coverage],
        "sample_count": [sample_count],
        "sensor_confidence": [confidence],
        "valid": [valid],
    })


# ---------------------------------------------------------
# Schema
# ---------------------------------------------------------

def test_schema_normalizes_required_columns():

    df = canonical_row()

    out = normalize_quality_observations(
        df,
        cfg(),
    )

    expected = {
        "patient_id",
        "timestamp",
        "signal",
        "value",
        "coverage",
        "sample_count",
        "sensor_confidence",
        "valid",
    }

    assert expected <= set(out.columns)


def test_missing_required_columns_raise_clear_error():

    with pytest.raises(
        ValueError,
        match="patient id",
    ):
        normalize_quality_observations(
            pd.DataFrame({
                "timestamp": ["2025-01-01"],
                "signal": ["hr"],
                "value": [70],
            }),
            cfg(),
        )


# ---------------------------------------------------------
# Quality behavior
# ---------------------------------------------------------

def test_good_quality_observation():

    out = engine().assess(
        canonical_row(
            coverage=0.95,
            sample_count=150,
            valid=True,
        )
    )

    row = out.iloc[0]

    assert row.quality_status == "GOOD"
    assert 0.8 <= row.quality_score <= 1.0


def test_moderate_quality_becomes_fair():

    out = engine().assess(
        canonical_row(
            coverage=0.60,
            sample_count=50,
            valid=True,
        )
    )

    row = out.iloc[0]

    assert row.quality_status in {
        "FAIR",
        "GOOD",
    }

    assert 0 <= row.quality_score <= 1


def test_critically_low_coverage_is_poor():

    out = engine().assess(
        canonical_row(
            coverage=0.10,
            sample_count=200,
            valid=True,
        )
    )

    row = out.iloc[0]

    assert row.quality_status == "POOR"
    assert (
        "critically_low_coverage"
        in row.quality_reasons
    )


def test_invalid_observation_is_always_poor():

    out = engine().assess(
        canonical_row(
            coverage=1.0,
            sample_count=500,
            valid=False,
        )
    )

    row = out.iloc[0]

    assert row.quality_status == "POOR"
    assert (
        "invalid_observation"
        in row.quality_reasons
    )


def test_missing_signal_value_is_poor():

    out = engine().assess(
        canonical_row(
            value=np.nan,
            coverage=1.0,
            sample_count=500,
            valid=True,
        )
    )

    row = out.iloc[0]

    assert row.quality_status == "POOR"
    assert (
        "missing_signal_value"
        in row.quality_reasons
    )


def test_insufficient_samples_is_poor():

    out = engine().assess(
        canonical_row(
            coverage=0.95,
            sample_count=1,
            valid=True,
        )
    )

    row = out.iloc[0]

    assert row.quality_status == "POOR"
    assert (
        "insufficient_samples"
        in row.quality_reasons
    )


def test_score_always_between_zero_and_one():

    df = pd.concat(
        [
            canonical_row(
                coverage=0.0,
                sample_count=0,
                valid=False,
            ),
            canonical_row(
                coverage=0.5,
                sample_count=50,
                valid=True,
            ),
            canonical_row(
                coverage=1.0,
                sample_count=1000,
                valid=True,
            ),
        ],
        ignore_index=True,
    )

    df["timestamp"] = pd.date_range(
        "2025-01-01",
        periods=len(df),
        freq="D",
    )

    out = engine().assess(df)

    assert (
        out["quality_score"]
        .between(0, 1)
        .all()
    )


# ---------------------------------------------------------
# Missing optional evidence
# ---------------------------------------------------------

def test_missing_optional_component_is_renormalized():

    df = canonical_row(
        coverage=1.0,
        sample_count=100,
        valid=True,
        confidence=np.nan,
    )

    out = engine().assess(df)

    row = out.iloc[0]

    assert row.quality_status == "GOOD"

    # confidence is disabled/missing and must not
    # automatically reduce the observation to poor quality.
    assert row.quality_score > 0.8


# ---------------------------------------------------------
# Row preservation
# ---------------------------------------------------------

def test_engine_does_not_remove_observations():

    df = pd.concat(
        [
            canonical_row(
                value=70,
                coverage=0.95,
            ),
            canonical_row(
                value=np.nan,
                coverage=0.0,
                valid=False,
            ),
        ],
        ignore_index=True,
    )

    df["timestamp"] = [
        "2025-01-01",
        "2025-01-02",
    ]

    out = engine().assess(df)

    assert len(out) == len(df)


# ---------------------------------------------------------
# PMData adapter
# ---------------------------------------------------------

def test_pmdata_adapter():

    raw = pd.DataFrame({
        "participant": ["p01"],
        "date": ["2025-01-01"],
        "hr_rest_median": [58.0],
        "hr_rest_n": [1846],
        "hr_coverage": [0.98],
        "hr_valid_day": [True],
        "hr_conf_mean": [2.3],
    })

    out = from_pmdata_daily(raw)

    assert out.loc[0, "patient_id"] == "p01"
    assert out.loc[0, "signal"] == "hr"
    assert out.loc[0, "value"] == 58.0
    assert out.loc[0, "coverage"] == 0.98
    assert out.loc[0, "sample_count"] == 1846
    assert bool(out.loc[0, "valid"])


def test_pmdata_adapter_then_engine():

    raw = pd.DataFrame({
        "participant": ["p01"],
        "date": ["2025-01-01"],
        "hr_rest_median": [58.0],
        "hr_rest_n": [1846],
        "hr_coverage": [0.98],
        "hr_valid_day": [True],
        "hr_conf_mean": [2.3],
    })

    adapted = from_pmdata_daily(raw)

    out = engine().assess(adapted)

    assert len(out) == 1
    assert out.loc[0, "quality_status"] == "GOOD"
    assert out.loc[0, "quality_score"] > 0.8


# ---------------------------------------------------------
# Determinism / no future leakage
# ---------------------------------------------------------

def test_future_rows_do_not_change_past_quality():

    first = canonical_row(
        coverage=0.8,
        sample_count=80,
        valid=True,
    )

    future = canonical_row(
        value=200,
        coverage=0.01,
        sample_count=0,
        valid=False,
    )

    future["timestamp"] = "2025-03-01"

    e = engine()

    before = e.assess(first)

    combined = pd.concat(
        [first, future],
        ignore_index=True,
    )

    after = e.assess(combined)

    past = after[
        after["timestamp"]
        == pd.Timestamp("2025-01-01")
    ]

    assert before.iloc[0].quality_score == pytest.approx(
        past.iloc[0].quality_score
    )

    assert before.iloc[0].quality_status == (
        past.iloc[0].quality_status
    )


# ---------------------------------------------------------
# Explainability
# ---------------------------------------------------------

def test_explain():

    e = engine()

    out = e.assess(
        canonical_row()
    )

    txt = e.explain(
        out,
        "p01",
        "2025-01-01",
        "hr",
    )

    assert "quality=" in txt
    assert "GOOD" in txt