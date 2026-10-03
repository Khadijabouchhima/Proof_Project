"""Tests for BioVance Temporal Engine v1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biovance.temporal import (
    TemporalEngine,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "temporal.yaml"
)


@pytest.fixture
def engine():
    return TemporalEngine.from_yaml(
        CONFIG_PATH
    )


def make_df(
    scores,
    minutes=None,
    codes=None,
    phase="EVALUATION",
    context="WAKE",
    reliability=1.0,
):
    """Construct synthetic temporal input."""

    if minutes is None:
        minutes = [
            30 * i
            for i in range(
                len(scores)
            )
        ]

    if codes is None:
        codes = [
            "OK"
            for _ in scores
        ]

    start = pd.Timestamp(
        "2025-01-01 08:00:00"
    )

    rows = []

    for i, (
        score,
        minute,
        code,
    ) in enumerate(
        zip(
            scores,
            minutes,
            codes,
        )
    ):

        magnitude = (
            abs(score)
            if score is not None
            else np.nan
        )

        rows.append({
            "patient_id":
                "p01",

            "timestamp":
                start
                + pd.Timedelta(
                    minutes=minute
                ),

            "signal":
                "sbp",

            "deviation_score":
                score,

            "deviation_magnitude":
                magnitude,

            "risk_aligned_score":
                score,

            "deviation_code":
                code,

            "phase":
                phase,

            "context_state":
                context,

            "reference_reliability":
                reliability,
        })

    return pd.DataFrame(
        rows
    )


def test_preserves_row_count(
    engine,
):
    df = make_df(
        [0.2, 2.5, 2.8]
    )

    out = engine.transform(
        df
    )

    assert len(out) == len(df)


def test_preserves_original_order(
    engine,
):
    df = make_df(
        [0.2, 2.5, 2.8]
    )

    df = df.iloc[
        [2, 0, 1]
    ].reset_index(
        drop=True
    )

    expected = list(
        df[
            "timestamp"
        ]
    )

    out = engine.transform(
        df
    )

    assert list(
        out[
            "timestamp"
        ]
    ) == expected


def test_reference_row_not_evaluated(
    engine,
):
    df = make_df(
        [2.5]
    )

    df["phase"] = "REFERENCE"

    out = engine.transform(
        df
    )

    assert (
        out.loc[
            0,
            "temporal_code",
        ]
        == "NOT_EVALUATION"
    )


def test_unusable_current_deviation(
    engine,
):
    df = make_df(
        [np.nan],
        codes=[
            "MISSING_OR_FAILED_MEASUREMENT"
        ],
    )

    out = engine.transform(
        df
    )

    assert (
        out.loc[
            0,
            "temporal_code",
        ]
        ==
        "CURRENT_DEVIATION_UNAVAILABLE"
    )


def test_initial_rows_have_insufficient_evidence(
    engine,
):
    df = make_df(
        [2.5]
    )

    out = engine.transform(
        df
    )

    assert (
        out.loc[
            0,
            "temporal_state",
        ]
        ==
        "INSUFFICIENT_EVIDENCE"
    )


def test_no_current_elevation(
    engine,
):
    df = make_df(
        [
            0.5,
            0.6,
            0.7,
        ]
    )

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "temporal_state"
        ]
        ==
        "NO_CURRENT_ELEVATION"
    )


def test_two_elevated_observations_form_persistence(
    engine,
):
    df = make_df(
        [
            0.2,
            2.5,
            2.8,
        ]
    )

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "persistence_observation_count"
        ]
        == 2
    )

    assert (
        last[
            "persistence_duration_minutes"
        ]
        == pytest.approx(
            30.0
        )
    )

    assert (
        "PERSISTENT"
        in last[
            "temporal_state"
        ]
    )


def test_normal_observation_breaks_persistence(
    engine,
):
    df = make_df(
        [
            2.5,
            0.5,
            2.8,
        ]
    )

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "persistence_observation_count"
        ]
        == 1
    )


def test_large_time_gap_breaks_persistence(
    engine,
):
    df = make_df(
        [
            2.5,
            2.8,
        ],
        minutes=[
            0,
            180,
        ],
    )

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "persistence_observation_count"
        ]
        == 1
    )


def test_recurrence_detects_separate_episodes(
    engine,
):
    df = make_df(
        [
            2.5,
            0.5,
            2.6,
        ]
    )

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "recurrence_count"
        ]
        == 2
    )


def test_recurrence_can_be_separated_by_time_gap(
    engine,
):
    df = make_df(
        [
            2.5,
            2.6,
            2.7,
        ],
        minutes=[
            0,
            30,
            240,
        ],
    )

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "recurrence_count"
        ]
        == 2
    )


def test_direction_consistency_same_direction(
    engine,
):
    df = make_df(
        [
            2.5,
            2.8,
            3.0,
        ]
    )

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "direction_consistency"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        out.iloc[-1][
            "direction_dominant"
        ]
        == "HIGH"
    )


def test_direction_consistency_mixed(
    engine,
):
    df = make_df(
        [
            2.5,
            -2.8,
            3.0,
            -3.1,
        ]
    )

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "direction_consistency"
        ]
        == pytest.approx(
            0.5
        )
    )


def test_positive_risk_aligned_trend_is_positive(
    engine,
):
    df = make_df(
        [
            0.5,
            1.0,
            2.0,
            3.0,
        ]
    )

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "trend_slope_z_per_hour"
        ]
        > 0
    )

    assert (
        out.iloc[-1][
            "trend_score"
        ]
        > 0
    )


def test_negative_risk_aligned_trend_is_negative(
    engine,
):
    df = make_df(
        [
            3.0,
            2.0,
            1.0,
            0.5,
        ]
    )

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "trend_slope_z_per_hour"
        ]
        < 0
    )

    assert (
        out.iloc[-1][
            "trend_score"
        ]
        < 0
    )


def test_evidence_fraction_detects_failed_rows(
    engine,
):
    df = make_df(
        [
            2.5,
            np.nan,
            2.8,
        ],
        codes=[
            "OK",
            "MISSING_OR_FAILED_MEASUREMENT",
            "OK",
        ],
    )

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "opportunity_count"
        ]
        == 3
    )

    assert (
        last[
            "evidence_count"
        ]
        == 2
    )

    assert (
        last[
            "evidence_fraction"
        ]
        == pytest.approx(
            2 / 3
        )
    )


def test_reference_reliability_affects_confidence(
    engine,
):
    high = make_df(
        [
            0.5,
            0.6,
            0.7,
            0.8,
        ],
        reliability=1.0,
    )

    low = make_df(
        [
            0.5,
            0.6,
            0.7,
            0.8,
        ],
        reliability=0.2,
    )

    high_out = engine.transform(
        high
    )

    low_out = engine.transform(
        low
    )

    assert (
        high_out.iloc[-1][
            "temporal_confidence"
        ]
        >
        low_out.iloc[-1][
            "temporal_confidence"
        ]
    )


def test_contexts_do_not_mix(
    engine,
):
    df = make_df(
        [
            3.0,
            3.0,
            3.0,
        ]
    )

    df.loc[
        1,
        "context_state",
    ] = "SLEEP"

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    # WAKE row at t0 and WAKE row at t60 only.
    assert (
        last[
            "evidence_count"
        ]
        == 2
    )


def test_future_rows_do_not_change_past_output(
    engine,
):
    first = make_df(
        [
            0.5,
            2.5,
            2.8,
        ]
    )

    extended = make_df(
        [
            0.5,
            2.5,
            2.8,
            9.0,
            -8.0,
        ]
    )

    out_first = engine.transform(
        first
    )

    out_extended = engine.transform(
        extended
    )

    columns = [
        "temporal_state",
        "persistence_score",
        "recurrence_count",
        "trend_score",
        "temporal_confidence",
    ]

    for column in columns:

        a = out_first.loc[
            2,
            column,
        ]

        b = out_extended.loc[
            2,
            column,
        ]

        if (
            pd.isna(a)
            and pd.isna(b)
        ):
            continue

        assert a == pytest.approx(
            b
        ) if isinstance(
            a,
            (float, np.floating),
        ) else a == b


def test_reason_codes_serializable(
    engine,
):
    df = make_df(
        [
            0.5,
            2.5,
            2.8,
        ]
    )

    out = engine.transform(
        df
    )

    value = out.iloc[-1][
        "temporal_reason_codes"
    ]

    assert isinstance(
        value,
        str,
    )

    assert value.startswith(
        "["
    )


def test_explanation_created(
    engine,
):
    df = make_df(
        [
            0.5,
            2.5,
            2.8,
        ]
    )

    out = engine.transform(
        df
    )

    explanation = out.iloc[-1][
        "temporal_explanation"
    ]

    assert isinstance(
        explanation,
        str,
    )

    assert "Temporal state" in explanation


def test_missing_required_column_raises(
    engine,
):
    df = make_df(
        [
            1.0,
            2.0,
        ]
    )

    df = df.drop(
        columns=[
            "risk_aligned_score"
        ]
    )

    with pytest.raises(
        ValueError
    ):
        engine.transform(
            df
        )