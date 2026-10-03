"""Tests for BioVance Fusion Engine v1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biovance.fusion import (
    FusionEngine,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "fusion.yaml"
)


@pytest.fixture
def engine():

    return FusionEngine.from_yaml(
        CONFIG_PATH
    )


def make_row(
    signal,
    minute,
    risk_score,
    temporal_code="OK",
    persistence=0.5,
    recurrence=0.5,
    direction=1.0,
    trend=0.5,
    confidence=0.9,
):

    timestamp = (
        pd.Timestamp(
            "2025-01-01 08:00"
        )
        +
        pd.Timedelta(
            minutes=minute
        )
    )

    return {
        "patient_id":
            "p01",

        "timestamp":
            timestamp,

        "signal":
            signal,

        "risk_aligned_score":
            risk_score,

        "deviation_magnitude":
            abs(
                risk_score
            ),

        "persistence_score":
            persistence,

        "recurrence_score":
            recurrence,

        "direction_consistency":
            direction,

        "trend_score":
            trend,

        "temporal_confidence":
            confidence,

        "reference_reliability":
            1.0,

        "temporal_code":
            temporal_code,

        "phase":
            "EVALUATION",

        "context_state":
            "WAKE",
    }


def make_df(
    rows,
):

    return pd.DataFrame(
        rows
    )


def test_preserves_row_count(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "dbp",
            0,
            2.5,
        ),
    ])

    out = engine.transform(
        df
    )

    assert len(out) == len(df)


def test_single_signal_is_insufficient(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "fusion_state"
        ]
        ==
        "INSUFFICIENT_EVIDENCE"
    )


def test_two_bp_signals_same_family(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "dbp",
            0,
            2.5,
        ),
    ])

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "supporting_signal_count"
        ]
        == 2
    )

    assert (
        last[
            "supporting_family_count"
        ]
        == 1
    )

    assert (
        last[
            "fusion_state"
        ]
        ==
        "MULTISIGNAL_SAME_FAMILY"
    )


def test_bp_and_hr_create_multimodal_support(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            0,
            2.5,
        ),
    ])

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "supporting_family_count"
        ]
        == 2
    )


def test_three_signals_can_form_consensus(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "dbp",
            0,
            2.5,
        ),
        make_row(
            "hr",
            0,
            3.2,
        ),
    ])

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "fusion_state"
        ]
        ==
        "MULTIMODAL_CONSENSUS"
    )


def test_normal_signals_do_not_support(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            0.5,
        ),
        make_row(
            "hr",
            0,
            0.3,
        ),
    ])

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "fusion_state"
        ]
        ==
        "NO_SUPPORT"
    )


def test_opposite_direction_creates_conflict(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            0,
            -3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "fusion_state"
        ]
        ==
        "CONFLICTING_EVIDENCE"
    )


def test_old_signal_is_not_used(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            120,
            3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "available_signal_count"
        ]
        == 1
    )


def test_signal_within_lookback_is_used(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            60,
            3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "available_signal_count"
        ]
        == 2
    )


def test_latest_signal_observation_is_used(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "sbp",
            30,
            0.2,
        ),
        make_row(
            "hr",
            30,
            3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "supporting_signal_count"
        ]
        == 1
    )


def test_contexts_do_not_mix(
    engine,
):

    wake = make_row(
        "sbp",
        0,
        3.0,
    )

    sleep = make_row(
        "hr",
        30,
        3.0,
    )

    sleep[
        "context_state"
    ] = "SLEEP"

    df = make_df([
        wake,
        sleep,
    ])

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "available_signal_count"
        ]
        == 1
    )


def test_temporal_unusable_signal_not_used(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
            temporal_code="OK",
        ),

        make_row(
            "dbp",
            0,
            3.0,
            temporal_code=(
                "INSUFFICIENT_TEMPORAL_EVIDENCE"
            ),
        ),

        make_row(
            "hr",
            0,
            3.0,
            temporal_code="OK",
        ),
    ])

    out = engine.transform(
        df
    )

    assert (
        out.iloc[-1][
            "available_signal_count"
        ]
        == 2
    )


def test_confidence_between_zero_and_one(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            0,
            3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    value = out.iloc[-1][
        "fusion_confidence"
    ]

    assert 0 <= value <= 1


def test_fusion_score_between_zero_and_one(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            4.0,
        ),
        make_row(
            "hr",
            0,
            4.0,
        ),
    ])

    out = engine.transform(
        df
    )

    value = out.iloc[-1][
        "fusion_score"
    ]

    assert 0 <= value <= 1


def test_stronger_evidence_increases_score(
    engine,
):

    weak = make_df([
        make_row(
            "sbp",
            0,
            2.1,
            persistence=0.2,
        ),
        make_row(
            "hr",
            0,
            2.1,
            persistence=0.2,
        ),
    ])

    strong = make_df([
        make_row(
            "sbp",
            0,
            4.0,
            persistence=1.0,
            recurrence=1.0,
            trend=1.0,
        ),
        make_row(
            "hr",
            0,
            4.0,
            persistence=1.0,
            recurrence=1.0,
            trend=1.0,
        ),
    ])

    weak_out = engine.transform(
        weak
    )

    strong_out = engine.transform(
        strong
    )

    assert (
        strong_out.iloc[-1][
            "fusion_score"
        ]
        >
        weak_out.iloc[-1][
            "fusion_score"
        ]
    )


def test_future_rows_do_not_change_past(
    engine,
):

    original = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            30,
            3.0,
        ),
    ])

    extended = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            30,
            3.0,
        ),
        make_row(
            "dbp",
            60,
            -8.0,
        ),
    ])

    first = engine.transform(
        original
    )

    second = engine.transform(
        extended
    )

    assert (
        first.iloc[1][
            "fusion_state"
        ]
        ==
        second.iloc[1][
            "fusion_state"
        ]
    )

    assert (
        first.iloc[1][
            "fusion_score"
        ]
        ==
        pytest.approx(
            second.iloc[1][
                "fusion_score"
            ]
        )
    )


def test_explanation_created(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            0,
            3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    assert isinstance(
        out.iloc[-1][
            "fusion_explanation"
        ],
        str,
    )


def test_missing_required_column_raises(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        )
    ])

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

def test_conflict_reduces_adjusted_fusion_score(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            0,
            -3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "fusion_state"
        ]
        ==
        "CONFLICTING_EVIDENCE"
    )

    assert (
        last[
            "opposition_fraction"
        ]
        > 0
    )

    assert (
        last[
            "fusion_score"
        ]
        <
        last[
            "fusion_evidence_strength"
        ]
    )

def test_no_opposition_keeps_full_evidence_strength(
    engine,
):

    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
        ),
        make_row(
            "hr",
            0,
            3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    last = out.iloc[-1]

    assert (
        last[
            "opposition_fraction"
        ]
        == pytest.approx(
            0.0
        )
    )

    assert (
        last[
            "fusion_score"
        ]
        ==
        pytest.approx(
            last[
                "fusion_evidence_strength"
            ]
        )
    )