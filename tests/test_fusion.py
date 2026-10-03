from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from biovance.fusion import (
    FusionEngine,
    load_fusion_config,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "fusion.yaml"
)

BASE_TIME = pd.Timestamp(
    "2025-01-01 08:00:00"
)


# ============================================================
# Helpers
# ============================================================

def make_row(
    signal: str,
    minute: int,
    risk_aligned_score: float,
    *,
    persistence_score: float = 0.5,
    recurrence_score: float = 0.5,
    direction_consistency: float = 1.0,
    trend_score: float = 0.5,
    temporal_confidence: float = 0.9,
    temporal_code: str = "OK",
    phase: str = "EVALUATION",
    context_state: str = "WAKE",
    patient_id: str = "p01",
) -> dict:
    """
    Create one standardized Temporal -> Fusion input row.
    """

    return {
        "patient_id": patient_id,
        "timestamp": (
            BASE_TIME
            + pd.Timedelta(
                minutes=minute
            )
        ),
        "signal": signal,
        "risk_aligned_score": (
            risk_aligned_score
        ),
        "deviation_magnitude": abs(
            risk_aligned_score
        ),
        "persistence_score": (
            persistence_score
        ),
        "recurrence_score": (
            recurrence_score
        ),
        "direction_consistency": (
            direction_consistency
        ),
        "trend_score": trend_score,
        "temporal_confidence": (
            temporal_confidence
        ),
        "temporal_code": temporal_code,
        "phase": phase,
        "context_state": context_state,
    }


def make_df(
    rows: list[dict],
) -> pd.DataFrame:
    return pd.DataFrame(rows)


def row_at(
    out: pd.DataFrame,
    minute: int,
) -> pd.Series:
    """
    Return the single Fusion result for a timestamp.
    """

    timestamp = (
        BASE_TIME
        + pd.Timedelta(
            minutes=minute
        )
    )

    selected = out[
        out["timestamp"] == timestamp
    ]

    assert len(selected) == 1

    return selected.iloc[0]


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def config():
    return load_fusion_config(
        CONFIG_PATH
    )


@pytest.fixture
def engine(
    config,
):
    return FusionEngine(
        config
    )


# ============================================================
# Atomic evaluation contract
# ============================================================

def test_one_result_per_patient_timestamp(
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

    assert len(out) == 1

    assert (
        out.loc[
            0,
            "patient_id",
        ]
        == "p01"
    )

    assert (
        out.loc[
            0,
            "timestamp",
        ]
        == BASE_TIME
    )

    assert (
        out.loc[
            0,
            "current_signal_count",
        ]
        == 2
    )

    signals = json.loads(
        out.loc[
            0,
            "current_signals",
        ]
    )

    assert set(signals) == {
        "sbp",
        "dbp",
    }


def test_same_timestamp_signal_order_does_not_change_result(
    engine,
):
    rows = [
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
    ]

    df_a = make_df(
        rows
    )

    df_b = make_df(
        list(
            reversed(
                rows
            )
        )
    )

    out_a = engine.transform(
        df_a
    )

    out_b = engine.transform(
        df_b
    )

    cols = [
        "fusion_code",
        "fusion_state",
        "fusion_evidence_strength",
        "fusion_score",
        "fusion_confidence",
        "available_signal_count",
        "supporting_signal_count",
        "opposing_signal_count",
        "available_family_count",
        "supporting_family_count",
        "support_fraction",
        "family_support_fraction",
        "direction_agreement",
    ]

    pd.testing.assert_frame_equal(
        out_a[
            cols
        ].reset_index(
            drop=True
        ),
        out_b[
            cols
        ].reset_index(
            drop=True
        ),
    )


def test_all_simultaneous_signals_are_seen_together(
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

    assert len(out) == 1

    result = out.iloc[0]

    assert (
        result[
            "available_signal_count"
        ]
        == 3
    )

    assert (
        result[
            "supporting_signal_count"
        ]
        == 3
    )

    assert (
        result[
            "fusion_state"
        ]
        == "MULTIMODAL_CONSENSUS"
    )


# ============================================================
# Evidence availability
# ============================================================

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

    result = out.iloc[0]

    assert (
        result["fusion_code"]
        == "INSUFFICIENT_MULTISIGNAL_EVIDENCE"
    )

    assert (
        result["fusion_state"]
        == "INSUFFICIENT_EVIDENCE"
    )

    assert (
        result[
            "available_signal_count"
        ]
        == 1
    )


# ============================================================
# Signal families
# ============================================================

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
            3.0,
        ),
    ])

    out = engine.transform(
        df
    )

    result = out.iloc[0]

    assert (
        result["fusion_code"]
        == "OK"
    )

    assert (
        result[
            "supporting_signal_count"
        ]
        == 2
    )

    assert (
        result[
            "supporting_family_count"
        ]
        == 1
    )

    assert (
        result["fusion_state"]
        == "MULTISIGNAL_SAME_FAMILY"
    )


def test_bp_and_hr_create_multimodal_support(
    engine,
):
    # Three signals are available.
    #
    # SBP and HR support concern.
    # DBP is neutral.
    #
    # support fraction = 2/3, which is below the configured
    # 0.67 consensus threshold, so this is MULTIMODAL_SUPPORT.
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
        make_row(
            "dbp",
            0,
            0.0,
        ),
    ])

    out = engine.transform(
        df
    )

    result = out.iloc[0]

    assert (
        result[
            "supporting_signal_count"
        ]
        == 2
    )

    assert (
        result[
            "supporting_family_count"
        ]
        == 2
    )

    assert (
        result["fusion_state"]
        == "MULTIMODAL_SUPPORT"
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
            3.5,
        ),
    ])

    out = engine.transform(
        df
    )

    result = out.iloc[0]

    assert (
        result[
            "available_signal_count"
        ]
        == 3
    )

    assert (
        result[
            "supporting_signal_count"
        ]
        == 3
    )

    assert (
        result[
            "supporting_family_count"
        ]
        == 2
    )

    assert (
        result["fusion_state"]
        == "MULTIMODAL_CONSENSUS"
    )


# ============================================================
# Support / opposition
# ============================================================

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
            0.5,
        ),
    ])

    out = engine.transform(
        df
    )

    result = out.iloc[0]

    assert (
        result[
            "supporting_signal_count"
        ]
        == 0
    )

    assert (
        result[
            "opposing_signal_count"
        ]
        == 0
    )

    assert (
        result["fusion_state"]
        == "NO_SUPPORT"
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

    result = out.iloc[0]

    assert (
        result[
            "supporting_signal_count"
        ]
        == 1
    )

    assert (
        result[
            "opposing_signal_count"
        ]
        == 1
    )

    assert (
        result["fusion_state"]
        == "CONFLICTING_EVIDENCE"
    )

    assert (
        result[
            "opposition_fraction"
        ]
        > 0.0
    )


# ============================================================
# Lookback behavior
# ============================================================

def test_old_signal_is_not_used(
    engine,
):
    # SBP is 120 minutes old.
    # Fusion lookback is 90 minutes.
    df = make_df([
        make_row(
            "sbp",
            0,
            4.0,
        ),
        make_row(
            "hr",
            120,
            3.0,
        ),
        make_row(
            "dbp",
            120,
            0.0,
        ),
    ])

    out = engine.transform(
        df
    )

    result = row_at(
        out,
        120,
    )

    contributing = set(
        json.loads(
            result[
                "contributing_signals"
            ]
        )
    )

    assert "sbp" not in contributing

    assert contributing == {
        "hr",
        "dbp",
    }

    assert (
        result[
            "available_signal_count"
        ]
        == 2
    )


def test_signal_within_lookback_is_used(
    engine,
):
    # SBP is 80 minutes old and therefore should still be
    # available inside the 90 minute lookback.
    df = make_df([
        make_row(
            "sbp",
            40,
            3.0,
        ),
        make_row(
            "hr",
            120,
            3.0,
        ),
        make_row(
            "dbp",
            120,
            0.0,
        ),
    ])

    out = engine.transform(
        df
    )

    result = row_at(
        out,
        120,
    )

    contributing = set(
        json.loads(
            result[
                "contributing_signals"
            ]
        )
    )

    assert "sbp" in contributing

    assert (
        result[
            "available_signal_count"
        ]
        == 3
    )

    assert (
        result[
            "supporting_signal_count"
        ]
        == 2
    )


def test_latest_signal_observation_is_used(
    engine,
):
    # Older SBP supports concern.
    df = make_df([
        make_row(
            "sbp",
            0,
            4.0,
        ),

        # Latest SBP is normal.
        make_row(
            "sbp",
            30,
            0.0,
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

    result = row_at(
        out,
        60,
    )

    # Latest SBP should replace the older elevated SBP,
    # leaving HR as the only supporting signal.
    assert (
        result[
            "available_signal_count"
        ]
        == 2
    )

    assert (
        result[
            "supporting_signal_count"
        ]
        == 1
    )

    assert (
        result["fusion_state"]
        == "SINGLE_SIGNAL_SUPPORT"
    )


# ============================================================
# Context isolation
# ============================================================

def test_contexts_do_not_mix(
    engine,
):
    df = make_df([
        make_row(
            "sbp",
            0,
            4.0,
            context_state="WAKE",
        ),
        make_row(
            "hr",
            30,
            3.0,
            context_state="SLEEP",
        ),
        make_row(
            "dbp",
            30,
            0.0,
            context_state="SLEEP",
        ),
    ])

    out = engine.transform(
        df
    )

    result = row_at(
        out,
        30,
    )

    contributing = set(
        json.loads(
            result[
                "contributing_signals"
            ]
        )
    )

    # WAKE SBP must not be combined with the SLEEP moment.
    assert "sbp" not in contributing

    assert contributing == {
        "hr",
        "dbp",
    }

    assert (
        result[
            "supporting_signal_count"
        ]
        == 1
    )


# ============================================================
# Temporal usability
# ============================================================

def test_temporal_unusable_signal_not_used(
    engine,
):
    df = make_df([
        make_row(
            "sbp",
            0,
            4.0,
            temporal_code="INSUFFICIENT",
        ),
        make_row(
            "hr",
            0,
            3.0,
        ),
        make_row(
            "dbp",
            0,
            0.0,
        ),
    ])

    out = engine.transform(
        df
    )

    result = out.iloc[0]

    contributing = set(
        json.loads(
            result[
                "contributing_signals"
            ]
        )
    )

    assert "sbp" not in contributing

    assert contributing == {
        "hr",
        "dbp",
    }

    assert (
        result[
            "available_signal_count"
        ]
        == 2
    )


# ============================================================
# Score bounds
# ============================================================

def test_confidence_between_zero_and_one(
    engine,
):
    df = make_df([
        make_row(
            "sbp",
            0,
            3.0,
            temporal_confidence=0.8,
        ),
        make_row(
            "hr",
            0,
            3.0,
            temporal_confidence=0.9,
        ),
    ])

    out = engine.transform(
        df
    )

    confidence = out.loc[
        0,
        "fusion_confidence",
    ]

    assert (
        0.0
        <= confidence
        <= 1.0
    )


def test_fusion_score_between_zero_and_one(
    engine,
):
    df = make_df([
        make_row(
            "sbp",
            0,
            10.0,
        ),
        make_row(
            "dbp",
            0,
            10.0,
        ),
        make_row(
            "hr",
            0,
            10.0,
        ),
    ])

    out = engine.transform(
        df
    )

    score = out.loc[
        0,
        "fusion_score",
    ]

    assert (
        0.0
        <= score
        <= 1.0
    )


# ============================================================
# Evidence strength
# ============================================================

def test_stronger_evidence_increases_score(
    engine,
):
    weak = make_df([
        make_row(
            "sbp",
            0,
            2.1,
        ),
        make_row(
            "dbp",
            0,
            2.1,
        ),
        make_row(
            "hr",
            0,
            2.1,
        ),
    ])

    strong = make_df([
        make_row(
            "sbp",
            0,
            4.0,
        ),
        make_row(
            "dbp",
            0,
            4.0,
        ),
        make_row(
            "hr",
            0,
            4.0,
        ),
    ])

    weak_out = engine.transform(
        weak
    )

    strong_out = engine.transform(
        strong
    )

    assert (
        strong_out.loc[
            0,
            "fusion_score",
        ]
        >
        weak_out.loc[
            0,
            "fusion_score",
        ]
    )


# ============================================================
# Leakage protection
# ============================================================

def test_future_rows_do_not_change_past(
    engine,
):
    past = make_df([
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
            3.0,
        ),
    ])

    with_future = make_df([
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
            3.0,
        ),

        # Extreme future observations.
        make_row(
            "sbp",
            60,
            -10.0,
        ),
        make_row(
            "dbp",
            60,
            -10.0,
        ),
        make_row(
            "hr",
            60,
            -10.0,
        ),
    ])

    past_out = engine.transform(
        past
    )

    full_out = engine.transform(
        with_future
    )

    past_result = row_at(
        past_out,
        0,
    )

    full_past_result = row_at(
        full_out,
        0,
    )

    cols = [
        "fusion_code",
        "fusion_state",
        "fusion_evidence_strength",
        "fusion_score",
        "fusion_confidence",
        "available_signal_count",
        "supporting_signal_count",
        "opposing_signal_count",
        "support_fraction",
        "direction_agreement",
    ]

    for col in cols:
        if isinstance(
            past_result[col],
            float,
        ):
            assert (
                full_past_result[col]
                == pytest.approx(
                    past_result[col]
                )
            )
        else:
            assert (
                full_past_result[col]
                == past_result[col]
            )


# ============================================================
# Explainability
# ============================================================

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
            "dbp",
            0,
            2.5,
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

    explanation = out.loc[
        0,
        "fusion_explanation",
    ]

    assert isinstance(
        explanation,
        str,
    )

    assert len(
        explanation
    ) > 0

    assert (
        "Fusion state:"
        in explanation
    )


# ============================================================
# Schema validation
# ============================================================

def test_missing_required_column_raises(
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

    df = df.drop(
        columns=[
            "risk_aligned_score"
        ]
    )

    with pytest.raises(
        ValueError,
        match="risk_aligned_score",
    ):
        engine.transform(
            df
        )


# ============================================================
# Opposition penalty
# ============================================================

def test_conflict_reduces_adjusted_fusion_score(
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
            -4.0,
        ),
    ])

    out = engine.transform(
        df
    )

    result = out.iloc[0]

    assert (
        result["fusion_state"]
        == "CONFLICTING_EVIDENCE"
    )

    assert (
        result[
            "opposition_fraction"
        ]
        > 0.0
    )

    assert (
        result[
            "fusion_score"
        ]
        <
        result[
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

    result = out.iloc[0]

    assert (
        result[
            "opposition_fraction"
        ]
        == pytest.approx(
            0.0
        )
    )

    assert (
        result[
            "fusion_score"
        ]
        == pytest.approx(
            result[
                "fusion_evidence_strength"
            ]
        )
    )