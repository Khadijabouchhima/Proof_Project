from __future__ import annotations

import pandas as pd
import pytest

from biovance.explainability import (
    ExplainabilityEngine,
    load_explainability_config,
)


# ============================================================
# Fixtures / helpers
# ============================================================

@pytest.fixture
def engine():
    config = load_explainability_config()
    return ExplainabilityEngine(
        config
    )


def make_row(
    *,
    decision_state: str = "MONITOR",
    decision_code: str = "MONITOR_NO_ESCALATION",
    decision_reason_codes: str = "NO_WARNING_CRITERIA_MET",
    fusion_state: str | None = "NO_SUPPORT",
    fusion_score: float | None = 0.20,
    fusion_confidence: float | None = 0.90,
    supporting_signals: str | None = "[]",
    opposing_signals: str | None = "[]",
    available_signal_count: int | None = 3,
    supporting_signal_count: int | None = 0,
    opposing_signal_count: int | None = 0,
    risk_probability: float | None = None,
    risk_input_completeness: float | None = None,
    uncertainty_score: float = 0.35,
    uncertainty_level: str = "MODERATE",
    uncertainty_code: str = "PARTIAL_EVIDENCE",
    uncertainty_reason_codes: str = "RISK_UNAVAILABLE",
) -> dict:

    return {
        "decision_state": decision_state,
        "decision_code": decision_code,
        "decision_reason_codes": decision_reason_codes,

        "fusion_state": fusion_state,
        "fusion_score": fusion_score,
        "fusion_confidence": fusion_confidence,
        "supporting_signals": supporting_signals,
        "opposing_signals": opposing_signals,
        "available_signal_count": available_signal_count,
        "supporting_signal_count": supporting_signal_count,
        "opposing_signal_count": opposing_signal_count,

        "risk_probability": risk_probability,
        "risk_input_completeness": risk_input_completeness,

        "uncertainty_score": uncertainty_score,
        "uncertainty_level": uncertainty_level,
        "uncertainty_code": uncertainty_code,
        "uncertainty_reason_codes": uncertainty_reason_codes,
    }


# ============================================================
# Contract
# ============================================================

def test_preserves_row_count(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(),
            make_row(),
            make_row(),
        ]
    )

    out = engine.explain(
        df
    )

    assert len(out) == len(df)


def test_does_not_return_new_decision(
    engine,
):
    df = pd.DataFrame(
        [
            make_row()
        ]
    )

    out = engine.explain(
        df
    )

    assert "decision_state" not in out.columns
    assert "decision_code" not in out.columns


def test_version_present(
    engine,
):
    df = pd.DataFrame(
        [
            make_row()
        ]
    )

    out = engine.explain(
        df
    )

    assert (
        out.loc[
            0,
            "explainability_version",
        ]
        == "explainability_v1"
    )


# ============================================================
# WARN
# ============================================================

def test_strong_fusion_warn_explained(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                decision_state="WARN",
                decision_code="WARN_STRONG_FUSION",
                decision_reason_codes=(
                    "STRONG_MULTIMODAL_FUSION"
                ),
                fusion_state="MULTIMODAL_CONSENSUS",
                fusion_score=0.88,
                supporting_signals='["sbp", "dbp", "hr"]',
                supporting_signal_count=3,
                risk_probability=None,
                uncertainty_score=0.36,
                uncertainty_level="MODERATE",
                uncertainty_reason_codes="RISK_UNAVAILABLE",
            )
        ]
    )

    out = engine.explain(
        df
    )

    summary = out.loc[
        0,
        "explanation_summary",
    ]

    assert "WARN" in summary
    assert "multimodal" in summary.lower()
    assert "heart rate" in summary.lower()


def test_warn_with_risk_mentions_background_context(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                decision_state="WARN",
                decision_code="WARN_FUSION_WITH_RISK",
                decision_reason_codes=(
                    "MULTIMODAL_FUSION|BACKGROUND_RISK_CONTEXT"
                ),
                fusion_state="MULTIMODAL_SUPPORT",
                fusion_score=0.68,
                supporting_signals='["sbp", "hr"]',
                supporting_signal_count=2,
                risk_probability=0.31,
                risk_input_completeness=1.0,
                uncertainty_score=0.25,
                uncertainty_level="LOW",
                uncertainty_reason_codes=(
                    "EVIDENCE_WELL_SUPPORTED"
                ),
            )
        ]
    )

    out = engine.explain(
        df
    )

    assert (
        "background cardiovascular risk"
        in out.loc[
            0,
            "explanation_summary",
        ].lower()
    )


# ============================================================
# MONITOR
# ============================================================

def test_single_signal_monitor_explained(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                fusion_state="SINGLE_SIGNAL_SUPPORT",
                supporting_signals='["hr"]',
                supporting_signal_count=1,
            )
        ]
    )

    out = engine.explain(
        df
    )

    assert (
        "single physiological signal"
        in out.loc[
            0,
            "explanation_summary",
        ].lower()
    )


def test_conflict_monitor_explained(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                fusion_state="CONFLICTING_EVIDENCE",
                fusion_score=0.30,
                supporting_signals='["sbp"]',
                opposing_signals='["hr"]',
                supporting_signal_count=1,
                opposing_signal_count=1,
                uncertainty_score=0.52,
                uncertainty_reason_codes=(
                    "RISK_UNAVAILABLE|"
                    "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
                ),
                decision_reason_codes=(
                    "NO_WARNING_CRITERIA_MET|"
                    "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
                ),
            )
        ]
    )

    out = engine.explain(
        df
    )

    assert (
        "conflicting"
        in out.loc[
            0,
            "explanation_summary",
        ].lower()
    )

    assert (
        "heart rate"
        in out.loc[
            0,
            "limiting_factors",
        ].lower()
    )


def test_consensus_blocked_by_uncertainty_explained(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                decision_state="MONITOR",
                decision_code="MONITOR_NO_ESCALATION",
                fusion_state="MULTIMODAL_CONSENSUS",
                fusion_score=0.82,
                supporting_signals='["sbp", "dbp", "hr"]',
                supporting_signal_count=3,
                uncertainty_score=0.403,
                uncertainty_level="MODERATE",
            )
        ]
    )

    out = engine.explain(
        df
    )

    assert (
        "uncertainty gate"
        in out.loc[
            0,
            "explanation_summary",
        ].lower()
    )


# ============================================================
# ABSTAIN
# ============================================================

def test_fusion_unavailable_abstain_explained(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                decision_state="ABSTAIN",
                decision_code="ABSTAIN_FUSION_UNAVAILABLE",
                decision_reason_codes="FUSION_UNAVAILABLE",
                fusion_state=None,
                fusion_score=None,
                fusion_confidence=None,
                supporting_signals=None,
                opposing_signals=None,
                available_signal_count=None,
                supporting_signal_count=None,
                opposing_signal_count=None,
                risk_probability=0.81,
                risk_input_completeness=1.0,
                uncertainty_score=0.65,
                uncertainty_level="HIGH",
                uncertainty_reason_codes="FUSION_UNAVAILABLE",
            )
        ]
    )

    out = engine.explain(
        df
    )

    summary = out.loc[
        0,
        "explanation_summary",
    ].lower()

    assert "abstained" in summary
    assert "fusion evidence was unavailable" in summary
    assert "risk alone cannot" in summary


def test_high_uncertainty_abstain_explained(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                decision_state="ABSTAIN",
                decision_code="ABSTAIN_HIGH_UNCERTAINTY",
                decision_reason_codes="HIGH_UNCERTAINTY",
                uncertainty_score=0.72,
                uncertainty_level="HIGH",
            )
        ]
    )

    out = engine.explain(
        df
    )

    assert (
        "uncertainty"
        in out.loc[
            0,
            "explanation_summary",
        ].lower()
    )


def test_insufficient_evidence_abstain_explained(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                decision_state="ABSTAIN",
                decision_code="ABSTAIN_INSUFFICIENT_EVIDENCE",
                decision_reason_codes="INSUFFICIENT_EVIDENCE",
                uncertainty_score=1.0,
                uncertainty_level="HIGH",
            )
        ]
    )

    out = engine.explain(
        df
    )

    assert (
        "insufficient"
        in out.loc[
            0,
            "explanation_summary",
        ].lower()
    )


# ============================================================
# Risk semantics
# ============================================================

def test_risk_described_as_ten_year_background_context(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                risk_probability=0.42,
                risk_input_completeness=1.0,
            )
        ]
    )

    out = engine.explain(
        df
    )

    text = out.loc[
        0,
        "risk_context",
    ].lower()

    assert "10-year chd risk" in text

    assert (
        "not a short-term deterioration probability"
        in text
    )


def test_missing_risk_not_invented(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                risk_probability=None,
                risk_input_completeness=None,
            )
        ]
    )

    out = engine.explain(
        df
    )

    assert (
        "risk was unavailable"
        in out.loc[
            0,
            "risk_context",
        ].lower()
    )


# ============================================================
# Input validation
# ============================================================

def test_missing_decision_state_raises(
    engine,
):
    df = pd.DataFrame(
        [
            make_row()
        ]
    ).drop(
        columns=[
            "decision_state"
        ]
    )

    with pytest.raises(
        ValueError,
        match="decision_state",
    ):
        engine.explain(
            df
        )


def test_invalid_decision_state_raises(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                decision_state="ALERT"
            )
        ]
    )

    with pytest.raises(
        ValueError,
        match="Unknown decision_state",
    ):
        engine.explain(
            df
        )


# ============================================================
# Determinism
# ============================================================

def test_explanations_are_deterministic(
    engine,
):
    df = pd.DataFrame(
        [
            make_row(
                fusion_state="MULTIMODAL_CONSENSUS",
                fusion_score=0.85,
                supporting_signals='["sbp", "dbp", "hr"]',
            )
        ]
    )

    a = engine.explain(
        df
    )

    b = engine.explain(
        df
    )

    pd.testing.assert_frame_equal(
        a,
        b,
    )