from __future__ import annotations

import pandas as pd
import pytest

from biovance.decision import (
    DecisionEngine,
    load_decision_config,
)


@pytest.fixture
def cfg():
    return load_decision_config()


@pytest.fixture
def engine(cfg):
    return DecisionEngine(cfg)


def test_config_loads(cfg):
    assert cfg.version == "decision_v1"

    assert cfg.abstain_uncertainty == pytest.approx(
        0.60
    )

    assert cfg.strong_fusion_warn == pytest.approx(
        0.75
    )

    assert cfg.fusion_warn_with_risk == pytest.approx(
        0.60
    )


def test_insufficient_evidence_abstains(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.90],
            "fusion_state": [
                "MULTIMODAL_CONSENSUS"
            ],
            "risk_probability": [0.30],
            "uncertainty_score": [0.90],
            "uncertainty_code": [
                "INSUFFICIENT_EVIDENCE"
            ],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "ABSTAIN"

    assert out.loc[
        0,
        "decision_code",
    ] == "ABSTAIN_INSUFFICIENT_EVIDENCE"


def test_missing_fusion_abstains_even_with_high_risk(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [None],
            "fusion_state": [None],
            "risk_probability": [0.80],
            "uncertainty_score": [0.30],
            "uncertainty_code": [
                "PARTIAL_EVIDENCE"
            ],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "ABSTAIN"

    assert out.loc[
        0,
        "decision_code",
    ] == "ABSTAIN_FUSION_UNAVAILABLE"


def test_missing_uncertainty_abstains(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.90],
            "fusion_state": [
                "MULTIMODAL_CONSENSUS"
            ],
            "risk_probability": [0.30],
            "uncertainty_score": [None],
            "uncertainty_code": [None],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "ABSTAIN"

    assert (
        "UNCERTAINTY_UNAVAILABLE"
        in out.loc[
            0,
            "decision_reason_codes",
        ]
    )


def test_high_uncertainty_abstains(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.90],
            "fusion_state": [
                "MULTIMODAL_CONSENSUS"
            ],
            "risk_probability": [0.30],
            "uncertainty_score": [0.70],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "ABSTAIN"

    assert out.loc[
        0,
        "decision_code",
    ] == "ABSTAIN_HIGH_UNCERTAINTY"


def test_strong_multimodal_fusion_warns(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.85],
            "fusion_state": [
                "MULTIMODAL_CONSENSUS"
            ],
            "risk_probability": [0.10],
            "uncertainty_score": [0.20],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "WARN"

    assert out.loc[
        0,
        "decision_code",
    ] == "WARN_STRONG_FUSION"


def test_multimodal_fusion_plus_risk_warns(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.65],
            "fusion_state": [
                "MULTIMODAL_SUPPORT"
            ],
            "risk_probability": [0.30],
            "uncertainty_score": [0.20],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "WARN"

    assert out.loc[
        0,
        "decision_code",
    ] == "WARN_FUSION_WITH_RISK"

    assert (
        "BACKGROUND_RISK_CONTEXT"
        in out.loc[
            0,
            "decision_reason_codes",
        ]
    )


def test_high_risk_alone_does_not_warn(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.20],
            "fusion_state": [
                "NO_SUPPORT"
            ],
            "risk_probability": [0.80],
            "uncertainty_score": [0.20],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "MONITOR"


def test_single_signal_support_does_not_warn(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.90],
            "fusion_state": [
                "SINGLE_SIGNAL_SUPPORT"
            ],
            "risk_probability": [0.50],
            "uncertainty_score": [0.20],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "MONITOR"


def test_conflicting_evidence_does_not_warn(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.90],
            "fusion_state": [
                "CONFLICTING_EVIDENCE"
            ],
            "risk_probability": [0.50],
            "uncertainty_score": [0.30],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "MONITOR"

    assert (
        "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
        in out.loc[
            0,
            "decision_reason_codes",
        ]
    )


def test_uncertainty_above_warn_limit_blocks_warn(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.90],
            "fusion_state": [
                "MULTIMODAL_CONSENSUS"
            ],
            "risk_probability": [0.50],
            "uncertainty_score": [0.50],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "MONITOR"


def test_warn_threshold_boundary_is_inclusive(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.75],
            "fusion_state": [
                "MULTIMODAL_CONSENSUS"
            ],
            "risk_probability": [0.10],
            "uncertainty_score": [0.40],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "WARN"


def test_abstain_threshold_boundary_is_inclusive(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.90],
            "fusion_state": [
                "MULTIMODAL_CONSENSUS"
            ],
            "risk_probability": [0.50],
            "uncertainty_score": [0.60],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "ABSTAIN"


def test_scores_are_clipped(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [2.0],
            "fusion_state": [
                "MULTIMODAL_CONSENSUS"
            ],
            "risk_probability": [2.0],
            "uncertainty_score": [-1.0],
            "uncertainty_code": ["OK"],
        }
    )

    out = engine.decide(df)

    assert out.loc[
        0,
        "decision_state",
    ] == "WARN"


def test_preserves_index(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [
                0.20,
                0.85,
            ],
            "fusion_state": [
                "NO_SUPPORT",
                "MULTIMODAL_CONSENSUS",
            ],
            "risk_probability": [
                0.10,
                0.20,
            ],
            "uncertainty_score": [
                0.20,
                0.20,
            ],
            "uncertainty_code": [
                "OK",
                "OK",
            ],
        },
        index=[
            10,
            20,
        ],
    )

    out = engine.decide(df)

    assert list(
        out.index
    ) == [
        10,
        20,
    ]


def test_deterministic(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_score": [0.65],
            "fusion_state": [
                "MULTIMODAL_SUPPORT"
            ],
            "risk_probability": [0.30],
            "uncertainty_score": [0.20],
            "uncertainty_code": ["OK"],
        }
    )

    a = engine.decide(df)
    b = engine.decide(df)

    pd.testing.assert_frame_equal(
        a,
        b,
    )