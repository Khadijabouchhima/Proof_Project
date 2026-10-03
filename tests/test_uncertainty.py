from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biovance.uncertainty import (
    UncertaintyEngine,
    load_uncertainty_config,
)


@pytest.fixture
def cfg():
    return load_uncertainty_config()


@pytest.fixture
def engine(cfg):
    return UncertaintyEngine(cfg)


def test_config_loads(cfg):
    assert cfg.version == "uncertainty_v1"

    total = (
        cfg.fusion_confidence_weight
        + cfg.risk_completeness_weight
        + cfg.source_coverage_weight
        + cfg.coherence_weight
    )

    assert total == pytest.approx(1.0)


def test_full_high_quality_evidence_has_low_uncertainty(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [1.0],
            "direction_agreement": [1.0],
            "fusion_state": [
                "MULTIMODAL_CONSENSUS"
            ],
            "risk_input_completeness": [1.0],
        }
    )

    out = engine.score(df)

    assert out.loc[
        0,
        "certainty_score",
    ] == pytest.approx(1.0)

    assert out.loc[
        0,
        "uncertainty_score",
    ] == pytest.approx(0.0)

    assert (
        out.loc[
            0,
            "uncertainty_level",
        ]
        == "LOW"
    )

    assert (
        out.loc[
            0,
            "uncertainty_code",
        ]
        == "OK"
    )


def test_missing_fusion_creates_partial_evidence(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [None],
            "direction_agreement": [None],
            "fusion_state": [None],
            "risk_input_completeness": [1.0],
        }
    )

    out = engine.score(df)

    assert out.loc[
        0,
        "source_coverage",
    ] == pytest.approx(0.5)

    assert (
        out.loc[
            0,
            "uncertainty_code",
        ]
        == "PARTIAL_EVIDENCE"
    )

    assert "FUSION_UNAVAILABLE" in out.loc[
        0,
        "uncertainty_reason_codes",
    ]


def test_missing_risk_creates_partial_evidence(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [0.9],
            "direction_agreement": [1.0],
            "fusion_state": ["NO_SUPPORT"],
            "risk_input_completeness": [None],
        }
    )

    out = engine.score(df)

    assert out.loc[
        0,
        "source_coverage",
    ] == pytest.approx(0.5)

    assert (
        out.loc[
            0,
            "uncertainty_code",
        ]
        == "PARTIAL_EVIDENCE"
    )

    assert "RISK_UNAVAILABLE" in out.loc[
        0,
        "uncertainty_reason_codes",
    ]


def test_no_sources_is_insufficient_evidence(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [None],
            "direction_agreement": [None],
            "fusion_state": [None],
            "risk_input_completeness": [None],
        }
    )

    out = engine.score(df)

    assert out.loc[
        0,
        "source_coverage",
    ] == pytest.approx(0.0)

    assert (
        out.loc[
            0,
            "uncertainty_code",
        ]
        == "INSUFFICIENT_EVIDENCE"
    )

    assert out.loc[
        0,
        "uncertainty_score",
    ] == pytest.approx(1.0)


def test_lower_direction_agreement_increases_uncertainty(
    engine,
):
    agreed = pd.DataFrame(
        {
            "fusion_confidence": [0.9],
            "direction_agreement": [1.0],
            "fusion_state": ["NO_SUPPORT"],
            "risk_input_completeness": [1.0],
        }
    )

    disagreement = pd.DataFrame(
        {
            "fusion_confidence": [0.9],
            "direction_agreement": [0.0],
            "fusion_state": [
                "CONFLICTING_EVIDENCE"
            ],
            "risk_input_completeness": [1.0],
        }
    )

    agreed_out = engine.score(
        agreed
    )

    disagreement_out = engine.score(
        disagreement
    )

    assert (
        disagreement_out.loc[
            0,
            "uncertainty_score",
        ]
        >
        agreed_out.loc[
            0,
            "uncertainty_score",
        ]
    )


def test_conflicting_state_is_flagged(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [0.8],
            "direction_agreement": [0.3],
            "fusion_state": [
                "CONFLICTING_EVIDENCE"
            ],
            "risk_input_completeness": [1.0],
        }
    )

    out = engine.score(df)

    assert (
        "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
        in out.loc[
            0,
            "uncertainty_reason_codes",
        ]
    )


def test_low_fusion_confidence_is_flagged(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [0.2],
            "direction_agreement": [1.0],
            "fusion_state": ["NO_SUPPORT"],
            "risk_input_completeness": [1.0],
        }
    )

    out = engine.score(df)

    assert "LOW_FUSION_CONFIDENCE" in out.loc[
        0,
        "uncertainty_reason_codes",
    ]


def test_low_risk_completeness_is_flagged(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [0.9],
            "direction_agreement": [1.0],
            "fusion_state": ["NO_SUPPORT"],
            "risk_input_completeness": [0.5],
        }
    )

    out = engine.score(df)

    assert (
        "LOW_RISK_INPUT_COMPLETENESS"
        in out.loc[
            0,
            "uncertainty_reason_codes",
        ]
    )


def test_scores_are_bounded(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [
                2.0,
                -1.0,
            ],
            "direction_agreement": [
                2.0,
                -1.0,
            ],
            "fusion_state": [
                "NO_SUPPORT",
                "CONFLICTING_EVIDENCE",
            ],
            "risk_input_completeness": [
                3.0,
                -2.0,
            ],
        }
    )

    out = engine.score(df)

    assert (
        (
            out[
                "certainty_score"
            ]
            >= 0.0
        )
        & (
            out[
                "certainty_score"
            ]
            <= 1.0
        )
    ).all()

    assert (
        (
            out[
                "uncertainty_score"
            ]
            >= 0.0
        )
        & (
            out[
                "uncertainty_score"
            ]
            <= 1.0
        )
    ).all()


def test_certainty_and_uncertainty_sum_to_one(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [
                0.8,
                0.4,
            ],
            "direction_agreement": [
                0.8,
                0.5,
            ],
            "fusion_state": [
                "NO_SUPPORT",
                "CONFLICTING_EVIDENCE",
            ],
            "risk_input_completeness": [
                0.9,
                0.6,
            ],
        }
    )

    out = engine.score(df)

    summed = (
        out[
            "certainty_score"
        ]
        + out[
            "uncertainty_score"
        ]
    )

    np.testing.assert_allclose(
        summed.to_numpy(),
        1.0,
        rtol=1e-9,
        atol=1e-9,
    )


def test_threshold_boundary_030_is_moderate(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [0.25],
            "direction_agreement": [1.0],
            "fusion_state": ["NO_SUPPORT"],
            "risk_input_completeness": [1.0],
        }
    )

    out = engine.score(df)

    assert out.loc[
        0,
        "uncertainty_score",
    ] == pytest.approx(0.30)

    assert (
        out.loc[
            0,
            "uncertainty_level",
        ]
        == "MODERATE"
    )


def test_preserves_index(
    engine,
):
    df = pd.DataFrame(
        {
            "fusion_confidence": [
                0.8,
                0.9,
            ],
            "direction_agreement": [
                1.0,
                0.9,
            ],
            "fusion_state": [
                "NO_SUPPORT",
                "SINGLE_SIGNAL_SUPPORT",
            ],
            "risk_input_completeness": [
                1.0,
                0.9,
            ],
        },
        index=[
            10,
            20,
        ],
    )

    out = engine.score(df)

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
            "fusion_confidence": [0.8],
            "direction_agreement": [0.9],
            "fusion_state": [
                "SINGLE_SIGNAL_SUPPORT"
            ],
            "risk_input_completeness": [0.9],
        }
    )

    a = engine.score(df)
    b = engine.score(df)

    pd.testing.assert_frame_equal(
        a,
        b,
    )