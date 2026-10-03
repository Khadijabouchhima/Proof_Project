from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biovance.risk import RiskEngine, load_risk_config


@pytest.fixture
def cfg():
    return load_risk_config()


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "male": [1, 0, 1, 0, 1, 0],
            "age": [45, 52, 61, 39, 67, 58],
            "currentSmoker": [1, 0, 1, 0, 0, 1],
            "cigsPerDay": [10, 0, 20, 0, 0, 15],
            "BPMeds": [0, 1, 1, 0, 1, 0],
            "prevalentStroke": [0, 0, 0, 0, 1, 0],
            "prevalentHyp": [0, 1, 1, 0, 1, 1],
            "diabetes": [0, 0, 1, 0, 1, 0],
            "totChol": [210, 240, 265, 180, 290, 225],
            "sysBP": [120, 138, 155, 112, 170, 145],
            "diaBP": [78, 88, 95, 72, 100, 90],
            "BMI": [24.5, 29.0, 31.2, 22.8, 33.5, 27.4],
            "heartRate": [70, 74, 80, 65, 85, 76],
            "glucose": [85, 92, 130, 82, 145, 100],
            "TenYearCHD": [0, 0, 1, 0, 1, 1],
        }
    )


def test_config_loads(cfg):
    assert cfg.target == "TenYearCHD"
    assert "age" in cfg.features
    assert "sysBP" in cfg.features
    assert "education" not in cfg.features


def test_predict_before_fit_raises(cfg, sample_df):
    engine = RiskEngine(cfg)

    with pytest.raises(RuntimeError, match="fitted"):
        engine.predict(sample_df)


def test_missing_required_feature_raises(cfg, sample_df):
    df = sample_df.drop(columns=["sysBP"])

    engine = RiskEngine(cfg)

    with pytest.raises(ValueError, match="sysBP"):
        engine.fit(df)


def test_missing_target_raises(cfg, sample_df):
    df = sample_df.drop(columns=["TenYearCHD"])

    engine = RiskEngine(cfg)

    with pytest.raises(ValueError, match="target"):
        engine.fit(df)


def test_nonbinary_target_raises(cfg, sample_df):
    df = sample_df.copy()
    df.loc[0, "TenYearCHD"] = 2

    engine = RiskEngine(cfg)

    with pytest.raises(ValueError, match="binary"):
        engine.fit(df)


def test_missing_predictor_values_are_imputed(cfg, sample_df):
    train = sample_df.copy()

    engine = RiskEngine(cfg).fit(train)

    test = sample_df.iloc[[0]].copy()
    test.loc[test.index[0], "glucose"] = np.nan
    test.loc[test.index[0], "BMI"] = np.nan

    result = engine.predict(test)

    assert len(result) == 1
    assert np.isfinite(result.iloc[0]["risk_probability"])


def test_probability_is_between_zero_and_one(cfg, sample_df):
    engine = RiskEngine(cfg).fit(sample_df)

    result = engine.predict(sample_df)

    assert (
        (result["risk_probability"] >= 0)
        & (result["risk_probability"] <= 1)
    ).all()


def test_log_odds_matches_probability(cfg, sample_df):
    engine = RiskEngine(cfg).fit(sample_df)

    result = engine.predict(sample_df)

    p = result["risk_probability"].to_numpy()
    expected = np.log(p / (1 - p))

    np.testing.assert_allclose(
        result["risk_log_odds"].to_numpy(),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_complete_input_has_completeness_one(cfg, sample_df):
    engine = RiskEngine(cfg).fit(sample_df)

    result = engine.predict(sample_df.iloc[[0]])

    assert result.iloc[0]["risk_input_completeness"] == pytest.approx(1.0)
    assert result.iloc[0]["risk_code"] == "OK"


def test_low_completeness_is_flagged(cfg, sample_df):
    engine = RiskEngine(cfg).fit(sample_df)

    test = sample_df.iloc[[0]].copy()

    # Leave only a few predictors populated.
    for col in cfg.features[4:]:
        test[col] = np.nan

    result = engine.predict(test)

    assert result.iloc[0]["risk_input_completeness"] < cfg.low_completeness_threshold
    assert result.iloc[0]["risk_code"] == "LOW_INPUT_COMPLETENESS"


def test_predictions_are_deterministic(cfg, sample_df):
    engine = RiskEngine(cfg).fit(sample_df)

    a = engine.predict(sample_df)
    b = engine.predict(sample_df)

    pd.testing.assert_frame_equal(a, b)


def test_output_preserves_row_count_and_index(cfg, sample_df):
    df = sample_df.copy()
    df.index = [10, 20, 30, 40, 50, 60]

    engine = RiskEngine(cfg).fit(df)

    result = engine.predict(df)

    assert len(result) == len(df)
    assert list(result.index) == list(df.index)

def test_save_load_roundtrip(
    cfg,
    sample_df,
    tmp_path,
):
    engine = RiskEngine(cfg).fit(sample_df)

    before = engine.predict(sample_df)

    path = (
        tmp_path
        / "risk_model.joblib"
    )

    engine.save(path)

    assert path.exists()

    loaded = RiskEngine.load(path)

    after = loaded.predict(sample_df)

    pd.testing.assert_frame_equal(
        before,
        after,
    )


def test_cannot_save_unfitted_engine(
    cfg,
    tmp_path,
):
    engine = RiskEngine(cfg)

    path = (
        tmp_path
        / "risk_model.joblib"
    )

    with pytest.raises(
        RuntimeError,
        match="before fitting",
    ):
        engine.save(path)