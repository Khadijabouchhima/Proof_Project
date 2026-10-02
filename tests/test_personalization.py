import numpy as np
import pandas as pd
import pytest

from biovance.personalization import (PersonalizationEngine, aggregate_daily, load_personalization_config,
                                      normalize_observations)
from helpers import auroc, make_cohort


@pytest.fixture(scope="module")
def cohort():
    return make_cohort(n=60, n_days=100, seed=1)


def cfg(**kw):
    return load_personalization_config("simulator", **kw)


def fitted(df, **kw):
    return PersonalizationEngine(cfg(**kw)).fit(df)


# ------------------------------------------------------------------ schema flexibility
def test_aliases_and_context_labels_are_normalized(cohort):
    df, _ = cohort
    obs = normalize_observations(df, cfg())
    assert {"patient_id", "timestamp", "context", "hr", "hr__q", "hrv", "hrv__q"} <= set(obs.columns)
    assert set(obs["context"].dropna()) == {"rest", "active", "sleep"}


def test_other_column_names_work():
    df, _ = make_cohort(n=6, n_days=5, seed=2, names=("subject_id", "ts", "heart_rate", "rmssd", "quality", "state"))
    obs = normalize_observations(df, cfg())
    assert "hr" in obs.columns and "hrv" in obs.columns and len(obs) > 0


def test_missing_required_columns_gives_clear_error():
    with pytest.raises(ValueError, match="patient id"):
        normalize_observations(pd.DataFrame({"x": [1]}), cfg())
    with pytest.raises(ValueError, match="no known signal"):
        normalize_observations(pd.DataFrame({"patient_id": ["a"], "timestamp": ["2025-01-01"]}), cfg())


# ------------------------------------------------------------------ daily aggregation
def test_low_quality_values_are_excluded_from_daily_median():
    ts = pd.date_range("2025-01-01", periods=5, freq="h")
    df = pd.DataFrame({"patient_id": "a", "timestamp": ts, "HR": [60, 60, 60, 200, 60],
                       "signal_quality": [.9, .9, .9, .1, .9], "context": "resting"})
    obs = normalize_observations(df, cfg())
    d = aggregate_daily(obs, cfg(), [s for s in cfg().signals if s.name == "hr"])
    assert d.loc[0, "value"] == 60 and d.loc[0, "n_eligible"] == 4 and d.loc[0, "n_raw"] == 5


def test_min_obs_per_day_rule():
    ts = pd.date_range("2025-01-01", periods=2, freq="h")  # hr needs >= 3 eligible obs/day
    df = pd.DataFrame({"patient_id": "a", "timestamp": ts, "HR": [60, 61], "signal_quality": .9, "context": "rest"})
    d = aggregate_daily(normalize_observations(df, cfg()), cfg(), [s for s in cfg().signals if s.name == "hr"])
    assert np.isnan(d.loc[0, "value"])


def test_missing_days_stay_in_grid():
    ts = pd.to_datetime(["2025-01-01 08:00", "2025-01-04 08:00"])
    df = pd.DataFrame({"patient_id": "a", "timestamp": ts, "SBP": [118, 119]})
    d = aggregate_daily(normalize_observations(df, cfg()), cfg(), [s for s in cfg().signals if s.name == "sbp"])
    assert len(d) == 4 and d["value"].isna().sum() == 2


# ------------------------------------------------------------------ baseline
def test_baseline_is_median_mad_of_window_and_robust_to_outlier_days():
    n_days = 40
    ts = [pd.Timestamp("2025-01-01") + pd.Timedelta(days=d, hours=8) for d in range(n_days)]
    vals = np.full(n_days, 60.0) + np.tile([-1.0, 0.0, 1.0, 0.0], 10)[:n_days]
    vals[3] = 120.0  # illness day inside the baseline window
    df = pd.DataFrame({"patient_id": "a", "timestamp": ts, "SBP": vals})
    e = PersonalizationEngine(cfg(mode="center_scale", baseline_days=28, min_valid_days=14))
    e.prior = {"sbp": dict(mu_c=60, tau2=100, s_pop=1.0, pop_mean=60, pop_sd=5, n_patients=9)}
    b = e.baselines(df).iloc[0]
    w = vals[:28]
    assert b.center == pytest.approx(np.median(w))
    assert b.scale_raw == pytest.approx(1.4826 * np.median(np.abs(w - np.median(w))))
    assert abs(b.center - 60) < 1.0          # the 120 outlier did not drag the baseline


def test_status_learning_established_and_cold_start(cohort):
    df, _ = cohort
    e = fitted(df)
    sc = e.score(df)
    hr = sc[sc.signal == "hr"]
    assert (hr[hr.day < 28].status == "LEARNING").all() and hr[hr.day < 28].z.isna().all()
    assert (hr[hr.day >= 28].status == "ESTABLISHED").all()
    short = df[df.timestamp < pd.Timestamp("2025-01-01") + pd.Timedelta(days=8)]   # only 8 days of history
    sc2 = e.score(short)
    assert (sc2.status != "ESTABLISHED").all() and sc2.z.isna().all()


# ------------------------------------------------------------------ RQ1: the personalization effect
def test_personal_z_is_centered_per_person_but_population_z_is_not(cohort):
    df, truth = cohort
    non = df[df.patient_id.isin([p for p, d in truth.items() if not d])]
    spread = {}
    for mode in ("population", "center_scale"):
        sc = fitted(non, mode=mode).score(non)
        h = sc[(sc.signal == "hr") & (sc.day >= 28) & (sc.day < 45)]
        per_person_mean = h.groupby("patient_id")["z_raw"].mean()
        spread[mode] = per_person_mean.std()
    assert spread["center_scale"] < 1.0          # each person is ~N(0,1) against their own normal
    assert spread["population"] > 3 * spread["center_scale"]   # population z mixes in who the person is


def test_personal_z_separates_drifters_population_z_does_not(cohort):
    df, truth = cohort
    res = {}
    for mode in ("population", "shrunk"):
        sc = fitted(df, mode=mode).score(df)
        late = sc[(sc.signal == "hr") & (sc.day >= 85)].groupby("patient_id")["z"].mean()
        pos = late[[p for p, d in truth.items() if d]]
        neg = late[[p for p, d in truth.items() if not d]]
        res[mode] = auroc(pos, neg)
    assert res["shrunk"] > 0.90
    assert res["population"] < 0.75
    assert res["shrunk"] - res["population"] > 0.2


def test_orientation_hrv_falls_means_positive_z(cohort):
    df, truth = cohort
    sc = fitted(df).score(df)
    late = sc[(sc.signal == "hrv") & (sc.day >= 85)].groupby("patient_id")["z"].mean()
    drift = late[[p for p, d in truth.items() if d]].mean()
    assert drift > late[[p for p, d in truth.items() if not d]].mean() + 1.0


# ------------------------------------------------------------------ leakage
def test_no_future_leakage_scores_unchanged_when_future_is_altered(cohort):
    df, _ = cohort
    e = fitted(df)
    t_cut = pd.Timestamp("2025-01-01") + pd.Timedelta(days=60)
    full = e.score(df)
    mutated = df.copy()
    fut = mutated.timestamp >= t_cut
    mutated.loc[fut, "HR"] = mutated.loc[fut, "HR"] + 50
    mutated.loc[fut, "HRV"] = np.nan
    alt = e.score(mutated)
    a = full[full.day < 60].set_index(["patient_id", "signal", "day"])["z"]
    b = alt[alt.day < 60].set_index(["patient_id", "signal", "day"])["z"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_baseline_frozen_after_window(cohort):
    df, _ = cohort
    e = fitted(df)
    b1 = e.baselines(df)
    late = df.copy()
    m = late.timestamp >= pd.Timestamp("2025-01-01") + pd.Timedelta(days=28)
    late.loc[m, "HR"] += 30
    b2 = e.baselines(late)
    pd.testing.assert_frame_equal(b1.reset_index(drop=True), b2.reset_index(drop=True))


def test_prior_depends_only_on_fit_patients(cohort):
    df, _ = cohort
    train = df[df.patient_id < "p030"]
    other = df[df.patient_id >= "p030"]
    e = fitted(train)
    p_before = e.to_json()
    e.score(other)           # scoring unseen patients must not change the prior
    assert e.to_json() == p_before
    assert set(e.fitted_patients) == set(train.patient_id.unique())


# ------------------------------------------------------------------ shrinkage and context
def test_shrinkage_reduces_scale_dispersion_with_short_noisy_baselines():
    df, _ = make_cohort(n=60, n_days=40, seed=3, hr_day_sd=2.0)
    c = dict(baseline_days=10, min_valid_days=6)
    raw = fitted(df, mode="center_scale", **c).baselines(df)
    shr = fitted(df, mode="shrunk", shrinkage_n0=14, **c).baselines(df)
    r, s = raw[raw.signal == "hr"], shr[shr.signal == "hr"]
    assert np.std(np.log(s.scale)) < np.std(np.log(r.scale))
    assert ((s.w_scale > 0) & (s.w_scale < 1)).all() and ((s.w_center > 0) & (s.w_center <= 1)).all()


def test_rest_only_context_excludes_exercise_from_baseline(cohort):
    df, _ = cohort
    none = fitted(df, context_mode="none").baselines(df)
    rest = fitted(df, context_mode="rest_only").baselines(df)
    n = none[none.signal == "hr"].set_index("patient_id")["center_raw"]
    r = rest[rest.signal == "hr"].set_index("patient_id")["center_raw"]
    # active hours (+35 bpm, 3 of 24 h) do not move a median much, but rest-only is never above it
    assert (r <= n + 0.5).mean() > 0.9
    with pytest.raises(ValueError, match="context"):
        PersonalizationEngine(cfg(context_mode="rest_only")).fit(df.drop(columns="context"))


def test_all_modes_score_the_same_days(cohort):
    df, _ = cohort
    idx = {}
    for mode in ("population", "center", "center_scale", "shrunk"):
        sc = fitted(df, mode=mode).score(df)
        idx[mode] = set(map(tuple, sc[sc.z.notna()][["patient_id", "signal", "day"]].to_numpy()))
    assert idx["population"] == idx["center"] == idx["center_scale"] == idx["shrunk"]


def test_serialization_roundtrip_and_explain(cohort):
    df, _ = cohort
    e = fitted(df)
    e2 = PersonalizationEngine.from_json(e.to_json(), cfg())
    pd.testing.assert_frame_equal(e.score(df), e2.score(df))
    txt = e.explain(e.baselines(df), "p000", "hr", " bpm")
    assert "personal baseline" in txt and "valid days" in txt


def test_real_profile_uses_dataset_baseline_length():
    c = load_personalization_config("real:baigutanova_hrv")
    assert (c.baseline_days, c.min_valid_days) == (7, 5)
    with pytest.raises(ValueError, match="no personal baseline"):
        load_personalization_config("real:dryad_24h")


def test_simulator_adapter_stacks_contract_tables():
    from biovance.personalization import from_simulator_tables
    hourly = pd.DataFrame({"subject_id": ["s1"] * 3, "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
                           "hr": [60., 61., 62.], "activity_state": ["rest", "rest", "walk"],
                           "hr_available": True, "hr_quality": [.9, .9, .9]})
    daily = pd.DataFrame({"subject_id": ["s1"], "date": pd.to_datetime(["2025-01-01"]), "hrv": [40.], "hrv_quality": [.9]})
    bp = pd.DataFrame({"subject_id": ["s1"], "timestamp": pd.to_datetime(["2025-01-01 08:00"]),
                       "sbp": [118.], "dbp": [74.], "bp_quality": [.95]})
    wide = from_simulator_tables(hourly, daily, bp)
    obs = normalize_observations(wide, cfg())
    assert {"hr", "hrv", "sbp", "dbp"} <= set(obs.columns)
    assert obs.loc[obs.sbp.notna(), "sbp__q"].iloc[0] == 0.95   # per-signal quality columns are honored
