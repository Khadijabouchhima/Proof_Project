"""Personalization engine: personal baselines and oriented deviation scores.

What is statistics and what is learned
--------------------------------------
* Per-person baseline = robust median / MAD over the FROZEN first `baseline_days` days (no ML).
* The only learned part is a population PRIOR fitted on TRAIN patients: between-person variance of
  baseline centers (tau^2), typical within-person scale (s_pop), pooled population mean/SD.
* Shrinkage (empirical Bayes for the center, n0-weighted for the scale) pulls noisy personal estimates
  toward that prior. n0 is tuned on calibration patients, never on test patients.

Leakage safety: the baseline uses days [0, baseline_days) only; the score of day t uses only day t's
data and the frozen baseline. Days inside the window are never scored (LEARNING).

Ablation modes (RQ1): population (P0) -> center (P1) -> center_scale (P2) -> shrunk (P3).
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from biovance.personalization.config import PersonalizationConfig
from biovance.personalization.schema import normalize_observations, signals_present

MAD_TO_SD = 1.4826


# ----------------------------------------------------------------------------- daily aggregation
def aggregate_daily(obs: pd.DataFrame, cfg: PersonalizationConfig, signals) -> pd.DataFrame:
    """One row per patient x signal x calendar day (complete grid from the patient's first to last
    observed day). `value` = median of ELIGIBLE observations (quality ok, context ok), NaN if fewer
    than min_obs_per_day. Missing days stay in the grid so coverage can be measured downstream."""
    if cfg.context_mode == "rest_only" and obs["context"].isna().all():
        raise ValueError("context_mode='rest_only' needs a context column (rest/active/sleep); none found")
    d = obs.assign(date=obs["timestamp"].dt.floor("D"))
    day0 = d.groupby("patient_id")["date"].min()
    last = d.groupby("patient_id")["date"].max()
    pids, dates = [], []
    for p in day0.index:
        rng = pd.date_range(day0[p], last[p], freq="D")
        pids.extend([p] * len(rng))
        dates.extend(rng)
    grid = pd.MultiIndex.from_arrays([pids, dates], names=["patient_id", "date"])

    frames = []
    for s in signals:
        v, q = d[s.name], d[s.name + "__q"]
        has = v.notna()
        qok = (q >= cfg.min_quality) | (q.isna() & cfg.nan_quality_ok)
        cok = (d["context"].isin(s.baseline_contexts)
               if (cfg.context_mode == "rest_only" and s.baseline_contexts) else pd.Series(True, index=d.index))
        elig = has & qok & cok
        elig_df = d[elig].copy()
        source_count_col = s.name + "__n_source"
        if source_count_col in elig_df.columns:
            agg = (
                elig_df
                .groupby(["patient_id", "date"])
                .agg(
                    value=(s.name, "median"),
                    n_eligible=(source_count_col, "sum"),
                )
            )
        else:
            agg = (
                elig_df
                .groupby(["patient_id", "date"])[s.name]
                .agg(
                    value="median",
                    n_eligible="count",
                )
            )
        raw = (d[has].assign(_q=q[has]).groupby(["patient_id", "date"])
               .agg(n_raw=(s.name, "size"), mean_quality=("_q", "mean")))
        f = pd.DataFrame(index=grid).join(agg).join(raw)
        f["n_eligible"] = f["n_eligible"].fillna(0).astype(int)
        f["n_raw"] = f["n_raw"].fillna(0).astype(int)
        f.loc[f["n_eligible"] < s.min_obs_per_day, "value"] = np.nan
        f["signal"] = s.name
        frames.append(f.reset_index())
    out = pd.concat(frames, ignore_index=True)
    out["day"] = (out["date"] - out["patient_id"].map(day0)).dt.days
    return out


# ----------------------------------------------------------------------------- engine
class PersonalizationEngine:
    def __init__(self, cfg: PersonalizationConfig):
        cfg.validate()
        self.cfg = cfg
        self.prior: Dict[str, dict] = {}
        self.fitted_patients: tuple = ()

    # -- helpers
    def _daily(self, df: pd.DataFrame):
        obs = normalize_observations(df, self.cfg)
        sigs = signals_present(obs, self.cfg)
        return aggregate_daily(obs, self.cfg, sigs), sigs

    def _raw_person_stats(self, daily: pd.DataFrame) -> pd.DataFrame:
        w = daily[(daily["day"] < self.cfg.baseline_days) & daily["value"].notna()]
        g = w.groupby(["patient_id", "signal"])["value"]
        st = g.agg(n="count", center="median")
        mad = g.apply(lambda x: MAD_TO_SD * np.median(np.abs(x - np.median(x))))
        st["scale"] = mad
        return st.reset_index()

    # -- fit (TRAIN patients only)
    def fit(self, train_df: pd.DataFrame) -> "PersonalizationEngine":
        daily, _ = self._daily(train_df)
        stats = self._raw_person_stats(daily)
        stats = stats[stats["n"] >= self.cfg.min_valid_days]
        self.prior = {}
        for sig, g in stats.groupby("signal"):
            if len(g) < 3:
                raise ValueError(f"need >= 3 train patients with an established baseline for '{sig}', got {len(g)}")
            s_pop = float(np.sqrt(np.mean(g["scale"] ** 2)))
            s_i = np.maximum(g["scale"].to_numpy(), self.cfg.scale_floor_frac * s_pop)
            var_b = float(np.var(g["center"], ddof=1))
            tau2 = max(var_b - float(np.mean(s_i ** 2 / g["n"].to_numpy())), 0.1 * var_b)
            win = daily[(daily["signal"] == sig) & (daily["day"] < self.cfg.baseline_days)]["value"].dropna()
            self.prior[sig] = dict(mu_c=float(g["center"].mean()), tau2=tau2, s_pop=s_pop,
                                   pop_mean=float(win.mean()), pop_sd=float(win.std(ddof=1)),
                                   n_patients=int(len(g)))
        self.fitted_patients = tuple(sorted(train_df[self._pid_col(train_df)].astype(str).unique()))
        return self

    def _pid_col(self, df):
        cols = {c.lower(): c for c in df.columns}
        for a in self.cfg.column_aliases["patient_id"]:
            if a.lower() in cols:
                return cols[a.lower()]
        raise ValueError("patient id column not found")

    # -- baselines
    def _finalize(self, st: pd.DataFrame) -> pd.DataFrame:
        if not self.prior:
            raise RuntimeError("call fit() on train patients first")
        c = self.cfg
        rows = []
        for r in st.itertuples(index=False):
            pr = self.prior.get(r.signal)
            if pr is None:
                continue
            s_raw = max(r.scale, c.scale_floor_frac * pr["s_pop"])
            wc = ws = np.nan
            if c.mode == "population":
                center, scale = pr["pop_mean"], pr["pop_sd"]
            elif c.mode == "center":
                center, scale = r.center, pr["s_pop"]
            elif c.mode == "center_scale":
                center, scale = r.center, s_raw
            else:  # shrunk
                wc = pr["tau2"] / (pr["tau2"] + s_raw ** 2 / r.n)
                ws = r.n / (r.n + c.shrinkage_n0)
                center = wc * r.center + (1 - wc) * pr["mu_c"]
                scale = float(np.sqrt(ws * s_raw ** 2 + (1 - ws) * pr["s_pop"] ** 2))
            rows.append(dict(patient_id=r.patient_id, signal=r.signal, baseline_n=int(r.n),
                             center_raw=r.center, scale_raw=r.scale, center=center, scale=scale,
                             w_center=wc, w_scale=ws,
                             baseline_status="ESTABLISHED" if r.n >= c.min_valid_days else "INSUFFICIENT"))
        return pd.DataFrame(rows)

    def baselines(self, df: pd.DataFrame) -> pd.DataFrame:
        daily, _ = self._daily(df)
        return self._finalize(self._raw_person_stats(daily))

    # -- score
    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Long table: patient_id, signal, date, day, value, n_eligible, mean_quality, observed,
        center, scale, z_raw, z (oriented toward risk), status, baseline_n."""
        daily, sigs = self._daily(df)
        base = self._finalize(self._raw_person_stats(daily))
        sc = daily.merge(base, on=["patient_id", "signal"], how="left")
        orient = {s.name: s.orientation for s in sigs}
        sc["z_raw"] = (sc["value"] - sc["center"]) / sc["scale"]
        o = sc["signal"].map(orient)
        sc["z"] = np.where(o == 0, sc["z_raw"].abs(), o * sc["z_raw"])
        in_window = sc["day"] < self.cfg.baseline_days
        no_base = sc["baseline_status"].ne("ESTABLISHED")
        sc["status"] = np.where(in_window, "LEARNING", np.where(no_base, "INSUFFICIENT_BASELINE", "ESTABLISHED"))
        hide = (in_window & (not self.cfg.score_during_window)) | no_base
        sc.loc[hide, ["z", "z_raw"]] = np.nan
        sc["observed"] = sc["value"].notna()
        cols = ["patient_id", "signal", "date", "day", "value", "n_eligible", "mean_quality", "observed",
                "center", "scale", "z_raw", "z", "status", "baseline_n"]
        return sc[cols].sort_values(["patient_id", "signal", "day"]).reset_index(drop=True)

    @staticmethod
    def to_wide(scored: pd.DataFrame, col: str = "z") -> pd.DataFrame:
        return scored.pivot_table(index=["patient_id", "date", "day"], columns="signal", values=col, dropna=False)

    # -- interpretability: "relative to WHOSE baseline"
    def explain(self, baselines: pd.DataFrame, patient_id: str, signal: str, unit: str = "") -> str:
        r = baselines[(baselines.patient_id == patient_id) & (baselines.signal == signal)]
        if r.empty:
            return f"{signal}: no baseline for {patient_id}"
        r = r.iloc[0]
        ctx = "rest-only" if self.cfg.context_mode == "rest_only" else "all contexts"
        shr = "" if np.isnan(r.w_center) else f", weights center={r.w_center:.2f} scale={r.w_scale:.2f}"
        return (f"{signal}: personal baseline {r.center:.1f}{unit} (scale {r.scale:.1f}) from "
                f"{r.baseline_n} valid days, {ctx}, mode={self.cfg.mode}{shr}; status={r.baseline_status}")

    # -- persistence of the learned prior
    def to_json(self) -> str:
        return json.dumps({"mode": self.cfg.mode, "shrinkage_n0": self.cfg.shrinkage_n0, "prior": self.prior,
                           "fitted_patients": list(self.fitted_patients)}, indent=2)

    @classmethod
    def from_json(cls, s: str, cfg: PersonalizationConfig) -> "PersonalizationEngine":
        d = json.loads(s)
        e = cls(cfg)
        e.prior = d["prior"]
        e.fitted_patients = tuple(d["fitted_patients"])
        return e
