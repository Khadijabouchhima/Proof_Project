"""Tiny synthetic cohort for testing the engine (NOT the benchmark simulator)."""
import numpy as np
import pandas as pd


def make_cohort(n=60, n_days=100, seed=0, drifter_frac=0.5, drift_hr=6.0, onset=45, dur=45,
                hr_day_sd=2.0, junk=True, names=("patient_id", "timestamp", "HR", "HRV", "signal_quality", "context")):
    """Wide table: hourly HR (rest/active), one HRV per night, quality column. Drifters get HR up / HRV down."""
    rng = np.random.default_rng(seed)
    pid_n, ts_n, hr_n, hrv_n, q_n, c_n = names
    rows = []
    truth = {}
    t0 = pd.Timestamp("2025-01-01")
    for i in range(n):
        pid = f"p{i:03d}"
        drifter = i < int(n * drifter_frac)
        truth[pid] = drifter
        base_hr = float(np.clip(rng.normal(68, 9), 50, 90))
        base_hrv = float(np.exp(rng.normal(np.log(40), 0.3)))
        hr_day = rng.normal(0, hr_day_sd, n_days)
        for d in range(n_days):
            u = np.clip((d - onset) / dur, 0, 1) if drifter else 0.0
            for h in range(24):
                active = h in (8, 17, 18)
                hr = base_hr + hr_day[d] + 35 * active + drift_hr * u + rng.normal(0, 2.0)
                q = rng.normal(0.92, 0.04)
                if junk and rng.random() < 0.03:
                    hr, q = hr + rng.normal(40, 10), 0.2          # corrupted, flagged low quality
                rows.append((pid, t0 + pd.Timedelta(days=d, hours=h), hr, np.nan, q,
                             "active" if active else "resting"))
            hrv = base_hrv * (1 - 0.20 * u) * np.exp(rng.normal(0, 0.1))
            rows.append((pid, t0 + pd.Timedelta(days=d, hours=4), np.nan, hrv, 0.9, "asleep"))
    df = pd.DataFrame(rows, columns=[pid_n, ts_n, hr_n, hrv_n, q_n, c_n])
    return df, truth


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
