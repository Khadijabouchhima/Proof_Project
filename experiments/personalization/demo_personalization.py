"""Demo of the personalization engine on a tiny synthetic cohort (NOT the benchmark).
Prints (1) the 'two subjects' example, (2) the P0..P3 ablation on separating drifters from non-drifters."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "tests"))
from biovance.personalization import PersonalizationEngine, load_personalization_config
from tests.helpers import auroc, make_cohort

df, truth = make_cohort(n=100, n_days=100, seed=7)
drifters = [p for p, d in truth.items() if d]
non = [p for p, d in truth.items() if not d]
rows = []
for label, mode in [("P0 population", "population"), ("P1 personal center", "center"),
                    ("P2 center + scale", "center_scale"), ("P3 + shrinkage", "shrunk")]:
    cfg = load_personalization_config("simulator", mode=mode)
    eng = PersonalizationEngine(cfg).fit(df)
    sc = eng.score(df)
    hr = sc[sc.signal == "hr"]
    late = hr[hr.day >= 85].groupby("patient_id")["z"].mean()
    quiet = hr[(hr.day >= 28) & (hr.day < 45)].groupby("patient_id")["z_raw"].mean()
    rows.append((label, auroc(late[drifters], late[non]), quiet[non].std()))
print("\nAblation (HR only): can the mean z over days 85-99 tell drifters from non-drifters?")
print(f"{'mode':22s} {'AUROC':>7s}  {'between-person spread of mean z (no-drift period)':>50s}")
for l, a, s in rows:
    print(f"{l:22s} {a:7.3f}  {s:50.2f}")

eng = PersonalizationEngine(load_personalization_config("simulator", mode="shrunk")).fit(df)
b = eng.baselines(df)
hrb = b[b.signal == "hr"].set_index("patient_id")
lo, hi = hrb["center_raw"].idxmin(), hrb["center_raw"].idxmax()
sc = eng.score(df)
eng_p0 = PersonalizationEngine(load_personalization_config("simulator", mode="population")).fit(df)
sc0 = eng_p0.score(df)
print("\nTwo people, same day (day 60), resting HR daily median:")
for who in (lo, hi):
    r = sc[(sc.signal == "hr") & (sc.patient_id == who) & (sc.day == 60)].iloc[0]
    r0 = sc0[(sc0.signal == "hr") & (sc0.patient_id == who) & (sc0.day == 60)].iloc[0]
    print(f"  {who} (drifter={truth[who]}): HR={r.value:5.1f}  personal z={r.z:+5.2f}  population z={r0.z:+5.2f}")
    print("   ", eng.explain(b, who, "hr", " bpm"))
