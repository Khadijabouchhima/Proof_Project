import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from biovance.personalization import (
    PersonalizationEngine,
    load_personalization_config,
    from_pmdata_daily,
)

# ---------------------------------------------------------
# 1. Load real PMData
# ---------------------------------------------------------
csv_path = (
    ROOT
    / "src"
    / "biovance"
    / "data"
    / "processed"
    / "pmdata_model_ready.csv"
)

print("Reading PMData from:")
print(csv_path)

if not csv_path.exists():
    raise FileNotFoundError(
        f"PMData file not found at: {csv_path}"
    )

raw = pd.read_csv(csv_path)

print("\nRaw shape:", raw.shape)

# ---------------------------------------------------------
# 2. Adapt PMData to personalization schema
# ---------------------------------------------------------
pmdata = from_pmdata_daily(raw)

print("\nAdapted shape:", pmdata.shape)
print("\nAdapted columns:")
print(pmdata.columns.tolist())

print("\nAdapted sample:")
print(pmdata.head())

# ---------------------------------------------------------
# 3. Load real PMData personalization config
# ---------------------------------------------------------
cfg = load_personalization_config(
    "real:pmdata",
    context_mode="rest_only",
)

print("\nConfig:")
print("baseline_days =", cfg.baseline_days)
print("min_valid_days =", cfg.min_valid_days)
print("context_mode =", cfg.context_mode)
print("mode =", cfg.mode)

# ---------------------------------------------------------
# 4. Split participants at patient level
# ---------------------------------------------------------
patients = sorted(pmdata["patient_id"].unique())

split = int(len(patients) * 0.7)

train_ids = patients[:split]
test_ids = patients[split:]

train_df = pmdata[
    pmdata["patient_id"].isin(train_ids)
].copy()

test_df = pmdata[
    pmdata["patient_id"].isin(test_ids)
].copy()

print("\nParticipants:")
print("Total:", len(patients))
print("Train:", len(train_ids))
print("Test:", len(test_ids))

print("\nTrain IDs:")
print(train_ids)

print("\nTest IDs:")
print(test_ids)

# ---------------------------------------------------------
# 5. Fit population prior on TRAIN participants only
# ---------------------------------------------------------
engine = PersonalizationEngine(cfg)

engine.fit(train_df)

print("\nPopulation prior fitted successfully.")

# ---------------------------------------------------------
# 6. Build baselines for test participants
# ---------------------------------------------------------
baselines = engine.baselines(test_df)

print("\n=== BASELINES ===")

if baselines.empty:
    print("No baselines were produced.")
else:
    print(
        baselines[
            [
                "patient_id",
                "signal",
                "baseline_n",
                "center",
                "scale",
                "baseline_status",
            ]
        ].head(20)
    )

# ---------------------------------------------------------
# 7. Score real test data
# ---------------------------------------------------------
scores = engine.score(test_df)

print("\n=== SCORE SAMPLE ===")

score_sample = scores.dropna(subset=["z"])

if score_sample.empty:
    print("No non-null z scores were produced.")
else:
    print(
        score_sample[
            [
                "patient_id",
                "signal",
                "date",
                "value",
                "center",
                "scale",
                "z",
                "status",
            ]
        ].head(20)
    )

# ---------------------------------------------------------
# 8. Sanity checks
# ---------------------------------------------------------
print("\n=== SANITY CHECKS ===")

print("\nTotal baselines:")
print(len(baselines))

if not baselines.empty:

    print("\nBaseline status counts:")
    print(
        baselines["baseline_status"].value_counts(
            dropna=False
        )
    )

    print("\nMissing center:")
    print(
        baselines["center"].isna().sum()
    )

    print("\nMissing scale:")
    print(
        baselines["scale"].isna().sum()
    )

    print("\nZero/non-positive scale:")
    print(
        (baselines["scale"] <= 0).sum()
    )

    print("\nBaseline center summary:")
    print(
        baselines["center"].describe()
    )

    print("\nBaseline scale summary:")
    print(
        baselines["scale"].describe()
    )

print("\nScore status counts:")
print(
    scores["status"].value_counts(
        dropna=False
    )
)

print("\nNumber of non-null z scores:")
print(
    scores["z"].notna().sum()
)

print("\nZ-score summary:")
print(
    scores["z"].describe()
)

# ---------------------------------------------------------
# 9. Inspect one test participant
# ---------------------------------------------------------
if test_ids:
    example_pid = test_ids[0]

    print(
        f"\n=== EXAMPLE PARTICIPANT: "
        f"{example_pid} ==="
    )

    print("\nBaseline:")
    print(
        baselines[
            baselines["patient_id"] == example_pid
        ]
    )

    print("\nFirst scored observations:")

    participant_scores = scores[
        (scores["patient_id"] == example_pid)
        & scores["z"].notna()
    ][
        [
            "date",
            "signal",
            "value",
            "center",
            "scale",
            "z",
            "status",
        ]
    ]

    print(participant_scores.head(20))