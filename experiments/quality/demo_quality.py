import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from biovance.quality import (
    DataQualityEngine,
    from_pmdata_daily,
    load_quality_config,
)


csv_path = (
    ROOT
    / "data"
    / "processed"
    / "pmdata_model_ready.csv"
)


print("Reading:")
print(csv_path)


raw = pd.read_csv(csv_path)


print("\nRaw shape:")
print(raw.shape)


quality_input = from_pmdata_daily(raw)


print("\nAdapted quality input:")
print(quality_input.head())


cfg = load_quality_config(
    "pmdata"
)


engine = DataQualityEngine(cfg)


assessed = engine.assess(
    quality_input
)


print("\n=== QUALITY OUTPUT ===")

print(
    assessed[
        [
            "patient_id",
            "timestamp",
            "signal",
            "value",
            "coverage",
            "sample_count",
            "quality_score",
            "quality_status",
            "quality_reasons",
        ]
    ].head(20)
)


print("\n=== STATUS COUNTS ===")

print(
    assessed[
        "quality_status"
    ].value_counts()
)


print("\n=== QUALITY SCORE SUMMARY ===")

print(
    assessed[
        "quality_score"
    ].describe()
)


print("\n=== WORST QUALITY OBSERVATIONS ===")

print(
    assessed
    .sort_values(
        "quality_score"
    )[
        [
            "patient_id",
            "timestamp",
            "value",
            "coverage",
            "sample_count",
            "valid",
            "quality_score",
            "quality_status",
            "quality_reasons",
        ]
    ]
    .head(20)
)


print("\n=== BEST QUALITY OBSERVATIONS ===")

print(
    assessed
    .sort_values(
        "quality_score",
        ascending=False,
    )[
        [
            "patient_id",
            "timestamp",
            "value",
            "coverage",
            "sample_count",
            "valid",
            "quality_score",
            "quality_status",
        ]
    ]
    .head(20)
)