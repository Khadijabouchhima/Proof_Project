"""Diagnostics for the BioVance Deviation Engine on PMData.

Purpose
-------
Investigate whether:

1. MISSING_VALUE rows are the same observations that the Quality Engine
   labels POOR.
2. The original PMData row actually contained resting HR on those days.
3. Deviation distributions change over chronological time.
4. High-deviation participants show isolated spikes or sustained runs.

This script is diagnostic only.

It does NOT:
    - change Deviation Engine thresholds
    - calculate warnings
    - implement the Temporal Engine
    - modify any source data
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


# ------------------------------------------------------------
# BioVance imports
# ------------------------------------------------------------

from biovance.personalization import (
    PersonalizationEngine,
    load_personalization_config,
    from_pmdata_daily as personalization_from_pmdata_daily,
)

from biovance.quality import (
    DataQualityEngine,
    load_quality_config,
    from_pmdata_daily as quality_from_pmdata_daily,
)

from biovance.deviation import (
    DeviationEngine,
    load_deviation_config,
)


# ============================================================
# Paths
# ============================================================

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "pmdata_model_ready.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "deviation"
    / "diagnostics"
)

ROW_DIAGNOSTIC_PATH = (
    OUTPUT_DIR
    / "pmdata_deviation_row_diagnostics.csv"
)

PARTICIPANT_DIAGNOSTIC_PATH = (
    OUTPUT_DIR
    / "pmdata_deviation_participant_diagnostics.csv"
)

RUN_DIAGNOSTIC_PATH = (
    OUTPUT_DIR
    / "pmdata_deviation_high_runs.csv"
)

TIME_DIAGNOSTIC_PATH = (
    OUTPUT_DIR
    / "pmdata_deviation_time_segments.csv"
)


# ============================================================
# Experiment configuration
# ============================================================

SIGNAL = "hr"

TRAIN_FRACTION = 0.70

# A "high deviation" run here is purely diagnostic.
#
# This is NOT yet Temporal Engine logic.
HIGH_DEVIATION_THRESHOLD = 2.0

# Minimum consecutive eligible high-deviation days to print prominently.
RUN_REVIEW_LENGTH = 3

# Number of chronological segments used to inspect distribution drift.
N_TIME_SEGMENTS = 4


# ============================================================
# Helpers
# ============================================================

def participant_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
) -> tuple[list[str], list[str]]:
    """Deterministic participant-level split."""

    participants = sorted(
        df["patient_id"]
        .astype(str)
        .unique()
    )

    if len(participants) < 4:
        raise ValueError(
            "Need at least 4 participants for this diagnostic."
        )

    n_train = int(
        len(participants)
        * train_fraction
    )

    n_train = max(
        3,
        n_train,
    )

    n_train = min(
        n_train,
        len(participants) - 1,
    )

    return (
        participants[:n_train],
        participants[n_train:],
    )


def run_quality(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Run the PMData Quality Engine."""

    quality_input = (
        quality_from_pmdata_daily(
            raw
        )
    )

    cfg = (
        load_quality_config(
            "pmdata"
        )
    )

    engine = DataQualityEngine(
        cfg
    )

    assessed = engine.assess(
        quality_input
    )

    assessed["timestamp"] = pd.to_datetime(
        assessed["timestamp"],
        errors="coerce",
        utc=True,
    )

    assessed["date"] = (
        assessed["timestamp"]
        .dt.normalize()
    )

    if "signal" in assessed.columns:

        assessed = assessed[
            assessed["signal"]
            .astype(str)
            .str.lower()
            == SIGNAL
        ].copy()

    return assessed


def run_personalization(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and score PMData personalization."""

    personalization_input = (
        personalization_from_pmdata_daily(
            raw
        )
    )

    cfg = load_personalization_config(
        "real:pmdata"
    )

    (
        train_ids,
        test_ids,
    ) = participant_split(
        personalization_input,
        TRAIN_FRACTION,
    )

    print("\n=== PERSONALIZATION SPLIT ===")
    print("Train:", train_ids)
    print("Holdout:", test_ids)

    train = personalization_input[
        personalization_input["patient_id"]
        .astype(str)
        .isin(train_ids)
    ].copy()

    engine = PersonalizationEngine(
        cfg
    )

    engine.fit(
        train
    )

    scored = engine.score(
        personalization_input
    )

    baselines = engine.baselines(
        personalization_input
    )

    scored = scored[
        scored["signal"]
        == SIGNAL
    ].copy()

    baselines = baselines[
        baselines["signal"]
        == SIGNAL
    ].copy()

    return (
        scored,
        baselines,
    )


def build_deviation_input(
    personal_scores: pd.DataFrame,
    baselines: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    """Build Deviation Engine input."""

    scores = personal_scores.copy()

    scores["timestamp"] = pd.to_datetime(
        scores["date"],
        errors="coerce",
        utc=True,
    )

    meta_cols = [
        "patient_id",
        "signal",
        "w_center",
        "w_scale",
    ]

    available = [
        c
        for c in meta_cols
        if c in baselines.columns
    ]

    if "w_center" in available:

        scores = scores.merge(
            baselines[available],
            on=[
                "patient_id",
                "signal",
            ],
            how="left",
        )

    else:

        scores["w_center"] = np.nan
        scores["w_scale"] = np.nan

    quality_daily = (
        quality
        .groupby(
            [
                "patient_id",
                "date",
            ],
            as_index=False,
        )
        .agg(
            quality_score=(
                "quality_score",
                "mean",
            ),

            quality_status=(
                "quality_status",
                lambda x: (
                    "POOR"
                    if (
                        x.astype(str)
                        .str.upper()
                        == "POOR"
                    ).any()
                    else (
                        "FAIR"
                        if (
                            x.astype(str)
                            .str.upper()
                            == "FAIR"
                        ).any()
                        else "GOOD"
                    )
                ),
            ),
        )
    )

    scores = scores.merge(
        quality_daily,
        left_on=[
            "patient_id",
            "timestamp",
        ],
        right_on=[
            "patient_id",
            "date",
        ],
        how="left",
        suffixes=(
            "",
            "_quality",
        ),
    )

    scores["quality_status"] = (
        scores["quality_status"]
        .fillna("UNKNOWN")
    )

    scores["baseline_status"] = (
        scores["status"]
    )

    scores["baseline_personal_weight"] = (
        scores["w_center"]
    )

    scores["baseline_population_weight"] = (
        1.0
        - scores["w_center"]
    )

    # PMData hr_rest_median already represents resting context.
    scores["context_state"] = "REST"
    scores["context_confidence"] = 1.0

    return pd.DataFrame({

        "patient_id":
            scores["patient_id"],

        "timestamp":
            scores["timestamp"],

        "signal":
            scores["signal"],

        "value":
            scores["value"],

        "baseline_center":
            scores["center"],

        "baseline_scale":
            scores["scale"],

        "baseline_n":
            scores["baseline_n"],

        "baseline_status":
            scores["baseline_status"],

        "baseline_personal_weight":
            scores[
                "baseline_personal_weight"
            ],

        "baseline_population_weight":
            scores[
                "baseline_population_weight"
            ],

        "quality_score":
            scores["quality_score"],

        "quality_status":
            scores["quality_status"],

        "context_state":
            scores["context_state"],

        "context_confidence":
            scores["context_confidence"],
    })


def original_pmdata_lookup(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Build original-PMData lookup for HR diagnostics."""

    required = [
        "participant",
        "date",
    ]

    missing = [
        c
        for c in required
        if c not in raw.columns
    ]

    if missing:
        raise ValueError(
            "PMData missing required columns: "
            f"{missing}"
        )

    out = pd.DataFrame({

        "patient_id":
            raw[
                "participant"
            ].astype(str),

        "timestamp":
            pd.to_datetime(
                raw["date"],
                errors="coerce",
                utc=True,
            ).dt.normalize(),
    })

    # Original HR fields where available.
    candidate_columns = [
        "hr_rest_median",
        "hr_rest_n",
        "hr_coverage",
        "hr_valid_day",
        "hr_conf_mean",
        "hr_mean",
        "hr_median",
        "rhr",
    ]

    for column in candidate_columns:

        if column in raw.columns:
            out[
                f"raw_{column}"
            ] = raw[column]

    return out


def longest_true_run(
    values: pd.Series,
) -> int:
    """Return longest consecutive True run."""

    longest = 0
    current = 0

    for value in values.fillna(False):

        if bool(value):
            current += 1
            longest = max(
                longest,
                current,
            )

        else:
            current = 0

    return longest


def identify_high_runs(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Identify consecutive high-deviation runs.

    This is descriptive only and is NOT the Temporal Engine.
    """

    rows = []

    for patient_id, group in scored.groupby(
        "patient_id"
    ):

        group = (
            group
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        # Only usable deviation observations.
        group["high"] = (
            (
                group[
                    "deviation_code"
                ]
                == "OK"
            )
            &
            (
                group[
                    "deviation_magnitude"
                ]
                >= HIGH_DEVIATION_THRESHOLD
            )
        )

        run_start = None
        run_rows = []

        for _, row in group.iterrows():

            if bool(
                row["high"]
            ):

                if run_start is None:
                    run_start = row[
                        "timestamp"
                    ]

                run_rows.append(
                    row
                )

            else:

                if run_rows:

                    rows.append(
                        summarize_run(
                            patient_id,
                            run_start,
                            run_rows,
                        )
                    )

                run_start = None
                run_rows = []

        if run_rows:

            rows.append(
                summarize_run(
                    patient_id,
                    run_start,
                    run_rows,
                )
            )

    return pd.DataFrame(
        rows
    )


def summarize_run(
    patient_id,
    run_start,
    run_rows,
) -> dict:
    """Summarize one consecutive high-deviation run."""

    frame = pd.DataFrame(
        run_rows
    )

    return {

        "patient_id":
            patient_id,

        "run_start":
            run_start,

        "run_end":
            frame[
                "timestamp"
            ].iloc[-1],

        "run_length":
            len(frame),

        "mean_deviation_score":
            frame[
                "deviation_score"
            ].mean(),

        "mean_magnitude":
            frame[
                "deviation_magnitude"
            ].mean(),

        "max_magnitude":
            frame[
                "deviation_magnitude"
            ].max(),

        "all_toward_risk":
            frame[
                "toward_risk"
            ]
            .fillna(False)
            .astype(bool)
            .all(),

        "toward_risk_fraction":
            frame[
                "toward_risk"
            ]
            .fillna(False)
            .astype(bool)
            .mean(),
    }


def time_segment_summary(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Split each participant chronologically into equal segments."""

    rows = []

    for patient_id, group in scored.groupby(
        "patient_id"
    ):

        group = (
            group
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        group["segment"] = pd.qcut(
            np.arange(
                len(group)
            ),
            q=min(
                N_TIME_SEGMENTS,
                len(group),
            ),
            labels=False,
            duplicates="drop",
        )

        for segment, part in group.groupby(
            "segment"
        ):

            usable = part[
                part[
                    "deviation_code"
                ]
                == "OK"
            ]

            rows.append({

                "patient_id":
                    patient_id,

                "segment":
                    int(segment),

                "start":
                    part[
                        "timestamp"
                    ].min(),

                "end":
                    part[
                        "timestamp"
                    ].max(),

                "n_rows":
                    len(part),

                "n_usable":
                    len(usable),

                "usable_fraction":
                    (
                        len(usable)
                        / len(part)
                        if len(part)
                        else np.nan
                    ),

                "mean_deviation_score":
                    usable[
                        "deviation_score"
                    ].mean(),

                "median_deviation_score":
                    usable[
                        "deviation_score"
                    ].median(),

                "mean_magnitude":
                    usable[
                        "deviation_magnitude"
                    ].mean(),

                "large_or_extreme_fraction":
                    (
                        usable[
                            "deviation_status"
                        ]
                        .isin(
                            [
                                "LARGE",
                                "EXTREME",
                            ]
                        )
                        .mean()
                        if len(usable)
                        else np.nan
                    ),

                "elevated_or_higher_fraction":
                    (
                        usable[
                            "deviation_magnitude"
                        ]
                        .ge(
                            HIGH_DEVIATION_THRESHOLD
                        )
                        .mean()
                        if len(usable)
                        else np.nan
                    ),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    print("Reading:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"PMData not found: {DATA_PATH}"
        )

    raw = pd.read_csv(
        DATA_PATH
    )

    print("\nRaw shape:")
    print(raw.shape)

    # ========================================================
    # Run engines
    # ========================================================

    quality = run_quality(
        raw
    )

    (
        personal_scores,
        baselines,
    ) = run_personalization(
        raw
    )

    deviation_input = (
        build_deviation_input(
            personal_scores,
            baselines,
            quality,
        )
    )

    deviation_engine = (
        DeviationEngine(
            load_deviation_config()
        )
    )

    scored = deviation_engine.score(
        deviation_input
    )

    raw_lookup = (
        original_pmdata_lookup(
            raw
        )
    )

    # ========================================================
    # Merge original PMData evidence
    # ========================================================

    diagnostic = scored.merge(
        raw_lookup,
        on=[
            "patient_id",
            "timestamp",
        ],
        how="left",
    )

    # ========================================================
    # 1. Missing value vs quality
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "1. MISSING VALUE VS QUALITY"
    )

    print(
        "=" * 72
    )

    crosstab = pd.crosstab(
        diagnostic[
            "deviation_code"
        ],
        diagnostic[
            "quality_status"
        ],
        margins=True,
    )

    print(
        crosstab
    )

    missing_rows = diagnostic[
        diagnostic[
            "deviation_code"
        ]
        == "MISSING_VALUE"
    ].copy()

    print(
        "\nMissing-value rows:"
    )

    print(
        len(
            missing_rows
        )
    )

    print(
        "\nQuality status among MISSING_VALUE:"
    )

    print(
        missing_rows[
            "quality_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    # ========================================================
    # 2. Was original resting HR present?
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "2. ORIGINAL PMDATA VALUE CHECK"
    )

    print(
        "=" * 72
    )

    if (
        "raw_hr_rest_median"
        in diagnostic.columns
    ):

        missing_rows[
            "raw_hr_rest_present"
        ] = (
            missing_rows[
                "raw_hr_rest_median"
            ].notna()
        )

        print(
            "\nAmong deviation MISSING_VALUE rows:"
        )

        print(
            missing_rows[
                "raw_hr_rest_present"
            ]
            .value_counts(
                dropna=False
            )
        )

        present_but_missing = missing_rows[
            missing_rows[
                "raw_hr_rest_present"
            ]
        ]

        print(
            "\nRows where original hr_rest_median existed "
            "but Deviation received MISSING_VALUE:"
        )

        print(
            len(
                present_but_missing
            )
        )

        if len(
            present_but_missing
        ):

            columns = [
                "patient_id",
                "timestamp",
                "raw_hr_rest_median",
                "raw_hr_rest_n",
                "raw_hr_coverage",
                "raw_hr_valid_day",
                "quality_score",
                "quality_status",
                "deviation_code",
            ]

            columns = [
                c
                for c in columns
                if c
                in present_but_missing.columns
            ]

            print(
                present_but_missing[
                    columns
                ]
                .head(30)
                .to_string(
                    index=False
                )
            )

    else:

        print(
            "raw_hr_rest_median not present in PMData."
        )

    # ========================================================
    # 3. Participant diagnostics
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "3. PARTICIPANT DEVIATION DIAGNOSTICS"
    )

    print(
        "=" * 72
    )

    participant_rows = []

    for patient_id, group in diagnostic.groupby(
        "patient_id"
    ):

        group = (
            group
            .sort_values(
                "timestamp"
            )
        )

        usable = group[
            group[
                "deviation_code"
            ]
            == "OK"
        ]

        high = (
            usable[
                "deviation_magnitude"
            ]
            >= HIGH_DEVIATION_THRESHOLD
        )

        participant_rows.append({

            "patient_id":
                patient_id,

            "n_total":
                len(group),

            "n_usable":
                len(usable),

            "usable_fraction":
                (
                    len(usable)
                    / len(group)
                    if len(group)
                    else np.nan
                ),

            "mean_deviation":
                usable[
                    "deviation_score"
                ].mean(),

            "median_deviation":
                usable[
                    "deviation_score"
                ].median(),

            "mean_magnitude":
                usable[
                    "deviation_magnitude"
                ].mean(),

            "max_magnitude":
                usable[
                    "deviation_magnitude"
                ].max(),

            "high_fraction":
                (
                    high.mean()
                    if len(usable)
                    else np.nan
                ),

            "large_or_extreme_fraction":
                (
                    usable[
                        "deviation_status"
                    ]
                    .isin(
                        [
                            "LARGE",
                            "EXTREME",
                        ]
                    )
                    .mean()
                    if len(usable)
                    else np.nan
                ),

            "toward_risk_fraction":
                (
                    usable[
                        "toward_risk"
                    ]
                    .fillna(False)
                    .astype(bool)
                    .mean()
                    if len(usable)
                    else np.nan
                ),

            "longest_high_run":
                longest_true_run(
                    (
                        group[
                            "deviation_code"
                        ]
                        == "OK"
                    )
                    &
                    (
                        group[
                            "deviation_magnitude"
                        ]
                        >= HIGH_DEVIATION_THRESHOLD
                    )
                ),
        })

    participant_summary = pd.DataFrame(
        participant_rows
    )

    print(
        participant_summary
        .sort_values(
            "high_fraction",
            ascending=False,
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 4. Chronological segments
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "4. CHRONOLOGICAL DEVIATION SEGMENTS"
    )

    print(
        "=" * 72
    )

    time_segments = (
        time_segment_summary(
            diagnostic
        )
    )

    print(
        time_segments
        .sort_values(
            [
                "patient_id",
                "segment",
            ]
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 5. High-deviation runs
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "5. HIGH-DEVIATION RUNS"
    )

    print(
        "=" * 72
    )

    runs = identify_high_runs(
        diagnostic
    )

    if runs.empty:

        print(
            "No high-deviation runs found."
        )

    else:

        print(
            "\nLongest runs:"
        )

        print(
            runs
            .sort_values(
                [
                    "run_length",
                    "max_magnitude",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(30)
            .round(4)
            .to_string(
                index=False
            )
        )

        sustained = runs[
            runs[
                "run_length"
            ]
            >= RUN_REVIEW_LENGTH
        ]

        print(
            f"\nRuns with length >= "
            f"{RUN_REVIEW_LENGTH}:"
        )

        print(
            len(
                sustained
            )
        )

        if len(
            sustained
        ):

            print(
                sustained
                .sort_values(
                    [
                        "run_length",
                        "max_magnitude",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
                .round(4)
                .to_string(
                    index=False
                )
            )

    # ========================================================
    # 6. Focus on notable participants
    # ========================================================

    notable_ids = (
        participant_summary
        .sort_values(
            "high_fraction",
            ascending=False,
        )
        .head(5)[
            "patient_id"
        ]
        .tolist()
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "6. TOP HIGH-DEVIATION PARTICIPANTS"
    )

    print(
        "=" * 72
    )

    print(
        notable_ids
    )

    for patient_id in notable_ids:

        subset = diagnostic[
            (
                diagnostic[
                    "patient_id"
                ]
                == patient_id
            )
            &
            (
                diagnostic[
                    "deviation_code"
                ]
                == "OK"
            )
        ].copy()

        subset = subset.sort_values(
            "timestamp"
        )

        print(
            f"\n--- {patient_id} ---"
        )

        display = [
            "timestamp",
            "value",
            "baseline_center",
            "baseline_scale",
            "quality_status",
            "deviation_score",
            "deviation_magnitude",
            "deviation_status",
            "toward_risk",
        ]

        print(
            subset[
                display
            ]
            .tail(40)
            .round(3)
            .to_string(
                index=False
            )
        )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic.to_csv(
        ROW_DIAGNOSTIC_PATH,
        index=False,
    )

    participant_summary.to_csv(
        PARTICIPANT_DIAGNOSTIC_PATH,
        index=False,
    )

    runs.to_csv(
        RUN_DIAGNOSTIC_PATH,
        index=False,
    )

    time_segments.to_csv(
        TIME_DIAGNOSTIC_PATH,
        index=False,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SAVED"
    )

    print(
        "=" * 72
    )

    print(
        ROW_DIAGNOSTIC_PATH
    )

    print(
        PARTICIPANT_DIAGNOSTIC_PATH
    )

    print(
        RUN_DIAGNOSTIC_PATH
    )

    print(
        TIME_DIAGNOSTIC_PATH
    )

    print(
        "\nDiagnostic complete."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "High-deviation runs here are descriptive diagnostics only."
    )

    print(
        "Persistence logic still belongs to the future Temporal Engine."
    )


if __name__ == "__main__":
    main()