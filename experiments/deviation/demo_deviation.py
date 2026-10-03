"""Real-data demo for the BioVance Deviation Engine using PMData.

Purpose
-------
Run resting heart-rate observations through:

    PMData
        ↓
    Personalization Engine
        ↓
    Data Quality Engine
        ↓
    Deviation Engine

This script is an empirical validation / calibration demo.

It does NOT:
    - model persistence
    - generate warnings
    - calculate temporal trajectories
    - calculate clinical risk

The main questions are:

1. How often do NORMAL / ELEVATED / LARGE / EXTREME deviations occur?
2. How many observations cannot be scored, and why?
3. Are the current descriptive bands (2.0 / 3.0 / 4.5 SD)
   reasonable on real PMData?
4. Do deviation rates vary substantially across participants?
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
# Import engines.
#
# Both personalization and quality packages have a
# from_pmdata_daily adapter, so alias them explicitly.
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
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "pmdata_hr_deviation_output.parquet"
)

PARTICIPANT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "pmdata_hr_deviation_by_participant.csv"
)

BAND_SUMMARY_PATH = (
    OUTPUT_DIR
    / "pmdata_hr_deviation_band_summary.csv"
)


# ============================================================
# Experiment configuration
# ============================================================

SIGNAL = "hr"

# Deterministic participant-level split.
#
# Personalization learns its population prior ONLY from
# training participants.
TRAIN_FRACTION = 0.70


# ============================================================
# Helpers
# ============================================================

def participant_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
) -> tuple[list[str], list[str]]:
    """Deterministically split participants.

    Sorting avoids randomness and makes this demo reproducible.

    IMPORTANT:
    The split is participant-level, not row-level.
    """

    participants = sorted(
        df[
            "patient_id"
        ]
        .astype(str)
        .unique()
    )

    if len(participants) < 4:
        raise ValueError(
            "Need at least 4 participants for this demo."
        )

    n_train = int(
        len(participants)
        * train_fraction
    )

    # Keep at least 3 training people because the
    # Personalization Engine population prior requires >=3.
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


def build_quality_daily(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Run PMData through the Data Quality Engine.

    Returns one quality record per patient / day / signal.
    """

    quality_input = (
        quality_from_pmdata_daily(
            raw
        )
    )

    quality_cfg = (
        load_quality_config(
            "pmdata"
        )
    )

    quality_engine = (
        DataQualityEngine(
            quality_cfg
        )
    )

    assessed = (
        quality_engine.assess(
            quality_input
        )
    )

    # Normalize join timestamp to calendar day.
    assessed[
        "date"
    ] = (
        pd.to_datetime(
            assessed[
                "timestamp"
            ],
            errors="coerce",
            utc=True,
        )
        .dt.normalize()
    )

    # Keep HR only.
    if (
        "signal"
        in assessed.columns
    ):

        assessed = assessed[
            assessed[
                "signal"
            ].astype(str).str.lower()
            == SIGNAL
        ]

    # There should normally be one HR quality row per patient/day,
    # but aggregate defensively if duplicates exist.
    quality_daily = (
        assessed
        .sort_values(
            [
                "patient_id",
                "date",
            ]
        )
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

    return quality_daily


def build_personalization_scores(
    raw: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    PersonalizationEngine,
]:
    """Fit Personalization prior on train participants and score PMData."""

    personalization_input = (
        personalization_from_pmdata_daily(
            raw
        )
    )

    cfg = (
        load_personalization_config(
            "real:pmdata"
        )
    )

    (
        train_ids,
        test_ids,
    ) = participant_split(
        personalization_input,
        TRAIN_FRACTION,
    )

    print(
        "\n=== PARTICIPANT SPLIT ==="
    )

    print(
        "Total participants:",
        personalization_input[
            "patient_id"
        ].nunique(),
    )

    print(
        "Train participants:",
        len(train_ids),
        train_ids,
    )

    print(
        "Holdout participants:",
        len(test_ids),
        test_ids,
    )

    train = (
        personalization_input[
            personalization_input[
                "patient_id"
            ].astype(str).isin(
                train_ids
            )
        ]
        .copy()
    )

    engine = (
        PersonalizationEngine(
            cfg
        )
    )

    # Population prior is learned from TRAIN participants only.
    engine.fit(
        train
    )

    # Score everybody with the frozen population prior.
    scored = (
        engine.score(
            personalization_input
        )
    )

    baselines = (
        engine.baselines(
            personalization_input
        )
    )

    scored = scored[
        scored[
            "signal"
        ]
        == SIGNAL
    ].copy()

    baselines = baselines[
        baselines[
            "signal"
        ]
        == SIGNAL
    ].copy()

    return (
        scored,
        baselines,
        engine,
    )


def build_deviation_input(
    personal_scores: pd.DataFrame,
    baselines: pd.DataFrame,
    quality_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Build canonical input for the Deviation Engine."""

    scores = (
        personal_scores.copy()
    )

    # Personalization uses a calendar-date field.
    scores[
        "timestamp"
    ] = pd.to_datetime(
        scores[
            "date"
        ],
        errors="coerce",
        utc=True,
    )

    # --------------------------------------------------------
    # Merge shrinkage metadata.
    #
    # Personalization score() already contains center, scale,
    # baseline_n and lifecycle status, but not w_center/w_scale.
    # baselines() provides those values.
    # --------------------------------------------------------

    baseline_meta_columns = [
        "patient_id",
        "signal",
        "w_center",
        "w_scale",
    ]

    available_meta = [
        c
        for c in baseline_meta_columns
        if c in baselines.columns
    ]

    if (
        "w_center"
        in available_meta
    ):

        scores = scores.merge(
            baselines[
                available_meta
            ],
            on=[
                "patient_id",
                "signal",
            ],
            how="left",
        )

    else:

        scores[
            "w_center"
        ] = np.nan

        scores[
            "w_scale"
        ] = np.nan

    # --------------------------------------------------------
    # Quality metadata
    # --------------------------------------------------------

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

    # Remove the quality helper join date if present.
    if (
        "date_quality"
        in scores.columns
    ):

        scores = scores.drop(
            columns=[
                "date_quality"
            ]
        )

    # --------------------------------------------------------
    # Missing quality metadata
    #
    # Don't silently call missing quality GOOD.
    # UNKNOWN lets us see this during the demo.
    # --------------------------------------------------------

    scores[
        "quality_status"
    ] = (
        scores[
            "quality_status"
        ]
        .fillna(
            "UNKNOWN"
        )
    )

    # --------------------------------------------------------
    # Context
    #
    # PMData source signal here is hr_rest_median, so this
    # particular observation represents resting HR evidence.
    #
    # We are NOT joining BAIGUTANOVA Context Engine results:
    # those are different participants/datasets.
    # --------------------------------------------------------

    scores[
        "context_state"
    ] = "REST"

    scores[
        "context_confidence"
    ] = 1.0

    # --------------------------------------------------------
    # Map Personalization lifecycle to Deviation lifecycle.
    #
    # Personalization score status:
    #   LEARNING
    #   ESTABLISHED
    #   INSUFFICIENT_BASELINE
    # --------------------------------------------------------

    scores[
        "baseline_status"
    ] = scores[
        "status"
    ]

    # --------------------------------------------------------
    # Shrinkage metadata
    #
    # w_center is the empirical-Bayes personal-center weight.
    # It is useful metadata for the future Uncertainty Engine.
    #
    # NOTE:
    # Personalization also has w_scale. The current Deviation
    # schema carries only one generic personal/population pair,
    # so for now we expose the CENTER shrinkage weight here.
    # --------------------------------------------------------

    scores[
        "baseline_personal_weight"
    ] = scores[
        "w_center"
    ]

    scores[
        "baseline_population_weight"
    ] = (
        1.0
        - scores[
            "w_center"
        ]
    )

    # --------------------------------------------------------
    # Canonical Deviation input
    # --------------------------------------------------------

    deviation_input = pd.DataFrame({

        "patient_id":
            scores[
                "patient_id"
            ],

        "timestamp":
            scores[
                "timestamp"
            ],

        "signal":
            scores[
                "signal"
            ],

        "value":
            scores[
                "value"
            ],

        "baseline_center":
            scores[
                "center"
            ],

        "baseline_scale":
            scores[
                "scale"
            ],

        "baseline_n":
            scores[
                "baseline_n"
            ],

        "baseline_status":
            scores[
                "baseline_status"
            ],

        "baseline_personal_weight":
            scores[
                "baseline_personal_weight"
            ],

        "baseline_population_weight":
            scores[
                "baseline_population_weight"
            ],

        "quality_score":
            scores[
                "quality_score"
            ],

        "quality_status":
            scores[
                "quality_status"
            ],

        "context_state":
            scores[
                "context_state"
            ],

        "context_confidence":
            scores[
                "context_confidence"
            ],
    })

    return deviation_input


def participant_summary(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """Create participant-level deviation diagnostics."""

    rows = []

    for patient_id, group in scored.groupby(
        "patient_id"
    ):

        usable = group[
            group[
                "deviation_code"
            ]
            == "OK"
        ]

        n_total = len(
            group
        )

        n_usable = len(
            usable
        )

        if n_usable:

            normal_fraction = (
                usable[
                    "deviation_status"
                ]
                .eq(
                    "NORMAL"
                )
                .mean()
            )

            elevated_fraction = (
                usable[
                    "deviation_status"
                ]
                .eq(
                    "ELEVATED"
                )
                .mean()
            )

            large_fraction = (
                usable[
                    "deviation_status"
                ]
                .eq(
                    "LARGE"
                )
                .mean()
            )

            extreme_fraction = (
                usable[
                    "deviation_status"
                ]
                .eq(
                    "EXTREME"
                )
                .mean()
            )

            toward_risk_fraction = (
                usable[
                    "toward_risk"
                ]
                .astype(bool)
                .mean()
            )

            mean_z = (
                usable[
                    "deviation_score"
                ]
                .mean()
            )

            mean_risk_z = (
                usable[
                    "risk_aligned_score"
                ]
                .mean()
            )

            max_abs_z = (
                usable[
                    "deviation_magnitude"
                ]
                .max()
            )

        else:

            normal_fraction = np.nan
            elevated_fraction = np.nan
            large_fraction = np.nan
            extreme_fraction = np.nan
            toward_risk_fraction = np.nan
            mean_z = np.nan
            mean_risk_z = np.nan
            max_abs_z = np.nan

        rows.append({

            "patient_id":
                patient_id,

            "n_total_days":
                n_total,

            "n_usable_days":
                n_usable,

            "usable_fraction":
                (
                    n_usable
                    / n_total
                    if n_total
                    else np.nan
                ),

            "normal_fraction":
                normal_fraction,

            "elevated_fraction":
                elevated_fraction,

            "large_fraction":
                large_fraction,

            "extreme_fraction":
                extreme_fraction,

            "toward_risk_fraction":
                toward_risk_fraction,

            "mean_deviation_score":
                mean_z,

            "mean_risk_aligned_score":
                mean_risk_z,

            "max_deviation_magnitude":
                max_abs_z,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Reading PMData:"
    )

    print(
        DATA_PATH
    )

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"PMData not found: {DATA_PATH}"
        )

    raw = pd.read_csv(
        DATA_PATH
    )

    print(
        "\nRaw shape:"
    )

    print(
        raw.shape
    )

    print(
        "\nRaw participants:"
    )

    print(
        raw[
            "participant"
        ].nunique()
        if "participant" in raw.columns
        else "participant column not found"
    )

    # ========================================================
    # QUALITY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "QUALITY ENGINE"
    )

    print(
        "=" * 70
    )

    quality_daily = (
        build_quality_daily(
            raw
        )
    )

    print(
        "\nQuality rows:"
    )

    print(
        len(
            quality_daily
        )
    )

    print(
        "\nQuality status:"
    )

    print(
        quality_daily[
            "quality_status"
        ]
        .value_counts(
            dropna=False
        )
    )

    # ========================================================
    # PERSONALIZATION
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PERSONALIZATION ENGINE"
    )

    print(
        "=" * 70
    )

    (
        personal_scores,
        baselines,
        personalization_engine,
    ) = build_personalization_scores(
        raw
    )

    print(
        "\nPersonalization HR rows:"
    )

    print(
        len(
            personal_scores
        )
    )

    print(
        "\nPersonalization status:"
    )

    print(
        personal_scores[
            "status"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\n=== HR BASELINES ==="
    )

    baseline_display = [
        "patient_id",
        "baseline_n",
        "center",
        "scale",
        "baseline_status",
    ]

    for column in (
        "w_center",
        "w_scale",
    ):

        if column in baselines.columns:

            baseline_display.append(
                column
            )

    print(
        baselines[
            baseline_display
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # BUILD DEVIATION INPUT
    # ========================================================

    deviation_input = (
        build_deviation_input(
            personal_scores,
            baselines,
            quality_daily,
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DEVIATION INPUT"
    )

    print(
        "=" * 70
    )

    print(
        "\nShape:"
    )

    print(
        deviation_input.shape
    )

    print(
        "\nSample:"
    )

    print(
        deviation_input
        .head(10)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # DEVIATION ENGINE
    # ========================================================

    deviation_cfg = (
        load_deviation_config()
    )

    deviation_engine = (
        DeviationEngine(
            deviation_cfg
        )
    )

    result = (
        deviation_engine.score(
            deviation_input
        )
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DEVIATION SANITY CHECKS"
    )

    print(
        "=" * 70
    )

    print(
        "\nInput rows:",
        len(
            deviation_input
        ),
    )

    print(
        "Output rows:",
        len(
            result
        ),
    )

    print(
        "Rows preserved:",
        len(
            deviation_input
        )
        == len(
            result
        ),
    )

    print(
        "Missing deviation_code:",
        result[
            "deviation_code"
        ].isna().sum(),
    )

    # ========================================================
    # AVAILABILITY / BLOCKING CODES
    # ========================================================

    print(
        "\n=== DEVIATION CODE COUNTS ==="
    )

    print(
        result[
            "deviation_code"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\n=== DEVIATION CODE FRACTIONS ==="
    )

    print(
        result[
            "deviation_code"
        ]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .round(4)
    )

    # ========================================================
    # USABLE OBSERVATIONS
    # ========================================================

    usable = result[
        result[
            "deviation_code"
        ]
        == "OK"
    ].copy()

    print(
        "\nUsable deviation rows:"
    )

    print(
        len(
            usable
        )
    )

    print(
        "Usable fraction:"
    )

    print(
        round(
            len(
                usable
            )
            / len(
                result
            ),
            4,
        )
    )

    # ========================================================
    # BANDS
    # ========================================================

    print(
        "\n=== DEVIATION STATUS COUNTS: USABLE ROWS ==="
    )

    band_counts = (
        usable[
            "deviation_status"
        ]
        .value_counts()
        .reindex(
            [
                "NORMAL",
                "ELEVATED",
                "LARGE",
                "EXTREME",
            ],
            fill_value=0,
        )
    )

    print(
        band_counts
    )

    print(
        "\n=== DEVIATION STATUS FRACTIONS: USABLE ROWS ==="
    )

    if len(
        usable
    ):

        band_fraction = (
            band_counts
            / len(
                usable
            )
        )

    else:

        band_fraction = (
            band_counts.astype(
                float
            )
        )

    print(
        band_fraction.round(
            4
        )
    )

    # ========================================================
    # SCORE DISTRIBUTIONS
    # ========================================================

    print(
        "\n=== PHYSICAL DEVIATION SCORE ==="
    )

    print(
        usable[
            "deviation_score"
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    print(
        "\n=== DEVIATION MAGNITUDE ==="
    )

    print(
        usable[
            "deviation_magnitude"
        ].describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.975,
                0.99,
            ]
        )
    )

    print(
        "\n=== RISK-ALIGNED SCORE ==="
    )

    print(
        usable[
            "risk_aligned_score"
        ].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )

    # ========================================================
    # DIRECTION
    # ========================================================

    print(
        "\n=== PHYSICAL DIRECTION ==="
    )

    print(
        usable[
            "deviation_direction"
        ]
        .value_counts(
            normalize=True
        )
        .round(4)
    )

    print(
        "\n=== TOWARD-RISK FRACTION ==="
    )

    if len(
        usable
    ):

        print(
            round(
                usable[
                    "toward_risk"
                ]
                .astype(bool)
                .mean(),
                4,
            )
        )

    # ========================================================
    # BASELINE RELIABILITY
    # ========================================================

    print(
        "\n=== BASELINE RELIABILITY ==="
    )

    print(
        result[
            "baseline_reliability"
        ].describe()
    )

    # ========================================================
    # LARGEST DEVIATIONS
    # ========================================================

    print(
        "\n=== LARGEST ABSOLUTE DEVIATIONS ==="
    )

    display_columns = [
        "patient_id",
        "timestamp",
        "value",
        "baseline_center",
        "baseline_scale",
        "baseline_n",
        "quality_status",
        "delta_units",
        "percent_change",
        "deviation_score",
        "risk_aligned_score",
        "deviation_status",
        "toward_risk",
        "deviation_explanation",
    ]

    print(
        usable
        .sort_values(
            "deviation_magnitude",
            ascending=False,
        )[
            display_columns
        ]
        .head(20)
        .round(
            {
                "value": 2,
                "baseline_center": 2,
                "baseline_scale": 2,
                "delta_units": 2,
                "percent_change": 2,
                "deviation_score": 2,
                "risk_aligned_score": 2,
            }
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # PARTICIPANT-LEVEL ANALYSIS
    # ========================================================

    participant = (
        participant_summary(
            result
        )
    )

    print(
        "\n=== PARTICIPANT-LEVEL SUMMARY ==="
    )

    print(
        participant
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n=== HIGHEST LARGE/EXTREME FRACTION ==="
    )

    participant[
        "large_or_extreme_fraction"
    ] = (
        participant[
            "large_fraction"
        ].fillna(0)
        +
        participant[
            "extreme_fraction"
        ].fillna(0)
    )

    print(
        participant[
            [
                "patient_id",
                "n_usable_days",
                "elevated_fraction",
                "large_fraction",
                "extreme_fraction",
                "large_or_extreme_fraction",
                "max_deviation_magnitude",
            ]
        ]
        .sort_values(
            "large_or_extreme_fraction",
            ascending=False,
        )
        .head(10)
        .round(4)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # EMPIRICAL BAND FIRING TABLE
    # ========================================================

    band_summary = pd.DataFrame({

        "band": [
            "NORMAL",
            "ELEVATED",
            "LARGE",
            "EXTREME",
        ],

        "count": [
            int(
                band_counts.get(
                    "NORMAL",
                    0,
                )
            ),
            int(
                band_counts.get(
                    "ELEVATED",
                    0,
                )
            ),
            int(
                band_counts.get(
                    "LARGE",
                    0,
                )
            ),
            int(
                band_counts.get(
                    "EXTREME",
                    0,
                )
            ),
        ],

        "fraction_of_usable": [
            float(
                band_fraction.get(
                    "NORMAL",
                    np.nan,
                )
            ),
            float(
                band_fraction.get(
                    "ELEVATED",
                    np.nan,
                )
            ),
            float(
                band_fraction.get(
                    "LARGE",
                    np.nan,
                )
            ),
            float(
                band_fraction.get(
                    "EXTREME",
                    np.nan,
                )
            ),
        ],
    })

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    participant.to_csv(
        PARTICIPANT_SUMMARY_PATH,
        index=False,
    )

    band_summary.to_csv(
        BAND_SUMMARY_PATH,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SAVED OUTPUTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nDeviation output:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nParticipant summary:"
    )

    print(
        PARTICIPANT_SUMMARY_PATH
    )

    print(
        "\nBand summary:"
    )

    print(
        BAND_SUMMARY_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These deviation bands are descriptive only."
    )

    print(
        "No persistence, warning, or clinical-risk decision "
        "is made in this script."
    )


if __name__ == "__main__":
    main()