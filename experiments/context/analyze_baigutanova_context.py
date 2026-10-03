"""
Exploratory analysis for the BioVance Context Engine using BAIGUTANOVA.

Purpose
-------
Understand how movement features behave during diary-defined sleep versus
non-sleep periods BEFORE selecting REST / ACTIVE thresholds.

This script does NOT generate final context labels.

It analyzes:
    - sleep overlap
    - accelerometer magnitude
    - participant-specific dynamic acceleration
    - gyroscope magnitude
    - steps
    - distance
    - calories
    - missingness
    - sleep timing by hour

Outputs are written to:
    results/context/baigutanova/

The results will be used to define the Context Engine configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Project paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "baigutanova_context_input.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "context"
    / "baigutanova"
)


# =============================================================================
# Analysis configuration
# =============================================================================

# Strong sleep evidence:
# at least 80% of the sensor window overlaps the diary sleep interval.
SLEEP_OVERLAP_THRESHOLD = 0.80

# Completely outside known diary sleep interval.
AWAKE_OVERLAP_THRESHOLD = 0.0

# Minimum number of strong-sleep accelerometer windows required to estimate
# a participant-specific low-motion accelerometer reference.
MIN_SLEEP_REFERENCE_WINDOWS = 10


# =============================================================================
# Utilities
# =============================================================================

def validate_input(df: pd.DataFrame) -> None:
    """Validate that the processed dataset has the expected fields."""

    required = {
        "patient_id",
        "timestamp_start",
        "timestamp_end",
        "acc_magnitude",
        "gyro_magnitude",
        "missingness_score",
        "inside_sleep_interval",
        "sleep_overlap_fraction",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Processed BAIGUTANOVA context dataset is missing "
            f"required columns: {sorted(missing)}"
        )


def safe_quantiles(
    series: pd.Series,
    quantiles=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
) -> pd.Series:
    """Return useful quantiles after removing missing/infinite values."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if values.empty:
        return pd.Series(
            {
                f"p{int(q * 100):02d}": np.nan
                for q in quantiles
            }
        )

    return pd.Series(
        {
            f"p{int(q * 100):02d}": values.quantile(q)
            for q in quantiles
        }
    )


def add_sleep_group(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create exploratory sleep-evidence groups.

    These are NOT final Context Engine labels.

    STRONG_SLEEP:
        >= 80% diary overlap

    AWAKE_REFERENCE:
        exactly 0% diary overlap

    PARTIAL_SLEEP:
        between 0 and 80% overlap
    """

    out = df.copy()

    overlap = out["sleep_overlap_fraction"]

    out["sleep_group"] = np.select(
        [
            overlap >= SLEEP_OVERLAP_THRESHOLD,
            overlap <= AWAKE_OVERLAP_THRESHOLD,
        ],
        [
            "STRONG_SLEEP",
            "AWAKE_REFERENCE",
        ],
        default="PARTIAL_SLEEP",
    )

    return out


def add_personal_acc_reference(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive participant-specific accelerometer movement.

    Raw accelerometer magnitude is strongly affected by gravity.

    Instead of directly treating ~9.8 as activity, estimate each person's
    low-motion reference from diary-defined sleep windows.

    If a participant does not have enough strong-sleep windows, fall back
    to that participant's overall median accelerometer magnitude.

    dynamic_acc =
        abs(current_acc_magnitude - personal_low_motion_reference)

    This is still only a candidate movement feature, not an engine output.
    """

    out = df.copy()

    references = {}

    for patient_id, group in out.groupby(
        "patient_id"
    ):

        sleep_acc = group.loc[
            (
                group["sleep_group"]
                == "STRONG_SLEEP"
            )
            & group["acc_magnitude"].notna(),
            "acc_magnitude",
        ]

        all_acc = group[
            "acc_magnitude"
        ].dropna()

        if (
            len(sleep_acc)
            >= MIN_SLEEP_REFERENCE_WINDOWS
        ):
            reference = float(
                sleep_acc.median()
            )

            source = "sleep_median"

        elif len(all_acc) > 0:
            reference = float(
                all_acc.median()
            )

            source = "overall_median"

        else:
            reference = np.nan
            source = "unavailable"

        references[patient_id] = (
            reference,
            source,
        )

    out["acc_reference"] = (
        out["patient_id"]
        .map(
            {
                pid: values[0]
                for pid, values
                in references.items()
            }
        )
    )

    out["acc_reference_source"] = (
        out["patient_id"]
        .map(
            {
                pid: values[1]
                for pid, values
                in references.items()
            }
        )
    )

    out["dynamic_acc"] = (
        out["acc_magnitude"]
        - out["acc_reference"]
    ).abs()

    return out


def add_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add time-of-day fields for exploratory sleep alignment checks."""

    out = df.copy()

    out["hour_utc"] = (
        out["timestamp_start"].dt.hour
    )

    out["date_utc"] = (
        out["timestamp_start"].dt.date
    )

    return out


# =============================================================================
# Summary tables
# =============================================================================

def feature_summary_by_sleep_group(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare movement distributions by sleep evidence."""

    features = [
        "acc_magnitude",
        "dynamic_acc",
        "gyro_magnitude",
        "steps",
        "distance",
        "calories",
        "missingness_score",
    ]

    rows = []

    for group_name, group in df.groupby(
        "sleep_group"
    ):

        for feature in features:

            if feature not in group.columns:
                continue

            x = pd.to_numeric(
                group[feature],
                errors="coerce",
            )

            valid = x.dropna()

            row = {
                "sleep_group": group_name,
                "feature": feature,
                "n_total": len(group),
                "n_available": len(valid),
                "missing_fraction": x.isna().mean(),
                "mean": (
                    valid.mean()
                    if len(valid)
                    else np.nan
                ),
                "std": (
                    valid.std()
                    if len(valid)
                    else np.nan
                ),
                "min": (
                    valid.min()
                    if len(valid)
                    else np.nan
                ),
                "max": (
                    valid.max()
                    if len(valid)
                    else np.nan
                ),
            }

            row.update(
                safe_quantiles(valid).to_dict()
            )

            rows.append(row)

    return pd.DataFrame(rows)


def awake_quantiles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate candidate awake movement quantiles.

    These quantiles are descriptive only.

    Later we can use them to determine whether REST / ACTIVE thresholds
    should be absolute or participant-relative.
    """

    awake = df[
        df["sleep_group"]
        == "AWAKE_REFERENCE"
    ]

    features = [
        "dynamic_acc",
        "gyro_magnitude",
        "steps",
        "distance",
        "calories",
    ]

    rows = []

    for feature in features:

        if feature not in awake.columns:
            continue

        q = safe_quantiles(
            awake[feature],
            quantiles=(
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.80,
                0.90,
                0.95,
                0.99,
            ),
        )

        row = {
            "feature": feature,
            "n_available": awake[
                feature
            ].notna().sum(),
            "missing_fraction": awake[
                feature
            ].isna().mean(),
        }

        row.update(
            q.to_dict()
        )

        rows.append(row)

    return pd.DataFrame(rows)


def participant_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize context evidence separately for every participant."""

    rows = []

    for patient_id, group in df.groupby(
        "patient_id"
    ):

        sleep = group[
            group["sleep_group"]
            == "STRONG_SLEEP"
        ]

        awake = group[
            group["sleep_group"]
            == "AWAKE_REFERENCE"
        ]

        rows.append(
            {
                "patient_id": patient_id,

                "n_windows": len(group),

                "strong_sleep_windows": len(
                    sleep
                ),

                "awake_reference_windows": len(
                    awake
                ),

                "partial_sleep_windows": (
                    group[
                        "sleep_group"
                    ]
                    == "PARTIAL_SLEEP"
                ).sum(),

                "sleep_fraction": (
                    (
                        group["sleep_group"]
                        == "STRONG_SLEEP"
                    ).mean()
                ),

                "acc_reference": (
                    group[
                        "acc_reference"
                    ].iloc[0]
                ),

                "acc_reference_source": (
                    group[
                        "acc_reference_source"
                    ].iloc[0]
                ),

                "sleep_dynamic_acc_median": (
                    sleep[
                        "dynamic_acc"
                    ].median()
                ),

                "awake_dynamic_acc_median": (
                    awake[
                        "dynamic_acc"
                    ].median()
                ),

                "sleep_gyro_median": (
                    sleep[
                        "gyro_magnitude"
                    ].median()
                ),

                "awake_gyro_median": (
                    awake[
                        "gyro_magnitude"
                    ].median()
                ),

                "awake_steps_available_fraction": (
                    awake[
                        "steps"
                    ].notna().mean()
                    if "steps" in awake
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def sleep_by_hour(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether diary-defined sleep falls at plausible UTC hours."""

    table = (
        df.groupby("hour_utc")
        .agg(
            n_windows=(
                "patient_id",
                "size",
            ),
            strong_sleep_windows=(
                "sleep_group",
                lambda x: (
                    x == "STRONG_SLEEP"
                ).sum(),
            ),
        )
        .reset_index()
    )

    table["strong_sleep_fraction"] = (
        table["strong_sleep_windows"]
        / table["n_windows"]
    )

    return table


# =============================================================================
# Extreme awake examples
# =============================================================================

def extract_awake_extremes(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return lowest- and highest-movement awake windows."""

    awake = df[
        df["sleep_group"]
        == "AWAKE_REFERENCE"
    ].copy()

    columns = [
        "patient_id",
        "timestamp_start",
        "dynamic_acc",
        "gyro_magnitude",
        "steps",
        "distance",
        "calories",
        "missingness_score",
    ]

    columns = [
        c
        for c in columns
        if c in awake.columns
    ]

    # Use dynamic_acc + gyro as an exploratory ranking only.
    awake["exploratory_motion_rank"] = (
        awake[
            "dynamic_acc"
        ].rank(
            pct=True,
        )
        +
        awake[
            "gyro_magnitude"
        ].rank(
            pct=True,
        )
    ) / 2

    lowest = (
        awake
        .sort_values(
            "exploratory_motion_rank"
        )
        .head(50)
    )

    highest = (
        awake
        .sort_values(
            "exploratory_motion_rank",
            ascending=False,
        )
        .head(50)
    )

    columns += [
        "exploratory_motion_rank"
    ]

    return (
        lowest[columns],
        highest[columns],
    )


# =============================================================================
# Plots
# =============================================================================

def save_histogram(
    series: pd.Series,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 50,
) -> None:

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if x.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        x,
        bins=bins,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Window count")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


def save_sleep_hour_plot(
    table: pd.DataFrame,
    output_path: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        table["hour_utc"],
        table["strong_sleep_fraction"],
        marker="o",
    )

    ax.set_title(
        "Diary-defined sleep fraction by UTC hour"
    )

    ax.set_xlabel(
        "UTC hour"
    )

    ax.set_ylabel(
        "Fraction of windows with strong sleep overlap"
    )

    ax.set_xticks(
        range(24)
    )

    ax.set_ylim(
        0,
        1,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


def save_sleep_awake_boxplot(
    df: pd.DataFrame,
    feature: str,
    output_path: Path,
) -> None:

    groups = []

    labels = []

    for label in [
        "STRONG_SLEEP",
        "AWAKE_REFERENCE",
    ]:

        values = pd.to_numeric(
            df.loc[
                df["sleep_group"]
                == label,
                feature,
            ],
            errors="coerce",
        ).dropna()

        if len(values):
            groups.append(
                values.to_numpy()
            )
            labels.append(label)

    if len(groups) < 2:
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        groups,
        tick_labels=labels,
        showfliers=False,
    )

    ax.set_title(
        f"{feature}: sleep vs awake"
    )

    ax.set_ylabel(
        feature
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Reading processed BAIGUTANOVA context data:"
    )

    print(
        INPUT_PATH
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Processed BAIGUTANOVA file not found: "
            f"{INPUT_PATH}\n"
            "Run scripts/prepare_baigutanova_context.py first."
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    validate_input(df)

    print(
        "\nRaw processed shape:"
    )

    print(
        df.shape
    )

    print(
        "\nParticipants:"
    )

    print(
        df["patient_id"].nunique()
    )

    # -----------------------------------------------------------------
    # Derive exploratory analysis fields
    # -----------------------------------------------------------------

    df = add_sleep_group(
        df
    )

    df = add_personal_acc_reference(
        df
    )

    df = add_time_features(
        df
    )

    # -----------------------------------------------------------------
    # Basic context evidence
    # -----------------------------------------------------------------

    print(
        "\n=== SLEEP EVIDENCE GROUPS ==="
    )

    print(
        df[
            "sleep_group"
        ].value_counts()
    )

    print(
        "\n=== ACC REFERENCE SOURCES ==="
    )

    print(
        df[
            [
                "patient_id",
                "acc_reference_source",
            ]
        ]
        .drop_duplicates()
        ["acc_reference_source"]
        .value_counts()
    )

    # -----------------------------------------------------------------
    # Sleep vs awake summary
    # -----------------------------------------------------------------

    feature_summary = (
        feature_summary_by_sleep_group(
            df
        )
    )

    feature_summary.to_csv(
        OUTPUT_DIR
        / "feature_summary_by_sleep_group.csv",
        index=False,
    )

    print(
        "\n=== FEATURE SUMMARY: SLEEP VS AWAKE ==="
    )

    display_columns = [
        "sleep_group",
        "feature",
        "n_available",
        "missing_fraction",
        "p25",
        "p50",
        "p75",
        "p90",
    ]

    print(
        feature_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Candidate awake quantiles
    # -----------------------------------------------------------------

    awake_q = awake_quantiles(
        df
    )

    awake_q.to_csv(
        OUTPUT_DIR
        / "awake_feature_quantiles.csv",
        index=False,
    )

    print(
        "\n=== AWAKE MOVEMENT QUANTILES ==="
    )

    print(
        awake_q.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Participant-level analysis
    # -----------------------------------------------------------------

    patient_stats = participant_summary(
        df
    )

    patient_stats.to_csv(
        OUTPUT_DIR
        / "participant_context_summary.csv",
        index=False,
    )

    print(
        "\n=== PARTICIPANT SUMMARY ==="
    )

    print(
        patient_stats.head(
            20
        ).to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Sleep timing / timezone sanity check
    # -----------------------------------------------------------------

    hourly_sleep = sleep_by_hour(
        df
    )

    hourly_sleep.to_csv(
        OUTPUT_DIR
        / "sleep_fraction_by_utc_hour.csv",
        index=False,
    )

    print(
        "\n=== SLEEP FRACTION BY UTC HOUR ==="
    )

    print(
        hourly_sleep.to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Extreme awake examples
    # -----------------------------------------------------------------

    lowest_motion, highest_motion = (
        extract_awake_extremes(
            df
        )
    )

    lowest_motion.to_csv(
        OUTPUT_DIR
        / "lowest_motion_awake_windows.csv",
        index=False,
    )

    highest_motion.to_csv(
        OUTPUT_DIR
        / "highest_motion_awake_windows.csv",
        index=False,
    )

    print(
        "\n=== LOWEST-MOTION AWAKE WINDOWS ==="
    )

    print(
        lowest_motion.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        "\n=== HIGHEST-MOTION AWAKE WINDOWS ==="
    )

    print(
        highest_motion.head(
            10
        ).to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Missingness
    # -----------------------------------------------------------------

    print(
        "\n=== MISSINGNESS ==="
    )

    for feature in [
        "acc_magnitude",
        "dynamic_acc",
        "gyro_magnitude",
        "steps",
        "distance",
        "calories",
        "missingness_score",
    ]:

        if feature not in df.columns:
            continue

        print(
            f"{feature:20s} "
            f"{df[feature].isna().mean():.3%} missing"
        )

    # -----------------------------------------------------------------
    # Save exploratory plots
    # -----------------------------------------------------------------

    save_histogram(
        df["dynamic_acc"],
        "Participant-relative dynamic acceleration",
        "dynamic_acc",
        OUTPUT_DIR
        / "dynamic_acc_histogram.png",
    )

    save_histogram(
        df["gyro_magnitude"],
        "Gyroscope magnitude",
        "gyro_magnitude",
        OUTPUT_DIR
        / "gyro_magnitude_histogram.png",
    )

    save_sleep_awake_boxplot(
        df,
        "dynamic_acc",
        OUTPUT_DIR
        / "dynamic_acc_sleep_vs_awake.png",
    )

    save_sleep_awake_boxplot(
        df,
        "gyro_magnitude",
        OUTPUT_DIR
        / "gyro_sleep_vs_awake.png",
    )

    save_sleep_hour_plot(
        hourly_sleep,
        OUTPUT_DIR
        / "sleep_fraction_by_utc_hour.png",
    )

    # -----------------------------------------------------------------
    # Save analysis-enhanced table
    #
    # This file is exploratory only.
    # Do NOT treat sleep_group or exploratory_motion_rank as engine labels.
    # -----------------------------------------------------------------

    analysis_path = (
        OUTPUT_DIR
        / "baigutanova_context_analysis.parquet"
    )

    df.to_parquet(
        analysis_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print(
        "\n=== ANALYSIS COMPLETE ==="
    )

    print(
        "\nResults directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nFiles created:"
    )

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        print(
            " -",
            path.name,
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No REST / ACTIVE / SLEEP engine labels "
        "were generated by this analysis."
    )

    print(
        "Use these results to choose and justify "
        "Context Engine rules."
    )


if __name__ == "__main__":
    main()