"""Run BioVance Fusion Engine v1 on DRYAD Temporal output."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from biovance.fusion import FusionEngine


INPUT_PATH = (
    ROOT
    / "results"
    / "temporal"
    / "dryad_temporal_output.parquet"
)

CONFIG_PATH = (
    ROOT
    / "configs"
    / "fusion.yaml"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "fusion"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "dryad_fusion_output.parquet"
)

STATE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "dryad_fusion_state_summary.csv"
)

PARTICIPANT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "dryad_fusion_participant_summary.csv"
)


def main():

    print(
        "Reading DRYAD Temporal output:"
    )

    print(
        INPUT_PATH
    )

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing Temporal output:\n{INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        "\nInput shape:",
        df.shape,
    )

    print(
        "Participants:",
        df[
            "patient_id"
        ].nunique(),
    )

    print(
        "Signals:",
        sorted(
            df[
                "signal"
            ]
            .dropna()
            .unique()
        ),
    )

    engine = (
        FusionEngine
        .from_yaml(
            CONFIG_PATH
        )
    )

    print(
        "\nRunning Fusion Engine..."
    )

    result = engine.transform(
        df
    )

    print(
        "\nRows in:",
        len(df),
    )

    print(
        "Rows out:",
        len(result),
    )

    if len(result) != len(df):

        raise RuntimeError(
            "Fusion changed row count."
        )

    # ========================================================
    # Fusion codes
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FUSION CODE COUNTS"
    )

    print(
        "=" * 72
    )

    print(
        result[
            "fusion_code"
        ]
        .value_counts(
            dropna=False
        )
    )

    usable = result[
        result[
            "fusion_code"
        ]
        == "OK"
    ].copy()

    print(
        "\nUsable fusion rows:",
        len(
            usable
        ),
    )

    # ========================================================
    # States
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FUSION STATE COUNTS"
    )

    print(
        "=" * 72
    )

    print(
        usable[
            "fusion_state"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nFractions:"
    )

    print(
        usable[
            "fusion_state"
        ]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .round(4)
    )

    # ========================================================
    # Fusion scores
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FUSION SCORE"
    )

    print(
        "=" * 72
    )

    print(
        usable[
            "fusion_score"
        ].describe(
            percentiles=[
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
    # Agreement
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "MULTISIGNAL AGREEMENT"
    )

    print(
        "=" * 72
    )

    print(
        "\nSupporting signal count:"
    )

    print(
        usable[
            "supporting_signal_count"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nSupporting family count:"
    )

    print(
        usable[
            "supporting_family_count"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nSupport fraction:"
    )

    print(
        usable[
            "support_fraction"
        ].describe()
    )

    print(
        "\nDirection agreement:"
    )

    print(
        usable[
            "direction_agreement"
        ].describe()
    )

    # ========================================================
    # Components
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FUSION COMPONENTS"
    )

    print(
        "=" * 72
    )

    print(
        usable[
            [
                "deviation_component",
                "temporal_component",
                "agreement_component",
                "fusion_confidence",
            ]
        ]
        .describe()
        .round(4)
    )

    # ========================================================
    # Strong examples
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "STRONGEST MULTIMODAL EVIDENCE"
    )

    print(
        "=" * 72
    )

    strong = (
        usable[
            usable[
                "supporting_signal_count"
            ]
            >= 2
        ]
        .sort_values(
            [
                "fusion_score",
                "supporting_family_count",
                "fusion_confidence",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
    )

    columns = [
        "patient_id",
        "timestamp",
        "signal",

        "fusion_state",
        "fusion_score",
        "fusion_evidence_strength",
        "opposition_fraction",

        "available_signal_count",
        "supporting_signal_count",
        "opposing_signal_count",
        "supporting_family_count",

        "support_fraction",
        "direction_agreement",

        "deviation_component",
        "temporal_component",
        "agreement_component",
        "fusion_confidence",

        "supporting_signals",
        "opposing_signals",
    ]

    print(
        strong[
            columns
        ]
        .head(40)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Conflicting examples
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "CONFLICTING EVIDENCE EXAMPLES"
    )

    print(
        "=" * 72
    )

    conflicts = usable[
        usable[
            "fusion_state"
        ]
        ==
        "CONFLICTING_EVIDENCE"
    ].copy()

    print(
        conflicts[
            columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )
    # ========================================================
    # State summary
    # ========================================================

    state_summary = (
        usable[
            "fusion_state"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "fusion_state"
        )
        .reset_index(
            name="count"
        )
    )

    state_summary[
        "fraction"
    ] = (
        state_summary[
            "count"
        ]
        / len(
            usable
        )
    )

    # ========================================================
    # Participant summary
    # ========================================================

    participant_summary = (
        usable
        .groupby(
            "patient_id"
        )
        .agg(

            n_fusion_rows=(
                "fusion_state",
                "size",
            ),

            mean_fusion_score=(
                "fusion_score",
                "mean",
            ),

            max_fusion_score=(
                "fusion_score",
                "max",
            ),

            mean_support_fraction=(
                "support_fraction",
                "mean",
            ),

            mean_supporting_signals=(
                "supporting_signal_count",
                "mean",
            ),

            max_supporting_signals=(
                "supporting_signal_count",
                "max",
            ),

            mean_fusion_confidence=(
                "fusion_confidence",
                "mean",
            ),
        )
        .reset_index()
    )

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

    state_summary.to_csv(
        STATE_SUMMARY_PATH,
        index=False,
    )

    participant_summary.to_csv(
        PARTICIPANT_SUMMARY_PATH,
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
        OUTPUT_PATH
    )

    print(
        STATE_SUMMARY_PATH
    )

    print(
        PARTICIPANT_SUMMARY_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Fusion states are descriptive multimodal evidence."
    )

    print(
        "No WARN / MONITOR / ABSTAIN decision was made."
    )

    print(
        "No future observations were used."
    )


if __name__ == "__main__":
    main()