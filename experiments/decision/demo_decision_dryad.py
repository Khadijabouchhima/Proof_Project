from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0,
    str(ROOT / "src"),
)

from biovance.decision import (
    DecisionEngine,
    load_decision_config,
)


FUSION_PATH = (
    ROOT
    / "results"
    / "fusion"
    / "dryad_fusion_output.parquet"
)

UNCERTAINTY_PATH = (
    ROOT
    / "results"
    / "uncertainty"
    / "dryad_uncertainty_output.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "decision"
)


def main() -> None:
    print("Reading:")
    print(FUSION_PATH)
    print(UNCERTAINTY_PATH)

    fusion = pd.read_parquet(
        FUSION_PATH
    )

    uncertainty = pd.read_parquet(
        UNCERTAINTY_PATH
    )

    print("\nFusion shape:")
    print(fusion.shape)

    print("\nUncertainty shape:")
    print(uncertainty.shape)

    key = [
        "patient_id",
        "timestamp",
    ]

    required_fusion = [
        "patient_id",
        "timestamp",
        "fusion_score",
        "fusion_state",
        "fusion_code",
    ]

    required_uncertainty = [
        "patient_id",
        "timestamp",
        "uncertainty_score",
        "uncertainty_level",
        "uncertainty_code",
    ]

    missing_fusion = [
        col
        for col in required_fusion
        if col not in fusion.columns
    ]

    missing_uncertainty = [
        col
        for col in required_uncertainty
        if col not in uncertainty.columns
    ]

    if missing_fusion:
        raise ValueError(
            "Missing Fusion columns: "
            + ", ".join(
                missing_fusion
            )
        )

    if missing_uncertainty:
        raise ValueError(
            "Missing Uncertainty columns: "
            + ", ".join(
                missing_uncertainty
            )
        )

    fusion[
        "timestamp"
    ] = pd.to_datetime(
        fusion[
            "timestamp"
        ]
    )

    uncertainty[
        "timestamp"
    ] = pd.to_datetime(
        uncertainty[
            "timestamp"
        ]
    )

    # ---------------------------------------------------------
    # Validate one row per decision moment.
    # ---------------------------------------------------------
    if fusion.duplicated(
        subset=key,
        keep=False,
    ).any():
        raise ValueError(
            "Fusion contains duplicate patient/timestamp "
            "moments. Correct Fusion first."
        )

    if uncertainty.duplicated(
        subset=key,
        keep=False,
    ).any():
        raise ValueError(
            "Uncertainty contains duplicate patient/timestamp "
            "moments. Correct Uncertainty first."
        )

    # ---------------------------------------------------------
    # Join by semantic key, not row position.
    # ---------------------------------------------------------
    uncertainty_selected = (
        uncertainty[
            [
                "patient_id",
                "timestamp",
                "uncertainty_score",
                "uncertainty_level",
                "uncertainty_code",
                "certainty_score",
                "source_coverage",
                "coherence_score",
                "uncertainty_reason_codes",
            ]
        ]
        .copy()
    )

    combined = fusion.merge(
        uncertainty_selected,
        on=key,
        how="left",
        validate="one_to_one",
    )

    missing_uncertainty_rows = int(
        combined[
            "uncertainty_score"
        ]
        .isna()
        .sum()
    )

    print(
        "\nFusion moments without uncertainty result:"
    )
    print(
        missing_uncertainty_rows
    )

    if missing_uncertainty_rows > 0:
        raise ValueError(
            "Some Fusion moments have no matching "
            "Uncertainty result."
        )

    # ---------------------------------------------------------
    # Risk is unavailable for DRYAD.
    #
    # We must NOT attach unrelated Framingham predictions.
    # ---------------------------------------------------------
    decision_input = pd.DataFrame(
        {
            "fusion_score": (
                combined[
                    "fusion_score"
                ]
            ),

            "fusion_state": (
                combined[
                    "fusion_state"
                ]
            ),

            "risk_probability": (
                pd.Series(
                    [pd.NA] * len(combined),
                    index=combined.index,
                    dtype="Float64",
                )
            ),

            "uncertainty_score": (
                combined[
                    "uncertainty_score"
                ]
            ),

            "uncertainty_code": (
                combined[
                    "uncertainty_code"
                ]
            ),
        },
        index=combined.index,
    )

    config = load_decision_config()

    engine = DecisionEngine(
        config
    )

    decided = engine.decide(
        decision_input
    )

    result = pd.concat(
        [
            combined.reset_index(
                drop=True
            ),

            decided.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # One final decision per moment.
    # ---------------------------------------------------------
    duplicated = result.duplicated(
        subset=key,
        keep=False,
    )

    assert not duplicated.any(), (
        "Decision output must contain exactly one row per "
        "patient_id + timestamp."
    )

    usable = result[
        result[
            "fusion_code"
        ]
        == "OK"
    ].copy()

    print("\nUsable decision moments:")
    print(len(usable))

    # ---------------------------------------------------------
    # Decision states
    # ---------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )
    print(
        "DECISION STATES — USABLE DRYAD MOMENTS"
    )
    print(
        "=" * 80
    )

    state_summary = (
        usable[
            "decision_state"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "decision_state"
        )
        .reset_index(
            name="count"
        )
    )

    if len(usable) > 0:
        state_summary[
            "fraction"
        ] = (
            state_summary[
                "count"
            ]
            / len(usable)
        )
    else:
        state_summary[
            "fraction"
        ] = 0.0

    print(
        state_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Decision codes
    # ---------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )
    print(
        "DECISION CODES — USABLE DRYAD MOMENTS"
    )
    print(
        "=" * 80
    )

    code_summary = (
        usable[
            "decision_code"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "decision_code"
        )
        .reset_index(
            name="count"
        )
    )

    if len(usable) > 0:
        code_summary[
            "fraction"
        ] = (
            code_summary[
                "count"
            ]
            / len(usable)
        )
    else:
        code_summary[
            "fraction"
        ] = 0.0

    print(
        code_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Decision by Fusion state
    # ---------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )
    print(
        "DECISION BY FUSION STATE"
    )
    print(
        "=" * 80
    )

    if len(usable) > 0:
        fusion_decision = pd.crosstab(
            usable[
                "fusion_state"
            ],
            usable[
                "decision_state"
            ],
        )

        print(
            fusion_decision
        )

    # ---------------------------------------------------------
    # Counts
    # ---------------------------------------------------------
    warn = usable[
        usable[
            "decision_state"
        ]
        == "WARN"
    ].copy()

    monitor = usable[
        usable[
            "decision_state"
        ]
        == "MONITOR"
    ].copy()

    abstain = usable[
        usable[
            "decision_state"
        ]
        == "ABSTAIN"
    ].copy()

    print("\nWARN moments:")
    print(len(warn))

    print("MONITOR moments:")
    print(len(monitor))

    print("ABSTAIN moments:")
    print(len(abstain))

    # ---------------------------------------------------------
    # WARN by patient
    # ---------------------------------------------------------
    if len(warn) > 0:
        print(
            "\n"
            + "=" * 80
        )
        print(
            "WARN MOMENTS BY PATIENT"
        )
        print(
            "=" * 80
        )

        print(
            warn[
                "patient_id"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    # ---------------------------------------------------------
    # Highest-scoring examples
    # ---------------------------------------------------------
    example_cols = [
        col
        for col in [
            "patient_id",
            "timestamp",
            "context_state",
            "current_signals",
            "fusion_state",
            "fusion_score",
            "fusion_confidence",
            "supporting_signal_count",
            "opposing_signal_count",
            "uncertainty_score",
            "uncertainty_level",
            "decision_state",
            "decision_code",
            "decision_reason_codes",
        ]
        if col in usable.columns
    ]

    if len(usable) > 0:
        print(
            "\n"
            + "=" * 80
        )
        print(
            "HIGHEST FUSION SCORE DECISION MOMENTS"
        )
        print(
            "=" * 80
        )

        print(
            usable
            .sort_values(
                "fusion_score",
                ascending=False,
            )[
                example_cols
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "dryad_decision_output.parquet"
    )

    state_path = (
        OUTPUT_DIR
        / "dryad_decision_state_summary.csv"
    )

    code_path = (
        OUTPUT_DIR
        / "dryad_decision_code_summary.csv"
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    state_summary.to_csv(
        state_path,
        index=False,
    )

    code_summary.to_csv(
        code_path,
        index=False,
    )

    print("\nSaved:")
    print(output_path)
    print(state_path)
    print(code_path)

    print("\nIMPORTANT:")
    print(
        "Each Decision row now represents exactly one "
        "patient/timestamp physiological moment."
    )
    print(
        "DRYAD has no matching Framingham risk estimate."
    )
    print(
        "This experiment therefore validates Decision behavior "
        "from current physiology + uncertainty only."
    )


if __name__ == "__main__":
    main()