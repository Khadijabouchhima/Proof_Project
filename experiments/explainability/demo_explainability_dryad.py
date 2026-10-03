from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


from biovance.explainability import (
    ExplainabilityEngine,
    load_explainability_config,
)


INPUT_PATH = (
    ROOT
    / "results"
    / "decision"
    / "dryad_decision_output.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "explainability"
)


def main() -> None:

    print(
        "Reading:"
    )

    print(
        INPUT_PATH
    )

    decision = pd.read_parquet(
        INPUT_PATH
    )

    print(
        "\nInput shape:"
    )

    print(
        decision.shape
    )

    # --------------------------------------------------------
    # Preserve corrected Fusion/Decision moment contract
    # --------------------------------------------------------

    required_keys = [
        "patient_id",
        "timestamp",
    ]

    for column in required_keys:
        if column not in decision.columns:
            raise ValueError(
                f"Missing required DRYAD key: {column}"
            )

    duplicate_moments = decision.duplicated(
        subset=[
            "patient_id",
            "timestamp",
        ],
        keep=False,
    )

    assert not duplicate_moments.any(), (
        "Explainability received duplicate DRYAD "
        "patient/timestamp decision moments."
    )

    config = load_explainability_config()

    engine = ExplainabilityEngine(
        config
    )

    explanations = engine.explain(
        decision
    )

    assert len(
        explanations
    ) == len(
        decision
    )

    result = pd.concat(
        [
            decision.reset_index(
                drop=True
            ),

            explanations.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Decision summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "EXPLAINED DRYAD DECISIONS"
    )

    print(
        "=" * 100
    )

    summary = (
        result[
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

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # WARN examples
    # --------------------------------------------------------

    warns = result[
        result[
            "decision_state"
        ]
        == "WARN"
    ].copy()

    print(
        "\nWARN explanations:"
    )

    print(
        len(
            warns
        )
    )

    if len(
        warns
    ) > 0:

        columns = [
            "patient_id",
            "timestamp",
            "fusion_state",
            "fusion_score",
            "uncertainty_score",
            "decision_state",
            "explanation_summary",
        ]

        columns = [
            column
            for column in columns
            if column in warns.columns
        ]

        print(
            "\n"
            + warns[
                columns
            ]
            .head(
                10
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Conflict MONITOR examples
    # --------------------------------------------------------

    conflict = result[
        (
            result[
                "decision_state"
            ]
            == "MONITOR"
        )
        &
        (
            result[
                "fusion_state"
            ]
            == "CONFLICTING_EVIDENCE"
        )
    ].copy()

    print(
        "\nConflict MONITOR explanations:"
    )

    print(
        len(
            conflict
        )
    )

    if len(
        conflict
    ) > 0:

        print(
            conflict[
                [
                    "patient_id",
                    "timestamp",
                    "fusion_score",
                    "uncertainty_score",
                    "explanation_summary",
                    "limiting_factors",
                ]
            ]
            .head(
                10
            )
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Consensus but MONITOR
    # --------------------------------------------------------

    blocked = result[
        (
            result[
                "decision_state"
            ]
            == "MONITOR"
        )
        &
        (
            result[
                "fusion_state"
            ]
            == "MULTIMODAL_CONSENSUS"
        )
    ].copy()

    print(
        "\nConsensus moments that remained MONITOR:"
    )

    print(
        len(
            blocked
        )
    )

    if len(
        blocked
    ) > 0:

        print(
            blocked[
                [
                    "patient_id",
                    "timestamp",
                    "fusion_score",
                    "uncertainty_score",
                    "explanation_summary",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "dryad_explainability_output.parquet"
    )

    summary_path = (
        OUTPUT_DIR
        / "dryad_explainability_decision_summary.csv"
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        summary_path
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Explainability preserved the existing "
        "Decision results."
    )

    print(
        "No decision or upstream score was recomputed."
    )


if __name__ == "__main__":
    main()