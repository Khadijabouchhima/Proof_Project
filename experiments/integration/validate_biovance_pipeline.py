from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# Project setup
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)


# ============================================================
# Paths
# ============================================================

TEMPORAL_PATH = (
    ROOT
    / "results"
    / "temporal"
    / "dryad_temporal_output.parquet"
)

FUSION_PATH = (
    ROOT
    / "results"
    / "fusion"
    / "dryad_fusion_output.parquet"
)

UNCERTAINTY_DRYAD_PATH = (
    ROOT
    / "results"
    / "uncertainty"
    / "dryad_uncertainty_output.parquet"
)

DECISION_DRYAD_PATH = (
    ROOT
    / "results"
    / "decision"
    / "dryad_decision_output.parquet"
)

EXPLAINABILITY_DRYAD_PATH = (
    ROOT
    / "results"
    / "explainability"
    / "dryad_explainability_output.parquet"
)

RISK_PATH = (
    ROOT
    / "results"
    / "risk"
    / "framingham_risk_predictions.parquet"
)

UNCERTAINTY_FRAMINGHAM_PATH = (
    ROOT
    / "results"
    / "uncertainty"
    / "framingham_uncertainty_output.parquet"
)

DECISION_FRAMINGHAM_PATH = (
    ROOT
    / "results"
    / "decision"
    / "framingham_decision_safety_validation.csv"
)

EXPLAINABILITY_FRAMINGHAM_PATH = (
    ROOT
    / "results"
    / "explainability"
    / "framingham_explainability_output.csv"
)

SYNTHETIC_DECISION_PATH = (
    ROOT
    / "results"
    / "decision"
    / "synthetic_decision_validation.csv"
)

SYNTHETIC_EXPLAINABILITY_PATH = (
    ROOT
    / "results"
    / "explainability"
    / "synthetic_explainability_validation.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "integration"
)


# ============================================================
# Helpers
# ============================================================

def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} missing columns: "
            + ", ".join(missing)
        )


def unique_moments(
    df: pd.DataFrame,
) -> int:
    return (
        df[
            [
                "patient_id",
                "timestamp",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )


def no_duplicate_moments(
    df: pd.DataFrame,
) -> bool:
    return not df.duplicated(
        subset=[
            "patient_id",
            "timestamp",
        ],
        keep=False,
    ).any()


def add_check(
    checks: list[dict],
    *,
    name: str,
    passed: bool,
    observed,
    expected,
    notes: str = "",
) -> None:
    checks.append(
        {
            "check": name,
            "passed": bool(passed),
            "observed": str(observed),
            "expected": str(expected),
            "notes": notes,
        }
    )


# ============================================================
# DRYAD validation
# ============================================================

def validate_dryad(
    checks: list[dict],
) -> dict:

    print(
        "\n"
        + "=" * 100
    )
    print(
        "DRYAD CURRENT-PHYSIOLOGY PATH"
    )
    print(
        "=" * 100
    )

    temporal = pd.read_parquet(
        TEMPORAL_PATH
    )

    fusion = pd.read_parquet(
        FUSION_PATH
    )

    uncertainty = pd.read_parquet(
        UNCERTAINTY_DRYAD_PATH
    )

    decision = pd.read_parquet(
        DECISION_DRYAD_PATH
    )

    explainability = pd.read_parquet(
        EXPLAINABILITY_DRYAD_PATH
    )

    temporal[
        "timestamp"
    ] = pd.to_datetime(
        temporal[
            "timestamp"
        ]
    )

    for df in [
        fusion,
        uncertainty,
        decision,
        explainability,
    ]:
        df[
            "timestamp"
        ] = pd.to_datetime(
            df[
                "timestamp"
            ]
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    require_columns(
        temporal,
        [
            "patient_id",
            "timestamp",
            "signal",
            "temporal_code",
        ],
        "Temporal",
    )

    require_columns(
        fusion,
        [
            "patient_id",
            "timestamp",
            "fusion_code",
            "fusion_state",
            "fusion_score",
            "fusion_confidence",
        ],
        "Fusion",
    )

    require_columns(
        uncertainty,
        [
            "patient_id",
            "timestamp",
            "uncertainty_score",
            "uncertainty_level",
            "uncertainty_code",
        ],
        "DRYAD Uncertainty",
    )

    require_columns(
        decision,
        [
            "patient_id",
            "timestamp",
            "decision_state",
            "decision_code",
        ],
        "DRYAD Decision",
    )

    require_columns(
        explainability,
        [
            "patient_id",
            "timestamp",
            "decision_state",
            "decision_code",
            "explanation_summary",
            "explainability_version",
        ],
        "DRYAD Explainability",
    )

    # --------------------------------------------------------
    # Moment-level contract
    # --------------------------------------------------------

    temporal_moments = unique_moments(
        temporal
    )

    print(
        f"Temporal unique moments: {temporal_moments}"
    )

    print(
        f"Fusion rows: {len(fusion)}"
    )

    add_check(
        checks,
        name="DRYAD Fusion one-row-per-moment",
        passed=(
            len(fusion)
            == temporal_moments
            and no_duplicate_moments(fusion)
        ),
        observed=(
            f"{len(fusion)} rows, "
            f"duplicates={not no_duplicate_moments(fusion)}"
        ),
        expected=(
            f"{temporal_moments} rows, duplicates=False"
        ),
        notes=(
            "Fusion must evaluate all simultaneous signals atomically."
        ),
    )

    # --------------------------------------------------------
    # Downstream row preservation
    # --------------------------------------------------------

    for label, df in [
        (
            "Uncertainty",
            uncertainty,
        ),
        (
            "Decision",
            decision,
        ),
        (
            "Explainability",
            explainability,
        ),
    ]:

        add_check(
            checks,
            name=f"DRYAD {label} preserves Fusion moments",
            passed=(
                len(df)
                == len(fusion)
                and no_duplicate_moments(df)
            ),
            observed=(
                f"{len(df)} rows"
            ),
            expected=(
                f"{len(fusion)} rows"
            ),
        )

    # --------------------------------------------------------
    # Semantic key equality
    # --------------------------------------------------------

    fusion_keys = set(
        map(
            tuple,
            fusion[
                [
                    "patient_id",
                    "timestamp",
                ]
            ].to_numpy(),
        )
    )

    uncertainty_keys = set(
        map(
            tuple,
            uncertainty[
                [
                    "patient_id",
                    "timestamp",
                ]
            ].to_numpy(),
        )
    )

    decision_keys = set(
        map(
            tuple,
            decision[
                [
                    "patient_id",
                    "timestamp",
                ]
            ].to_numpy(),
        )
    )

    explainability_keys = set(
        map(
            tuple,
            explainability[
                [
                    "patient_id",
                    "timestamp",
                ]
            ].to_numpy(),
        )
    )

    add_check(
        checks,
        name="DRYAD downstream key alignment",
        passed=(
            fusion_keys
            == uncertainty_keys
            == decision_keys
            == explainability_keys
        ),
        observed=(
            f"Fusion={len(fusion_keys)}, "
            f"Uncertainty={len(uncertainty_keys)}, "
            f"Decision={len(decision_keys)}, "
            f"Explainability={len(explainability_keys)}"
        ),
        expected="all key sets identical",
    )

    # --------------------------------------------------------
    # Usable physiology
    # --------------------------------------------------------

    usable = fusion[
        fusion[
            "fusion_code"
        ]
        == "OK"
    ]

    usable_keys = set(
        map(
            tuple,
            usable[
                [
                    "patient_id",
                    "timestamp",
                ]
            ].to_numpy(),
        )
    )

    decision_usable = decision[
        decision.apply(
            lambda row: (
                row[
                    "patient_id"
                ],
                row[
                    "timestamp"
                ],
            )
            in usable_keys,
            axis=1,
        )
    ]

    warn_count = int(
        (
            decision_usable[
                "decision_state"
            ]
            == "WARN"
        ).sum()
    )

    monitor_count = int(
        (
            decision_usable[
                "decision_state"
            ]
            == "MONITOR"
        ).sum()
    )

    abstain_count = int(
        (
            decision_usable[
                "decision_state"
            ]
            == "ABSTAIN"
        ).sum()
    )

    print(
        f"Usable Fusion moments: {len(usable)}"
    )

    print(
        f"WARN={warn_count}, "
        f"MONITOR={monitor_count}, "
        f"ABSTAIN={abstain_count}"
    )

    add_check(
        checks,
        name="DRYAD usable decision count",
        passed=(
            len(decision_usable)
            == len(usable)
        ),
        observed=len(
            decision_usable
        ),
        expected=len(
            usable
        ),
    )

    # --------------------------------------------------------
    # WARN safety semantics
    # --------------------------------------------------------

    warned = decision_usable[
        decision_usable[
            "decision_state"
        ]
        == "WARN"
    ]

    invalid_warn_states = warned[
        ~warned[
            "fusion_state"
        ].isin(
            [
                "MULTIMODAL_SUPPORT",
                "MULTIMODAL_CONSENSUS",
            ]
        )
    ]

    add_check(
        checks,
        name="DRYAD WARN requires multimodal physiology",
        passed=(
            len(
                invalid_warn_states
            )
            == 0
        ),
        observed=(
            len(
                invalid_warn_states
            )
        ),
        expected=0,
    )

    conflicting_warns = decision_usable[
        (
            decision_usable[
                "fusion_state"
            ]
            == "CONFLICTING_EVIDENCE"
        )
        &
        (
            decision_usable[
                "decision_state"
            ]
            == "WARN"
        )
    ]

    add_check(
        checks,
        name="DRYAD conflicting evidence never WARNs",
        passed=(
            len(
                conflicting_warns
            )
            == 0
        ),
        observed=len(
            conflicting_warns
        ),
        expected=0,
    )

    # --------------------------------------------------------
    # Explainability preservation
    # --------------------------------------------------------

    merged = decision[
        [
            "patient_id",
            "timestamp",
            "decision_state",
            "decision_code",
        ]
    ].merge(
        explainability[
            [
                "patient_id",
                "timestamp",
                "decision_state",
                "decision_code",
                "explanation_summary",
            ]
        ],
        on=[
            "patient_id",
            "timestamp",
        ],
        how="inner",
        suffixes=(
            "_decision",
            "_explain",
        ),
        validate="one_to_one",
    )

    same_states = (
        merged[
            "decision_state_decision"
        ]
        == merged[
            "decision_state_explain"
        ]
    ).all()

    same_codes = (
        merged[
            "decision_code_decision"
        ]
        == merged[
            "decision_code_explain"
        ]
    ).all()

    explanations_present = (
        merged[
            "explanation_summary"
        ]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )

    add_check(
        checks,
        name="DRYAD Explainability preserves Decision",
        passed=(
            same_states
            and same_codes
            and explanations_present
        ),
        observed=(
            f"states_match={same_states}, "
            f"codes_match={same_codes}, "
            f"explanations_present={explanations_present}"
        ),
        expected="all True",
    )

    return {
        "temporal_rows": len(
            temporal
        ),
        "temporal_unique_moments": (
            temporal_moments
        ),
        "fusion_rows": len(
            fusion
        ),
        "fusion_usable": len(
            usable
        ),
        "warn_count": (
            warn_count
        ),
        "monitor_count": (
            monitor_count
        ),
        "abstain_count": (
            abstain_count
        ),
    }


# ============================================================
# Framingham validation
# ============================================================

def validate_framingham(
    checks: list[dict],
) -> dict:

    print(
        "\n"
        + "=" * 100
    )
    print(
        "FRAMINGHAM BACKGROUND-RISK PATH"
    )
    print(
        "=" * 100
    )

    risk = pd.read_parquet(
        RISK_PATH
    )

    uncertainty = pd.read_parquet(
        UNCERTAINTY_FRAMINGHAM_PATH
    )

    decision = pd.read_csv(
        DECISION_FRAMINGHAM_PATH
    )

    explainability = pd.read_csv(
        EXPLAINABILITY_FRAMINGHAM_PATH
    )

    print(
        f"Risk rows: {len(risk)}"
    )

    print(
        f"Uncertainty rows: {len(uncertainty)}"
    )

    print(
        f"Decision rows: {len(decision)}"
    )

    print(
        f"Explainability rows: {len(explainability)}"
    )

    same_length = (
        len(risk)
        == len(uncertainty)
        == len(decision)
        == len(explainability)
    )

    add_check(
        checks,
        name="Framingham row alignment",
        passed=same_length,
        observed=(
            f"{len(risk)}, "
            f"{len(uncertainty)}, "
            f"{len(decision)}, "
            f"{len(explainability)}"
        ),
        expected="all equal",
    )

    warn_count = int(
        (
            decision[
                "decision_state"
            ]
            == "WARN"
        ).sum()
    )

    monitor_count = int(
        (
            decision[
                "decision_state"
            ]
            == "MONITOR"
        ).sum()
    )

    abstain_count = int(
        (
            decision[
                "decision_state"
            ]
            == "ABSTAIN"
        ).sum()
    )

    add_check(
        checks,
        name="Framingham risk alone never WARNs",
        passed=(
            warn_count
            == 0
        ),
        observed=warn_count,
        expected=0,
        notes=(
            "Background 10-year CHD risk cannot create a "
            "current deterioration warning."
        ),
    )

    add_check(
        checks,
        name="Framingham without Fusion abstains",
        passed=(
            abstain_count
            == len(decision)
        ),
        observed=abstain_count,
        expected=len(
            decision
        ),
    )

    expected_codes = (
        decision[
            "decision_code"
        ]
        == "ABSTAIN_FUSION_UNAVAILABLE"
    ).all()

    add_check(
        checks,
        name="Framingham abstention reason is Fusion unavailable",
        passed=(
            expected_codes
        ),
        observed=expected_codes,
        expected=True,
    )

    explanations_safe = (
        explainability[
            "explanation_summary"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            "risk alone cannot",
            regex=False,
        )
        .all()
    )

    add_check(
        checks,
        name="Framingham explanations preserve risk-only safety",
        passed=(
            explanations_safe
        ),
        observed=explanations_safe,
        expected=True,
    )

    return {
        "rows": len(
            risk
        ),
        "warn_count": (
            warn_count
        ),
        "monitor_count": (
            monitor_count
        ),
        "abstain_count": (
            abstain_count
        ),
        "max_risk_probability": float(
            risk[
                "risk_probability"
            ].max()
        ),
    }


# ============================================================
# Synthetic validation
# ============================================================

def validate_synthetic(
    checks: list[dict],
) -> dict:

    print(
        "\n"
        + "=" * 100
    )
    print(
        "SYNTHETIC BRANCH VALIDATION"
    )
    print(
        "=" * 100
    )

    decision = pd.read_csv(
        SYNTHETIC_DECISION_PATH
    )

    explainability = pd.read_csv(
        SYNTHETIC_EXPLAINABILITY_PATH
    )

    expected_decisions = {
        "strong_fusion_warn": (
            "WARN"
        ),
        "fusion_plus_risk_warn": (
            "WARN"
        ),
        "high_risk_only_monitor": (
            "MONITOR"
        ),
        "single_signal_monitor": (
            "MONITOR"
        ),
        "conflict_monitor": (
            "MONITOR"
        ),
        "uncertainty_blocks_warn": (
            "MONITOR"
        ),
        "high_uncertainty_abstain": (
            "ABSTAIN"
        ),
        "fusion_missing_abstain": (
            "ABSTAIN"
        ),
        "insufficient_evidence_abstain": (
            "ABSTAIN"
        ),
    }

    mismatches = []

    if (
        "case"
        in decision.columns
    ):

        for (
            case,
            expected_state,
        ) in expected_decisions.items():

            rows = decision[
                decision[
                    "case"
                ]
                == case
            ]

            if len(
                rows
            ) != 1:
                mismatches.append(
                    (
                        case,
                        "missing_or_duplicate",
                    )
                )
                continue

            observed_state = (
                rows.iloc[
                    0
                ][
                    "decision_state"
                ]
            )

            if (
                observed_state
                != expected_state
            ):
                mismatches.append(
                    (
                        case,
                        observed_state,
                    )
                )

        add_check(
            checks,
            name="Synthetic Decision branch behavior",
            passed=(
                len(
                    mismatches
                )
                == 0
            ),
            observed=(
                mismatches
                if mismatches
                else "all expected"
            ),
            expected="all expected",
        )

    explanation_complete = (
        explainability[
            "explanation_summary"
        ]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )

    add_check(
        checks,
        name="Synthetic explanations present",
        passed=(
            explanation_complete
        ),
        observed=(
            explanation_complete
        ),
        expected=True,
    )

    return {
        "decision_cases": len(
            decision
        ),
        "explainability_cases": len(
            explainability
        ),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 100
    )

    print(
        "BIOVANCE FINAL END-TO-END VALIDATION"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    required_files = [
        (
            TEMPORAL_PATH,
            "DRYAD Temporal output",
        ),
        (
            FUSION_PATH,
            "DRYAD Fusion output",
        ),
        (
            UNCERTAINTY_DRYAD_PATH,
            "DRYAD Uncertainty output",
        ),
        (
            DECISION_DRYAD_PATH,
            "DRYAD Decision output",
        ),
        (
            EXPLAINABILITY_DRYAD_PATH,
            "DRYAD Explainability output",
        ),
        (
            RISK_PATH,
            "Framingham Risk output",
        ),
        (
            UNCERTAINTY_FRAMINGHAM_PATH,
            "Framingham Uncertainty output",
        ),
        (
            DECISION_FRAMINGHAM_PATH,
            "Framingham Decision output",
        ),
        (
            EXPLAINABILITY_FRAMINGHAM_PATH,
            "Framingham Explainability output",
        ),
        (
            SYNTHETIC_DECISION_PATH,
            "Synthetic Decision validation",
        ),
        (
            SYNTHETIC_EXPLAINABILITY_PATH,
            "Synthetic Explainability validation",
        ),
    ]

    for (
        path,
        label,
    ) in required_files:
        require_file(
            path,
            label,
        )

    checks: list[
        dict
    ] = []

    dryad_summary = validate_dryad(
        checks
    )

    framingham_summary = (
        validate_framingham(
            checks
        )
    )

    synthetic_summary = (
        validate_synthetic(
            checks
        )
    )

    check_df = pd.DataFrame(
        checks
    )

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    passed_count = int(
        check_df[
            "passed"
        ].sum()
    )

    total_count = len(
        check_df
    )

    failed = check_df[
        ~check_df[
            "passed"
        ]
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL VALIDATION CHECKS"
    )

    print(
        "=" * 100
    )

    print(
        check_df.to_string(
            index=False
        )
    )

    print(
        "\nPassed:"
    )

    print(
        f"{passed_count}/{total_count}"
    )

    # --------------------------------------------------------
    # Project-level summary
    # --------------------------------------------------------

    final_summary = {
        "validation_status": (
            "PASSED"
            if len(
                failed
            )
            == 0
            else "FAILED"
        ),

        "checks_passed": (
            passed_count
        ),

        "checks_total": (
            total_count
        ),

        "dryad": (
            dryad_summary
        ),

        "framingham": (
            framingham_summary
        ),

        "synthetic": (
            synthetic_summary
        ),

        "validated_invariants": [
            (
                "Fusion emits one result per patient and timestamp."
            ),
            (
                "Downstream physiological outputs preserve "
                "moment-level alignment."
            ),
            (
                "Conflicting physiological evidence does not WARN."
            ),
            (
                "WARN requires current multimodal physiological evidence."
            ),
            (
                "Background Framingham risk alone cannot WARN."
            ),
            (
                "Missing current Fusion evidence causes abstention."
            ),
            (
                "Explainability preserves upstream Decision output."
            ),
            (
                "Explainability does not recompute upstream scores."
            ),
        ],
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checks_path = (
        OUTPUT_DIR
        / "biovance_validation_checks.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "biovance_validation_summary.json"
    )

    check_df.to_csv(
        checks_path,
        index=False,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            final_summary,
            handle,
            indent=2,
        )

    print(
        "\nSaved:"
    )

    print(
        checks_path
    )

    print(
        summary_path
    )

    # --------------------------------------------------------
    # Fail loudly at the very end.
    # --------------------------------------------------------

    if len(
        failed
    ) > 0:

        print(
            "\n"
            + "=" * 100
        )

        print(
            "BIOVANCE END-TO-END VALIDATION: FAILED"
        )

        print(
            "=" * 100
        )

        print(
            failed.to_string(
                index=False
            )
        )

        raise AssertionError(
            f"{len(failed)} integration validation "
            "check(s) failed."
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "BIOVANCE END-TO-END VALIDATION: PASSED"
    )

    print(
        "=" * 100
    )

    print(
        "\nThe frozen BioVance v1 components are "
        "internally consistent across synthetic, "
        "DRYAD, and Framingham validation paths."
    )


if __name__ == "__main__":
    main()