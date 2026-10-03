"""BioVance Fusion Engine v1.

Fusion asks:

    Do multiple trustworthy physiological signals provide
    mutually supportive evidence at approximately the same time?

Important
---------
Fusion does NOT:

- calculate personalized baselines
- calculate deviation
- calculate temporal persistence
- calculate clinical risk
- issue WARN / MONITOR / ABSTAIN

Atomic evaluation rule
----------------------
Fusion evaluates one physiological moment at a time.

All signal rows sharing the same:

    patient_id + timestamp

are treated as simultaneous observations and are evaluated together.

This prevents the order of signal rows at the same timestamp from changing
the Fusion result.

All fusion calculations at time t use observations with timestamp <= t.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from biovance.fusion.config import (
    FusionConfig,
    load_fusion_config,
)

from biovance.fusion.schema import (
    FusionCode,
    FusionState,
    validate_fusion_input,
)


class FusionEngine:

    def __init__(
        self,
        config: FusionConfig,
    ):
        self.config = config
        self.config.validate()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "FusionEngine":
        return cls(
            load_fusion_config(
                path
            )
        )

    # ========================================================
    # Public API
    # ========================================================

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce one Fusion result per patient/timestamp.

        All signals observed at the same timestamp are considered
        simultaneously.

        This is intentionally different from scoring each long-format
        signal row independently. Fusion is a multisignal engine, so its
        natural evaluation unit is a physiological moment rather than an
        individual signal row.
        """

        validate_fusion_input(
            df
        )

        work = df.copy()

        work["timestamp"] = pd.to_datetime(
            work["timestamp"],
            errors="coerce",
        )

        if work[
            "timestamp"
        ].isna().any():
            raise ValueError(
                "Fusion input contains invalid timestamps."
            )

        work["patient_id"] = (
            work[
                "patient_id"
            ]
            .astype(str)
        )

        work["signal"] = (
            work[
                "signal"
            ]
            .astype(str)
            .str.lower()
        )

        # Stable order used only for deterministic processing.
        work[
            "_fusion_original_index"
        ] = np.arange(
            len(work)
        )

        ordered = (
            work
            .sort_values(
                [
                    "patient_id",
                    "timestamp",
                    "signal",
                    "_fusion_original_index",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        output_rows = []

        for patient_id, patient in ordered.groupby(
            "patient_id",
            sort=False,
        ):

            patient = (
                patient
                .sort_values(
                    [
                        "timestamp",
                        "signal",
                        "_fusion_original_index",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            timestamps = (
                patient[
                    "timestamp"
                ]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )

            for current_time in timestamps:

                # ------------------------------------------------
                # All rows belonging to this physiological moment.
                # ------------------------------------------------
                current_rows = patient[
                    patient[
                        "timestamp"
                    ]
                    == current_time
                ].copy()

                # ------------------------------------------------
                # IMPORTANT:
                #
                # History includes ALL observations at the current
                # timestamp, not merely the rows encountered before
                # the current signal in sort order.
                # ------------------------------------------------
                history = patient[
                    patient[
                        "timestamp"
                    ]
                    <= current_time
                ].copy()

                context = self._moment_context(
                    current_rows
                )

                result = self._score_moment(
                    current_rows=current_rows,
                    history=history,
                    current_time=current_time,
                    context=context,
                )

                current_signals = sorted(
                    current_rows[
                        "signal"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                output_row = {
                    "patient_id": patient_id,
                    "timestamp": current_time,
                    "context_state": context,
                    "current_signal_count": len(
                        current_signals
                    ),
                    "current_signals": json.dumps(
                        current_signals
                    ),
                }

                output_row.update(
                    result
                )

                output_rows.append(
                    output_row
                )

        result = pd.DataFrame(
            output_rows
        )

        if result.empty:
            return result

        result = (
            result
            .sort_values(
                [
                    "patient_id",
                    "timestamp",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # ========================================================
    # Moment scoring
    # ========================================================

    def _score_moment(
        self,
        current_rows: pd.DataFrame,
        history: pd.DataFrame,
        current_time: pd.Timestamp,
        context,
    ) -> dict:

        # ----------------------------------------------------
        # Evaluation phase
        # ----------------------------------------------------
        if "phase" in current_rows.columns:

            phases = (
                current_rows[
                    "phase"
                ]
                .dropna()
                .astype(str)
                .str.upper()
            )

            if (
                len(phases) > 0
                and not (
                    phases
                    == "EVALUATION"
                ).any()
            ):
                return self._empty_result(
                    code=FusionCode.NOT_EVALUATION,
                    explanation=(
                        "Fusion unavailable because this "
                        "physiological moment is not part of "
                        "the evaluation phase."
                    ),
                    reasons=[
                        "CURRENT_MOMENT_NOT_EVALUATION",
                    ],
                )

        # ----------------------------------------------------
        # At least one current signal must have usable
        # temporal evidence.
        # ----------------------------------------------------
        current_temporal_usable = (
            current_rows[
                "temporal_code"
            ]
            .astype(str)
            .isin(
                self.config
                .usable_temporal_codes
            )
        )

        if not current_temporal_usable.any():

            return self._empty_result(
                code=(
                    FusionCode
                    .CURRENT_TEMPORAL_UNAVAILABLE
                ),
                explanation=(
                    "Fusion unavailable because no signal "
                    "at the current physiological moment has "
                    "usable temporal evidence."
                ),
                reasons=[
                    "CURRENT_TEMPORAL_EVIDENCE_NOT_USABLE",
                ],
            )

        # ----------------------------------------------------
        # Latest usable observation for each signal.
        # ----------------------------------------------------
        evidence = self._latest_signal_evidence(
            history=history,
            current_time=current_time,
            context=context,
        )

        available_signal_count = len(
            evidence
        )

        if (
            available_signal_count
            < self.config.min_available_signals
        ):

            return self._insufficient_result(
                available_signal_count=(
                    available_signal_count
                )
            )

        evidence = evidence.copy()

        # ----------------------------------------------------
        # Per-signal components
        # ----------------------------------------------------
        evidence[
            "deviation_support"
        ] = evidence.apply(
            self._deviation_support,
            axis=1,
        )

        evidence[
            "temporal_support"
        ] = evidence.apply(
            self._temporal_support,
            axis=1,
        )

        evidence[
            "signal_confidence"
        ] = evidence.apply(
            self._signal_confidence,
            axis=1,
        )

        evidence[
            "supports_concern"
        ] = (
            evidence[
                "risk_aligned_score"
            ]
            >= (
                self.config
                .support_risk_aligned_z
            )
        )

        evidence[
            "opposes_concern"
        ] = (
            evidence[
                "risk_aligned_score"
            ]
            <= (
                self.config
                .opposition_risk_aligned_z
            )
        )

        evidence[
            "family"
        ] = evidence[
            "signal"
        ].map(
            self.config.signal_families
        )

        evidence[
            "family"
        ] = evidence[
            "family"
        ].fillna(
            evidence[
                "signal"
            ].map(
                lambda value:
                f"signal:{value}"
            )
        )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------
        supporting = evidence[
            evidence[
                "supports_concern"
            ]
        ]

        opposing = evidence[
            evidence[
                "opposes_concern"
            ]
        ]

        supporting_signal_count = len(
            supporting
        )

        opposing_signal_count = len(
            opposing
        )

        opposition_fraction = (
            opposing_signal_count
            / available_signal_count
        )

        available_family_count = (
            evidence[
                "family"
            ]
            .nunique()
        )

        supporting_family_count = (
            supporting[
                "family"
            ]
            .nunique()
        )

        support_fraction = (
            supporting_signal_count
            / available_signal_count
        )

        family_support_fraction = (
            supporting_family_count
            / available_family_count
            if available_family_count
            else 0.0
        )

        # ----------------------------------------------------
        # Direction agreement
        # ----------------------------------------------------
        direction_agreement = (
            self._direction_agreement(
                evidence
            )
        )

        # ----------------------------------------------------
        # Component aggregation
        # ----------------------------------------------------
        deviation_component = (
            self._weighted_mean(
                values=(
                    evidence[
                        "deviation_support"
                    ]
                ),
                weights=(
                    evidence[
                        "signal_confidence"
                    ]
                ),
            )
        )

        temporal_component = (
            self._weighted_mean(
                values=(
                    evidence[
                        "temporal_support"
                    ]
                ),
                weights=(
                    evidence[
                        "signal_confidence"
                    ]
                ),
            )
        )

        confidence_component = float(
            evidence[
                "signal_confidence"
            ].mean()
        )

        agreement_component = float(
            np.clip(
                (
                    0.50
                    * support_fraction
                    +
                    0.30
                    * family_support_fraction
                    +
                    0.20
                    * direction_agreement
                ),
                0.0,
                1.0,
            )
        )

        weights = (
            self.config
            .fusion_weights
        )

        fusion_evidence_strength = (
            weights.deviation
            * deviation_component

            +
            weights.temporal
            * temporal_component

            +
            weights.agreement
            * agreement_component

            +
            weights.confidence
            * confidence_component
        )

        fusion_evidence_strength = float(
            np.clip(
                fusion_evidence_strength,
                0.0,
                1.0,
            )
        )

        fusion_score = (
            fusion_evidence_strength
            * (
                1.0
                - opposition_fraction
            )
        )

        fusion_score = float(
            np.clip(
                fusion_score,
                0.0,
                1.0,
            )
        )

        # ----------------------------------------------------
        # Fusion state
        # ----------------------------------------------------
        state = self._fusion_state(
            supporting_signal_count=(
                supporting_signal_count
            ),
            opposing_signal_count=(
                opposing_signal_count
            ),
            support_fraction=(
                support_fraction
            ),
            supporting_family_count=(
                supporting_family_count
            ),
        )

        # ----------------------------------------------------
        # Reasons
        # ----------------------------------------------------
        reasons = self._reason_codes(
            available_signal_count=(
                available_signal_count
            ),
            supporting_signal_count=(
                supporting_signal_count
            ),
            opposing_signal_count=(
                opposing_signal_count
            ),
            supporting_family_count=(
                supporting_family_count
            ),
            support_fraction=(
                support_fraction
            ),
            direction_agreement=(
                direction_agreement
            ),
        )

        supporting_signals = sorted(
            supporting[
                "signal"
            ].tolist()
        )

        opposing_signals = sorted(
            opposing[
                "signal"
            ].tolist()
        )

        contributing_signals = sorted(
            evidence[
                "signal"
            ].tolist()
        )

        explanation = self._explanation(
            state=state,
            contributing_signals=(
                contributing_signals
            ),
            supporting_signals=(
                supporting_signals
            ),
            opposing_signals=(
                opposing_signals
            ),
            fusion_evidence_strength=(
                fusion_evidence_strength
            ),
            fusion_score=(
                fusion_score
            ),
            opposition_fraction=(
                opposition_fraction
            ),
            support_fraction=(
                support_fraction
            ),
            confidence=(
                confidence_component
            ),
        )

        return {
            "fusion_code":
                FusionCode.OK.value,

            "fusion_state":
                state.value,

            "fusion_evidence_strength":
                fusion_evidence_strength,

            "fusion_score":
                fusion_score,

            "opposition_fraction":
                float(
                    opposition_fraction
                ),

            "deviation_component":
                float(
                    deviation_component
                ),

            "temporal_component":
                float(
                    temporal_component
                ),

            "agreement_component":
                float(
                    agreement_component
                ),

            "fusion_confidence":
                float(
                    confidence_component
                ),

            "available_signal_count":
                int(
                    available_signal_count
                ),

            "supporting_signal_count":
                int(
                    supporting_signal_count
                ),

            "opposing_signal_count":
                int(
                    opposing_signal_count
                ),

            "available_family_count":
                int(
                    available_family_count
                ),

            "supporting_family_count":
                int(
                    supporting_family_count
                ),

            "support_fraction":
                float(
                    support_fraction
                ),

            "family_support_fraction":
                float(
                    family_support_fraction
                ),

            "direction_agreement":
                float(
                    direction_agreement
                ),

            "contributing_signals":
                json.dumps(
                    contributing_signals
                ),

            "supporting_signals":
                json.dumps(
                    supporting_signals
                ),

            "opposing_signals":
                json.dumps(
                    opposing_signals
                ),

            "fusion_reason_codes":
                json.dumps(
                    reasons
                ),

            "fusion_explanation":
                explanation,
        }

    # ========================================================
    # Moment helpers
    # ========================================================

    @staticmethod
    def _moment_context(
        current_rows: pd.DataFrame,
    ):
        """
        Return the shared context at a timestamp when it is
        unambiguous.

        If context is absent or multiple different contexts are
        recorded at exactly the same timestamp, no context filter
        is imposed.
        """

        if (
            "context_state"
            not in current_rows.columns
        ):
            return None

        values = (
            current_rows[
                "context_state"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(values) == 1:
            return values[0]

        return None

    # ========================================================
    # Evidence selection
    # ========================================================

    def _latest_signal_evidence(
        self,
        history: pd.DataFrame,
        current_time: pd.Timestamp,
        context,
    ) -> pd.DataFrame:
        """
        Select the latest usable observation per signal.

        All observations must satisfy:

            timestamp <= current_time

        so no future information can enter Fusion.
        """

        start_time = (
            current_time
            - pd.Timedelta(
                minutes=(
                    self.config
                    .lookback_minutes
                )
            )
        )

        candidates = history[
            (
                history[
                    "timestamp"
                ]
                >= start_time
            )
            &
            (
                history[
                    "timestamp"
                ]
                <= current_time
            )
            &
            (
                history[
                    "temporal_code"
                ]
                .astype(str)
                .isin(
                    self.config
                    .usable_temporal_codes
                )
            )
        ].copy()

        if (
            context is not None
            and
            pd.notna(
                context
            )
            and
            "context_state"
            in candidates.columns
        ):

            candidates = candidates[
                candidates[
                    "context_state"
                ]
                .astype(str)
                ==
                str(
                    context
                )
            ]

        if candidates.empty:
            return candidates

        candidates = (
            candidates
            .sort_values(
                [
                    "signal",
                    "timestamp",
                    "_fusion_original_index",
                ]
            )
        )

        latest = (
            candidates
            .groupby(
                "signal",
                as_index=False,
            )
            .tail(1)
        )

        return (
            latest
            .sort_values(
                "signal"
            )
            .reset_index(
                drop=True
            )
        )

    # ========================================================
    # Per-signal components
    # ========================================================

    def _deviation_support(
        self,
        row: pd.Series,
    ) -> float:

        risk_score = float(
            row[
                "risk_aligned_score"
            ]
        )

        positive = max(
            risk_score,
            0.0,
        )

        return float(
            np.clip(
                positive
                /
                (
                    self.config
                    .deviation_full_strength_z
                ),
                0.0,
                1.0,
            )
        )

    def _temporal_support(
        self,
        row: pd.Series,
    ) -> float:

        weights = (
            self.config
            .temporal_component_weights
        )

        persistence = self._bounded(
            row.get(
                "persistence_score",
                0.0,
            )
        )

        recurrence = self._bounded(
            row.get(
                "recurrence_score",
                0.0,
            )
        )

        direction = self._bounded(
            row.get(
                "direction_consistency",
                0.0,
            )
        )

        trend = row.get(
            "trend_score",
            np.nan,
        )

        if pd.isna(
            trend
        ):
            worsening = 0.0
        else:
            worsening = float(
                np.clip(
                    float(
                        trend
                    ),
                    0.0,
                    1.0,
                )
            )

        score = (
            weights.persistence
            * persistence

            +
            weights.recurrence
            * recurrence

            +
            weights.worsening
            * worsening

            +
            weights.direction_consistency
            * direction
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    def _signal_confidence(
        self,
        row: pd.Series,
    ) -> float:

        values = []

        temporal_confidence = row.get(
            "temporal_confidence",
            np.nan,
        )

        if pd.notna(
            temporal_confidence
        ):
            values.append(
                self._bounded(
                    temporal_confidence
                )
            )

        if (
            "reference_reliability"
            in row.index
            and pd.notna(
                row.get(
                    "reference_reliability"
                )
            )
        ):
            values.append(
                self._bounded(
                    row[
                        "reference_reliability"
                    ]
                )
            )

        if (
            "quality_score"
            in row.index
            and pd.notna(
                row.get(
                    "quality_score"
                )
            )
        ):
            values.append(
                self._bounded(
                    row[
                        "quality_score"
                    ]
                )
            )

        if not values:
            return 1.0

        return float(
            np.mean(
                values
            )
        )

    # ========================================================
    # Agreement
    # ========================================================

    def _direction_agreement(
        self,
        evidence: pd.DataFrame,
    ) -> float:

        elevated = evidence[
            evidence[
                "risk_aligned_score"
            ]
            .abs()
            >= (
                self.config
                .support_risk_aligned_z
            )
        ]

        if elevated.empty:
            return 0.0

        signs = np.sign(
            elevated[
                "risk_aligned_score"
            ].astype(float)
        )

        positive = int(
            (
                signs > 0
            ).sum()
        )

        negative = int(
            (
                signs < 0
            ).sum()
        )

        dominant = max(
            positive,
            negative,
        )

        return float(
            dominant
            / len(
                signs
            )
        )

    # ========================================================
    # State
    # ========================================================

    def _fusion_state(
        self,
        supporting_signal_count: int,
        opposing_signal_count: int,
        support_fraction: float,
        supporting_family_count: int,
    ) -> FusionState:

        if (
            supporting_signal_count > 0
            and
            opposing_signal_count > 0
        ):
            return (
                FusionState
                .CONFLICTING_EVIDENCE
            )

        if supporting_signal_count == 0:
            return (
                FusionState
                .NO_SUPPORT
            )

        if supporting_signal_count == 1:
            return (
                FusionState
                .SINGLE_SIGNAL_SUPPORT
            )

        if (
            supporting_family_count
            < 2
        ):
            return (
                FusionState
                .MULTISIGNAL_SAME_FAMILY
            )

        consensus = (
            self.config
            .consensus
        )

        if (
            supporting_signal_count
            >= consensus.min_supporting_signals

            and
            support_fraction
            >= consensus.min_support_fraction

            and
            supporting_family_count
            >= consensus.min_supporting_families
        ):
            return (
                FusionState
                .MULTIMODAL_CONSENSUS
            )

        return (
            FusionState
            .MULTIMODAL_SUPPORT
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _bounded(
        value,
    ) -> float:

        if pd.isna(
            value
        ):
            return 0.0

        return float(
            np.clip(
                float(
                    value
                ),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _weighted_mean(
        values: pd.Series,
        weights: pd.Series,
    ) -> float:

        values = np.asarray(
            values,
            dtype=float,
        )

        weights = np.asarray(
            weights,
            dtype=float,
        )

        valid = (
            np.isfinite(
                values
            )
            &
            np.isfinite(
                weights
            )
        )

        values = values[
            valid
        ]

        weights = weights[
            valid
        ]

        if len(
            values
        ) == 0:
            return 0.0

        if weights.sum() <= 0:
            return float(
                values.mean()
            )

        return float(
            np.average(
                values,
                weights=weights,
            )
        )

    # ========================================================
    # Explainability
    # ========================================================

    def _reason_codes(
        self,
        available_signal_count: int,
        supporting_signal_count: int,
        opposing_signal_count: int,
        supporting_family_count: int,
        support_fraction: float,
        direction_agreement: float,
    ) -> list[str]:

        reasons = []

        if available_signal_count >= 2:
            reasons.append(
                "MULTIPLE_SIGNALS_AVAILABLE"
            )

        if supporting_signal_count == 0:
            reasons.append(
                "NO_SIGNAL_SUPPORTS_CONCERN"
            )

        if supporting_signal_count == 1:
            reasons.append(
                "SINGLE_SIGNAL_SUPPORT"
            )

        if supporting_signal_count >= 2:
            reasons.append(
                "MULTISIGNAL_SUPPORT"
            )

        if supporting_family_count >= 2:
            reasons.append(
                "MULTIPLE_SIGNAL_FAMILIES_SUPPORT"
            )

        if opposing_signal_count > 0:
            reasons.append(
                "OPPOSING_SIGNAL_EVIDENCE"
            )

        if support_fraction >= 0.67:
            reasons.append(
                "HIGH_SUPPORT_FRACTION"
            )

        if direction_agreement >= 0.75:
            reasons.append(
                "HIGH_DIRECTION_AGREEMENT"
            )

        return reasons

    def _explanation(
        self,
        state: FusionState,
        contributing_signals: list[str],
        supporting_signals: list[str],
        opposing_signals: list[str],
        fusion_evidence_strength: float,
        fusion_score: float,
        opposition_fraction: float,
        support_fraction: float,
        confidence: float,
    ) -> str:

        return (
            f"Fusion state: {state.value}. "
            f"Contributing signals: "
            f"{', '.join(contributing_signals)}. "
            f"Supporting signals: "
            f"{', '.join(supporting_signals) if supporting_signals else 'none'}. "
            f"Opposing signals: "
            f"{', '.join(opposing_signals) if opposing_signals else 'none'}. "
            f"Support fraction: {support_fraction:.2f}. "
            f"Opposition fraction: {opposition_fraction:.2f}. "
            f"Evidence strength: {fusion_evidence_strength:.2f}. "
            f"Adjusted fusion score: {fusion_score:.2f}. "
            f"Fusion confidence: {confidence:.2f}."
        )

    # ========================================================
    # Empty states
    # ========================================================

    def _empty_result(
        self,
        code: FusionCode,
        explanation: str,
        reasons: list[str],
    ) -> dict:

        return {
            "fusion_code":
                code.value,

            "fusion_state":
                FusionState.UNKNOWN.value,

            "fusion_evidence_strength":
                np.nan,

            "fusion_score":
                np.nan,

            "opposition_fraction":
                np.nan,

            "deviation_component":
                np.nan,

            "temporal_component":
                np.nan,

            "agreement_component":
                np.nan,

            "fusion_confidence":
                0.0,

            "available_signal_count":
                0,

            "supporting_signal_count":
                0,

            "opposing_signal_count":
                0,

            "available_family_count":
                0,

            "supporting_family_count":
                0,

            "support_fraction":
                np.nan,

            "family_support_fraction":
                np.nan,

            "direction_agreement":
                np.nan,

            "contributing_signals":
                "[]",

            "supporting_signals":
                "[]",

            "opposing_signals":
                "[]",

            "fusion_reason_codes":
                json.dumps(
                    reasons
                ),

            "fusion_explanation":
                explanation,
        }

    def _insufficient_result(
        self,
        available_signal_count: int,
    ) -> dict:

        result = self._empty_result(
            code=(
                FusionCode
                .INSUFFICIENT_MULTISIGNAL_EVIDENCE
            ),
            explanation=(
                "Fusion requires additional usable signals "
                "within the configured lookback window."
            ),
            reasons=[
                "INSUFFICIENT_AVAILABLE_SIGNALS",
            ],
        )

        result[
            "fusion_state"
        ] = (
            FusionState
            .INSUFFICIENT_EVIDENCE
            .value
        )

        result[
            "available_signal_count"
        ] = int(
            available_signal_count
        )

        return result