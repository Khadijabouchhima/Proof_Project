from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DecisionConfig
from .schema import (
    DecisionCode,
    DecisionState,
)


class DecisionEngine:
    """
    BioVance Decision Engine v1.

    Purpose
    -------
    Convert current physiological evidence,
    background cardiovascular risk, and
    uncertainty into:

        WARN
        MONITOR
        ABSTAIN

    Important
    ---------
    Background risk cannot generate WARN by itself.

    WARN always requires meaningful current
    multimodal physiological evidence.
    """

    def __init__(
        self,
        config: DecisionConfig,
    ):
        self.config = config

    @staticmethod
    def _clip01(
        value: float,
    ) -> float:
        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    def decide(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        rows = []

        for _, row in df.iterrows():

            # ---------------------------------------------
            # Read upstream evidence
            # ---------------------------------------------
            fusion_score_raw = row.get(
                "fusion_score",
                np.nan,
            )

            fusion_state = row.get(
                "fusion_state",
                None,
            )

            risk_raw = row.get(
                "risk_probability",
                np.nan,
            )

            uncertainty_raw = row.get(
                "uncertainty_score",
                np.nan,
            )

            uncertainty_code = row.get(
                "uncertainty_code",
                None,
            )

            fusion_available = pd.notna(
                fusion_score_raw
            )

            risk_available = pd.notna(
                risk_raw
            )

            uncertainty_available = pd.notna(
                uncertainty_raw
            )

            if fusion_available:
                fusion_score = self._clip01(
                    float(
                        fusion_score_raw
                    )
                )
            else:
                fusion_score = np.nan

            if risk_available:
                risk_probability = self._clip01(
                    float(
                        risk_raw
                    )
                )
            else:
                risk_probability = np.nan

            if uncertainty_available:
                uncertainty_score = self._clip01(
                    float(
                        uncertainty_raw
                    )
                )
            else:
                uncertainty_score = np.nan

            reasons = []

            # =============================================
            # RULE 1
            # Insufficient upstream evidence
            # =============================================
            if (
                uncertainty_code
                == "INSUFFICIENT_EVIDENCE"
            ):
                state = (
                    DecisionState
                    .ABSTAIN
                    .value
                )

                code = (
                    DecisionCode
                    .ABSTAIN_INSUFFICIENT_EVIDENCE
                    .value
                )

                reasons.append(
                    "INSUFFICIENT_EVIDENCE"
                )

            # =============================================
            # RULE 2
            # Fusion unavailable
            #
            # We do not WARN from background risk alone.
            # =============================================
            elif not fusion_available:
                state = (
                    DecisionState
                    .ABSTAIN
                    .value
                )

                code = (
                    DecisionCode
                    .ABSTAIN_FUSION_UNAVAILABLE
                    .value
                )

                reasons.append(
                    "FUSION_UNAVAILABLE"
                )

            # =============================================
            # RULE 3
            # Uncertainty unavailable
            #
            # Conservative behavior.
            # =============================================
            elif not uncertainty_available:
                state = (
                    DecisionState
                    .ABSTAIN
                    .value
                )

                code = (
                    DecisionCode
                    .ABSTAIN_INSUFFICIENT_EVIDENCE
                    .value
                )

                reasons.append(
                    "UNCERTAINTY_UNAVAILABLE"
                )

            # =============================================
            # RULE 4
            # High uncertainty
            # =============================================
            elif (
                uncertainty_score
                >= self.config.abstain_uncertainty
            ):
                state = (
                    DecisionState
                    .ABSTAIN
                    .value
                )

                code = (
                    DecisionCode
                    .ABSTAIN_HIGH_UNCERTAINTY
                    .value
                )

                reasons.append(
                    "HIGH_UNCERTAINTY"
                )

            else:
                # =========================================
                # Evidence is sufficiently usable.
                # =========================================

                multimodal_state = (
                    fusion_state
                    in self.config.warn_fusion_states
                )

                uncertainty_allows_warning = (
                    uncertainty_score
                    <= self.config.warn_max_uncertainty
                )

                strong_fusion = (
                    multimodal_state
                    and fusion_score
                    >= self.config.strong_fusion_warn
                )

                risk_context_present = (
                    risk_available
                    and risk_probability
                    >= self.config.background_risk_context
                )

                fusion_plus_risk = (
                    multimodal_state
                    and fusion_score
                    >= self.config.fusion_warn_with_risk
                    and risk_context_present
                )

                # =========================================
                # RULE 5
                # Strong current physiological evidence
                # =========================================
                if (
                    uncertainty_allows_warning
                    and strong_fusion
                ):
                    state = (
                        DecisionState
                        .WARN
                        .value
                    )

                    code = (
                        DecisionCode
                        .WARN_STRONG_FUSION
                        .value
                    )

                    reasons.append(
                        "STRONG_MULTIMODAL_FUSION"
                    )

                # =========================================
                # RULE 6
                # Moderate multimodal evidence supported by
                # elevated background-risk context.
                # =========================================
                elif (
                    uncertainty_allows_warning
                    and fusion_plus_risk
                ):
                    state = (
                        DecisionState
                        .WARN
                        .value
                    )

                    code = (
                        DecisionCode
                        .WARN_FUSION_WITH_RISK
                        .value
                    )

                    reasons.append(
                        "MULTIMODAL_FUSION"
                    )

                    reasons.append(
                        "BACKGROUND_RISK_CONTEXT"
                    )

                # =========================================
                # RULE 7
                # Usable evidence without enough support
                # for escalation.
                # =========================================
                else:
                    state = (
                        DecisionState
                        .MONITOR
                        .value
                    )

                    code = (
                        DecisionCode
                        .MONITOR_NO_ESCALATION
                        .value
                    )

                    reasons.append(
                        "NO_WARNING_CRITERIA_MET"
                    )

                    if (
                        fusion_state
                        == "CONFLICTING_EVIDENCE"
                    ):
                        reasons.append(
                            "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
                        )

                    if risk_context_present:
                        reasons.append(
                            "BACKGROUND_RISK_CONTEXT"
                        )

            explanation = (
                f"Decision={state}. "
                f"Fusion state={fusion_state}, "
                f"fusion score="
                f"{fusion_score if fusion_available else 'NA'}, "
                f"risk probability="
                f"{risk_probability if risk_available else 'NA'}, "
                f"uncertainty="
                f"{uncertainty_score if uncertainty_available else 'NA'}. "
                f"Reasons: {', '.join(reasons)}."
            )

            rows.append(
                {
                    "decision_state": state,
                    "decision_code": code,
                    "decision_reason_codes": "|".join(
                        reasons
                    ),
                    "decision_explanation": explanation,
                    "decision_version": (
                        self.config.version
                    ),
                }
            )

        return pd.DataFrame(
            rows,
            index=df.index,
        )