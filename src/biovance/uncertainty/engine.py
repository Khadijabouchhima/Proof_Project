from __future__ import annotations

import numpy as np
import pandas as pd

from .config import UncertaintyConfig
from .schema import (
    UncertaintyCode,
    UncertaintyLevel,
)


class UncertaintyEngine:
    """
    BioVance Uncertainty Engine v1.

    Purpose
    -------
    Estimate how trustworthy the currently available
    evidence is.

    This engine does NOT:
    - predict deterioration
    - calculate long-term cardiovascular risk
    - issue WARN / MONITOR / ABSTAIN decisions

    Expected optional inputs
    ------------------------
    fusion_confidence
    fusion_state
    risk_input_completeness

    Notes
    -----
    Physiological coherence is determined from fusion_state.

    Only CONFLICTING_EVIDENCE is treated as incoherent.

    NO_SUPPORT is not considered conflict because signals may
    coherently indicate no concerning physiological support.
    """

    def __init__(
        self,
        config: UncertaintyConfig,
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

    def _level(
        self,
        uncertainty: float,
    ) -> str:
        """
        Convert uncertainty score into LOW / MODERATE / HIGH.
        """

        # Avoid floating-point boundary problems such as
        # 0.29999999999999993 instead of exactly 0.30.
        uncertainty = round(
            float(uncertainty),
            12,
        )

        if (
            uncertainty
            >= self.config.high_uncertainty_threshold
        ):
            return UncertaintyLevel.HIGH.value

        if (
            uncertainty
            >= self.config.moderate_uncertainty_threshold
        ):
            return UncertaintyLevel.MODERATE.value

        return UncertaintyLevel.LOW.value

    def score(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score uncertainty row by row.

        Missing upstream sources are allowed.

        Missing Fusion or Risk evidence lowers source coverage
        and therefore increases uncertainty.
        """

        rows = []

        for _, row in df.iterrows():

            # -------------------------------------------------
            # Source availability
            # -------------------------------------------------
            fusion_available = (
                "fusion_confidence" in df.columns
                and pd.notna(
                    row.get(
                        "fusion_confidence"
                    )
                )
            )

            risk_available = (
                "risk_input_completeness" in df.columns
                and pd.notna(
                    row.get(
                        "risk_input_completeness"
                    )
                )
            )

            available_sources = (
                int(fusion_available)
                + int(risk_available)
            )

            expected_count = len(
                self.config.expected_sources
            )

            source_coverage = (
                available_sources
                / expected_count
            )

            # -------------------------------------------------
            # Fusion confidence
            # -------------------------------------------------
            if fusion_available:
                fusion_confidence = self._clip01(
                    float(
                        row[
                            "fusion_confidence"
                        ]
                    )
                )
            else:
                fusion_confidence = 0.0

            # -------------------------------------------------
            # Risk input completeness
            # -------------------------------------------------
            if risk_available:
                risk_completeness = self._clip01(
                    float(
                        row[
                            "risk_input_completeness"
                        ]
                    )
                )
            else:
                risk_completeness = 0.0

            # -------------------------------------------------
            # Physiological coherence
            #
            # Only an explicit Fusion conflict state is treated
            # as incoherent.
            #
            # NO_SUPPORT is not conflict: signals may coherently
            # indicate no concerning physiological support.
            # -------------------------------------------------
            fusion_state = row.get(
                "fusion_state",
                None,
            )

            if not fusion_available:
                coherence = 0.0

            elif (
                pd.notna(fusion_state)
                and fusion_state == "CONFLICTING_EVIDENCE"
            ):
                coherence = 0.0

            else:
                coherence = 1.0

            # -------------------------------------------------
            # Certainty score
            # -------------------------------------------------
            certainty = (
                self.config.fusion_confidence_weight
                * fusion_confidence

                + self.config.risk_completeness_weight
                * risk_completeness

                + self.config.source_coverage_weight
                * source_coverage

                + self.config.coherence_weight
                * coherence
            )

            certainty = self._clip01(
                certainty
            )

            uncertainty = self._clip01(
                1.0 - certainty
            )

            # -------------------------------------------------
            # Reason codes
            # -------------------------------------------------
            reasons = []

            if not fusion_available:
                reasons.append(
                    "FUSION_UNAVAILABLE"
                )

            if not risk_available:
                reasons.append(
                    "RISK_UNAVAILABLE"
                )

            if (
                fusion_available
                and fusion_confidence < 0.5
            ):
                reasons.append(
                    "LOW_FUSION_CONFIDENCE"
                )

            if (
                risk_available
                and risk_completeness < 0.75
            ):
                reasons.append(
                    "LOW_RISK_INPUT_COMPLETENESS"
                )

            if (
                pd.notna(fusion_state)
                and fusion_state == "CONFLICTING_EVIDENCE"
            ):
                reasons.append(
                    "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
                )

            if not reasons:
                reasons.append(
                    "EVIDENCE_WELL_SUPPORTED"
                )

            # -------------------------------------------------
            # Evidence availability code
            # -------------------------------------------------
            if available_sources == 0:
                code = (
                    UncertaintyCode
                    .INSUFFICIENT_EVIDENCE
                    .value
                )

            elif (
                available_sources
                < expected_count
            ):
                code = (
                    UncertaintyCode
                    .PARTIAL_EVIDENCE
                    .value
                )

            else:
                code = (
                    UncertaintyCode
                    .OK
                    .value
                )

            # -------------------------------------------------
            # LOW / MODERATE / HIGH uncertainty level
            # -------------------------------------------------
            level = self._level(
                uncertainty
            )

            # -------------------------------------------------
            # Explanation
            # -------------------------------------------------
            explanation = (
                f"Evidence certainty={certainty:.3f}, "
                f"uncertainty={uncertainty:.3f}, "
                f"source coverage={source_coverage:.2f}, "
                f"coherence={coherence:.3f}. "
                f"Reasons: {', '.join(reasons)}."
            )

            # -------------------------------------------------
            # Output row
            # -------------------------------------------------
            rows.append(
                {
                    "certainty_score": certainty,
                    "uncertainty_score": uncertainty,
                    "uncertainty_level": level,
                    "uncertainty_code": code,
                    "source_coverage": source_coverage,
                    "coherence_score": coherence,
                    "uncertainty_reason_codes": "|".join(
                        reasons
                    ),
                    "uncertainty_explanation": explanation,
                    "uncertainty_version": (
                        self.config.version
                    ),
                }
            )

        return pd.DataFrame(
            rows,
            index=df.index,
        )