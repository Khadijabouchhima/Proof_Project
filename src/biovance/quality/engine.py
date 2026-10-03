"""BioVance Data Quality Engine.

The engine evaluates whether physiological observations are trustworthy.

It does NOT:
- detect physiological abnormality
- compute personalized deviation
- diagnose disease
- remove observations
- make WARN / MONITOR / ABSTAIN decisions

It only annotates observations with reliability information.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from biovance.quality.config import (
    QualityConfig,
    SignalQualitySpec,
)
from biovance.quality.schema import (
    normalize_quality_observations,
)


def _clip01(x):
    if pd.isna(x):
        return np.nan

    return float(
        np.clip(float(x), 0.0, 1.0)
    )


class DataQualityEngine:
    """Deterministic observation-level quality assessment."""

    def __init__(
        self,
        cfg: QualityConfig,
    ):
        cfg.validate()
        self.cfg = cfg

    # ---------------------------------------------------------
    # Component scoring
    # ---------------------------------------------------------

    @staticmethod
    def _coverage_score(
        coverage,
    ) -> float:
        """Coverage is assumed to already be expressed on [0, 1]."""

        return _clip01(coverage)

    @staticmethod
    def _sample_score(
        sample_count,
        spec: SignalQualitySpec,
    ) -> float:
        """Gradually reward sample sufficiency.

        target_samples receives score 1.
        Values above the target remain capped at 1.
        """

        if pd.isna(sample_count):
            return np.nan

        count = max(
            float(sample_count),
            0.0,
        )

        return float(
            np.clip(
                count / spec.target_samples,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _validity_score(
        valid,
    ) -> float:
        if pd.isna(valid):
            return np.nan

        return 1.0 if bool(valid) else 0.0

    @staticmethod
    def _confidence_score(
        confidence,
        spec: SignalQualitySpec,
    ) -> float:
        """Normalize confidence only if its scale is explicitly known."""

        if pd.isna(confidence):
            return np.nan

        if (
            spec.confidence_min is None
            or spec.confidence_max is None
        ):
            return np.nan

        normalized = (
            float(confidence)
            - spec.confidence_min
        ) / (
            spec.confidence_max
            - spec.confidence_min
        )

        return float(
            np.clip(
                normalized,
                0.0,
                1.0,
            )
        )

    # ---------------------------------------------------------
    # Weighted quality score
    # ---------------------------------------------------------

    def _weighted_score(
        self,
        coverage_score,
        sample_score,
        confidence_score,
        validity_score,
    ) -> tuple[float, int]:

        components = (
            (
                coverage_score,
                self.cfg.coverage_weight,
            ),
            (
                sample_score,
                self.cfg.sample_weight,
            ),
            (
                confidence_score,
                self.cfg.confidence_weight,
            ),
            (
                validity_score,
                self.cfg.validity_weight,
            ),
        )

        numerator = 0.0
        denominator = 0.0
        evidence_count = 0

        for value, weight in components:

            # A zero-weight component is deliberately disabled.
            if weight <= 0:
                continue

            if pd.isna(value):
                continue

            numerator += float(value) * weight
            denominator += weight
            evidence_count += 1

        if denominator <= 0:
            return 0.0, 0

        return (
            float(
                np.clip(
                    numerator / denominator,
                    0.0,
                    1.0,
                )
            ),
            evidence_count,
        )

    # ---------------------------------------------------------
    # Status
    # ---------------------------------------------------------

    def _status(
        self,
        quality_score: float,
        hard_failure: bool,
    ) -> str:

        if hard_failure:
            return "POOR"

        if quality_score >= self.cfg.good_threshold:
            return "GOOD"

        if quality_score >= self.cfg.fair_threshold:
            return "FAIR"

        return "POOR"

    # ---------------------------------------------------------
    # Reasons + hard failures
    # ---------------------------------------------------------

    def _reasons(
        self,
        row,
        spec: SignalQualitySpec,
        coverage_score,
        sample_score,
        confidence_score,
        validity_score,
    ) -> tuple[list[str], bool]:

        reasons = []
        hard_failure = False

        # Missing physiological value
        if pd.isna(row.value):
            reasons.append("missing_signal_value")
            hard_failure = True

        # Dataset explicitly states invalid
        if (
            not pd.isna(row.valid)
            and not bool(row.valid)
        ):
            reasons.append("invalid_observation")
            hard_failure = True

        # Coverage
        if pd.isna(row.coverage):
            reasons.append("coverage_unavailable")

        elif row.coverage < self.cfg.hard_min_coverage:
            reasons.append("critically_low_coverage")
            hard_failure = True

        elif coverage_score < self.cfg.good_threshold:
            reasons.append("reduced_coverage")

        # Sample count
        if pd.isna(row.sample_count):
            reasons.append("sample_count_unavailable")

        elif row.sample_count < spec.hard_min_samples:
            reasons.append("insufficient_samples")
            hard_failure = True

        elif sample_score < 1.0:
            reasons.append("below_target_sample_count")

        # Confidence
        if (
            self.cfg.confidence_weight > 0
            and pd.isna(confidence_score)
        ):
            reasons.append("confidence_unavailable")

        # No quality metadata at all
        component_values = (
            coverage_score,
            sample_score,
            confidence_score,
            validity_score,
        )

        if all(
            pd.isna(v)
            for v in component_values
        ):
            reasons.append("no_quality_evidence")
            hard_failure = True

        if not reasons:
            reasons.append("quality_evidence_adequate")

        return reasons, hard_failure

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def assess(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assess quality without deleting any observations."""

        obs = normalize_quality_observations(
            df,
            self.cfg,
        )

        rows = []

        for row in obs.itertuples(index=False):

            spec = self.cfg.signal_spec(
                row.signal
            )

            coverage_score = self._coverage_score(
                row.coverage
            )

            sample_score = self._sample_score(
                row.sample_count,
                spec,
            )

            validity_score = self._validity_score(
                row.valid
            )

            confidence_score = self._confidence_score(
                row.sensor_confidence,
                spec,
            )

            quality_score, evidence_count = (
                self._weighted_score(
                    coverage_score,
                    sample_score,
                    confidence_score,
                    validity_score,
                )
            )

            reasons, hard_failure = self._reasons(
                row,
                spec,
                coverage_score,
                sample_score,
                confidence_score,
                validity_score,
            )

            # Hard failures should never accidentally receive
            # a high quality score merely because other components
            # were strong.
            if hard_failure:
                quality_score = min(
                    quality_score,
                    self.cfg.fair_threshold - 1e-6,
                )

            status = self._status(
                quality_score,
                hard_failure,
            )

            rows.append({
                "patient_id": row.patient_id,
                "timestamp": row.timestamp,
                "signal": row.signal,
                "value": row.value,

                "coverage": row.coverage,
                "sample_count": row.sample_count,
                "sensor_confidence": row.sensor_confidence,
                "valid": row.valid,

                "coverage_score": coverage_score,
                "sample_score": sample_score,
                "confidence_score": confidence_score,
                "validity_score": validity_score,

                "quality_score": quality_score,
                "quality_status": status,
                "quality_evidence_count": evidence_count,

                # String rather than Python list makes the output
                # easy to save to CSV/Parquet.
                "quality_reasons": ";".join(reasons),
            })

        return pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Explainability
    # ---------------------------------------------------------

    @staticmethod
    def explain(
        assessed: pd.DataFrame,
        patient_id: str,
        timestamp,
        signal: str,
    ) -> str:

        timestamp = pd.Timestamp(timestamp)

        r = assessed[
            (assessed["patient_id"].astype(str) == str(patient_id))
            & (assessed["timestamp"] == timestamp)
            & (assessed["signal"] == signal)
        ]

        if r.empty:
            return (
                f"{signal}: no quality assessment found "
                f"for {patient_id} at {timestamp}"
            )

        row = r.iloc[0]

        return (
            f"{signal} quality={row.quality_score:.2f} "
            f"({row.quality_status}); "
            f"coverage={row.coverage_score:.2f} "
            f"sample={row.sample_score:.2f} "
            f"validity={row.validity_score:.2f}; "
            f"reasons={row.quality_reasons}"
        )

    # ---------------------------------------------------------
    # Configuration serialization
    # ---------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the non-learned engine settings."""

        return json.dumps(
            {
                "good_threshold": self.cfg.good_threshold,
                "fair_threshold": self.cfg.fair_threshold,
                "hard_min_coverage": self.cfg.hard_min_coverage,
                "weights": {
                    "coverage": self.cfg.coverage_weight,
                    "samples": self.cfg.sample_weight,
                    "confidence": self.cfg.confidence_weight,
                    "validity": self.cfg.validity_weight,
                },
            },
            indent=2,
        )