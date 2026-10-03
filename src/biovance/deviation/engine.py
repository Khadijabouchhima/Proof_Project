"""BioVance Deviation Engine.

This module answers:

    "How far is the current physiological observation from the
     person's established personalized baseline?"

It deliberately does NOT answer:

    - Is the deviation persistent?
    - Is it recurrent?
    - Is risk increasing?
    - Should a warning be generated?

Those belong to downstream engines.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from biovance.deviation.config import (
    DeviationConfig,
)

from biovance.deviation.schema import (
    normalize_deviation_input,
)


VALID_BASELINE_STATES = {
    "ESTABLISHED",
    "READY",
    "OK",
}


LEARNING_BASELINE_STATES = {
    "LEARNING",
    "BASELINE_LEARNING",
}


INSUFFICIENT_BASELINE_STATES = {
    "INSUFFICIENT",
    "INSUFFICIENT_BASELINE",
    "BASELINE_INSUFFICIENT",
}


class DeviationEngine:
    """Per-observation personalized physiological deviation."""

    def __init__(
        self,
        cfg: DeviationConfig,
    ):

        cfg.validate()

        self.cfg = cfg

    # ========================================================
    # Reliability
    # ========================================================

    def _baseline_reliability(
        self,
        baseline_n,
    ) -> float:

        if pd.isna(
            baseline_n
        ):

            return np.nan

        return float(
            np.clip(
                float(
                    baseline_n
                )
                / self.cfg.baseline_target_n,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # Magnitude bands
    # ========================================================

    def _status_from_magnitude(
        self,
        magnitude: float,
    ) -> str:

        if magnitude < self.cfg.elevated_threshold:

            return "NORMAL"

        if magnitude < self.cfg.large_threshold:

            return "ELEVATED"

        if magnitude < self.cfg.extreme_threshold:

            return "LARGE"

        return "EXTREME"

    # ========================================================
    # Eligibility / blocking state
    # ========================================================

    def _blocking_code(
        self,
        row,
        signal_spec,
    ) -> tuple[str | None, list[str]]:

        reasons = []

        # ----------------------------------------------------
        # Missing physiological observation
        # ----------------------------------------------------

        if pd.isna(
            row.value
        ):

            return (
                "MISSING_VALUE",
                [
                    "MISSING_SIGNAL_VALUE"
                ],
            )

        # ----------------------------------------------------
        # Baseline lifecycle
        # ----------------------------------------------------

        baseline_status = str(
            row.baseline_status
        ).upper()

        if (
            baseline_status
            in LEARNING_BASELINE_STATES
        ):

            return (
                "BASELINE_LEARNING",
                [
                    "BASELINE_STILL_LEARNING"
                ],
            )

        if (
            baseline_status
            in INSUFFICIENT_BASELINE_STATES
        ):

            return (
                "BASELINE_INSUFFICIENT",
                [
                    "INSUFFICIENT_BASELINE_EVIDENCE"
                ],
            )

        # ----------------------------------------------------
        # Invalid baseline statistics
        # ----------------------------------------------------

        if pd.isna(
            row.baseline_center
        ):

            return (
                "BASELINE_INSUFFICIENT",
                [
                    "MISSING_BASELINE_CENTER"
                ],
            )

        if (
            pd.isna(
                row.baseline_scale
            )
            or row.baseline_scale <= 0
        ):

            return (
                "INVALID_BASELINE",
                [
                    "INVALID_BASELINE_SCALE"
                ],
            )

        # ----------------------------------------------------
        # Explicit poor quality
        # ----------------------------------------------------

        quality_status = str(
            row.quality_status
        ).upper()

        if (
            quality_status
            in self.cfg.blocking_quality_statuses
        ):

            return (
                "POOR_QUALITY",
                [
                    "POOR_SIGNAL_QUALITY"
                ],
            )

        # ----------------------------------------------------
        # Context eligibility
        # ----------------------------------------------------

        context_state = str(
            row.context_state
        ).upper()

        if (
            context_state
            not in signal_spec.eligible_contexts
        ):

            return (
                "CONTEXT_INELIGIBLE",
                [
                    "CONTEXT_NOT_ELIGIBLE_FOR_BASELINE"
                ],
            )

        return (
            None,
            reasons,
        )

    # ========================================================
    # Explanation
    # ========================================================

    @staticmethod
    def _format_number(
        value,
        digits=1,
    ) -> str:

        if pd.isna(
            value
        ):

            return "NA"

        return f"{float(value):.{digits}f}"

    def _explanation(
        self,
        *,
        signal,
        unit,
        value,
        baseline_center,
        delta,
        deviation_score,
        direction,
        context,
        baseline_n,
        status,
    ) -> str:

        signal_name = (
            signal.upper()
        )

        if direction == "HIGH":

            direction_text = (
                "above"
            )

        elif direction == "LOW":

            direction_text = (
                "below"
            )

        else:

            direction_text = (
                "at"
            )

        delta_abs = abs(
            delta
        )

        text = (
            f"{signal_name} "
            f"{self._format_number(delta_abs)} {unit} "
            f"({self._format_number(abs(deviation_score), 2)} SD) "
            f"{direction_text} personal baseline "
            f"{self._format_number(baseline_center)} {unit}"
        )

        if context != "UNKNOWN":

            text += (
                f"; {context.lower()} context"
            )

        if not pd.isna(
            baseline_n
        ):

            text += (
                f"; baseline from "
                f"{int(baseline_n)} observations"
            )

        text += (
            f"; magnitude={status}"
        )

        return text

    # ========================================================
    # Score one row
    # ========================================================

    def _score_row(
        self,
        row,
    ) -> dict:

        signal = str(
            row.signal
        ).lower()

        if signal not in self.cfg.signals:

            return {
                "deviation_code":
                    "UNSUPPORTED_SIGNAL",

                "deviation_status":
                    "UNKNOWN",

                "reason_codes":
                    "UNSUPPORTED_SIGNAL",

                "deviation_score":
                    np.nan,

                "deviation_magnitude":
                    np.nan,

                "deviation_direction":
                    "NONE",

                "risk_aligned_score":
                    np.nan,

                "toward_risk":
                    None,

                "delta_units":
                    np.nan,

                "percent_change":
                    np.nan,

                "baseline_reliability":
                    self._baseline_reliability(
                        row.baseline_n
                    ),

                "deviation_explanation":
                    (
                        f"Unsupported signal: {signal}"
                    ),
            }

        spec = self.cfg.signals[
            signal
        ]

        # ----------------------------------------------------
        # Check whether this row is usable
        # ----------------------------------------------------

        (
            blocking_code,
            blocking_reasons,
        ) = self._blocking_code(
            row,
            spec,
        )

        reliability = (
            self._baseline_reliability(
                row.baseline_n
            )
        )

        if blocking_code is not None:

            return {
                "deviation_code":
                    blocking_code,

                "deviation_status":
                    "UNKNOWN",

                "reason_codes":
                    ";".join(
                        blocking_reasons
                    ),

                "deviation_score":
                    np.nan,

                "deviation_magnitude":
                    np.nan,

                "deviation_direction":
                    "NONE",

                "risk_aligned_score":
                    np.nan,

                "toward_risk":
                    None,

                "delta_units":
                    (
                        row.value
                        - row.baseline_center
                        if (
                            not pd.isna(row.value)
                            and not pd.isna(
                                row.baseline_center
                            )
                        )
                        else np.nan
                    ),

                "percent_change":
                    np.nan,

                "baseline_reliability":
                    reliability,

                "deviation_explanation":
                    (
                        f"Deviation unavailable: "
                        f"{blocking_code}"
                    ),
            }

        # ----------------------------------------------------
        # Physical deviation
        # ----------------------------------------------------

        delta = (
            float(
                row.value
            )
            - float(
                row.baseline_center
            )
        )

        deviation_score = (
            delta
            / float(
                row.baseline_scale
            )
        )

        deviation_magnitude = abs(
            deviation_score
        )

        # ----------------------------------------------------
        # Physical direction
        # ----------------------------------------------------

        if deviation_score > 0:

            direction = "HIGH"

        elif deviation_score < 0:

            direction = "LOW"

        else:

            direction = "NONE"

        # ----------------------------------------------------
        # Risk-aligned orientation
        # ----------------------------------------------------

        risk_aligned_score = (
            deviation_score
            * spec.orientation
        )

        toward_risk = bool(
            risk_aligned_score
            > 0
        )

        # ----------------------------------------------------
        # Descriptive magnitude band
        # ----------------------------------------------------

        deviation_status = (
            self._status_from_magnitude(
                deviation_magnitude
            )
        )

        # ----------------------------------------------------
        # Change in percent
        #
        # Appropriate for identity-transformed signals.
        # ----------------------------------------------------

        if (
            row.baseline_center != 0
            and not pd.isna(
                row.baseline_center
            )
        ):

            percent_change = (
                delta
                / abs(
                    float(
                        row.baseline_center
                    )
                )
                * 100.0
            )

        else:

            percent_change = (
                np.nan
            )

        # ----------------------------------------------------
        # Machine-readable reasons
        # ----------------------------------------------------

        reasons = [
            "BASELINE_ESTABLISHED",
        ]

        if direction == "HIGH":

            reasons.append(
                "ABOVE_PERSONAL_BASELINE"
            )

        elif direction == "LOW":

            reasons.append(
                "BELOW_PERSONAL_BASELINE"
            )

        else:

            reasons.append(
                "AT_PERSONAL_BASELINE"
            )

        if toward_risk:

            reasons.append(
                "RISK_ALIGNED_DIRECTION"
            )

        else:

            reasons.append(
                "NOT_RISK_ALIGNED_DIRECTION"
            )

        reasons.append(
            f"MAGNITUDE_{deviation_status}"
        )

        if (
            row.context_state
            != "UNKNOWN"
        ):

            reasons.append(
                f"CONTEXT_{row.context_state}"
            )

        # ----------------------------------------------------
        # Human-readable explanation
        # ----------------------------------------------------

        explanation = (
            self._explanation(
                signal=signal,
                unit=spec.unit,
                value=row.value,
                baseline_center=
                    row.baseline_center,
                delta=delta,
                deviation_score=
                    deviation_score,
                direction=direction,
                context=
                    row.context_state,
                baseline_n=
                    row.baseline_n,
                status=
                    deviation_status,
            )
        )

        return {
            "deviation_code":
                "OK",

            "deviation_status":
                deviation_status,

            "reason_codes":
                ";".join(
                    reasons
                ),

            "deviation_score":
                float(
                    deviation_score
                ),

            "deviation_magnitude":
                float(
                    deviation_magnitude
                ),

            "deviation_direction":
                direction,

            "risk_aligned_score":
                float(
                    risk_aligned_score
                ),

            "toward_risk":
                toward_risk,

            "delta_units":
                float(
                    delta
                ),

            "percent_change":
                (
                    float(
                        percent_change
                    )
                    if not pd.isna(
                        percent_change
                    )
                    else np.nan
                ),

            "baseline_reliability":
                reliability,

            "deviation_explanation":
                explanation,
        }

    # ========================================================
    # Public API
    # ========================================================

    def score(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Score per-observation deviations.

        Row count is always preserved.
        """

        obs = (
            normalize_deviation_input(
                df
            )
        )

        rows = []

        for row in obs.itertuples(
            index=False
        ):

            result = self._score_row(
                row
            )

            output = {
                "patient_id":
                    row.patient_id,

                "timestamp":
                    row.timestamp,

                "signal":
                    row.signal,

                "value":
                    row.value,

                "baseline_center":
                    row.baseline_center,

                "baseline_scale":
                    row.baseline_scale,

                "baseline_n":
                    row.baseline_n,

                "baseline_status":
                    row.baseline_status,

                "baseline_personal_weight":
                    row.baseline_personal_weight,

                "baseline_population_weight":
                    row.baseline_population_weight,

                "baseline_reliability":
                    result[
                        "baseline_reliability"
                    ],

                "quality_score":
                    row.quality_score,

                "quality_status":
                    row.quality_status,

                "context_state":
                    row.context_state,

                "context_confidence":
                    row.context_confidence,

                "delta_units":
                    result[
                        "delta_units"
                    ],

                "percent_change":
                    result[
                        "percent_change"
                    ],

                "deviation_score":
                    result[
                        "deviation_score"
                    ],

                "deviation_magnitude":
                    result[
                        "deviation_magnitude"
                    ],

                "deviation_direction":
                    result[
                        "deviation_direction"
                    ],

                "risk_aligned_score":
                    result[
                        "risk_aligned_score"
                    ],

                "toward_risk":
                    result[
                        "toward_risk"
                    ],

                "deviation_status":
                    result[
                        "deviation_status"
                    ],

                "deviation_code":
                    result[
                        "deviation_code"
                    ],

                "reason_codes":
                    result[
                        "reason_codes"
                    ],

                "deviation_explanation":
                    result[
                        "deviation_explanation"
                    ],
            }

            rows.append(
                output
            )

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # Explain
    # ========================================================

    @staticmethod
    def explain(
        scored: pd.DataFrame,
        patient_id: str,
        timestamp,
        signal: str,
    ) -> str:

        timestamp = pd.Timestamp(
            timestamp
        )

        if timestamp.tzinfo is None:

            timestamp = (
                timestamp.tz_localize(
                    "UTC"
                )
            )

        found = scored[
            (
                scored[
                    "patient_id"
                ].astype(str)
                == str(patient_id)
            )
            &
            (
                scored[
                    "timestamp"
                ]
                == timestamp
            )
            &
            (
                scored[
                    "signal"
                ].astype(str)
                == signal.lower()
            )
        ]

        if found.empty:

            return (
                "No deviation result found."
            )

        return str(
            found.iloc[0][
                "deviation_explanation"
            ]
        )

    # ========================================================
    # Serialization
    # ========================================================

    def to_json(
        self,
    ) -> str:

        payload = {

            "elevated_threshold":
                self.cfg.elevated_threshold,

            "large_threshold":
                self.cfg.large_threshold,

            "extreme_threshold":
                self.cfg.extreme_threshold,

            "baseline_target_n":
                self.cfg.baseline_target_n,

            "signals": {
                name: {
                    "unit":
                        spec.unit,

                    "orientation":
                        spec.orientation,

                    "eligible_contexts":
                        list(
                            spec.eligible_contexts
                        ),
                }

                for name, spec
                in self.cfg.signals.items()
            },
        }

        return json.dumps(
            payload,
            indent=2,
        )