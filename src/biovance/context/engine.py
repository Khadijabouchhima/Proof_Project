"""BioVance Context Engine.

The engine determines observational context:

    SLEEP
    REST
    ACTIVE
    UNKNOWN

Gyroscope movement thresholds can be learned from a historical
reference period and personalized per participant.

The engine does NOT diagnose disease and does NOT determine whether
physiological signals are abnormal.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from biovance.context.config import (
    ContextConfig,
)

from biovance.context.schema import (
    normalize_context_observations,
)


def _clip01(
    value,
) -> float:

    if pd.isna(value):
        return np.nan

    return float(
        np.clip(
            float(value),
            0.0,
            1.0,
        )
    )


class ContextEngine:

    def __init__(
        self,
        cfg: ContextConfig,
    ):

        cfg.validate()

        self.cfg = cfg

        self._fitted = False

        self.global_rest_gyro = (
            cfg.fallback_rest_gyro
        )

        self.global_active_gyro = (
            cfg.fallback_active_gyro
        )

        self.global_reference_n = 0

        self.patient_thresholds: dict[
            str,
            dict,
        ] = {}

    # ========================================================
    # Fit personalized movement thresholds
    # ========================================================

    def fit(
        self,
        reference_df: pd.DataFrame,
    ) -> "ContextEngine":
        """Fit gyro thresholds using historical reference data only.

        Only windows with no sleep overlap and valid gyro magnitude
        are used.

        The fitted thresholds must subsequently be frozen before
        classifying future observations.
        """

        obs = normalize_context_observations(
            reference_df,
            self.cfg,
        )

        awake = obs[
            (
                obs[
                    "sleep_overlap_fraction"
                ].notna()
            )
            &
            (
                obs[
                    "sleep_overlap_fraction"
                ]
                <= self.cfg.awake_overlap_max
            )
            &
            (
                obs[
                    "gyro_magnitude"
                ].notna()
            )
        ].copy()

        # ----------------------------------------------------
        # Global thresholds learned from reference data
        # ----------------------------------------------------

        n_global = len(
            awake
        )

        self.global_reference_n = (
            n_global
        )

        if (
            n_global
            >= self.cfg.min_global_awake_windows
        ):

            global_rest = float(
                awake[
                    "gyro_magnitude"
                ].quantile(
                    self.cfg.rest_gyro_quantile
                )
            )

            global_active = float(
                awake[
                    "gyro_magnitude"
                ].quantile(
                    self.cfg.active_gyro_quantile
                )
            )

            if (
                global_active
                >
                global_rest
                + self.cfg.min_gyro_threshold_gap
            ):

                self.global_rest_gyro = (
                    global_rest
                )

                self.global_active_gyro = (
                    global_active
                )

        # ----------------------------------------------------
        # Participant-specific thresholds
        # ----------------------------------------------------

        self.patient_thresholds = {}

        for patient_id, group in awake.groupby(
            "patient_id"
        ):

            values = (
                group[
                    "gyro_magnitude"
                ]
                .dropna()
            )

            n = len(
                values
            )

            # -----------------------------------------------
            # Too little personal evidence:
            # global fallback
            # -----------------------------------------------

            if (
                n
                < self.cfg.min_personal_awake_windows
            ):

                self.patient_thresholds[
                    str(patient_id)
                ] = {
                    "rest": self.global_rest_gyro,
                    "active": self.global_active_gyro,
                    "n": n,
                    "source": "global_fallback",
                    "personal_weight": 0.0,
                }

                continue

            personal_rest = float(
                values.quantile(
                    self.cfg.rest_gyro_quantile
                )
            )

            personal_active = float(
                values.quantile(
                    self.cfg.active_gyro_quantile
                )
            )

            # -----------------------------------------------
            # Invalid / collapsed personal thresholds
            # -----------------------------------------------

            if (
                personal_active
                <= personal_rest
                + self.cfg.min_gyro_threshold_gap
            ):

                self.patient_thresholds[
                    str(patient_id)
                ] = {
                    "rest": self.global_rest_gyro,
                    "active": self.global_active_gyro,
                    "n": n,
                    "source": "global_fallback",
                    "personal_weight": 0.0,
                }

                continue

            # -----------------------------------------------
            # Empirical shrinkage
            #
            # n / (n + k)
            # -----------------------------------------------

            if (
                self.cfg.shrinkage_strength
                == 0
            ):

                weight = 1.0

            else:

                weight = (
                    n
                    /
                    (
                        n
                        + self.cfg.shrinkage_strength
                    )
                )

            shrunk_rest = (
                weight
                * personal_rest
                +
                (
                    1.0
                    - weight
                )
                * self.global_rest_gyro
            )

            shrunk_active = (
                weight
                * personal_active
                +
                (
                    1.0
                    - weight
                )
                * self.global_active_gyro
            )

            if (
                shrunk_active
                <= shrunk_rest
                + self.cfg.min_gyro_threshold_gap
            ):

                shrunk_rest = (
                    self.global_rest_gyro
                )

                shrunk_active = (
                    self.global_active_gyro
                )

                source = (
                    "global_fallback"
                )

                weight = 0.0

            else:

                source = (
                    "personalized_shrunk"
                )

            self.patient_thresholds[
                str(patient_id)
            ] = {
                "rest": float(
                    shrunk_rest
                ),
                "active": float(
                    shrunk_active
                ),
                "n": n,
                "source": source,
                "personal_weight": float(
                    weight
                ),
            }

        self._fitted = True

        return self

    # ========================================================
    # Threshold lookup
    # ========================================================

    def _thresholds_for(
        self,
        patient_id,
    ) -> dict:

        patient_id = str(
            patient_id
        )

        if (
            self.cfg.personalize_gyro
            and not self._fitted
        ):

            raise RuntimeError(
                "ContextEngine must be fitted before classification "
                "when personalize_gyro=True."
            )

        if not self.cfg.personalize_gyro:

            return {
                "rest": self.cfg.fallback_rest_gyro,
                "active": self.cfg.fallback_active_gyro,
                "n": 0,
                "source": "configured_global",
                "personal_weight": 0.0,
            }

        return self.patient_thresholds.get(
            patient_id,
            {
                "rest": self.global_rest_gyro,
                "active": self.global_active_gyro,
                "n": 0,
                "source": "global_unseen_patient",
                "personal_weight": 0.0,
            },
        )

    # ========================================================
    # Activity scores
    # ========================================================

    def _gyro_activity_score(
        self,
        gyro,
        active_threshold,
    ):

        if pd.isna(
            gyro
        ):

            return np.nan

        return _clip01(
            float(
                gyro
            )
            / float(
                active_threshold
            )
        )

    def _steps_activity_score(
        self,
        steps,
    ):

        if pd.isna(
            steps
        ):

            return np.nan

        return _clip01(
            float(
                steps
            )
            / self.cfg.steps_activity_reference
        )

    def _dynamic_acc_activity_score(
        self,
        dynamic_acc,
    ):

        if pd.isna(
            dynamic_acc
        ):

            return np.nan

        return _clip01(
            float(
                dynamic_acc
            )
            / self.cfg.dynamic_acc_activity_reference
        )

    def _activity_level(
        self,
        gyro_score,
        steps_score,
        dynamic_score,
    ):

        components = [
            (
                gyro_score,
                self.cfg.gyro_weight,
            ),
            (
                steps_score,
                self.cfg.steps_weight,
            ),
        ]

        if self.cfg.use_dynamic_acc:

            components.append(
                (
                    dynamic_score,
                    self.cfg.dynamic_acc_weight,
                )
            )

        numerator = 0.0
        denominator = 0.0
        evidence_count = 0

        for value, weight in components:

            if weight <= 0:
                continue

            if pd.isna(
                value
            ):
                continue

            numerator += (
                float(
                    value
                )
                * weight
            )

            denominator += weight

            evidence_count += 1

        if denominator == 0:

            return (
                np.nan,
                0,
            )

        return (
            _clip01(
                numerator
                / denominator
            ),
            evidence_count,
        )

    # ========================================================
    # Classification
    # ========================================================

    def _classify_row(
        self,
        row,
    ) -> dict:

        thresholds = (
            self._thresholds_for(
                row.patient_id
            )
        )

        rest_threshold = (
            thresholds[
                "rest"
            ]
        )

        active_threshold = (
            thresholds[
                "active"
            ]
        )

        gyro_score = (
            self._gyro_activity_score(
                row.gyro_magnitude,
                active_threshold,
            )
        )

        steps_score = (
            self._steps_activity_score(
                row.steps
            )
        )

        dynamic_score = (
            self._dynamic_acc_activity_score(
                row.dynamic_acc
            )
        )

        (
            activity_level,
            activity_evidence_count,
        ) = self._activity_level(
            gyro_score,
            steps_score,
            dynamic_score,
        )

        sleep_overlap = (
            row.sleep_overlap_fraction
        )

        # ----------------------------------------------------
        # SLEEP
        # ----------------------------------------------------

        if (
            not pd.isna(
                sleep_overlap
            )
            and sleep_overlap
            >= self.cfg.sleep_overlap_threshold
        ):

            return {
                "context_state": "SLEEP",
                "activity_level": activity_level,
                "context_confidence":
                    _clip01(
                        sleep_overlap
                    ),
                "context_reasons":
                    "strong_sleep_diary_overlap",
                "activity_evidence_count":
                    activity_evidence_count,
                **thresholds,
            }

        # ----------------------------------------------------
        # PARTIAL SLEEP -> UNKNOWN
        # ----------------------------------------------------

        if (
            self.cfg.partial_sleep_is_unknown
            and not pd.isna(
                sleep_overlap
            )
            and 0
            < sleep_overlap
            < self.cfg.sleep_overlap_threshold
        ):

            return {
                "context_state": "UNKNOWN",
                "activity_level": activity_level,
                "context_confidence":
                    min(
                        float(
                            sleep_overlap
                        ),
                        self.cfg.unknown_confidence_cap,
                    ),
                "context_reasons":
                    "partial_sleep_overlap",
                "activity_evidence_count":
                    activity_evidence_count,
                **thresholds,
            }

        # ----------------------------------------------------
        # Movement evidence
        # ----------------------------------------------------

        active_reasons = []
        rest_conditions = []
        rest_reasons = []

        # Gyro
        if not pd.isna(
            row.gyro_magnitude
        ):

            if (
                row.gyro_magnitude
                >= active_threshold
            ):

                active_reasons.append(
                    "high_gyro_motion"
                )

            low = (
                row.gyro_magnitude
                <= rest_threshold
            )

            rest_conditions.append(
                low
            )

            if low:

                rest_reasons.append(
                    "low_gyro_motion"
                )

        # Steps
        if not pd.isna(
            row.steps
        ):

            if (
                row.steps
                >= self.cfg.active_steps_min
            ):

                active_reasons.append(
                    "steps_present"
                )

            low_steps = (
                row.steps
                <= self.cfg.rest_steps_max
            )

            rest_conditions.append(
                low_steps
            )

            if low_steps:

                rest_reasons.append(
                    "no_steps"
                )

        # Optional dynamic acceleration
        if (
            self.cfg.use_dynamic_acc
            and not pd.isna(
                row.dynamic_acc
            )
        ):

            if (
                row.dynamic_acc
                >= self.cfg.active_dynamic_acc_min
            ):

                active_reasons.append(
                    "high_dynamic_acceleration"
                )

            low_dynamic = (
                row.dynamic_acc
                <= self.cfg.rest_dynamic_acc_max
            )

            rest_conditions.append(
                low_dynamic
            )

            if low_dynamic:

                rest_reasons.append(
                    "low_dynamic_acceleration"
                )

        # ----------------------------------------------------
        # ACTIVE
        # ----------------------------------------------------

        if active_reasons:

            agreeing_sources = len(
                active_reasons
            )

            confidence = (
                0.85
                if agreeing_sources >= 2
                else self.cfg.single_source_confidence_cap
            )

            return {
                "context_state": "ACTIVE",
                "activity_level": activity_level,
                "context_confidence":
                    _clip01(
                        confidence
                    ),
                "context_reasons":
                    ";".join(
                        active_reasons
                    ),
                "activity_evidence_count":
                    activity_evidence_count,
                **thresholds,
            }

        # ----------------------------------------------------
        # REST
        # ----------------------------------------------------

        if (
            rest_conditions
            and all(
                rest_conditions
            )
        ):

            confidence = (
                0.80
                if len(
                    rest_conditions
                ) >= 2
                else self.cfg.single_source_confidence_cap
            )

            return {
                "context_state": "REST",
                "activity_level": activity_level,
                "context_confidence":
                    confidence,
                "context_reasons":
                    ";".join(
                        rest_reasons
                    ),
                "activity_evidence_count":
                    activity_evidence_count,
                **thresholds,
            }

        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        if (
            activity_evidence_count
            == 0
        ):

            reason = (
                "insufficient_motion_evidence"
            )

            confidence = 0.0

        else:

            reason = (
                "ambiguous_motion_evidence"
            )

            confidence = min(
                0.35,
                self.cfg.unknown_confidence_cap,
            )

        return {
            "context_state": "UNKNOWN",
            "activity_level": activity_level,
            "context_confidence": confidence,
            "context_reasons": reason,
            "activity_evidence_count":
                activity_evidence_count,
            **thresholds,
        }

    # ========================================================
    # Public classification API
    # ========================================================

    def classify(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        obs = normalize_context_observations(
            df,
            self.cfg,
        )

        rows = []

        for row in obs.itertuples(
            index=False
        ):

            r = self._classify_row(
                row
            )

            rows.append(
                {
                    "patient_id":
                        row.patient_id,

                    "timestamp_start":
                        row.timestamp_start,

                    "timestamp_end":
                        row.timestamp_end,

                    "sleep_overlap_fraction":
                        row.sleep_overlap_fraction,

                    "gyro_magnitude":
                        row.gyro_magnitude,

                    "dynamic_acc":
                        row.dynamic_acc,

                    "steps":
                        row.steps,

                    "gyro_rest_threshold":
                        r["rest"],

                    "gyro_active_threshold":
                        r["active"],

                    "gyro_reference_n":
                        r["n"],

                    "gyro_threshold_source":
                        r["source"],

                    "gyro_personal_weight":
                        r["personal_weight"],

                    "activity_level":
                        r["activity_level"],

                    "context_state":
                        r["context_state"],

                    "context_confidence":
                        r["context_confidence"],

                    "activity_evidence_count":
                        r[
                            "activity_evidence_count"
                        ],

                    "context_reasons":
                        r["context_reasons"],
                }
            )

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # Threshold summary
    # ========================================================

    def threshold_table(
        self,
    ) -> pd.DataFrame:

        rows = []

        for patient_id, x in sorted(
            self.patient_thresholds.items()
        ):

            rows.append(
                {
                    "patient_id":
                        patient_id,

                    "gyro_rest_threshold":
                        x["rest"],

                    "gyro_active_threshold":
                        x["active"],

                    "gyro_reference_n":
                        x["n"],

                    "gyro_threshold_source":
                        x["source"],

                    "gyro_personal_weight":
                        x["personal_weight"],
                }
            )

        return pd.DataFrame(
            rows
        )

    # ========================================================
    # Explain
    # ========================================================

    @staticmethod
    def explain(
        classified,
        patient_id,
        timestamp,
    ):

        timestamp = pd.Timestamp(
            timestamp
        )

        if timestamp.tzinfo is None:

            timestamp = (
                timestamp.tz_localize(
                    "UTC"
                )
            )

        r = classified[
            (
                classified[
                    "patient_id"
                ].astype(str)
                == str(
                    patient_id
                )
            )
            &
            (
                classified[
                    "timestamp_start"
                ]
                == timestamp
            )
        ]

        if r.empty:

            return (
                "no context classification found"
            )

        row = r.iloc[0]

        return (
            f"context={row.context_state}; "
            f"confidence={row.context_confidence:.2f}; "
            f"activity_level={row.activity_level:.2f}; "
            f"gyro_rest_threshold="
            f"{row.gyro_rest_threshold:.3f}; "
            f"gyro_active_threshold="
            f"{row.gyro_active_threshold:.3f}; "
            f"threshold_source="
            f"{row.gyro_threshold_source}; "
            f"reasons={row.context_reasons}"
        )

    # ========================================================
    # Serialization
    # ========================================================

    def to_json(
        self,
    ) -> str:

        payload = {
            "fitted":
                self._fitted,

            "global_rest_gyro":
                self.global_rest_gyro,

            "global_active_gyro":
                self.global_active_gyro,

            "global_reference_n":
                self.global_reference_n,

            "patient_thresholds":
                self.patient_thresholds,
        }

        return json.dumps(
            payload,
            indent=2,
        )