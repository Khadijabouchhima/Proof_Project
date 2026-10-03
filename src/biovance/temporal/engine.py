"""BioVance Temporal Engine v1.

Research question
-----------------
Given a sequence of deviation observations:

    - does the deviation persist?
    - does it recur?
    - is its direction consistent?
    - is risk-aligned deviation worsening?
    - how much usable evidence supports the interpretation?

The Temporal Engine does NOT issue warnings.

Important
---------
All features at timestamp t use observations with timestamp <= t.

Future observations are never used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from biovance.temporal.config import (
    TemporalConfig,
    load_temporal_config,
)

from biovance.temporal.schema import (
    TemporalCode,
    TemporalState,
    validate_temporal_input,
)


class TemporalEngine:
    """Deterministic elapsed-time-aware temporal evidence engine."""

    def __init__(
        self,
        config: TemporalConfig,
    ):
        self.config = config
        self.config.validate()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "TemporalEngine":
        return cls(
            load_temporal_config(
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
        """Calculate row-level temporal evidence."""

        validate_temporal_input(
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
                "Temporal input contains invalid timestamps."
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

        # Preserve exact incoming row identity.
        work[
            "_temporal_original_index"
        ] = np.arange(
            len(work)
        )

        work = (
            work
            .sort_values(
                [
                    "patient_id",
                    "signal",
                    "timestamp",
                    "_temporal_original_index",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        output_rows: list[dict] = []

        for (
            patient_id,
            signal,
        ), group in work.groupby(
            [
                "patient_id",
                "signal",
            ],
            sort=False,
        ):

            group = (
                group
                .sort_values(
                    [
                        "timestamp",
                        "_temporal_original_index",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            for position in range(
                len(group)
            ):

                current = group.iloc[
                    position
                ]

                history = group.iloc[
                    : position + 1
                ].copy()

                result = self._score_row(
                    current=current,
                    history=history,
                )

                row = current.to_dict()

                row.update(
                    result
                )

                output_rows.append(
                    row
                )

        output = pd.DataFrame(
            output_rows
        )

        output = (
            output
            .sort_values(
                "_temporal_original_index"
            )
            .drop(
                columns=[
                    "_temporal_original_index"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return output

    # ========================================================
    # Current-row scoring
    # ========================================================

    def _score_row(
        self,
        current: pd.Series,
        history: pd.DataFrame,
    ) -> dict:

        # ----------------------------------------------------
        # Reference / learning rows are not temporal evaluation
        # rows when a phase column is available.
        # ----------------------------------------------------

        if (
            "phase"
            in current.index
            and pd.notna(
                current.get(
                    "phase"
                )
            )
            and str(
                current.get(
                    "phase"
                )
            ).upper()
            != "EVALUATION"
        ):

            return self._empty_result(
                code=TemporalCode.NOT_EVALUATION,
                explanation=(
                    "Temporal interpretation unavailable because "
                    "the current row is not an evaluation observation."
                ),
                reasons=[
                    "CURRENT_ROW_NOT_EVALUATION",
                ],
            )

        # ----------------------------------------------------
        # Current deviation must be usable.
        # ----------------------------------------------------

        if not self._row_usable(
            current
        ):

            return self._empty_result(
                code=(
                    TemporalCode
                    .CURRENT_DEVIATION_UNAVAILABLE
                ),
                explanation=(
                    "Temporal interpretation unavailable because "
                    "the current deviation observation is not usable."
                ),
                reasons=[
                    "CURRENT_DEVIATION_NOT_USABLE",
                ],
            )

        current_time = current[
            "timestamp"
        ]

        # ----------------------------------------------------
        # Context-aware history.
        #
        # DRYAD v1 compares WAKE temporal evidence with WAKE.
        # If context_state is absent, the engine remains generic.
        # ----------------------------------------------------

        history = self._same_context_history(
            history=history,
            current=current,
        )

        # Exclude reference-period rows from evidence.
        if "phase" in history.columns:

            history = history[
                history[
                    "phase"
                ]
                .astype(str)
                .str.upper()
                != "REFERENCE"
            ].copy()

        long_start = (
            current_time
            - pd.Timedelta(
                hours=(
                    self.config
                    .windows
                    .long_hours
                )
            )
        )

        long_history = history[
            (
                history[
                    "timestamp"
                ]
                >= long_start
            )
            &
            (
                history[
                    "timestamp"
                ]
                <= current_time
            )
        ].copy()

        medium_start = (
            current_time
            - pd.Timedelta(
                hours=(
                    self.config
                    .windows
                    .medium_hours
                )
            )
        )

        short_start = (
            current_time
            - pd.Timedelta(
                hours=(
                    self.config
                    .windows
                    .short_hours
                )
            )
        )

        usable_long = self._usable_rows(
            long_history
        )

        usable_medium = usable_long[
            usable_long[
                "timestamp"
            ]
            >= medium_start
        ].copy()

        usable_short = usable_long[
            usable_long[
                "timestamp"
            ]
            >= short_start
        ].copy()

        evidence_count = len(
            usable_long
        )

        opportunity_count = len(
            long_history
        )

        evidence_fraction = (
            evidence_count
            / opportunity_count
            if opportunity_count
            else 0.0
        )

        # ----------------------------------------------------
        # Temporal coverage / continuity
        # ----------------------------------------------------

        gap_continuity = (
            self._gap_continuity(
                usable_long
            )
        )

        # ----------------------------------------------------
        # Current elevation
        # ----------------------------------------------------

        current_elevated = bool(
            float(
                current[
                    "deviation_magnitude"
                ]
            )
            >= (
                self.config
                .elevated_magnitude_threshold
            )
        )

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        (
            persistence_count,
            persistence_duration,
        ) = self._current_elevated_run(
            usable_long
        )

        short_elevated_fraction = (
            self._elevated_fraction(
                usable_short
            )
        )

        persistence_score = (
            self._persistence_score(
                count=persistence_count,
                duration_minutes=(
                    persistence_duration
                ),
                elevated_fraction=(
                    short_elevated_fraction
                ),
            )
        )

        persistent = bool(
            current_elevated
            and
            persistence_count
            >= (
                self.config
                .persistence
                .min_elevated_observations
            )
            and
            short_elevated_fraction
            >= (
                self.config
                .persistence
                .min_elevated_fraction
            )
        )

        # ----------------------------------------------------
        # Recurrence
        # ----------------------------------------------------

        recurrence_count = (
            self._count_elevated_episodes(
                usable_long
            )
        )

        recurrence_score = float(
            np.clip(
                recurrence_count
                /
                (
                    self.config
                    .recurrence
                    .min_separate_episodes
                ),
                0.0,
                1.0,
            )
        )

        recurrent = bool(
            current_elevated
            and
            recurrence_count
            >= (
                self.config
                .recurrence
                .min_separate_episodes
            )
        )

        # ----------------------------------------------------
        # Direction consistency
        # ----------------------------------------------------

        (
            direction_consistency,
            direction_dominant,
        ) = self._direction_consistency(
            usable_medium
        )

        # ----------------------------------------------------
        # Trend
        # ----------------------------------------------------

        (
            trend_slope,
            trend_score,
        ) = self._trend(
            usable_medium
        )

        worsening_trend = bool(
            pd.notna(
                trend_score
            )
            and
            trend_score
            >= (
                self.config
                .trend
                .worsening_score_threshold
            )
        )

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        reference_reliability = (
            self._reference_reliability(
                current
            )
        )

        temporal_confidence = (
            self._temporal_confidence(
                evidence_count=evidence_count,
                evidence_fraction=(
                    evidence_fraction
                ),
                gap_continuity=(
                    gap_continuity
                ),
                reference_reliability=(
                    reference_reliability
                ),
            )
        )

        # ----------------------------------------------------
        # Evidence lifecycle
        # ----------------------------------------------------

        if (
            evidence_count
            < (
                self.config
                .confidence
                .min_observations
            )
        ):

            state = (
                TemporalState
                .INSUFFICIENT_EVIDENCE
            )

            code = (
                TemporalCode
                .INSUFFICIENT_TEMPORAL_EVIDENCE
            )

        else:

            state = self._temporal_state(
                current_elevated=(
                    current_elevated
                ),
                persistent=persistent,
                recurrent=recurrent,
                worsening=(
                    worsening_trend
                ),
            )

            code = TemporalCode.OK

        reasons = self._reason_codes(
            current_elevated=(
                current_elevated
            ),
            persistent=persistent,
            recurrent=recurrent,
            worsening=worsening_trend,
            direction_consistency=(
                direction_consistency
            ),
            evidence_count=(
                evidence_count
            ),
            evidence_fraction=(
                evidence_fraction
            ),
            gap_continuity=(
                gap_continuity
            ),
        )

        explanation = (
            self._explanation(
                state=state,
                evidence_count=(
                    evidence_count
                ),
                persistence_count=(
                    persistence_count
                ),
                persistence_duration=(
                    persistence_duration
                ),
                recurrence_count=(
                    recurrence_count
                ),
                direction_consistency=(
                    direction_consistency
                ),
                trend_score=(
                    trend_score
                ),
                confidence=(
                    temporal_confidence
                ),
            )
        )

        return {
            "temporal_state":
                state.value,

            "temporal_code":
                code.value,

            "current_elevated":
                current_elevated,

            "persistence_observation_count":
                int(
                    persistence_count
                ),

            "persistence_duration_minutes":
                float(
                    persistence_duration
                ),

            "short_elevated_fraction":
                float(
                    short_elevated_fraction
                ),

            "persistence_score":
                float(
                    persistence_score
                ),

            "recurrence_count":
                int(
                    recurrence_count
                ),

            "recurrence_score":
                float(
                    recurrence_score
                ),

            "direction_consistency":
                (
                    float(
                        direction_consistency
                    )
                    if pd.notna(
                        direction_consistency
                    )
                    else np.nan
                ),

            "direction_dominant":
                direction_dominant,

            "trend_slope_z_per_hour":
                (
                    float(
                        trend_slope
                    )
                    if pd.notna(
                        trend_slope
                    )
                    else np.nan
                ),

            "trend_score":
                (
                    float(
                        trend_score
                    )
                    if pd.notna(
                        trend_score
                    )
                    else np.nan
                ),

            "worsening_trend":
                worsening_trend,

            "evidence_count":
                int(
                    evidence_count
                ),

            "opportunity_count":
                int(
                    opportunity_count
                ),

            "evidence_fraction":
                float(
                    evidence_fraction
                ),

            "gap_continuity":
                float(
                    gap_continuity
                ),

            "temporal_confidence":
                float(
                    temporal_confidence
                ),

            "temporal_reason_codes":
                json.dumps(
                    reasons
                ),

            "temporal_explanation":
                explanation,
        }

    # ========================================================
    # Eligibility helpers
    # ========================================================

    def _row_usable(
        self,
        row: pd.Series,
    ) -> bool:

        code = str(
            row.get(
                "deviation_code",
                "",
            )
        )

        if (
            code
            not in self.config.usable_codes
        ):
            return False

        needed = (
            row.get(
                "deviation_score"
            ),
            row.get(
                "deviation_magnitude"
            ),
            row.get(
                "risk_aligned_score"
            ),
        )

        return all(
            pd.notna(
                x
            )
            for x in needed
        )

    def _usable_rows(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        usable = df[
            df[
                "deviation_code"
            ]
            .astype(str)
            .isin(
                self.config.usable_codes
            )
            &
            df[
                "deviation_score"
            ].notna()
            &
            df[
                "deviation_magnitude"
            ].notna()
            &
            df[
                "risk_aligned_score"
            ].notna()
        ].copy()

        return (
            usable
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

    def _same_context_history(
        self,
        history: pd.DataFrame,
        current: pd.Series,
    ) -> pd.DataFrame:

        if (
            "context_state"
            not in history.columns
        ):
            return history.copy()

        current_context = (
            current.get(
                "context_state"
            )
        )

        if pd.isna(
            current_context
        ):
            return history.copy()

        return history[
            history[
                "context_state"
            ]
            .astype(str)
            ==
            str(
                current_context
            )
        ].copy()

    # ========================================================
    # Elevated helpers
    # ========================================================

    def _is_elevated_series(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        return (
            df[
                "deviation_magnitude"
            ]
            >= (
                self.config
                .elevated_magnitude_threshold
            )
        )

    def _elevated_fraction(
        self,
        df: pd.DataFrame,
    ) -> float:

        if df.empty:
            return 0.0

        return float(
            self._is_elevated_series(
                df
            ).mean()
        )

    # ========================================================
    # Persistence
    # ========================================================

    def _current_elevated_run(
        self,
        usable: pd.DataFrame,
    ) -> tuple[int, float]:

        if usable.empty:
            return 0, 0.0

        usable = (
            usable
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        last = usable.iloc[-1]

        if (
            float(
                last[
                    "deviation_magnitude"
                ]
            )
            < (
                self.config
                .elevated_magnitude_threshold
            )
        ):
            return 0, 0.0

        run_indices = [
            len(usable) - 1
        ]

        current_index = (
            len(usable) - 1
        )

        while current_index > 0:

            current_row = usable.iloc[
                current_index
            ]

            previous_row = usable.iloc[
                current_index - 1
            ]

            gap_minutes = (
                (
                    current_row[
                        "timestamp"
                    ]
                    -
                    previous_row[
                        "timestamp"
                    ]
                )
                .total_seconds()
                / 60.0
            )

            if (
                gap_minutes
                > (
                    self.config
                    .gap_control
                    .max_gap_minutes
                )
            ):
                break

            if (
                float(
                    previous_row[
                        "deviation_magnitude"
                    ]
                )
                < (
                    self.config
                    .elevated_magnitude_threshold
                )
            ):
                break

            run_indices.append(
                current_index - 1
            )

            current_index -= 1

        run = usable.iloc[
            sorted(
                run_indices
            )
        ]

        count = len(
            run
        )

        if count <= 1:
            duration = 0.0

        else:
            duration = (
                (
                    run[
                        "timestamp"
                    ].iloc[-1]
                    -
                    run[
                        "timestamp"
                    ].iloc[0]
                )
                .total_seconds()
                / 60.0
            )

        return (
            int(count),
            float(duration),
        )

    def _persistence_score(
        self,
        count: int,
        duration_minutes: float,
        elevated_fraction: float,
    ) -> float:
        """Bounded descriptive persistence score.

        Components:
        - support from number of elevated observations
        - fraction elevated in short window
        - duration relative to short window

        This is an engineering evidence score, not clinical risk.
        """

        count_support = float(
            np.clip(
                count
                /
                (
                    self.config
                    .persistence
                    .min_elevated_observations
                ),
                0.0,
                1.0,
            )
        )

        short_minutes = (
            self.config
            .windows
            .short_hours
            * 60.0
        )

        duration_support = float(
            np.clip(
                duration_minutes
                / short_minutes,
                0.0,
                1.0,
            )
        )

        score = (
            0.40
            * elevated_fraction
            +
            0.35
            * count_support
            +
            0.25
            * duration_support
        )

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # Recurrence
    # ========================================================

    def _count_elevated_episodes(
        self,
        usable: pd.DataFrame,
    ) -> int:
        """Count separate elevated episodes in the current risk direction.

        Episodes are counted only when their risk-aligned direction matches
        the direction of the current elevated observation.

        This prevents a positive deviation and a later negative deviation
        from being interpreted as recurrence of the same temporal pattern.
        """

        if usable.empty:
            return 0

        usable = (
            usable
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        current = usable.iloc[-1]

        if (
            float(current["deviation_magnitude"])
            < self.config.elevated_magnitude_threshold
        ):
            return 0

        current_direction = np.sign(
            float(current["risk_aligned_score"])
        )

        if current_direction == 0:
            return 0

        episode_count = 0
        in_episode = False
        previous_elevated_time = None

        for _, row in usable.iterrows():

            magnitude_elevated = (
                float(row["deviation_magnitude"])
                >= self.config.elevated_magnitude_threshold
            )

            same_direction = (
                np.sign(
                    float(row["risk_aligned_score"])
                )
                == current_direction
            )

            elevated = (
                magnitude_elevated
                and same_direction
            )

            if not elevated:
                in_episode = False
                previous_elevated_time = None
                continue

            timestamp = row["timestamp"]

            start_new = not in_episode

            if (
                in_episode
                and previous_elevated_time is not None
            ):

                gap_minutes = (
                    (
                        timestamp
                        - previous_elevated_time
                    ).total_seconds()
                    / 60.0
                )

                if (
                    gap_minutes
                    > self.config.recurrence.episode_gap_minutes
                ):
                    start_new = True

            if start_new:
                episode_count += 1

            in_episode = True
            previous_elevated_time = timestamp

        return int(episode_count)

    # ========================================================
    # Direction consistency
    # ========================================================

    def _direction_consistency(
        self,
        usable: pd.DataFrame,
    ) -> tuple[float, str]:

        if usable.empty:
            return np.nan, "NONE"

        elevated = usable[
            self._is_elevated_series(
                usable
            )
        ].copy()

        if elevated.empty:
            return np.nan, "NONE"

        signs = np.sign(
            elevated[
                "deviation_score"
            ].astype(float)
        )

        signs = signs[
            signs != 0
        ]

        if len(signs) == 0:
            return np.nan, "NONE"

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

        dominant_count = max(
            positive,
            negative,
        )

        consistency = (
            dominant_count
            / len(signs)
        )

        if positive > negative:
            dominant = "HIGH"

        elif negative > positive:
            dominant = "LOW"

        else:
            dominant = "MIXED"

        return (
            float(consistency),
            dominant,
        )

    # ========================================================
    # Trend
    # ========================================================

    def _trend(
        self,
        usable: pd.DataFrame,
    ) -> tuple[float, float]:

        if (
            len(usable)
            < (
                self.config
                .trend
                .min_points
            )
        ):
            return np.nan, np.nan

        group = (
            usable
            .sort_values(
                "timestamp"
            )
            .copy()
        )

        start_time = (
            group[
                "timestamp"
            ].iloc[0]
        )

        hours = (
            (
                group[
                    "timestamp"
                ]
                -
                start_time
            )
            .dt
            .total_seconds()
            / 3600.0
        ).to_numpy(
            dtype=float
        )

        values = (
            group[
                "risk_aligned_score"
            ]
            .to_numpy(
                dtype=float
            )
        )

        # Degenerate timing cannot support trend.
        if (
            len(
                np.unique(
                    hours
                )
            )
            < 2
        ):
            return np.nan, np.nan

        slope = float(
            np.polyfit(
                hours,
                values,
                deg=1,
            )[0]
        )

        score = float(
            np.tanh(
                slope
                /
                (
                    self.config
                    .trend
                    .slope_scale_z_per_hour
                )
            )
        )

        return (
            slope,
            score,
        )

    # ========================================================
    # Evidence quality
    # ========================================================

    def _gap_continuity(
        self,
        usable: pd.DataFrame,
    ) -> float:

        if len(usable) <= 1:
            return 1.0

        timestamps = (
            usable[
                "timestamp"
            ]
            .sort_values()
        )

        gaps = (
            timestamps
            .diff()
            .dt
            .total_seconds()
            .div(
                60.0
            )
            .dropna()
        )

        if gaps.empty:
            return 1.0

        return float(
            (
                gaps
                <= (
                    self.config
                    .gap_control
                    .max_gap_minutes
                )
            )
            .mean()
        )

    def _reference_reliability(
        self,
        current: pd.Series,
    ) -> float:

        if (
            "reference_reliability"
            not in current.index
            or pd.isna(
                current.get(
                    "reference_reliability"
                )
            )
        ):
            return 1.0

        return float(
            np.clip(
                float(
                    current[
                        "reference_reliability"
                    ]
                ),
                0.0,
                1.0,
            )
        )

    def _temporal_confidence(
        self,
        evidence_count: int,
        evidence_fraction: float,
        gap_continuity: float,
        reference_reliability: float,
    ) -> float:

        observation_support = float(
            np.clip(
                evidence_count
                /
                (
                    self.config
                    .confidence
                    .target_observations
                ),
                0.0,
                1.0,
            )
        )

        weights = (
            self.config
            .confidence
            .weights
        )

        confidence = (
            weights.observation_support
            * observation_support

            +
            weights.evidence_fraction
            * evidence_fraction

            +
            weights.gap_continuity
            * gap_continuity

            +
            weights.reference_reliability
            * reference_reliability
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

    # ========================================================
    # State
    # ========================================================

    def _temporal_state(
        self,
        current_elevated: bool,
        persistent: bool,
        recurrent: bool,
        worsening: bool,
    ) -> TemporalState:

        if not current_elevated:
            return (
                TemporalState
                .NO_CURRENT_ELEVATION
            )

        if (
            persistent
            and recurrent
            and worsening
        ):
            return (
                TemporalState
                .PERSISTENT_RECURRENT_WORSENING
            )

        if (
            persistent
            and worsening
        ):
            return (
                TemporalState
                .PERSISTENT_WORSENING
            )

        if (
            recurrent
            and worsening
        ):
            return (
                TemporalState
                .RECURRENT_WORSENING
            )

        if (
            persistent
            and recurrent
        ):
            return (
                TemporalState
                .PERSISTENT_RECURRENT
            )

        if persistent:
            return TemporalState.PERSISTENT

        if recurrent:
            return TemporalState.RECURRENT

        if worsening:
            return TemporalState.WORSENING

        return (
            TemporalState
            .ISOLATED_ELEVATION
        )

    # ========================================================
    # Explainability
    # ========================================================

    def _reason_codes(
        self,
        current_elevated: bool,
        persistent: bool,
        recurrent: bool,
        worsening: bool,
        direction_consistency: float,
        evidence_count: int,
        evidence_fraction: float,
        gap_continuity: float,
    ) -> list[str]:

        reasons: list[str] = []

        if current_elevated:
            reasons.append(
                "CURRENT_DEVIATION_ELEVATED"
            )
        else:
            reasons.append(
                "CURRENT_DEVIATION_NOT_ELEVATED"
            )

        if persistent:
            reasons.append(
                "PERSISTENT_ELEVATION"
            )

        if recurrent:
            reasons.append(
                "RECURRENT_ELEVATION"
            )

        if worsening:
            reasons.append(
                "RISK_ALIGNED_TREND_INCREASING"
            )

        if pd.notna(
            direction_consistency
        ):

            if (
                direction_consistency
                >= (
                    self.config
                    .direction
                    .min_consistency_fraction
                )
            ):
                reasons.append(
                    "DIRECTION_CONSISTENT"
                )
            else:
                reasons.append(
                    "DIRECTION_MIXED"
                )

        if (
            evidence_count
            < (
                self.config
                .confidence
                .min_observations
            )
        ):
            reasons.append(
                "LIMITED_TEMPORAL_EVIDENCE"
            )

        if evidence_fraction < 0.5:
            reasons.append(
                "LOW_EVIDENCE_FRACTION"
            )

        if gap_continuity < 0.75:
            reasons.append(
                "TEMPORAL_GAPS_PRESENT"
            )

        return reasons

    def _explanation(
        self,
        state: TemporalState,
        evidence_count: int,
        persistence_count: int,
        persistence_duration: float,
        recurrence_count: int,
        direction_consistency: float,
        trend_score: float,
        confidence: float,
    ) -> str:

        parts = [
            f"Temporal state: {state.value}.",
            f"Usable evidence observations: {evidence_count}.",
            (
                "Current elevated run: "
                f"{persistence_count} observations over "
                f"{persistence_duration:.1f} minutes."
            ),
            (
                "Elevated episodes in the long window: "
                f"{recurrence_count}."
            ),
        ]

        if pd.notna(
            direction_consistency
        ):
            parts.append(
                "Directional consistency: "
                f"{direction_consistency:.2f}."
            )

        if pd.notna(
            trend_score
        ):
            parts.append(
                "Risk-aligned trend score: "
                f"{trend_score:.2f}."
            )

        parts.append(
            "Temporal confidence: "
            f"{confidence:.2f}."
        )

        return " ".join(
            parts
        )

    # ========================================================
    # Empty result
    # ========================================================

    def _empty_result(
        self,
        code: TemporalCode,
        explanation: str,
        reasons: list[str],
    ) -> dict:

        return {
            "temporal_state":
                TemporalState.UNKNOWN.value,

            "temporal_code":
                code.value,

            "current_elevated":
                False,

            "persistence_observation_count":
                0,

            "persistence_duration_minutes":
                0.0,

            "short_elevated_fraction":
                np.nan,

            "persistence_score":
                np.nan,

            "recurrence_count":
                0,

            "recurrence_score":
                np.nan,

            "direction_consistency":
                np.nan,

            "direction_dominant":
                "NONE",

            "trend_slope_z_per_hour":
                np.nan,

            "trend_score":
                np.nan,

            "worsening_trend":
                False,

            "evidence_count":
                0,

            "opportunity_count":
                0,

            "evidence_fraction":
                np.nan,

            "gap_continuity":
                np.nan,

            "temporal_confidence":
                0.0,

            "temporal_reason_codes":
                json.dumps(
                    reasons
                ),

            "temporal_explanation":
                explanation,
        }