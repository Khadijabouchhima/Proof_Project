"""Configuration for the BioVance Temporal Engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WindowConfig:
    short_hours: float = 2.0
    medium_hours: float = 6.0
    long_hours: float = 12.0


@dataclass(frozen=True)
class GapConfig:
    max_gap_minutes: float = 120.0


@dataclass(frozen=True)
class PersistenceConfig:
    min_elevated_observations: int = 2
    min_elevated_fraction: float = 0.50


@dataclass(frozen=True)
class RecurrenceConfig:
    min_separate_episodes: int = 2
    episode_gap_minutes: float = 120.0


@dataclass(frozen=True)
class DirectionConfig:
    min_consistency_fraction: float = 0.75


@dataclass(frozen=True)
class TrendConfig:
    min_points: int = 4
    min_span_minutes: float = 90.0
    slope_scale_z_per_hour: float = 0.50
    worsening_score_threshold: float = 0.50


@dataclass(frozen=True)
class ConfidenceWeights:
    observation_support: float = 0.30
    evidence_fraction: float = 0.30
    gap_continuity: float = 0.20
    reference_reliability: float = 0.20

    def validate(self) -> None:
        total = (
            self.observation_support
            + self.evidence_fraction
            + self.gap_continuity
            + self.reference_reliability
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Temporal confidence weights must sum to 1.0. "
                f"Observed total={total}"
            )


@dataclass(frozen=True)
class ConfidenceConfig:
    min_observations: int = 3
    target_observations: int = 8
    weights: ConfidenceWeights = ConfidenceWeights()


@dataclass(frozen=True)
class TemporalConfig:
    version: int = 1

    usable_codes: tuple[str, ...] = ("OK",)

    elevated_magnitude_threshold: float = 2.0

    windows: WindowConfig = WindowConfig()
    gap_control: GapConfig = GapConfig()
    persistence: PersistenceConfig = PersistenceConfig()
    recurrence: RecurrenceConfig = RecurrenceConfig()
    direction: DirectionConfig = DirectionConfig()
    trend: TrendConfig = TrendConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()

    def validate(self) -> None:
        if self.elevated_magnitude_threshold <= 0:
            raise ValueError(
                "elevated_magnitude_threshold must be > 0"
            )

        if not (
            0.0
            <= self.persistence.min_elevated_fraction
            <= 1.0
        ):
            raise ValueError(
                "min_elevated_fraction must be in [0, 1]"
            )

        if not (
            0.0
            <= self.direction.min_consistency_fraction
            <= 1.0
        ):
            raise ValueError(
                "min_consistency_fraction must be in [0, 1]"
            )

        if (
            self.windows.short_hours
            > self.windows.medium_hours
            or self.windows.medium_hours
            > self.windows.long_hours
        ):
            raise ValueError(
                "Expected short_hours <= medium_hours <= long_hours"
            )

        if self.gap_control.max_gap_minutes <= 0:
            raise ValueError(
                "max_gap_minutes must be > 0"
            )

        if self.recurrence.episode_gap_minutes <= 0:
            raise ValueError(
                "episode_gap_minutes must be > 0"
            )
        if self.trend.min_span_minutes < 0:
            raise ValueError(
                "min_span_minutes must be >= 0"
            )

        if self.trend.slope_scale_z_per_hour <= 0:
            raise ValueError(
                "slope_scale_z_per_hour must be > 0"
            )

        if self.confidence.target_observations < 1:
            raise ValueError(
                "target_observations must be >= 1"
            )

        if self.confidence.min_observations < 1:
            raise ValueError(
                "min_observations must be >= 1"
            )

        self.confidence.weights.validate()


def _get(
    mapping: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    value = mapping.get(key, default)

    if value is None:
        return default

    return value


def load_temporal_config(
    path: str | Path,
) -> TemporalConfig:
    """Load Temporal Engine YAML configuration."""

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = yaml.safe_load(handle) or {}

    windows_raw = raw.get(
        "windows",
        {},
    )

    gap_raw = raw.get(
        "gap_control",
        {},
    )

    persistence_raw = raw.get(
        "persistence",
        {},
    )

    recurrence_raw = raw.get(
        "recurrence",
        {},
    )

    direction_raw = raw.get(
        "direction",
        {},
    )

    trend_raw = raw.get(
        "trend",
        {},
    )

    confidence_raw = raw.get(
        "confidence",
        {},
    )

    confidence_weights_raw = confidence_raw.get(
        "weights",
        {},
    )

    config = TemporalConfig(

        version=int(
            _get(
                raw,
                "version",
                1,
            )
        ),

        usable_codes=tuple(
            str(x)
            for x in _get(
                raw,
                "usable_codes",
                ["OK"],
            )
        ),

        elevated_magnitude_threshold=float(
            _get(
                raw,
                "elevated_magnitude_threshold",
                2.0,
            )
        ),

        windows=WindowConfig(
            short_hours=float(
                _get(
                    windows_raw,
                    "short_hours",
                    2.0,
                )
            ),
            medium_hours=float(
                _get(
                    windows_raw,
                    "medium_hours",
                    6.0,
                )
            ),
            long_hours=float(
                _get(
                    windows_raw,
                    "long_hours",
                    12.0,
                )
            ),
        ),

        gap_control=GapConfig(
            max_gap_minutes=float(
                _get(
                    gap_raw,
                    "max_gap_minutes",
                    120.0,
                )
            ),
        ),

        persistence=PersistenceConfig(
            min_elevated_observations=int(
                _get(
                    persistence_raw,
                    "min_elevated_observations",
                    2,
                )
            ),
            min_elevated_fraction=float(
                _get(
                    persistence_raw,
                    "min_elevated_fraction",
                    0.50,
                )
            ),
        ),

        recurrence=RecurrenceConfig(
            min_separate_episodes=int(
                _get(
                    recurrence_raw,
                    "min_separate_episodes",
                    2,
                )
            ),
            episode_gap_minutes=float(
                _get(
                    recurrence_raw,
                    "episode_gap_minutes",
                    120.0,
                )
            ),
        ),

        direction=DirectionConfig(
            min_consistency_fraction=float(
                _get(
                    direction_raw,
                    "min_consistency_fraction",
                    0.75,
                )
            ),
        ),

        trend=TrendConfig(
            min_points=int(
                _get(
                    trend_raw,
                    "min_points",
                    4,
                )
            ),

            min_span_minutes=float(
                _get(
                    trend_raw,
                    "min_span_minutes",
                    90.0,
                )
            ),

            slope_scale_z_per_hour=float(
                _get(
                    trend_raw,
                    "slope_scale_z_per_hour",
                    0.50,
                )
            ),

            worsening_score_threshold=float(
                _get(
                    trend_raw,
                    "worsening_score_threshold",
                    0.50,
                )
            ),
        ),

        confidence=ConfidenceConfig(
            min_observations=int(
                _get(
                    confidence_raw,
                    "min_observations",
                    3,
                )
            ),
            target_observations=int(
                _get(
                    confidence_raw,
                    "target_observations",
                    8,
                )
            ),
            weights=ConfidenceWeights(
                observation_support=float(
                    _get(
                        confidence_weights_raw,
                        "observation_support",
                        0.30,
                    )
                ),
                evidence_fraction=float(
                    _get(
                        confidence_weights_raw,
                        "evidence_fraction",
                        0.30,
                    )
                ),
                gap_continuity=float(
                    _get(
                        confidence_weights_raw,
                        "gap_continuity",
                        0.20,
                    )
                ),
                reference_reliability=float(
                    _get(
                        confidence_weights_raw,
                        "reference_reliability",
                        0.20,
                    )
                ),
            ),
        ),
    )

    config.validate()

    return config