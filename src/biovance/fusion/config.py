"""Configuration for BioVance Fusion Engine v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TemporalComponentWeights:
    persistence: float = 0.40
    recurrence: float = 0.25
    worsening: float = 0.20
    direction_consistency: float = 0.15

    def validate(self) -> None:
        total = (
            self.persistence
            + self.recurrence
            + self.worsening
            + self.direction_consistency
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Temporal component weights must sum to 1.0. "
                f"Observed total={total}"
            )


@dataclass(frozen=True)
class FusionWeights:
    deviation: float = 0.35
    temporal: float = 0.30
    agreement: float = 0.20
    confidence: float = 0.15

    def validate(self) -> None:
        total = (
            self.deviation
            + self.temporal
            + self.agreement
            + self.confidence
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "Fusion weights must sum to 1.0. "
                f"Observed total={total}"
            )


@dataclass(frozen=True)
class ConsensusConfig:
    min_supporting_signals: int = 2
    min_support_fraction: float = 0.67
    min_supporting_families: int = 2


@dataclass(frozen=True)
class FusionConfig:
    version: int = 1

    usable_temporal_codes: tuple[str, ...] = ("OK",)

    lookback_minutes: float = 90.0

    support_risk_aligned_z: float = 2.0
    opposition_risk_aligned_z: float = -2.0

    deviation_full_strength_z: float = 4.0

    min_available_signals: int = 2

    signal_families: dict[str, str] = field(
        default_factory=lambda: {
            "sbp": "blood_pressure",
            "dbp": "blood_pressure",
            "hr": "heart_rate",
        }
    )

    temporal_component_weights: TemporalComponentWeights = (
        TemporalComponentWeights()
    )

    fusion_weights: FusionWeights = FusionWeights()

    consensus: ConsensusConfig = ConsensusConfig()

    def validate(self) -> None:

        if self.lookback_minutes <= 0:
            raise ValueError(
                "lookback_minutes must be > 0"
            )

        if self.support_risk_aligned_z <= 0:
            raise ValueError(
                "support_risk_aligned_z must be > 0"
            )

        if self.opposition_risk_aligned_z >= 0:
            raise ValueError(
                "opposition_risk_aligned_z must be < 0"
            )

        if self.deviation_full_strength_z <= 0:
            raise ValueError(
                "deviation_full_strength_z must be > 0"
            )

        if self.min_available_signals < 1:
            raise ValueError(
                "min_available_signals must be >= 1"
            )

        if not (
            0.0
            <= self.consensus.min_support_fraction
            <= 1.0
        ):
            raise ValueError(
                "min_support_fraction must be in [0, 1]"
            )

        self.temporal_component_weights.validate()
        self.fusion_weights.validate()


def _get(
    mapping: dict[str, Any],
    key: str,
    default: Any,
) -> Any:

    value = mapping.get(
        key,
        default,
    )

    if value is None:
        return default

    return value


def load_fusion_config(
    path: str | Path,
) -> FusionConfig:

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        raw = yaml.safe_load(handle) or {}

    temporal_weights_raw = raw.get(
        "temporal_component_weights",
        {},
    )

    fusion_weights_raw = raw.get(
        "fusion_weights",
        {},
    )

    consensus_raw = raw.get(
        "consensus",
        {},
    )

    families = {
        str(k).lower(): str(v)
        for k, v in raw.get(
            "signal_families",
            {},
        ).items()
    }

    config = FusionConfig(

        version=int(
            _get(
                raw,
                "version",
                1,
            )
        ),

        usable_temporal_codes=tuple(
            str(x)
            for x in _get(
                raw,
                "usable_temporal_codes",
                ["OK"],
            )
        ),

        lookback_minutes=float(
            _get(
                raw,
                "lookback_minutes",
                90.0,
            )
        ),

        support_risk_aligned_z=float(
            _get(
                raw,
                "support_risk_aligned_z",
                2.0,
            )
        ),

        opposition_risk_aligned_z=float(
            _get(
                raw,
                "opposition_risk_aligned_z",
                -2.0,
            )
        ),

        deviation_full_strength_z=float(
            _get(
                raw,
                "deviation_full_strength_z",
                4.0,
            )
        ),

        min_available_signals=int(
            _get(
                raw,
                "min_available_signals",
                2,
            )
        ),

        signal_families=(
            families
            if families
            else {
                "sbp": "blood_pressure",
                "dbp": "blood_pressure",
                "hr": "heart_rate",
            }
        ),

        temporal_component_weights=(
            TemporalComponentWeights(

                persistence=float(
                    _get(
                        temporal_weights_raw,
                        "persistence",
                        0.40,
                    )
                ),

                recurrence=float(
                    _get(
                        temporal_weights_raw,
                        "recurrence",
                        0.25,
                    )
                ),

                worsening=float(
                    _get(
                        temporal_weights_raw,
                        "worsening",
                        0.20,
                    )
                ),

                direction_consistency=float(
                    _get(
                        temporal_weights_raw,
                        "direction_consistency",
                        0.15,
                    )
                ),
            )
        ),

        fusion_weights=FusionWeights(

            deviation=float(
                _get(
                    fusion_weights_raw,
                    "deviation",
                    0.35,
                )
            ),

            temporal=float(
                _get(
                    fusion_weights_raw,
                    "temporal",
                    0.30,
                )
            ),

            agreement=float(
                _get(
                    fusion_weights_raw,
                    "agreement",
                    0.20,
                )
            ),

            confidence=float(
                _get(
                    fusion_weights_raw,
                    "confidence",
                    0.15,
                )
            ),
        ),

        consensus=ConsensusConfig(

            min_supporting_signals=int(
                _get(
                    consensus_raw,
                    "min_supporting_signals",
                    2,
                )
            ),

            min_support_fraction=float(
                _get(
                    consensus_raw,
                    "min_support_fraction",
                    0.67,
                )
            ),

            min_supporting_families=int(
                _get(
                    consensus_raw,
                    "min_supporting_families",
                    2,
                )
            ),
        ),
    )

    config.validate()

    return config