"""Configuration for the BioVance Deviation Engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from biovance import config as C


@dataclass(frozen=True)
class SignalDeviationSpec:
    """Signal-specific deviation semantics."""

    name: str
    unit: str

    # +1 = high is risk-aligned
    # -1 = low is risk-aligned
    orientation: int

    eligible_contexts: Tuple[str, ...]

    def validate(self) -> None:

        if self.orientation not in (-1, 1):
            raise ValueError(
                f"{self.name}: orientation must be +1 or -1"
            )

        if not self.eligible_contexts:
            raise ValueError(
                f"{self.name}: eligible_contexts cannot be empty"
            )


@dataclass(frozen=True)
class DeviationConfig:
    """Configuration for deviation interpretation."""

    elevated_threshold: float
    large_threshold: float
    extreme_threshold: float

    baseline_target_n: int

    blocking_quality_statuses: Tuple[str, ...]

    signals: Mapping[str, SignalDeviationSpec]

    def validate(self) -> None:

        if self.elevated_threshold <= 0:
            raise ValueError(
                "elevated_threshold must be > 0"
            )

        if (
            self.large_threshold
            <= self.elevated_threshold
        ):
            raise ValueError(
                "large_threshold must exceed elevated_threshold"
            )

        if (
            self.extreme_threshold
            <= self.large_threshold
        ):
            raise ValueError(
                "extreme_threshold must exceed large_threshold"
            )

        if self.baseline_target_n < 1:
            raise ValueError(
                "baseline_target_n must be >= 1"
            )

        for spec in self.signals.values():
            spec.validate()


def load_deviation_config() -> DeviationConfig:
    """Load configs/deviation.yaml."""

    raw = C._load_yaml(
        C.CONFIG_DIR
        / "deviation.yaml"
    )

    signals = {}

    for name, values in raw[
        "signals"
    ].items():

        signals[
            name.lower()
        ] = SignalDeviationSpec(
            name=name.lower(),

            unit=str(
                values["unit"]
            ),

            orientation=int(
                values["orientation"]
            ),

            eligible_contexts=tuple(
                str(x).upper()
                for x in values[
                    "eligible_contexts"
                ]
            ),
        )

    cfg = DeviationConfig(

        elevated_threshold=float(
            raw["bands"]["elevated"]
        ),

        large_threshold=float(
            raw["bands"]["large"]
        ),

        extreme_threshold=float(
            raw["bands"]["extreme"]
        ),

        baseline_target_n=int(
            raw["baseline"]["target_n"]
        ),

        blocking_quality_statuses=tuple(
            str(x).upper()
            for x in raw[
                "quality"
            ][
                "blocking_statuses"
            ]
        ),

        signals=signals,
    )

    cfg.validate()

    return cfg