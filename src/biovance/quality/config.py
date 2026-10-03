"""Configuration for the BioVance Data Quality Engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Tuple

from biovance import config as C


@dataclass(frozen=True)
class SignalQualitySpec:
    """Quality rules for one physiological signal."""

    name: str
    aliases: Tuple[str, ...]

    # Number of contributing samples that counts as fully sufficient.
    # Dataset/device dependent and should ultimately be calibrated.
    target_samples: float

    # Below this count, quality is considered a hard failure.
    hard_min_samples: float = 1.0

    # Optional normalization range for a sensor-confidence field.
    # If either bound is None, confidence is not used.
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None


@dataclass(frozen=True)
class QualityConfig:
    """Configuration for deterministic observation-quality assessment."""

    column_aliases: Mapping[str, Tuple[str, ...]]
    signals: Tuple[SignalQualitySpec, ...]

    good_threshold: float = 0.80
    fair_threshold: float = 0.50

    hard_min_coverage: float = 0.20

    coverage_weight: float = 0.45
    sample_weight: float = 0.35
    confidence_weight: float = 0.00
    validity_weight: float = 0.20

    def with_(self, **kw) -> "QualityConfig":
        new = replace(self, **kw)
        new.validate()
        return new

    def signal_spec(self, signal: str) -> SignalQualitySpec:
        for spec in self.signals:
            if spec.name == signal:
                return spec

        raise ValueError(
            f"no quality configuration found for signal '{signal}'"
        )

    def validate(self) -> None:
        if not 0 <= self.fair_threshold <= 1:
            raise ValueError(
                "fair_threshold must be between 0 and 1"
            )

        if not 0 <= self.good_threshold <= 1:
            raise ValueError(
                "good_threshold must be between 0 and 1"
            )

        if self.fair_threshold > self.good_threshold:
            raise ValueError(
                "fair_threshold cannot exceed good_threshold"
            )

        if not 0 <= self.hard_min_coverage <= 1:
            raise ValueError(
                "hard_min_coverage must be between 0 and 1"
            )

        weights = (
            self.coverage_weight,
            self.sample_weight,
            self.confidence_weight,
            self.validity_weight,
        )

        if any(w < 0 for w in weights):
            raise ValueError(
                "quality component weights cannot be negative"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "at least one quality component weight must be positive"
            )

        for spec in self.signals:
            if spec.target_samples <= 0:
                raise ValueError(
                    f"{spec.name}: target_samples must be > 0"
                )

            if spec.hard_min_samples < 0:
                raise ValueError(
                    f"{spec.name}: hard_min_samples cannot be negative"
                )

            if (
                spec.confidence_min is not None
                and spec.confidence_max is not None
                and spec.confidence_max <= spec.confidence_min
            ):
                raise ValueError(
                    f"{spec.name}: confidence_max must exceed confidence_min"
                )


def load_quality_config(
    profile: str = "pmdata",
    **overrides,
) -> QualityConfig:
    """Load Data Quality Engine configuration.

    profile currently selects a dataset-specific block from quality.yaml.
    """

    raw = C._load_yaml(C.CONFIG_DIR / "quality.yaml")

    profiles = raw.get("profiles", {})

    if profile not in profiles:
        raise ValueError(
            f"unknown quality profile '{profile}'; "
            f"available profiles: {sorted(profiles)}"
        )

    p = profiles[profile]

    signals = tuple(
        SignalQualitySpec(
            name=name,
            aliases=tuple(spec.get("aliases", (name,))),
            target_samples=float(spec["target_samples"]),
            hard_min_samples=float(
                spec.get("hard_min_samples", 1)
            ),
            confidence_min=(
                None
                if spec.get("confidence_min") is None
                else float(spec["confidence_min"])
            ),
            confidence_max=(
                None
                if spec.get("confidence_max") is None
                else float(spec["confidence_max"])
            ),
        )
        for name, spec in p["signals"].items()
    )

    weights = p["weights"]
    thresholds = p["status_thresholds"]

    cfg = QualityConfig(
        column_aliases={
            key: tuple(values)
            for key, values in raw["columns"].items()
        },
        signals=signals,
        good_threshold=float(thresholds["good"]),
        fair_threshold=float(thresholds["fair"]),
        hard_min_coverage=float(p["hard_min_coverage"]),
        coverage_weight=float(weights["coverage"]),
        sample_weight=float(weights["samples"]),
        confidence_weight=float(weights["confidence"]),
        validity_weight=float(weights["validity"]),
    )

    if overrides:
        cfg = replace(cfg, **overrides)

    cfg.validate()
    return cfg