from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class UncertaintyConfig:
    version: str

    fusion_confidence_weight: float
    risk_completeness_weight: float
    source_coverage_weight: float
    coherence_weight: float

    expected_sources: tuple[str, ...]

    high_uncertainty_threshold: float
    moderate_uncertainty_threshold: float


def load_uncertainty_config(
    path: str | Path = "configs/uncertainty.yaml",
) -> UncertaintyConfig:

    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    weights = raw["weights"]
    thresholds = raw["thresholds"]

    total_weight = sum(weights.values())

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError(
            "Uncertainty weights must sum to 1.0."
        )

    return UncertaintyConfig(
        version=raw["version"],

        fusion_confidence_weight=float(
            weights["fusion_confidence"]
        ),
        risk_completeness_weight=float(
            weights["risk_completeness"]
        ),
        source_coverage_weight=float(
            weights["source_coverage"]
        ),
        coherence_weight=float(
            weights["coherence"]
        ),

        expected_sources=tuple(
            raw["expected_sources"]
        ),

        high_uncertainty_threshold=float(
            thresholds["high_uncertainty"]
        ),
        moderate_uncertainty_threshold=float(
            thresholds["moderate_uncertainty"]
        ),
    )