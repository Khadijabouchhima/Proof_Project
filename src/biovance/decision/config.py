from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DecisionConfig:
    version: str

    abstain_uncertainty: float
    warn_max_uncertainty: float

    strong_fusion_warn: float
    fusion_warn_with_risk: float

    background_risk_context: float

    warn_fusion_states: tuple[str, ...]


def load_decision_config(
    path: str | Path = "configs/decision.yaml",
) -> DecisionConfig:

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        raw = yaml.safe_load(f)

    thresholds = raw["thresholds"]

    return DecisionConfig(
        version=raw["version"],

        abstain_uncertainty=float(
            thresholds[
                "abstain_uncertainty"
            ]
        ),

        warn_max_uncertainty=float(
            thresholds[
                "warn_max_uncertainty"
            ]
        ),

        strong_fusion_warn=float(
            thresholds[
                "strong_fusion_warn"
            ]
        ),

        fusion_warn_with_risk=float(
            thresholds[
                "fusion_warn_with_risk"
            ]
        ),

        background_risk_context=float(
            thresholds[
                "background_risk_context"
            ]
        ),

        warn_fusion_states=tuple(
            raw[
                "warn_fusion_states"
            ]
        ),
    )