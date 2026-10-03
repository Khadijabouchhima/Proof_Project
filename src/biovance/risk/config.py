from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass(frozen=True)
class RiskConfig:
    model_version: str
    target: str

    features: List[str]
    binary_features: List[str]
    continuous_features: List[str]

    C: float
    max_iter: int
    random_state: int

    low_completeness_threshold: float


def load_risk_config(
    path: str | Path = "configs/risk.yaml",
) -> RiskConfig:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    lr = raw["logistic_regression"]
    completeness = raw["input_completeness"]

    return RiskConfig(
        model_version=raw["model_version"],
        target=raw["target"],
        features=list(raw["features"]),
        binary_features=list(raw["binary_features"]),
        continuous_features=list(raw["continuous_features"]),
        C=float(lr["C"]),
        max_iter=int(lr["max_iter"]),
        random_state=int(lr["random_state"]),
        low_completeness_threshold=float(
            completeness["low_threshold"]
        ),
    )