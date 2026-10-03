from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExplainabilityDisplayConfig:
    probability_decimals: int = 3
    score_decimals: int = 3


@dataclass(frozen=True)
class ExplainabilityConfig:
    version: str = "explainability_v1"

    signal_labels: dict[str, str] = field(
        default_factory=lambda: {
            "sbp": "systolic blood pressure",
            "dbp": "diastolic blood pressure",
            "hr": "heart rate",
            "hrv": "heart-rate variability",
        }
    )

    display: ExplainabilityDisplayConfig = field(
        default_factory=ExplainabilityDisplayConfig
    )


def _default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "explainability.yaml"
    )


def load_explainability_config(
    path: str | Path | None = None,
) -> ExplainabilityConfig:
    """
    Load Explainability v1 configuration.

    Explainability configuration contains presentation metadata only.
    It must not contain decision thresholds or rules that could change
    the upstream Decision result.
    """

    config_path = (
        Path(path)
        if path is not None
        else _default_config_path()
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Explainability config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw: dict[str, Any] = (
            yaml.safe_load(handle)
            or {}
        )

    display_raw = (
        raw.get("display", {})
        or {}
    )

    display = ExplainabilityDisplayConfig(
        probability_decimals=int(
            display_raw.get(
                "probability_decimals",
                3,
            )
        ),
        score_decimals=int(
            display_raw.get(
                "score_decimals",
                3,
            )
        ),
    )

    labels = raw.get(
        "signal_labels",
        {},
    ) or {}

    return ExplainabilityConfig(
        version=str(
            raw.get(
                "version",
                "explainability_v1",
            )
        ),
        signal_labels={
            str(key): str(value)
            for key, value in labels.items()
        },
        display=display,
    )