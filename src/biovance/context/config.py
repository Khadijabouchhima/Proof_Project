"""Configuration for the BioVance Context Engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Tuple

from biovance import config as C


CONTEXT_STATES = (
    "SLEEP",
    "REST",
    "ACTIVE",
    "UNKNOWN",
)


@dataclass(frozen=True)
class ContextConfig:
    """Configuration for context classification."""

    column_aliases: Mapping[str, Tuple[str, ...]]

    # Sleep
    sleep_overlap_threshold: float = 0.80
    partial_sleep_is_unknown: bool = True

    # Personalized gyro thresholds
    personalize_gyro: bool = True

    awake_overlap_max: float = 0.0

    rest_gyro_quantile: float = 0.50
    active_gyro_quantile: float = 0.90

    min_personal_awake_windows: int = 30
    min_global_awake_windows: int = 100

    shrinkage_strength: float = 50.0

    min_gyro_threshold_gap: float = 0.05

    fallback_rest_gyro: float = 0.79
    fallback_active_gyro: float = 1.33

    # Steps
    rest_steps_max: float = 0.0
    active_steps_min: float = 1.0

    # Dynamic acceleration
    use_dynamic_acc: bool = False

    rest_dynamic_acc_max: float = 0.55
    active_dynamic_acc_min: float = 1.99

    # Activity level
    steps_activity_reference: float = 5.0
    dynamic_acc_activity_reference: float = 1.99

    gyro_weight: float = 0.70
    steps_weight: float = 0.30
    dynamic_acc_weight: float = 0.00

    # Confidence
    single_source_confidence_cap: float = 0.75
    unknown_confidence_cap: float = 0.49

    def with_(self, **kw) -> "ContextConfig":

        cfg = replace(
            self,
            **kw,
        )

        cfg.validate()

        return cfg

    def validate(self) -> None:

        if not 0 <= self.sleep_overlap_threshold <= 1:
            raise ValueError(
                "sleep_overlap_threshold must be in [0,1]"
            )

        if not 0 <= self.awake_overlap_max <= 1:
            raise ValueError(
                "awake_overlap_max must be in [0,1]"
            )

        if not 0 < self.rest_gyro_quantile < 1:
            raise ValueError(
                "rest_gyro_quantile must be in (0,1)"
            )

        if not 0 < self.active_gyro_quantile < 1:
            raise ValueError(
                "active_gyro_quantile must be in (0,1)"
            )

        if (
            self.active_gyro_quantile
            <= self.rest_gyro_quantile
        ):
            raise ValueError(
                "active_gyro_quantile must exceed "
                "rest_gyro_quantile"
            )

        if self.min_personal_awake_windows < 1:
            raise ValueError(
                "min_personal_awake_windows must be >= 1"
            )

        if self.min_global_awake_windows < 1:
            raise ValueError(
                "min_global_awake_windows must be >= 1"
            )

        if self.shrinkage_strength < 0:
            raise ValueError(
                "shrinkage_strength cannot be negative"
            )

        if self.min_gyro_threshold_gap <= 0:
            raise ValueError(
                "min_gyro_threshold_gap must be > 0"
            )

        if (
            self.fallback_active_gyro
            <= self.fallback_rest_gyro
        ):
            raise ValueError(
                "fallback_active_gyro must exceed "
                "fallback_rest_gyro"
            )

        if (
            self.active_dynamic_acc_min
            <= self.rest_dynamic_acc_max
        ):
            raise ValueError(
                "active_dynamic_acc_min must exceed "
                "rest_dynamic_acc_max"
            )

        if self.steps_activity_reference <= 0:
            raise ValueError(
                "steps_activity_reference must be > 0"
            )

        if self.dynamic_acc_activity_reference <= 0:
            raise ValueError(
                "dynamic_acc_activity_reference must be > 0"
            )

        weights = (
            self.gyro_weight,
            self.steps_weight,
            self.dynamic_acc_weight,
        )

        if any(
            w < 0
            for w in weights
        ):
            raise ValueError(
                "activity weights cannot be negative"
            )

        if sum(weights) <= 0:
            raise ValueError(
                "at least one activity weight must be positive"
            )

        if not (
            0
            <= self.single_source_confidence_cap
            <= 1
        ):
            raise ValueError(
                "single_source_confidence_cap must be in [0,1]"
            )

        if not (
            0
            <= self.unknown_confidence_cap
            <= 1
        ):
            raise ValueError(
                "unknown_confidence_cap must be in [0,1]"
            )


def load_context_config(
    profile: str = "baigutanova",
    **overrides,
) -> ContextConfig:

    raw = C._load_yaml(
        C.CONFIG_DIR
        / "context.yaml"
    )

    profiles = raw.get(
        "profiles",
        {},
    )

    if profile not in profiles:

        raise ValueError(
            f"unknown context profile '{profile}'; "
            f"available: {sorted(profiles)}"
        )

    p = profiles[
        profile
    ]

    weights = p[
        "activity_weights"
    ]

    cfg = ContextConfig(

        column_aliases={
            key: tuple(value)
            for key, value
            in raw["columns"].items()
        },

        sleep_overlap_threshold=float(
            p["sleep_overlap_threshold"]
        ),

        partial_sleep_is_unknown=bool(
            p["partial_sleep_is_unknown"]
        ),

        personalize_gyro=bool(
            p["personalize_gyro"]
        ),

        awake_overlap_max=float(
            p["awake_overlap_max"]
        ),

        rest_gyro_quantile=float(
            p["rest_gyro_quantile"]
        ),

        active_gyro_quantile=float(
            p["active_gyro_quantile"]
        ),

        min_personal_awake_windows=int(
            p["min_personal_awake_windows"]
        ),

        min_global_awake_windows=int(
            p["min_global_awake_windows"]
        ),

        shrinkage_strength=float(
            p["shrinkage_strength"]
        ),

        min_gyro_threshold_gap=float(
            p["min_gyro_threshold_gap"]
        ),

        fallback_rest_gyro=float(
            p["fallback_rest_gyro"]
        ),

        fallback_active_gyro=float(
            p["fallback_active_gyro"]
        ),

        rest_steps_max=float(
            p["rest_steps_max"]
        ),

        active_steps_min=float(
            p["active_steps_min"]
        ),

        use_dynamic_acc=bool(
            p["use_dynamic_acc"]
        ),

        rest_dynamic_acc_max=float(
            p["rest_dynamic_acc_max"]
        ),

        active_dynamic_acc_min=float(
            p["active_dynamic_acc_min"]
        ),

        steps_activity_reference=float(
            p["steps_activity_reference"]
        ),

        dynamic_acc_activity_reference=float(
            p[
                "dynamic_acc_activity_reference"
            ]
        ),

        gyro_weight=float(
            weights["gyro"]
        ),

        steps_weight=float(
            weights["steps"]
        ),

        dynamic_acc_weight=float(
            weights["dynamic_acc"]
        ),

        single_source_confidence_cap=float(
            p["single_source_confidence_cap"]
        ),

        unknown_confidence_cap=float(
            p["unknown_confidence_cap"]
        ),
    )

    if overrides:

        cfg = replace(
            cfg,
            **overrides,
        )

    cfg.validate()

    return cfg