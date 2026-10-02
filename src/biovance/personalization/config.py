"""Configuration of the personalization engine (aliases, signals, modes)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Tuple

from biovance import config as C

MODES = ("population", "center", "center_scale", "shrunk")
CONTEXT_MODES = ("none", "rest_only")


@dataclass(frozen=True)
class SignalSpec:
    name: str
    aliases: Tuple[str, ...]
    orientation: int
    min_obs_per_day: int
    baseline_contexts: Optional[Tuple[str, ...]]
    quality_aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PersonalizationConfig:
    column_aliases: Mapping[str, Tuple[str, ...]]
    context_map: Mapping[str, Tuple[str, ...]]
    signals: Tuple[SignalSpec, ...]
    min_quality: float
    nan_quality_ok: bool
    baseline_days: int
    min_valid_days: int
    mode: str = "shrunk"
    shrinkage_n0: float = 14.0
    scale_floor_frac: float = 0.25
    context_mode: str = "none"
    score_during_window: bool = False

    def with_(self, **kw) -> "PersonalizationConfig":
        new = replace(self, **kw)
        new.validate()
        return new

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}")
        if self.min_valid_days > self.baseline_days:
            raise ValueError("min_valid_days cannot exceed baseline_days")
        if self.shrinkage_n0 < 0 or not (0 < self.scale_floor_frac <= 1):
            raise ValueError("invalid shrinkage settings")
        for s in self.signals:
            if s.orientation not in (-1, 0, 1):
                raise ValueError(f"{s.name}: orientation must be -1, 0 or 1")


def load_personalization_config(profile: str = "simulator", **overrides) -> PersonalizationConfig:
    """profile: 'simulator' or 'real:<dataset>' (e.g. 'real:pmdata'). Window length is read from
    simulator.yaml / real_checks.yaml so there is a single source of truth."""
    raw = C._load_yaml(C.CONFIG_DIR / "personalization.yaml")
    if profile == "simulator":
        b = C.load_simulator_rules().baseline
        bdays, mdays = int(b["window_days"]), int(b["min_valid_days"])
    elif profile.startswith("real:"):
        name = profile.split(":", 1)[1]
        ds = C.load_real_checks()["datasets"].get(name)
        if ds is None or ds["baseline_days"] is None:
            raise ValueError(f"dataset '{name}' has no personal baseline (single day or anchor-only)")
        bdays, mdays = int(ds["baseline_days"]), int(ds["min_valid_days"])
    else:
        raise ValueError("profile must be 'simulator' or 'real:<dataset>'")
    sigs = tuple(
        SignalSpec(
            name=n, aliases=tuple(s["aliases"]), orientation=int(s["orientation"]),
            min_obs_per_day=int(s["min_obs_per_day"]),
            baseline_contexts=None if s["baseline_contexts"] is None else tuple(s["baseline_contexts"]),
            quality_aliases=tuple(s.get("quality_aliases", ())),
        )
        for n, s in raw["signals"].items()
    )
    d = raw["defaults"]
    cfg = PersonalizationConfig(
        column_aliases={k: tuple(v) for k, v in raw["columns"].items()},
        context_map={k: tuple(v) for k, v in raw["context_map"].items()},
        signals=sigs, min_quality=float(raw["min_quality"]), nan_quality_ok=bool(raw["nan_quality_ok"]),
        baseline_days=bdays, min_valid_days=mdays, mode=d["mode"], shrinkage_n0=float(d["shrinkage_n0"]),
        scale_floor_frac=float(d["scale_floor_frac"]), context_mode=d["context_mode"],
        score_during_window=bool(d["score_during_window"]),
    )
    cfg = replace(cfg, **overrides) if overrides else cfg
    cfg.validate()
    return cfg
