"""Typed, validated access to the frozen definitions, simulator rules and real-check rules.

Layout (see configs/):
  definitions.yaml   global definitions shared by simulator and real checks
  simulator.yaml     simulator-benchmark rules
  real_checks.yaml   real-dataset rules
  sim_train.yaml / sim_shifted.yaml   generation parameters
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

STATES = ("NORMAL", "EMERGING_RISK", "INSUFFICIENT_EVIDENCE", "POOR_QUALITY")
REAL_DATASETS = ("baigutanova_hrv", "pmdata", "dryad_24h", "framingham_teaching")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


@dataclass(frozen=True)
class Reference:
    sbp: float
    dbp: float
    consecutive_readings: int
    maximum_gap_days: int

    def is_elevated(self, sbp: float, dbp: float) -> bool:
        """One reading meets the reference: SBP >= sbp OR DBP >= dbp."""
        return (sbp >= self.sbp) or (dbp >= self.dbp)


@dataclass(frozen=True)
class Definitions:
    horizon_days: int
    min_lead_days: int
    reference: Reference
    sensitivity_reference: Reference
    persistence_k: int
    refractory_days: int
    fa_budgets: tuple
    primary_fa_budget: float
    states: tuple
    statistics: Mapping[str, Any]


@dataclass(frozen=True)
class SimulatorRules:
    reference_source: str
    baseline: Mapping[str, Any]
    splits: Mapping[str, Any]
    seeds: tuple


def load_definitions(path: Path | None = None) -> Definitions:
    raw = _load_yaml(path or CONFIG_DIR / "definitions.yaml")
    t, w = raw["task"], raw["warning"]
    d = Definitions(
        horizon_days=int(t["horizon_days"]),
        min_lead_days=int(t["min_lead_days"]),
        reference=Reference(**t["reference"]),
        sensitivity_reference=Reference(**t["sensitivity_reference"]),
        persistence_k=int(w["persistence_k"]),
        refractory_days=int(w["refractory_days"]),
        fa_budgets=tuple(w["false_alarm_budgets_per_subject_month"]),
        primary_fa_budget=float(w["primary_false_alarm_budget"]),
        states=tuple(raw["states"]),
        statistics=_freeze(raw["statistics"]),
    )
    validate_definitions(d)
    return d


def validate_definitions(d: Definitions) -> None:
    if d.states != STATES:
        raise ValueError(f"states must be exactly {STATES}, got {d.states}")
    if d.horizon_days <= d.min_lead_days:
        raise ValueError("horizon_days must exceed min_lead_days")
    if d.primary_fa_budget not in d.fa_budgets:
        raise ValueError("primary false-alarm budget must be one of the listed budgets")
    for ref in (d.reference, d.sensitivity_reference):
        if ref.consecutive_readings < 2:
            raise ValueError("reference must require >= 2 consecutive readings")
        if ref.maximum_gap_days < 1:
            raise ValueError("maximum_gap_days must be >= 1")
    if d.sensitivity_reference.sbp <= d.reference.sbp:
        raise ValueError("sensitivity reference must be stricter than primary")


def load_simulator_rules(path: Path | None = None) -> SimulatorRules:
    raw = _load_yaml(path or CONFIG_DIR / "simulator.yaml")
    r = SimulatorRules(
        reference_source=raw["reference_source"],
        baseline=_freeze(raw["baseline"]),
        splits=_freeze(raw["splits"]),
        seeds=tuple(raw["seeds"]),
    )
    validate_simulator_rules(r)
    return r


def validate_simulator_rules(r: SimulatorRules) -> None:
    if r.reference_source != "clinic_process":
        raise ValueError("T_ref must come from the clean clinic_process, not the wearable BP stream")
    s = r.splits
    if abs(s["train"] + s["calibration"] + s["test"] - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    if s["level"] != "subject":
        raise ValueError("splits must be at subject level")
    if r.baseline["min_valid_days"] > r.baseline["window_days"]:
        raise ValueError("min_valid_days cannot exceed the baseline window")


def load_real_checks() -> Mapping[str, Any]:
    raw = _load_yaml(CONFIG_DIR / "real_checks.yaml")
    validate_real_checks(raw)
    return _freeze(raw)


def validate_real_checks(c: dict[str, Any]) -> None:
    com = c["common"]
    if com["uses_reference_event"] or com["uses_task_horizon"] or com["supervised_risk_model"]:
        raise ValueError("real data has no onset labels: reference event, horizon and supervised model must be off")
    if set(c["datasets"]) != set(REAL_DATASETS):
        raise ValueError(f"datasets must be exactly {REAL_DATASETS}")
    for name, ds in c["datasets"].items():
        bd = ds.get("baseline_days")
        if bd is not None and ds.get("min_valid_days", 0) > bd:
            raise ValueError(f"{name}: min_valid_days exceeds baseline_days")
    if "reference_event" not in c["datasets"]["dryad_24h"].get("never_apply", []):
        raise ValueError("dryad_24h must forbid the reference event (ambulatory BP thresholds differ)")


def load_sim_config(name: str) -> Mapping[str, Any]:
    """Load a simulator generation config ('sim_train' or 'sim_shifted') and validate it."""
    raw = _load_yaml(CONFIG_DIR / f"{name}.yaml")
    validate_sim_config(raw)
    return _freeze(raw)


def validate_sim_config(c: dict[str, Any]) -> None:
    base_days = _load_yaml(CONFIG_DIR / "simulator.yaml")["baseline"]["window_days"]
    ref = _load_yaml(CONFIG_DIR / "definitions.yaml")["task"]["reference"]
    lo, hi = c["drift"]["onset_range"]
    if lo <= base_days:
        raise ValueError(
            f"onset_range starts at day {lo} but the frozen baseline window is {base_days} days: "
            "the baseline would be contaminated by drift"
        )
    if hi + c["drift"]["event_window"][0] > c["n_days"]:
        raise ValueError("latest onset + minimum event delay exceeds n_days")
    if c["drift"]["event_window"][1] > c["n_days"]:
        raise ValueError("event_window upper bound exceeds n_days")
    if not (0.0 < c["drifter_fraction"] < 1.0):
        raise ValueError("drifter_fraction must be in (0, 1)")
    if c["population"]["nondrifter_sbp_cap"] >= ref["sbp"]:
        raise ValueError("non-drifter SBP cap must be below the reference threshold")
    if c["population"]["nondrifter_dbp_cap"] >= ref["dbp"]:
        raise ValueError("non-drifter DBP cap must be below the reference threshold")
    if c["measurement"]["noise_level"] not in ("low", "medium", "high"):
        raise ValueError("noise_level must be low, medium or high")
