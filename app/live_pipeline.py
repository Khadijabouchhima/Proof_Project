from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import importlib
import inspect

import numpy as np
import pandas as pd

from biovance.personalization import PersonalizationEngine, load_personalization_config


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Engine loading
# ---------------------------------------------------------------------

def _import_package(name: str):
    return importlib.import_module(f"biovance.{name}")


def _find_engine_class(module, name: str):
    preferred = f"{name.capitalize()}Engine"
    if hasattr(module, preferred):
        return getattr(module, preferred)

    candidates = [
        getattr(module, attr)
        for attr in dir(module)
        if inspect.isclass(getattr(module, attr))
        and attr.endswith("Engine")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(f"No Engine class found in biovance.{name}")
    raise RuntimeError(
        f"More than one Engine class found in biovance.{name}; "
        "the live adapter cannot choose safely."
    )


def _build_engine(name: str):
    """Instantiate a frozen BioVance engine without replacing its logic."""
    module = _import_package(name)
    engine_cls = _find_engine_class(module, name)
    config_path = ROOT / "configs" / f"{name}.yaml"

    if hasattr(engine_cls, "from_yaml") and config_path.exists():
        return engine_cls.from_yaml(config_path)

    loader = getattr(module, f"load_{name}_config", None)
    if loader is not None:
        for arg in (config_path, str(config_path), None):
            try:
                config = loader() if arg is None else loader(arg)
                try:
                    return engine_cls(config)
                except TypeError:
                    pass
            except (TypeError, FileNotFoundError):
                continue

    try:
        return engine_cls()
    except TypeError as exc:
        raise RuntimeError(
            f"Could not instantiate {engine_cls.__name__}. "
            "Expected from_yaml(), a config loader, or a no-argument constructor."
        ) from exc


def _run_table_engine(
    engine: Any,
    df: pd.DataFrame,
    methods: tuple[str, ...],
) -> pd.DataFrame:
    """
    Run a real BioVance engine through one of its public table APIs.

    No clinical fallback logic exists here. If the bridge contract is wrong,
    the error is surfaced so it can be fixed rather than silently faked.
    """
    errors: list[str] = []

    for method_name in methods:
        method = getattr(engine, method_name, None)
        if method is None:
            continue
        try:
            result = method(df.copy())
        except Exception as exc:
            errors.append(f"{method_name}: {type(exc).__name__}: {exc}")
            continue

        if isinstance(result, pd.DataFrame):
            return result
        if isinstance(result, pd.Series):
            return result.to_frame().T
        if isinstance(result, dict):
            return pd.DataFrame([result])
        if isinstance(result, np.ndarray):
            if result.ndim == 1:
                return pd.DataFrame({"prediction": result})
            return pd.DataFrame(result)

        errors.append(
            f"{method_name}: unsupported return type {type(result).__name__}"
        )

    name = type(engine).__name__
    detail = " | ".join(errors[-3:]) if errors else "no supported public method"
    raise RuntimeError(f"{name} could not consume the live bridge input. {detail}")


def _latest_record(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    return df.iloc[-1].to_dict()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _attach_record(base: pd.DataFrame, record: dict[str, Any]) -> pd.DataFrame:
    """Attach already-computed evidence to a one-row schema carrier."""
    out = base.copy()
    if out.empty:
        out = pd.DataFrame([{}])
    if len(out) > 1:
        out = out.tail(1).copy()
    for key, value in record.items():
        out[key] = value
    return out


# ---------------------------------------------------------------------
# Synthetic source
# ---------------------------------------------------------------------

@dataclass
class SyntheticPatient:
    patient_id: str = "SIM-001"
    age: int = 42
    male: int = 0

    hr_center: float = 64.0
    hrv_center: float = 48.0
    sbp_center: float = 118.0
    dbp_center: float = 75.0

    # These are updated from the real learned baseline after initialization.
    hr_noise: float = 1.5
    hrv_noise: float = 2.0
    sbp_noise: float = 2.5
    dbp_noise: float = 1.8

    seed: int = 2026


@dataclass
class SimulatorState:
    patient: SyntheticPatient
    scenario: str = "Gradual deterioration"
    tick: int = 0
    history: pd.DataFrame = field(default_factory=pd.DataFrame)


def make_baseline_history(
    patient: SyntheticPatient,
    n_days: int = 35,
    n_reference_patients: int = 6,
) -> pd.DataFrame:
    """
    Build only the historical training cohort required by the real shrunk
    Personalization Engine. No downstream state is precomputed.
    """
    rng = np.random.default_rng(patient.seed)
    start = pd.Timestamp("2026-01-01 06:00")

    subjects: list[dict[str, Any]] = []
    for idx in range(n_reference_patients):
        subjects.append(
            {
                "patient_id": f"REF-{idx + 1:03d}",
                "hr_center": patient.hr_center + rng.normal(0, 6.0),
                "hrv_center": max(20.0, patient.hrv_center + rng.normal(0, 8.0)),
                "sbp_center": patient.sbp_center + rng.normal(0, 10.0),
                "dbp_center": patient.dbp_center + rng.normal(0, 7.0),
            }
        )

    subjects.append(
        {
            "patient_id": patient.patient_id,
            "hr_center": patient.hr_center,
            "hrv_center": patient.hrv_center,
            "sbp_center": patient.sbp_center,
            "dbp_center": patient.dbp_center,
        }
    )

    rows: list[dict[str, Any]] = []
    for subject in subjects:
        deterministic_offset = sum(ord(c) for c in subject["patient_id"])
        srng = np.random.default_rng(patient.seed + deterministic_offset)

        for day in range(n_days):
            for hour in (6, 10, 14, 18):
                ts = start + pd.Timedelta(days=day, hours=hour - 6)
                rows.append(
                    {
                        "patient_id": subject["patient_id"],
                        "timestamp": ts,
                        "HR": subject["hr_center"] + srng.normal(0, 2.0),
                        "HRV": subject["hrv_center"] + srng.normal(0, 3.0),
                        "SBP": subject["sbp_center"] + srng.normal(0, 4.0),
                        "DBP": subject["dbp_center"] + srng.normal(0, 3.0),
                        "signal_quality": 0.96,
                        "context": "rest",
                    }
                )
    return pd.DataFrame(rows)


def _baseline_value(
    baselines: pd.DataFrame,
    signal: str,
    names: tuple[str, ...],
    default=np.nan,
):
    rows = baselines[
        baselines["signal"].astype(str).str.lower() == signal.lower()
    ]
    if rows.empty:
        return default
    row = rows.iloc[0]
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def learn_personal_baselines(
    patient: SyntheticPatient,
) -> tuple[PersonalizationEngine, pd.DataFrame]:
    history = make_baseline_history(patient)
    config = load_personalization_config("simulator", mode="shrunk")
    engine = PersonalizationEngine(config).fit(history)
    baselines = engine.baselines(history)
    target = baselines[
        baselines["patient_id"].astype(str) == str(patient.patient_id)
    ].copy()

    if target.empty:
        raise RuntimeError(
            "Personalization did not produce an established baseline "
            "for the synthetic target patient."
        )
    return engine, target


def _align_generator_to_learned_baseline(
    patient: SyntheticPatient,
    baselines: pd.DataFrame,
) -> None:
    """
    Presentation-safety fix:
    stable live physiology is generated around the *actual learned baseline*,
    not around the pre-fit seed values.

    This prevents a valid shrinkage adjustment from looking like pathology.
    """
    mapping = {
        "hr": ("hr_center", "hr_noise"),
        "hrv": ("hrv_center", "hrv_noise"),
        "sbp": ("sbp_center", "sbp_noise"),
        "dbp": ("dbp_center", "dbp_noise"),
    }
    for signal, (center_attr, noise_attr) in mapping.items():
        center = _baseline_value(
            baselines, signal, ("center", "baseline_center", "center_raw")
        )
        scale = _baseline_value(
            baselines, signal, ("scale", "baseline_scale", "scale_raw")
        )
        if pd.notna(center):
            setattr(patient, center_attr, float(center))
        if pd.notna(scale) and float(scale) > 0:
            # Keep stable points mostly inside ±1 personal SD.
            setattr(patient, noise_attr, max(0.35, min(float(scale) * 0.45, 3.0)))


def generate_next_observation(state: SimulatorState) -> dict[str, Any]:
    """
    Generate ONE new raw synthetic observation.

    This generator never calculates deviation, temporal state, Fusion,
    uncertainty, decisions, or explanations.
    """
    p = state.patient
    tick = state.tick
    rng = np.random.default_rng(p.seed + 10000 + tick)

    timestamp = pd.Timestamp("2026-02-05 08:00") + pd.Timedelta(
        minutes=30 * tick
    )

    hr = p.hr_center + rng.normal(0, p.hr_noise)
    hrv = p.hrv_center + rng.normal(0, p.hrv_noise)
    sbp = p.sbp_center + rng.normal(0, p.sbp_noise)
    dbp = p.dbp_center + rng.normal(0, p.dbp_noise)

    hr_q = hrv_q = sbp_q = dbp_q = 0.96

    # Give the engines several baseline-like live points before a scenario starts.
    phase = max(0, tick - 5)

    if state.scenario == "Gradual deterioration":
        # Smooth risk-aligned drift across independent physiological families.
        hr += 0.90 * phase
        sbp += 1.50 * phase
        dbp += 0.85 * phase
        hrv -= 1.20 * phase

    elif state.scenario == "Conflicting signals":
        hr += 0.90 * phase
        sbp += 1.40 * phase
        dbp += 0.60 * phase
        # HRV deliberately moves in the reassuring direction.
        hrv += 1.20 * phase

    elif state.scenario == "Sensor dropout" and tick >= 8:
        hr = np.nan
        hrv = np.nan
        sbp = np.nan
        hr_q = hrv_q = sbp_q = 0.05
        dbp_q = 0.82

    return {
        "patient_id": p.patient_id,
        "timestamp": timestamp,
        "HR": hr,
        "HRV": hrv,
        "SBP": sbp,
        "DBP": dbp,
        "hr_quality": hr_q,
        "hrv_quality": hrv_q,
        "sbp_quality": sbp_q,
        "dbp_quality": dbp_q,
        "signal_quality": min(hr_q, hrv_q, sbp_q, dbp_q),
        "context": "rest",
        "phase": "EVALUATION",
    }


# ---------------------------------------------------------------------
# Personalization -> Deviation bridge
# ---------------------------------------------------------------------

def build_deviation_input(
    observation: dict[str, Any],
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Schema adapter only. It does not calculate a deviation.

    NOTE:
    The current presentation simulator supplies synthetic quality metadata and
    a fixed REST context. Real Quality/Context engines are validated elsewhere
    in the project and are not silently imitated here.
    """
    signal_map = {"hr": "HR", "hrv": "HRV", "sbp": "SBP", "dbp": "DBP"}
    rows = []

    for signal, raw_column in signal_map.items():
        quality = observation.get(
            f"{signal}_quality",
            observation.get("signal_quality", 1.0),
        )
        value = observation.get(raw_column, np.nan)

        rows.append(
            {
                "patient_id": observation["patient_id"],
                "timestamp": observation["timestamp"],
                "signal": signal,
                "value": value,
                "baseline_center": _baseline_value(
                    baselines,
                    signal,
                    ("center", "baseline_center", "center_raw"),
                ),
                "baseline_scale": _baseline_value(
                    baselines,
                    signal,
                    ("scale", "baseline_scale", "scale_raw"),
                ),
                "baseline_n": _baseline_value(
                    baselines,
                    signal,
                    ("n_valid", "baseline_n", "n", "valid_days"),
                ),
                "baseline_status": "ESTABLISHED",
                "quality_score": quality,
                "quality_status": (
                    "GOOD"
                    if pd.notna(quality) and float(quality) >= 0.70
                    else "POOR"
                ),
                "context_state": "REST",
                "context_confidence": 1.0,
                "phase": observation.get("phase", "EVALUATION"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Live pipeline
# ---------------------------------------------------------------------

class RealBioVanceLivePipeline:
    """
    Synthetic source + real frozen BioVance downstream engines.

    Real in live path:
      Personalization -> Deviation -> Temporal -> Fusion ->
      Uncertainty -> Decision -> Explainability

    Presentation adapters:
      synthetic quality metadata + fixed REST context

    Background Risk:
      unavailable in this live synthetic path unless explicitly connected.
      It is NEVER fabricated.
    """

    def __init__(self, patient: SyntheticPatient | None = None):
        self.patient = patient or SyntheticPatient()

        (
            self.personalization_engine,
            self.baselines,
        ) = learn_personal_baselines(self.patient)

        _align_generator_to_learned_baseline(self.patient, self.baselines)

        self.deviation_engine = _build_engine("deviation")
        self.temporal_engine = _build_engine("temporal")
        self.fusion_engine = _build_engine("fusion")
        self.uncertainty_engine = _build_engine("uncertainty")
        self.decision_engine = _build_engine("decision")
        self.explainability_engine = _build_engine("explainability")

        self.risk_probability = np.nan
        self.risk_input_completeness = np.nan

        self.deviation_history = pd.DataFrame()

    def reset(self) -> None:
        self.deviation_history = pd.DataFrame()

    def baseline_summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for signal in ("hr", "hrv", "sbp", "dbp"):
            out[signal] = {
                "center": float(
                    _baseline_value(
                        self.baselines,
                        signal,
                        ("center", "baseline_center", "center_raw"),
                    )
                ),
                "scale": float(
                    _baseline_value(
                        self.baselines,
                        signal,
                        ("scale", "baseline_scale", "scale_raw"),
                    )
                ),
            }
        return out

    def _current_fusion_row(
        self,
        fusion_output: pd.DataFrame,
        current_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return Fusion for exactly the current physiological moment.

        IMPORTANT:
        Do not fall back to a stale previous Fusion row. If the real Fusion
        engine emits nothing for the current moment, downstream uncertainty
        should see Fusion as unavailable.
        """
        if fusion_output is None or fusion_output.empty:
            return pd.DataFrame()

        if "timestamp" not in fusion_output.columns:
            # Only accept a timestamp-less result if the engine returned one row.
            return fusion_output.copy() if len(fusion_output) == 1 else pd.DataFrame()

        ts = pd.to_datetime(fusion_output["timestamp"])
        current = fusion_output[ts == current_timestamp].copy()

        if len(current) <= 1:
            return current

        # Atomic safety guard: choose one only if a legacy engine emitted more.
        if "fusion_score" in current.columns:
            return current.sort_values(
                "fusion_score", ascending=False, na_position="last"
            ).head(1)
        return current.head(1)

    def _missing_fusion_carrier(
        self,
        observation: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Schema carrier for a genuinely missing current Fusion result.
        This is not a synthetic Fusion decision: fields are explicitly missing.
        """
        return pd.DataFrame(
            [
                {
                    "patient_id": observation["patient_id"],
                    "timestamp": observation["timestamp"],
                    "fusion_state": np.nan,
                    "fusion_score": np.nan,
                    "fusion_confidence": np.nan,
                }
            ]
        )

    def process(self, observation: dict[str, Any]) -> dict[str, Any]:
        current_timestamp = pd.Timestamp(observation["timestamp"])

        # 1) Personalization baseline -> Deviation schema
        deviation_input = build_deviation_input(observation, self.baselines)

        # 2) Real Deviation
        deviation_output = _run_table_engine(
            self.deviation_engine,
            deviation_input,
            methods=("transform", "score", "evaluate", "run"),
        )

        self.deviation_history = pd.concat(
            [self.deviation_history, deviation_output],
            ignore_index=True,
        )

        # 3) Real Temporal on all causal history so far
        temporal_output = _run_table_engine(
            self.temporal_engine,
            self.deviation_history,
            methods=("transform", "score", "evaluate", "run"),
        )

        # 4) Real Fusion
        fusion_output = _run_table_engine(
            self.fusion_engine,
            temporal_output,
            methods=("transform", "score", "evaluate", "run"),
        )

        current_fusion = self._current_fusion_row(
            fusion_output, current_timestamp
        )

        fusion_available = not current_fusion.empty
        if fusion_available:
            downstream = current_fusion.copy()
        else:
            downstream = self._missing_fusion_carrier(observation)

        # Background risk is deliberately unavailable in this synthetic live path.
        downstream["risk_probability"] = self.risk_probability
        downstream["risk_input_completeness"] = self.risk_input_completeness

        # 5) Real Uncertainty
        uncertainty_output = _run_table_engine(
            self.uncertainty_engine,
            downstream,
            methods=("score", "transform", "evaluate", "run"),
        )

        uncertainty_record = _latest_record(uncertainty_output)

        # 6) REAL Decision — critical schema-preservation fix.
        #
        # The previous bridge passed uncertainty_output alone. If the
        # Uncertainty Engine returns only uncertainty-specific columns,
        # Fusion fields disappear and Decision incorrectly thinks Fusion is
        # unavailable. We preserve current Fusion/risk evidence and attach the
        # already-computed uncertainty fields.
        decision_input = _attach_record(downstream, uncertainty_record)

        decision_output = _run_table_engine(
            self.decision_engine,
            decision_input,
            methods=("transform", "decide", "score", "evaluate", "run"),
        )

        decision_record = _latest_record(decision_output)

        # 7) Explainability gets the same evidence plus final Decision.
        explainability_input = _attach_record(decision_input, decision_record)

        explanation_output = _run_table_engine(
            self.explainability_engine,
            explainability_input,
            methods=("transform", "explain", "score", "evaluate", "run"),
        )

        # Current temporal rows for UI.
        if "timestamp" in temporal_output.columns:
            temporal_current = temporal_output[
                pd.to_datetime(temporal_output["timestamp"]) == current_timestamp
            ].copy()
        else:
            temporal_current = temporal_output.tail(4).copy()

        return {
            "observation": dict(observation),
            "deviation_current": deviation_output.copy(),
            "temporal_current": temporal_current,
            "fusion": _latest_record(current_fusion),
            "fusion_available": fusion_available,
            "uncertainty": uncertainty_record,
            "decision": decision_record,
            "explanation": _latest_record(explanation_output),
            "risk": {
                "risk_probability": self.risk_probability,
                "risk_input_completeness": self.risk_input_completeness,
                "status": "UNAVAILABLE_IN_LIVE_SYNTHETIC_PATH",
            },
        }

