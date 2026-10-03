from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import importlib
import inspect

import numpy as np
import pandas as pd

from biovance.personalization import (
    PersonalizationEngine,
    load_personalization_config,
)


ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# Generic helpers for frozen BioVance engines
# ============================================================

def _import_package(name: str):
    return importlib.import_module(f"biovance.{name}")


def _find_engine_class(module, name: str):
    preferred = f"{name.capitalize()}Engine"
    if hasattr(module, preferred):
        return getattr(module, preferred)

    for attr_name in dir(module):
        if attr_name.endswith("Engine"):
            obj = getattr(module, attr_name)
            if inspect.isclass(obj):
                return obj

    raise RuntimeError(
        f"Could not find an Engine class in biovance.{name}"
    )


def _build_engine(name: str):
    """
    Instantiate a frozen BioVance engine using its package exports.

    Supports the common BioVance patterns:
      Engine.from_yaml(config_path)
      Engine(load_<name>_config(config_path))
      Engine()
    """
    module = _import_package(name)
    engine_cls = _find_engine_class(module, name)

    config_path = ROOT / "configs" / f"{name}.yaml"

    if hasattr(engine_cls, "from_yaml") and config_path.exists():
        return engine_cls.from_yaml(config_path)

    loader_name = f"load_{name}_config"
    loader = getattr(module, loader_name, None)

    if loader is not None:
        try:
            config = loader(config_path)
        except TypeError:
            config = loader(str(config_path))

        try:
            return engine_cls(config)
        except TypeError:
            pass

    try:
        return engine_cls()
    except TypeError as exc:
        raise RuntimeError(
            f"Could not instantiate {engine_cls.__name__}. "
            f"Expected either from_yaml(), {loader_name}(), "
            "or a no-argument constructor."
        ) from exc


def _run_table_engine(
    engine: Any,
    df: pd.DataFrame,
    methods: tuple[str, ...] = (
        "transform",
        "score",
        "predict",
        "decide",
        "explain",
        "run",
    ),
) -> pd.DataFrame:
    """
    Run an actual BioVance engine without re-implementing its logic.

    The frozen modules do not all use the same public verb, so the
    bridge looks for the module's own public table method.
    """
    last_error: Exception | None = None

    for method_name in methods:
        method = getattr(engine, method_name, None)
        if method is None:
            continue

        try:
            result = method(df.copy())
        except Exception as exc:
            last_error = exc
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

    engine_name = type(engine).__name__

    if last_error is not None:
        raise RuntimeError(
            f"{engine_name} was found, but its public table method "
            f"could not consume the simulator bridge input. "
            f"Last error: {last_error}"
        ) from last_error

    raise RuntimeError(
        f"{engine_name} has none of the supported public methods: "
        + ", ".join(methods)
    )


# ============================================================
# Synthetic patient source
# ============================================================

@dataclass
class SyntheticPatient:
    patient_id: str = "SIM-001"
    age: int = 42
    male: int = 0

    # Used only to generate synthetic physiology.
    hr_center: float = 64.0
    hrv_center: float = 48.0
    sbp_center: float = 118.0
    dbp_center: float = 75.0

    seed: int = 2026


@dataclass
class SimulatorState:
    patient: SyntheticPatient
    scenario: str = "Gradual deterioration"
    tick: int = 0
    history: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame()
    )


def make_baseline_history(
    patient: SyntheticPatient,
    n_days: int = 35,
) -> pd.DataFrame:
    """
    Create synthetic historical observations only.

    This history is used by the REAL Personalization Engine to
    establish the patient's baseline before the live demo starts.
    """
    rng = np.random.default_rng(patient.seed)

    start = pd.Timestamp("2026-01-01 06:00")

    rows = []

    # Multiple observations/day so the real daily aggregation
    # contract has enough eligible HR/BP observations.
    for day in range(n_days):
        for hour in (6, 10, 14, 18):
            ts = start + pd.Timedelta(days=day, hours=hour - 6)

            rows.append(
                {
                    "patient_id": patient.patient_id,
                    "timestamp": ts,
                    "HR": patient.hr_center + rng.normal(0, 2.0),
                    "HRV": patient.hrv_center + rng.normal(0, 3.0),
                    "SBP": patient.sbp_center + rng.normal(0, 4.0),
                    "DBP": patient.dbp_center + rng.normal(0, 3.0),
                    "signal_quality": 0.96,
                    "context": "rest",
                }
            )

    return pd.DataFrame(rows)


def generate_next_observation(
    state: SimulatorState,
) -> dict[str, Any]:
    """
    Generate ONE new live observation.

    The simulator generates only the raw synthetic measurement.
    It does not calculate z-scores, temporal evidence, Fusion,
    uncertainty, decisions, or explanations.
    """
    p = state.patient
    tick = state.tick

    rng = np.random.default_rng(p.seed + 10000 + tick)

    # Accelerated simulated time: each real UI refresh represents
    # 30 minutes of patient time.
    ts = (
        pd.Timestamp("2026-02-05 08:00")
        + pd.Timedelta(minutes=30 * tick)
    )

    hr = p.hr_center + rng.normal(0, 1.4)
    hrv = p.hrv_center + rng.normal(0, 2.0)
    sbp = p.sbp_center + rng.normal(0, 2.0)
    dbp = p.dbp_center + rng.normal(0, 1.5)

    hr_q = hrv_q = sbp_q = dbp_q = 0.96

    # Keep the first few live ticks stable, then introduce the
    # scenario gradually.
    phase = max(0, tick - 5)

    if state.scenario == "Gradual deterioration":
        hr += 2.4 * phase
        sbp += 3.7 * phase
        dbp += 2.3 * phase
        hrv -= 3.2 * phase

    elif state.scenario == "Conflicting signals":
        hr += 2.4 * phase
        sbp += 3.3 * phase
        # HRV moves in the opposite/reassuring direction.
        hrv += 3.0 * phase

    elif state.scenario == "Sensor dropout" and tick >= 8:
        hr = np.nan
        sbp = np.nan
        hrv = np.nan
        hr_q = sbp_q = hrv_q = 0.05
        dbp_q = 0.82

    return {
        "patient_id": p.patient_id,
        "timestamp": ts,
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


# ============================================================
# Real Personalization -> canonical Deviation bridge
# ============================================================

def learn_personal_baselines(
    patient: SyntheticPatient,
) -> tuple[PersonalizationEngine, pd.DataFrame]:
    baseline_history = make_baseline_history(patient)

    config = load_personalization_config(
        "simulator",
        mode="shrunk",
    )

    engine = PersonalizationEngine(config).fit(
        baseline_history
    )

    baselines = engine.baselines(
        baseline_history
    )

    target = baselines[
        baselines["patient_id"].astype(str)
        == str(patient.patient_id)
    ].copy()

    if target.empty:
        raise RuntimeError(
            "Real Personalization Engine did not produce a baseline "
            "for the synthetic patient."
        )

    return engine, target


def _baseline_value(
    baselines: pd.DataFrame,
    signal: str,
    names: tuple[str, ...],
    default=np.nan,
):
    row = baselines[
        baselines["signal"].astype(str).str.lower()
        == signal.lower()
    ]

    if row.empty:
        return default

    row = row.iloc[0]

    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]

    return default


def build_deviation_input(
    observation: dict[str, Any],
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adapt the synthetic current observation into the documented
    Deviation Engine canonical schema.

    Required canonical fields:
      patient_id, timestamp, signal, value,
      baseline_center, baseline_scale
    """
    signal_map = {
        "hr": "HR",
        "hrv": "HRV",
        "sbp": "SBP",
        "dbp": "DBP",
    }

    rows = []

    for signal, raw_col in signal_map.items():
        value = observation.get(raw_col, np.nan)

        center = _baseline_value(
            baselines,
            signal,
            ("center", "center_raw", "baseline_center"),
        )

        scale = _baseline_value(
            baselines,
            signal,
            ("scale", "scale_raw", "baseline_scale"),
        )

        baseline_n = _baseline_value(
            baselines,
            signal,
            ("n_valid", "baseline_n", "n", "valid_days"),
            default=np.nan,
        )

        quality = observation.get(
            f"{signal}_quality",
            observation.get("signal_quality", 1.0),
        )

        rows.append(
            {
                "patient_id": observation["patient_id"],
                "timestamp": observation["timestamp"],
                "signal": signal,
                "value": value,
                "baseline_center": center,
                "baseline_scale": scale,
                "baseline_n": baseline_n,
                "baseline_status": "ESTABLISHED",
                "quality_score": quality,
                "quality_status": (
                    "GOOD" if quality >= 0.70 else "POOR"
                ),
                "context_state": "REST",
                "context_confidence": 1.0,
                "phase": observation.get(
                    "phase",
                    "EVALUATION",
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Strict real-engine pipeline
# ============================================================

class RealBioVanceLivePipeline:
    """
    Simulator source + frozen BioVance engines.

    IMPORTANT:
    The simulator does NOT reproduce downstream engine logic.
    It only generates raw measurements and adapts schemas.

    Personalization, Deviation, Temporal, Fusion, Uncertainty,
    Decision, and Explainability are executed through the actual
    biovance.* packages from src/.
    """

    def __init__(
        self,
        patient: SyntheticPatient | None = None,
    ):
        self.patient = patient or SyntheticPatient()

        (
            self.personalization_engine,
            self.baselines,
        ) = learn_personal_baselines(
            self.patient
        )

        self.deviation_engine = _build_engine("deviation")
        self.temporal_engine = _build_engine("temporal")
        self.fusion_engine = _build_engine("fusion")
        self.uncertainty_engine = _build_engine("uncertainty")
        self.decision_engine = _build_engine("decision")
        self.explainability_engine = _build_engine("explainability")

        # Risk is intentionally unavailable in the first live
        # physiology demo. This is a valid BioVance pathway:
        # strong current physiology can WARN without Framingham,
        # while background risk alone can never create WARN.
        self.risk_probability = np.nan
        self.risk_input_completeness = np.nan

        self.deviation_history = pd.DataFrame()

    def reset(self):
        self.deviation_history = pd.DataFrame()

    def process(
        self,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        # ----------------------------------------------------
        # 1) Real Personalization baseline is already learned.
        # 2) Real Deviation Engine
        # ----------------------------------------------------
        deviation_input = build_deviation_input(
            observation,
            self.baselines,
        )

        deviation_output = _run_table_engine(
            self.deviation_engine,
            deviation_input,
            methods=("transform", "score", "run"),
        )

        self.deviation_history = pd.concat(
            [
                self.deviation_history,
                deviation_output,
            ],
            ignore_index=True,
        )

        # ----------------------------------------------------
        # 3) Real Temporal Engine on all past/current evidence
        # ----------------------------------------------------
        temporal_output = _run_table_engine(
            self.temporal_engine,
            self.deviation_history,
            methods=("transform", "score", "run"),
        )

        # ----------------------------------------------------
        # 4) Real Fusion Engine
        # ----------------------------------------------------
        fusion_output = _run_table_engine(
            self.fusion_engine,
            temporal_output,
            methods=("transform", "score", "run"),
        )

        # The final frozen Fusion version emits one atomic row
        # per patient/timestamp. If an older row-level build is
        # present locally, select the latest rows for the moment.
        current_ts = pd.Timestamp(observation["timestamp"])

        current_fusion = fusion_output[
            pd.to_datetime(
                fusion_output["timestamp"]
            )
            == current_ts
        ].copy()

        if current_fusion.empty:
            current_fusion = fusion_output.tail(1).copy()

        if len(current_fusion) > 1:
            # Prefer a single atomic row if present; otherwise
            # take the strongest current row only for display.
            if "fusion_score" in current_fusion.columns:
                current_fusion = current_fusion.sort_values(
                    "fusion_score",
                    ascending=False,
                    na_position="last",
                ).head(1)
            else:
                current_fusion = current_fusion.head(1)

        downstream = current_fusion.copy()

        downstream["risk_probability"] = (
            self.risk_probability
        )
        downstream["risk_input_completeness"] = (
            self.risk_input_completeness
        )

        # ----------------------------------------------------
        # 5) Real Uncertainty Engine
        # ----------------------------------------------------
        uncertainty_output = _run_table_engine(
            self.uncertainty_engine,
            downstream,
            methods=("score", "transform", "run"),
        )

        # ----------------------------------------------------
        # 6) Real Decision Engine
        # ----------------------------------------------------
        decision_output = _run_table_engine(
            self.decision_engine,
            uncertainty_output,
            methods=("transform", "decide", "score", "run"),
        )

        # ----------------------------------------------------
        # 7) Real Explainability Engine
        # ----------------------------------------------------
        explanation_output = _run_table_engine(
            self.explainability_engine,
            decision_output,
            methods=("transform", "explain", "score", "run"),
        )

        def latest_record(df: pd.DataFrame) -> dict[str, Any]:
            if df is None or df.empty:
                return {}
            return df.iloc[-1].to_dict()

        return {
            "observation": dict(observation),
            "deviation_current": deviation_output.copy(),
            "temporal_current": temporal_output[
                pd.to_datetime(temporal_output["timestamp"])
                == current_ts
            ].copy(),
            "fusion": latest_record(current_fusion),
            "uncertainty": latest_record(uncertainty_output),
            "decision": latest_record(decision_output),
            "explanation": latest_record(explanation_output),
        }
