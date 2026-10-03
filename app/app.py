from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from live_pipeline import (
    RealBioVanceLivePipeline,
    SimulatorState,
    SyntheticPatient,
    generate_next_observation,
)


st.set_page_config(
    page_title="PROOF Live Demonstrator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def first_value(mapping: dict, names: tuple[str, ...], default="—"):
    if not isinstance(mapping, dict):
        return default
    for name in names:
        value = mapping.get(name)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        return value
    return default


def fmt(value, decimals: int = 3) -> str:
    try:
        if pd.isna(value):
            return "—"
    except Exception:
        pass
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def decision_state(result: dict | None) -> str:
    if not result:
        return "WAITING"
    return str(
        first_value(
            result.get("decision", {}),
            ("decision_state", "decision", "state"),
            "WAITING",
        )
    )


def init():
    if "pipeline" not in st.session_state:
        with st.spinner("Learning the synthetic patient's personal baseline..."):
            st.session_state.pipeline = RealBioVanceLivePipeline(
                SyntheticPatient()
            )

    if "scenario" not in st.session_state:
        st.session_state.scenario = "Gradual deterioration"

    if "sim" not in st.session_state:
        st.session_state.sim = SimulatorState(
            patient=st.session_state.pipeline.patient,
            scenario=st.session_state.scenario,
        )

    if "results" not in st.session_state:
        st.session_state.results = []


def reset(scenario: str):
    st.session_state.pipeline.reset()
    st.session_state.sim = SimulatorState(
        patient=st.session_state.pipeline.patient,
        scenario=scenario,
    )
    st.session_state.results = []
    st.session_state.scenario = scenario


def advance_one():
    sim = st.session_state.sim
    obs = generate_next_observation(sim)
    result = st.session_state.pipeline.process(obs)

    sim.history = pd.concat(
        [sim.history, pd.DataFrame([obs])],
        ignore_index=True,
    )
    sim.tick += 1
    st.session_state.results.append(result)


def run_steps(n: int):
    for _ in range(n):
        advance_one()


init()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("PROOF")
st.subheader("Personalized Robust Observation Framework — live demonstrator")

st.caption(
    "The patient stream is synthetic. The live path executes the real "
    "Personalization, Deviation, Temporal, Fusion, Uncertainty, Decision, "
    "and Explainability engines. Quality values and REST context are "
    "synthetic bridge metadata in this demo; background Framingham risk is "
    "left unavailable rather than fabricated."
)

scenario = st.selectbox(
    "Scenario",
    (
        "Stable physiology",
        "Gradual deterioration",
        "Conflicting signals",
        "Sensor dropout",
    ),
    index=(
        (
            "Stable physiology",
            "Gradual deterioration",
            "Conflicting signals",
            "Sensor dropout",
        ).index(st.session_state.scenario)
    ),
)

if scenario != st.session_state.scenario:
    reset(scenario)

c1, c2, c3 = st.columns(3)
if c1.button("Advance one observation", use_container_width=True):
    try:
        advance_one()
    except Exception as exc:
        st.error("A real PROOF engine rejected the live bridge input.")
        st.exception(exc)

if c2.button("Run 12 observations", use_container_width=True):
    try:
        run_steps(12)
    except Exception as exc:
        st.error("A real PROOF engine rejected the live bridge input.")
        st.exception(exc)

if c3.button("Reset scenario", use_container_width=True):
    reset(scenario)
    st.rerun()

st.divider()


# ---------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------

with st.expander("Personal baseline learned by the real Personalization Engine"):
    base = st.session_state.pipeline.baseline_summary()
    cols = st.columns(4)
    labels = {
        "hr": ("HR", "bpm"),
        "hrv": ("HRV", "ms"),
        "sbp": ("SBP", "mmHg"),
        "dbp": ("DBP", "mmHg"),
    }
    for col, key in zip(cols, ("hr", "hrv", "sbp", "dbp")):
        label, unit = labels[key]
        col.metric(
            label,
            f"{base[key]['center']:.1f} {unit}",
            help=f"Robust learned scale: {base[key]['scale']:.2f}",
        )


sim = st.session_state.sim
results = st.session_state.results
final = results[-1] if results else None

top1, top2, top3 = st.columns([2, 1, 1])
top1.markdown("### Synthetic Patient A")
top1.caption(
    f"{scenario} · every observation represents 30 minutes of patient time"
)
top2.metric("Observations", sim.tick)
top3.metric("Current decision", decision_state(final))


# ---------------------------------------------------------------------
# Physiology
# ---------------------------------------------------------------------

st.markdown("### 1. Live physiology")

latest = (
    sim.history.iloc[-1].to_dict()
    if not sim.history.empty
    else {"HR": np.nan, "HRV": np.nan, "SBP": np.nan, "DBP": np.nan}
)

cols = st.columns(4)
for col, (field, label, unit) in zip(
    cols,
    (
        ("HR", "Heart rate", "bpm"),
        ("SBP", "Systolic BP", "mmHg"),
        ("DBP", "Diastolic BP", "mmHg"),
        ("HRV", "HRV", "ms"),
    ),
):
    value = latest.get(field, np.nan)
    col.metric(
        label,
        "Unavailable"
        if pd.isna(value)
        else f"{float(value):.1f} {unit}",
    )

if not sim.history.empty:
    st.line_chart(
        sim.history.set_index("timestamp")[["HR", "SBP", "DBP", "HRV"]],
        height=280,
    )

if final is None:
    st.info(
        "Advance one observation or run 12 observations. The first points stay "
        "close to the learned baseline before the selected scenario develops."
    )
    st.stop()


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

st.markdown("### 2. What the engines see now")

deviation_df = final["deviation_current"]
temporal_df = final["temporal_current"]
fusion = final["fusion"]
uncertainty = final["uncertainty"]
decision = final["decision"]
explanation = final["explanation"]

p1, p2, p3 = st.columns(3)

with p1:
    with st.container(border=True):
        st.markdown("**Personalization → Deviation**")
        if isinstance(deviation_df, pd.DataFrame) and not deviation_df.empty:
            for _, row in deviation_df.iterrows():
                signal = str(row.get("signal", "?")).upper()
                state = first_value(
                    row.to_dict(),
                    (
                        "deviation_state",
                        "deviation_status",
                        "deviation_level",
                        "deviation_code",
                    ),
                    "UNKNOWN",
                )
                score = first_value(
                    row.to_dict(),
                    ("risk_aligned_score", "z", "deviation_score"),
                    None,
                )
                text = f"{signal}: {state}"
                if score is not None:
                    text += f" · score {fmt(score, 2)}"
                st.write(text)
        else:
            st.write("No usable deviation yet")

with p2:
    with st.container(border=True):
        st.markdown("**Temporal**")
        if isinstance(temporal_df, pd.DataFrame) and not temporal_df.empty:
            for _, row in temporal_df.iterrows():
                signal = str(row.get("signal", "?")).upper()
                state = first_value(
                    row.to_dict(),
                    ("temporal_state", "temporal_status", "temporal_code"),
                    "UNKNOWN",
                )
                st.write(f"{signal}: {state}")
        else:
            st.write("No current temporal evidence yet")

with p3:
    with st.container(border=True):
        st.markdown("**Fusion**")
        if final.get("fusion_available", False):
            st.metric(
                "State",
                first_value(fusion, ("fusion_state",), "UNKNOWN"),
            )
            st.caption(
                "Score: "
                + fmt(first_value(fusion, ("fusion_score",), np.nan))
                + " · confidence: "
                + fmt(first_value(fusion, ("fusion_confidence",), np.nan))
            )
        else:
            st.metric("State", "UNAVAILABLE")
            st.caption(
                "The current moment did not produce a Fusion row. "
                "A previous Fusion result is never reused."
            )

p4, p5, p6 = st.columns(3)

with p4:
    with st.container(border=True):
        st.markdown("**Background risk**")
        st.metric("Live risk", "Not connected")
        st.caption(
            "Framingham risk is intentionally not fabricated for the synthetic "
            "live patient. The validated risk engine remains a separate branch."
        )

with p5:
    with st.container(border=True):
        st.markdown("**Uncertainty**")
        st.metric(
            "Level",
            first_value(
                uncertainty,
                ("uncertainty_level", "level"),
                "UNKNOWN",
            ),
        )
        st.caption(
            "Score: "
            + fmt(
                first_value(
                    uncertainty,
                    ("uncertainty_score", "uncertainty"),
                    np.nan,
                )
            )
        )

with p6:
    with st.container(border=True):
        st.markdown("**Decision**")
        state = decision_state(final)
        st.metric("State", state)
        reason = first_value(
            decision,
            ("decision_code", "primary_reason", "decision_reason"),
            "",
        )
        if reason:
            st.caption(str(reason))


# ---------------------------------------------------------------------
# Why
# ---------------------------------------------------------------------

st.markdown("### 3. Why did PROOF choose this state?")

explanation_text = first_value(
    explanation,
    ("explanation_summary", "primary_reason", "explanation"),
    "The Explainability Engine did not emit a recognized text field.",
)

state = decision_state(final)
if state == "WARN":
    st.error(str(explanation_text))
elif state == "ABSTAIN":
    st.info(str(explanation_text))
else:
    st.warning(str(explanation_text))

st.caption(
    "Presentation prototype · synthetic source only · no DRYAD or Framingham "
    "participant is being replayed · not a diagnostic system"
)

