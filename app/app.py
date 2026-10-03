from __future__ import annotations

import time
from pathlib import Path

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
    page_title="BioVance Live Simulator",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Helpers
# ============================================================

def first_value(
    mapping: dict,
    names: tuple[str, ...],
    default="Unavailable",
):
    for name in names:
        value = mapping.get(name)
        if value is not None:
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            return value
    return default


def fmt(
    value,
    decimals=3,
):
    if value == "Unavailable":
        return value
    try:
        if pd.isna(value):
            return "Unavailable"
    except Exception:
        pass
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def current_decision(result: dict) -> str:
    d = result.get("decision", {})
    return str(
        first_value(
            d,
            (
                "decision_state",
                "decision",
                "state",
            ),
            "MONITOR",
        )
    )


# ============================================================
# Session state
# ============================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RealBioVanceLivePipeline(
        SyntheticPatient()
    )

if "sim" not in st.session_state:
    st.session_state.sim = SimulatorState(
        patient=st.session_state.pipeline.patient
    )

if "running" not in st.session_state:
    st.session_state.running = False

if "results" not in st.session_state:
    st.session_state.results = []

if "scenario" not in st.session_state:
    st.session_state.scenario = "Gradual deterioration"


# ============================================================
# Controls
# ============================================================

st.title("BioVance")
st.subheader("Live synthetic patient → real BioVance engines")

st.caption(
    "The measurements are synthetic. The downstream pipeline "
    "uses the frozen BioVance engine classes from src/biovance."
)

scenario = st.selectbox(
    "Scenario",
    [
        "Stable physiology",
        "Gradual deterioration",
        "Conflicting signals",
        "Sensor dropout",
    ],
    index=1,
)

speed = st.select_slider(
    "Speed",
    options=[
        "Slow",
        "Normal",
        "Fast",
    ],
    value="Normal",
)

c1, c2, c3 = st.columns(3)

if c1.button(
    "Start",
    use_container_width=True,
):
    st.session_state.running = True

if c2.button(
    "Pause",
    use_container_width=True,
):
    st.session_state.running = False

if c3.button(
    "Reset",
    use_container_width=True,
):
    st.session_state.running = False
    st.session_state.results = []
    st.session_state.sim = SimulatorState(
        patient=st.session_state.pipeline.patient,
        scenario=scenario,
    )
    st.session_state.pipeline.reset()


if scenario != st.session_state.scenario:
    st.session_state.scenario = scenario
    st.session_state.running = False
    st.session_state.results = []
    st.session_state.sim = SimulatorState(
        patient=st.session_state.pipeline.patient,
        scenario=scenario,
    )
    st.session_state.pipeline.reset()


st.divider()


# ============================================================
# Live update fragment
# ============================================================

intervals = {
    "Slow": 2.0,
    "Normal": 1.0,
    "Fast": 0.45,
}


@st.fragment(run_every=0.45)
def live_view():
    sim = st.session_state.sim
    sim.scenario = st.session_state.scenario

    # Generate ONE genuinely new raw observation per refresh.
    if st.session_state.running:
        now = time.monotonic()

        last_emit = st.session_state.get(
            "last_emit",
            0.0,
        )

        if now - last_emit >= intervals[speed]:
            observation = generate_next_observation(sim)

            try:
                result = st.session_state.pipeline.process(
                    observation
                )
            except Exception as exc:
                st.session_state.running = False
                st.error(
                    "The simulator reached a real BioVance engine, "
                    "but the local engine contract did not match the "
                    "bridge input."
                )
                st.exception(exc)
                st.stop()

            sim.history = pd.concat(
                [
                    sim.history,
                    pd.DataFrame([observation]),
                ],
                ignore_index=True,
            )

            sim.tick += 1

            st.session_state.results.append(result)
            st.session_state.last_emit = now

    # --------------------------------------------------------
    # Patient / latest signal values
    # --------------------------------------------------------
    left, middle, right = st.columns(
        [2, 1, 1]
    )

    left.subheader("Simulated Patient A")
    left.caption(
        "Synthetic measurements · personal baseline learned "
        "by the real Personalization Engine"
    )

    middle.metric(
        "Generated observations",
        sim.tick,
    )

    if st.session_state.results:
        final = st.session_state.results[-1]
        decision = current_decision(final)
    else:
        final = None
        decision = "Waiting"

    right.metric(
        "Current decision",
        decision,
    )

    # --------------------------------------------------------
    # Live signals
    # --------------------------------------------------------
    st.markdown("### Live physiology")

    signal_cols = st.columns(4)

    if sim.history.empty:
        latest = {
            "HR": np.nan,
            "SBP": np.nan,
            "DBP": np.nan,
            "HRV": np.nan,
        }
    else:
        latest = sim.history.iloc[-1]

    for col, field, label, unit in zip(
        signal_cols,
        ["HR", "SBP", "DBP", "HRV"],
        [
            "Heart rate",
            "Systolic BP",
            "Diastolic BP",
            "HRV",
        ],
        [
            "bpm",
            "mmHg",
            "mmHg",
            "ms",
        ],
    ):
        value = latest[field]
        if pd.isna(value):
            col.metric(label, "Unavailable")
        else:
            col.metric(
                label,
                f"{float(value):.1f} {unit}",
            )

    if not sim.history.empty:
        chart = sim.history[
            [
                "timestamp",
                "HR",
                "SBP",
                "DBP",
                "HRV",
            ]
        ].copy()

        chart = chart.set_index(
            "timestamp"
        )

        st.line_chart(
            chart,
            height=300,
        )

    st.divider()

    # --------------------------------------------------------
    # Real pipeline states
    # --------------------------------------------------------
    st.markdown("### Real BioVance pipeline")

    if final is None:
        st.info(
            "Press Start. A new synthetic observation will be "
            "generated and passed through the real engines."
        )
        return

    deviation_df = final[
        "deviation_current"
    ]

    temporal_df = final[
        "temporal_current"
    ]

    fusion = final["fusion"]
    uncertainty = final["uncertainty"]
    decision_map = final["decision"]
    explanation = final["explanation"]

    # Personalized deviation
    deviation_states = []

    if (
        isinstance(deviation_df, pd.DataFrame)
        and not deviation_df.empty
    ):
        for _, row in deviation_df.iterrows():
            signal = str(
                row.get(
                    "signal",
                    "?",
                )
            ).upper()

            status = str(
                row.get(
                    "deviation_status",
                    row.get(
                        "deviation_code",
                        "UNKNOWN",
                    ),
                )
            )

            deviation_states.append(
                f"{signal}: {status}"
            )

    # Temporal
    temporal_states = []

    if (
        isinstance(temporal_df, pd.DataFrame)
        and not temporal_df.empty
    ):
        for _, row in temporal_df.iterrows():
            signal = str(
                row.get(
                    "signal",
                    "?",
                )
            ).upper()

            state = str(
                row.get(
                    "temporal_state",
                    row.get(
                        "temporal_code",
                        "UNKNOWN",
                    ),
                )
            )

            temporal_states.append(
                f"{signal}: {state}"
            )

    p1, p2, p3 = st.columns(3)

    with p1:
        with st.container(border=True):
            st.markdown("**Personalization + Deviation**")
            if deviation_states:
                for line in deviation_states:
                    st.write(line)
            else:
                st.write("No usable deviation yet")

    with p2:
        with st.container(border=True):
            st.markdown("**Temporal**")
            if temporal_states:
                for line in temporal_states:
                    st.write(line)
            else:
                st.write("Insufficient temporal evidence")

    with p3:
        with st.container(border=True):
            st.markdown("**Fusion**")
            st.metric(
                "State",
                first_value(
                    fusion,
                    ("fusion_state",),
                ),
            )
            st.caption(
                "Score: "
                + fmt(
                    first_value(
                        fusion,
                        ("fusion_score",),
                    )
                )
            )

    p4, p5 = st.columns(2)

    with p4:
        with st.container(border=True):
            st.markdown("**Uncertainty**")
            st.metric(
                "Level",
                first_value(
                    uncertainty,
                    (
                        "uncertainty_level",
                        "level",
                    ),
                ),
            )
            st.caption(
                "Score: "
                + fmt(
                    first_value(
                        uncertainty,
                        (
                            "uncertainty_score",
                            "uncertainty",
                        ),
                    )
                )
            )

    with p5:
        with st.container(border=True):
            st.markdown("**Decision**")
            state = current_decision(final)
            st.metric("State", state)

            reason = first_value(
                decision_map,
                (
                    "decision_code",
                    "primary_reason",
                    "decision_reason",
                ),
                "",
            )

            if reason:
                st.caption(str(reason))

    st.divider()

    st.markdown("### Why?")

    explanation_text = first_value(
        explanation,
        (
            "explanation_summary",
            "primary_reason",
            "explanation",
        ),
        "The real Explainability Engine did not emit a "
        "human-readable field for this row.",
    )

    if decision == "WARN":
        st.error(str(explanation_text))
    elif decision == "ABSTAIN":
        st.info(str(explanation_text))
    else:
        st.warning(str(explanation_text))

    st.caption(
        "Synthetic source only. No DRYAD or Framingham row is "
        "being replayed in this live simulation."
    )


live_view()

st.divider()
st.caption(
    "BioVance research prototype · Synthetic patient stream · "
    "Not a diagnostic system"
)
