# PROOF

**Personalized Robust Observation Framework**

PROOF is a multimodal early-warning framework built for the BioVance technical challenge. The main idea is simple: instead of asking only whether a physiological value is abnormal for the population, PROOF asks whether it is unusual for the person, whether the change persists over time, whether several signals agree, and whether there is enough evidence to make a reliable decision.

The system is designed as a sequence of small, testable engines rather than one black-box model.

---

## What problem does PROOF solve?

A single physiological measurement can be misleading.

A high heart rate may be normal during exercise.  
A slightly elevated blood-pressure reading may be temporary noise.  
A more concerning situation is when several signals move away from a person's own baseline, persist over time, and support the same interpretation.

PROOF was built around that idea.

The framework produces one of three final states:

- **WARN** — current physiological evidence is strong enough to justify escalation
- **MONITOR** — usable evidence exists, but warning conditions are not met
- **ABSTAIN** — there is not enough reliable evidence to make a current decision

This is a research prototype and is **not a medical diagnosis system**.

---

## Architecture

The final pipeline is:

```text
Current physiology

Quality
   ↓
Personalization
   ↓
Context
   ↓
Deviation
   ↓
Temporal
   ↓
Fusion
   │
   ├──────────────┐
   │              ▼
   │         Uncertainty
   │              │
   │              ▼
   └──────────→ Decision
                   │
                   ▼
             Explainability


Background context

Framingham variables
        ↓
     Risk Engine
        │
        ├────────→ Uncertainty
        └────────→ Decision
```

The two branches are intentionally different:

- **Fusion** answers: *What is happening physiologically right now?*
- **Risk** answers: *What is the person's longer-term cardiovascular background risk?*
- **Uncertainty** asks: *How much should the system trust the available evidence?*
- **Decision** asks: *Should the system WARN, MONITOR, or ABSTAIN?*

A key safety rule is that **background cardiovascular risk alone cannot create a current WARN**.

---

## Engines

### 1. Quality Engine

The Quality Engine checks whether an observation is reliable enough to contribute to downstream analysis.

It considers things such as:

- missingness
- signal availability
- measurement validity
- coverage
- confidence

It does not decide whether a value is abnormal. Its job is only to decide how trustworthy the observation is.

Example outputs:

```text
GOOD
FAIR
POOR
```

On the PMData evaluation used in the project:

- 1,199 observations were classified as GOOD
- 21 as FAIR
- 1,079 as POOR
- 2,299 observations were preserved in total

---

### 2. Personalization Engine

The Personalization Engine learns what is normal for each person.

The baseline is based on robust statistics:

- median for the personal center
- Median Absolute Deviation (MAD) for the personal scale
- shrinkage toward a population prior when personal history is limited

The baseline is frozen after the learning period. This is important because future deterioration should not slowly become the person's new normal.

The project also checks that future observations do not alter earlier scores.

The controlled personalization experiments compared:

```text
P0  population
P1  personal center
P2  personal center + scale
P3  personal center + scale + shrinkage
```

In the synthetic drift benchmark, the shrunk personalized model achieved AUROC above 0.90, while population-only scoring remained below 0.75.

---

### 3. Context Engine

Physiology depends on context.

The same heart rate can mean something very different during exercise, sleep, or rest.

The Context Engine uses four states:

```text
SLEEP
REST
ACTIVE
UNKNOWN
```

It was evaluated using BAIGUTANOVA motion and sleep information.

The system does not force a context label when evidence is unclear. `UNKNOWN` is treated as a valid output.

---

### 4. Deviation Engine

The Deviation Engine compares the current physiological value with the person's own baseline.

It produces a personalized, direction-aware deviation score.

The descriptive deviation bands are:

```text
NORMAL
ELEVATED
LARGE
EXTREME
```

The thresholds used in the research prototype are:

```text
NORMAL    |z| < 2.0
ELEVATED  2.0 ≤ |z| < 3.0
LARGE     3.0 ≤ |z| < 4.5
EXTREME   |z| ≥ 4.5
```

These are statistical deviation bands, not clinical diagnostic thresholds.

On PMData, the engine preserved all 2,299 rows and produced 785 usable deviations:

| State | Count |
|---|---:|
| NORMAL | 646 |
| ELEVATED | 69 |
| LARGE | 46 |
| EXTREME | 24 |

The Deviation Engine passed 33 unit tests.

---

### 5. Temporal Engine

One unusual reading is not enough to represent deterioration.

The Temporal Engine checks whether deviations:

- persist
- recur
- remain directionally consistent
- worsen over time

The final configuration uses 2-hour, 6-hour, and 12-hour windows, with a maximum allowed gap of 120 minutes.

All temporal calculations are causal: only current and past observations are used.

The Temporal Engine was evaluated on 4,848 DRYAD signal-level rows and passed 22 unit tests.

---

### 6. Fusion Engine

The Fusion Engine combines current physiological evidence across signals.

This is the point where separate signal-level evidence becomes one interpretation for a patient at a specific timestamp.

The final Fusion implementation is atomic:

```text
one patient + one timestamp = one Fusion result
```

This was an important integration fix. Earlier versions could produce multiple downstream results for the same physiological moment.

Final Fusion states include:

```text
NO_SUPPORT
SINGLE_SIGNAL_SUPPORT
MULTISIGNAL_SAME_FAMILY
MULTIMODAL_SUPPORT
MULTIMODAL_CONSENSUS
CONFLICTING_EVIDENCE
```

On the final DRYAD integration:

- 4,848 temporal rows
- 1,616 unique patient/timestamp moments
- 585 usable Fusion moments
- 0 duplicate patient/timestamp Fusion rows

Usable Fusion distribution:

| Fusion state | Count |
|---|---:|
| NO_SUPPORT | 382 |
| SINGLE_SIGNAL_SUPPORT | 128 |
| CONFLICTING_EVIDENCE | 23 |
| MULTISIGNAL_SAME_FAMILY | 19 |
| MULTIMODAL_CONSENSUS | 17 |
| MULTIMODAL_SUPPORT | 16 |

The Fusion Engine passed 20 unit tests.

---

### 7. Background Risk Engine

The Risk Engine is separate from the current physiological branch.

It uses a logistic-regression model trained on Framingham data to estimate long-term cardiovascular background risk.

It does **not** estimate current physiological deterioration.

The source dataset contained 4,240 records with a TenYearCHD prevalence of about 15.2%.

On the reserved test set, the logistic model achieved approximately:

| Metric | Result |
|---|---:|
| AUROC | 0.699 |
| AUPRC | 0.330 |
| Brier score | 0.118 |
| ECE | 0.026 |

The trained model is saved as:

```text
models/risk/risk_v1_logistic_framingham.joblib
```

---

### 8. Uncertainty Engine

The Uncertainty Engine measures how trustworthy the current evidence is.

It considers:

- Fusion confidence
- source availability
- risk-input completeness
- evidence coherence
- conflicting physiological evidence

The main uncertainty thresholds are:

```text
MODERATE ≥ 0.30
HIGH     ≥ 0.60
```

The important distinction is:

```text
NO_SUPPORT ≠ NO_EVIDENCE
```

`NO_SUPPORT` means usable physiological evidence exists, but it does not support escalation.

Missing Fusion means the system does not have enough current physiological evidence to make that judgment.

---

### 9. Decision Engine

The Decision Engine combines current Fusion evidence, background risk context, and uncertainty.

It returns:

```text
WARN
MONITOR
ABSTAIN
```

Examples of intended behavior:

- strong multimodal physiological support → WARN
- single-signal evidence → MONITOR
- conflicting physiological evidence → MONITOR
- missing current Fusion → ABSTAIN
- high background risk with no current Fusion → ABSTAIN

On the final usable DRYAD set:

- 16 WARN
- 569 MONITOR
- 0 ABSTAIN among the 585 usable Fusion moments

All 23 conflicting-evidence moments remained MONITOR.

On the Framingham risk-only safety pathway:

- 1,272 cases
- 0 WARN
- 0 MONITOR
- 1,272 ABSTAIN

This confirms the design rule that long-term risk alone cannot create a current warning.

The Decision Engine passed 16 unit tests.

---

### 10. Explainability Engine

Explainability is strictly downstream of Decision.

It does not recompute or change the final action.

Its job is to explain:

- what evidence supported the result
- what limited confidence
- whether signals agreed or conflicted
- whether background risk was available
- why the system warned, monitored, or abstained

The Explainability Engine passed 16 unit tests.

The integration also checks that the explanation preserves the original decision.

---

## Datasets

The project uses several datasets because no single available dataset contains all the signals, context, risk variables, and labels needed for the full problem.

| Dataset | Main use |
|---|---|
| PMData | quality, personalization, deviation |
| BAIGUTANOVA | context reasoning |
| DRYAD | temporal reasoning, Fusion, uncertainty, decision |
| Framingham | background cardiovascular risk |
| Synthetic data | controlled experiments, branch tests, live simulator |

The datasets are **not artificially joined at patient level**.

In particular, DRYAD participants and Framingham participants are unrelated.

---

## Final integration validation

After the engines were tested separately, the full downstream chain was validated:

```text
Deviation
→ Temporal
→ Fusion
→ Uncertainty
→ Decision
→ Explainability
```

The final integration checks included:

- one Fusion row per physiological moment
- downstream row alignment
- WARN requires current physiological support
- conflicting evidence never produces WARN
- background risk alone never produces WARN
- missing current Fusion leads to abstention
- correct abstention reason
- Explainability preserves Decision

Final result:

```text
16 / 16 integration checks passed
```

---

## Live simulator

The repository also contains a live synthetic-patient prototype.

The simulator generates new physiological measurements over time rather than replaying rows from DRYAD or Framingham.

Available scenarios include:

- stable physiology
- gradual deterioration
- conflicting signals
- sensor dropout

The purpose of the simulator is to make the framework behavior visible in real time.

It is a demonstration of the mechanism, not additional clinical evidence.

---

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

The project is run with `src` on the Python path:

```bash
export PYTHONPATH=src
```

or for a single command:

```bash
PYTHONPATH=src pytest
```

---

## Running the tests

Run the full test suite:

```bash
PYTHONPATH=src pytest
```

Individual engine tests can also be run separately, for example:

```bash
PYTHONPATH=src pytest tests/test_personalization.py
PYTHONPATH=src pytest tests/test_deviation.py
PYTHONPATH=src pytest tests/test_temporal.py
PYTHONPATH=src pytest tests/test_fusion.py
PYTHONPATH=src pytest tests/test_uncertainty.py
PYTHONPATH=src pytest tests/test_decision.py
```

---

## Running the live prototype

If Streamlit is installed:

```bash
PYTHONPATH=src python -m streamlit run app/app.py
```

If needed:

```bash
python -m pip install streamlit
```

---

## Project structure

A simplified view of the repository:

```text
BIOVANCE_4/
├── app/
│   ├── app.py
│   └── live_pipeline.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── experiments/
│   └── integration/
│
├── models/
│   └── risk/
│
├── results/
│   ├── integration/
│   ├── uncertainty/
│   └── ...
│
├── src/
│   └── biovance/
│       ├── quality/
│       ├── personalization/
│       ├── context/
│       ├── deviation/
│       ├── temporal/
│       ├── fusion/
│       ├── risk/
│       ├── uncertainty/
│       ├── decision/
│       └── explainability/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

## What PROOF does not claim

This project validates the **mechanism** of a personalized early-warning pipeline.

It does not yet prove:

- clinical sensitivity for future hypertension
- clinical specificity
- a validated warning lead time
- real-world medical effectiveness

A prospective longitudinal cohort containing synchronized wearable physiology, context, cardiovascular risk factors, and clinically confirmed transition points would be needed for that.

---

## Main takeaway

PROOF was built around one simple principle:

> A warning should not come from one unusual number.

A useful warning should depend on whether the data is reliable, whether the value is unusual for that person, whether the change continues over time, whether several signals agree, and whether the system has enough evidence to trust its own decision.

That is the role of the framework.
