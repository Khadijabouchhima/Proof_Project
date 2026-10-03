# Data Quality Engine

## Purpose

Assess whether each physiological observation is trustworthy enough to support downstream BioVance reasoning.

The Data Quality Engine does **not** diagnose disease, detect physiological abnormality, or make warning decisions. It annotates observations with a standardized measure of data reliability so later engines can decide how much confidence to place in the evidence.

Code: `src/biovance/quality/` · config: `configs/quality.yaml` · tests: `tests/test_quality.py`

---

## Role in the BioVance pipeline

```text
Raw / preprocessed observations
        ↓
Data Quality Engine
        ↓
quality_score
quality_status
quality_reasons
        ↓
Personalization
        ↓
Context
        ↓
Deviation
        ↓
Temporal
        ↓
Risk
        ↓
Uncertainty
        ↓
Decision
```

The engine follows a non-destructive design: **rows are never removed**. Poor-quality observations remain in the dataset with explicit quality annotations.

---

## Core question

For each patient, signal, and timestamp:

> **How trustworthy is this observation?**

This is intentionally different from:

> **How abnormal is this observation?**

For example, a heart rate of 170 bpm may be physiologically unusual in some contexts, but if it was measured with high coverage and sufficient samples it can still be a **high-quality observation**. Physiological abnormality belongs to the Context, Deviation, and Temporal engines.

---

## Canonical input fields

Required:

- `patient_id`
- `timestamp`
- `signal`
- `value`

Optional quality evidence:

- `coverage`
- `sample_count`
- `sensor_confidence`
- `valid`

Canonical signal names currently supported by configuration include:

- `hr`
- `hrv`
- `sbp`
- `dbp`
- `spo2`

Missing optional quality fields are allowed. The engine re-normalizes the weighted score using only the quality components that are actually available and enabled.

---

## PMData mapping

For the current PMData daily model-ready table:

```text
participant       -> patient_id
date              -> timestamp
"hr"              -> signal
hr_rest_median    -> value
hr_coverage       -> coverage
hr_rest_n         -> sample_count
hr_conf_mean      -> sensor_confidence
hr_valid_day      -> valid
```

The PMData adapter is implemented in:

```text
src/biovance/quality/adapters.py
```

using:

```python
from_pmdata_daily(...)
```

`hr_conf_mean` is currently preserved as metadata but is not used in the quality score until its device-specific scale is explicitly validated.

---

## Quality components

The first implementation uses four possible components.

### 1. Coverage score

Coverage represents how much of the expected observation period is represented.

```text
coverage_score = clip(coverage, 0, 1)
```

Example:

```text
coverage = 0.97
coverage_score = 0.97
```

---

### 2. Sample sufficiency score

Sample count is converted to a gradual score rather than a binary pass/fail rule.

```text
sample_score = clip(sample_count / target_samples, 0, 1)
```

Example with `target_samples = 100`:

```text
25 samples   -> 0.25
50 samples   -> 0.50
100 samples  -> 1.00
500 samples  -> 1.00
```

`target_samples` is signal- and dataset-specific and is configured in `quality.yaml`.

It should be treated as an engineering/calibration parameter rather than a universal physiological constant.

---

### 3. Sensor-confidence score

If a signal has a documented confidence scale, it can be normalized to `[0, 1]`:

```text
confidence_score =
    (confidence - confidence_min)
    /
    (confidence_max - confidence_min)
```

and then clipped to `[0, 1]`.

For PMData HR this component is currently disabled because the semantics of `hr_conf_mean` have not yet been formally validated.

---

### 4. Validity score

If an upstream dataset provides an explicit validity flag:

```text
valid = True  -> validity_score = 1.0
valid = False -> validity_score = 0.0
```

An explicitly invalid observation is treated as a hard quality failure.

---

## Overall quality score

For non-hard-failure observations, the score is a weighted mean over the available enabled components.

Current PMData defaults:

```text
coverage weight   = 0.45
sample weight     = 0.35
confidence weight = 0.00
validity weight   = 0.20
```

Conceptually:

```text
quality_score =
    weighted sum of available component scores
    /
    sum of their available weights
```

The result is always bounded to:

```text
0.0 <= quality_score <= 1.0
```

Missing optional components do **not** automatically contribute zero.

---

## Quality statuses

The continuous score is mapped to an interpretable status:

```text
GOOD  : quality_score >= 0.80
FAIR  : quality_score >= 0.50 and < 0.80
POOR  : quality_score < 0.50
```

Thresholds are stored in `configs/quality.yaml`.

These values are engineering defaults and should be calibrated or justified on training/calibration data rather than tuned on the final test set.

---

## Hard failures

Some situations override the weighted score because the observation is not considered usable evidence.

Current hard-failure conditions include:

- missing signal value
- `valid = False`
- critically low coverage
- sample count below the configured hard minimum
- no usable quality evidence

For these cases:

```text
quality_status = POOR
```

Recommended implementation:

```text
quality_score = 0.0
```

This avoids representing an unusable observation as being almost FAIR.

---

## Outputs

For every input row, the engine returns:

```text
patient_id
timestamp
signal
value

coverage
sample_count
sensor_confidence
valid

coverage_score
sample_score
confidence_score
validity_score

quality_score
quality_status
quality_evidence_count
quality_reasons
```

Example:

```text
patient_id       p01
timestamp        2019-11-01
signal           hr
value            58

coverage_score   0.99
sample_score     1.00
validity_score   1.00

quality_score    0.99
quality_status   GOOD

quality_reasons  quality_evidence_adequate
```

Poor-quality example:

```text
quality_score    0.00
quality_status   POOR

quality_reasons
missing_signal_value;
invalid_observation;
critically_low_coverage
```

---

## Explainability

The engine stores explicit reason codes rather than only a scalar score.

Examples:

- `quality_evidence_adequate`
- `missing_signal_value`
- `invalid_observation`
- `critically_low_coverage`
- `reduced_coverage`
- `sample_count_unavailable`
- `insufficient_samples`
- `below_target_sample_count`
- `confidence_unavailable`
- `no_quality_evidence`

The `explain(...)` helper can convert an assessed observation into a short human-readable summary.

Example:

```text
hr quality=0.94 (GOOD); coverage=0.96 sample=1.00 validity=1.00;
reasons=quality_evidence_adequate
```

---

## Downstream use

The Data Quality Engine does **not** directly modify a physiological deviation.

It should not implement:

```text
weighted_deviation = deviation_score * quality_score
```

inside this engine.

Instead:

```text
Deviation Engine
-> how abnormal is the physiology?

Data Quality Engine
-> how trustworthy is the observation?

Uncertainty Engine
-> how confident should the system be overall?

Decision Engine
-> WARN / MONITOR / ABSTAIN
```

This separation avoids hiding a large physiological deviation simply because data quality is imperfect.

The Personalization Engine can also use the standardized quality score as an eligibility input when constructing patient baselines. The existing personalization design already filters observations based on signal quality before daily aggregation.

---

## PMData demonstration

The current PMData demonstration uses:

```text
src/biovance/data/processed/pmdata_model_ready.csv
```

and:

```text
experiments/quality/demo_quality.py
```

Run:

```bash
python experiments/quality/demo_quality.py
```

The demo reports:

- status counts
- quality-score distribution
- lowest-quality observations
- highest-quality observations
- reason-code patterns

This is a smoke test of real-data compatibility, not a clinical performance benchmark.

---

## Tests

Run:

```bash
python -m pytest tests/test_quality.py -v
```

The test suite should verify at least:

- schema normalization
- missing required-column errors
- high-quality observations become `GOOD`
- reduced-quality observations produce lower scores
- critically low coverage becomes `POOR`
- invalid observations become `POOR`
- missing signal values become `POOR`
- insufficient samples become `POOR`
- scores stay within `[0, 1]`
- missing optional components are re-normalized rather than treated as zero
- observations are never removed
- PMData adapter field mapping
- PMData adapter -> engine integration
- deterministic behavior
- no future leakage
- explainability output

---

## No fitting stage

The current Data Quality Engine is deterministic and therefore has no `fit()` step.

Unlike the Personalization Engine, it does not learn patient baselines or population priors.

Any thresholds or weights that are later calibrated must be calibrated on designated training/calibration data only and then frozen before final evaluation.

---

## Known limitations

- The current PMData implementation focuses first on daily resting HR.
- The `hr_conf_mean` sensor-confidence scale is preserved but not yet used because its semantics require validation.
- `target_samples` is currently a configurable engineering parameter and should be justified from the underlying sampling process or calibration experiments.
- A daily aggregate can hide within-day clustering, irregular sampling, motion artifacts, or transient sensor failures.
- Richer raw-sensor quality analysis may later incorporate gap structure, sampling irregularity, motion artifacts, and modality-specific failure modes.
- `GOOD`, `FAIR`, and `POOR` are data-quality categories only; they are not health-risk labels.
- The engine does not diagnose disease or determine whether a physiological value is clinically abnormal.

---

## Design principle

The quality layer should answer only:

> **Can the system trust this observation enough to use it as evidence?**

It should preserve the observation, quantify reliability transparently, and pass that information downstream for personalization, uncertainty estimation, and final decision-making.
