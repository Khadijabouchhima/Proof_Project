# Data Quality Engine (RQ0)

## Purpose

Assess whether incoming physiological observations are reliable enough for downstream reasoning.

The engine does not estimate risk or detect abnormalities.

It only evaluates data trustworthiness.

---

## Pipeline

1. Normalize input fields.
2. Validate timestamps.
3. Detect missing values.
4. Detect impossible values.
5. Evaluate sensor confidence.
6. Estimate completeness.
7. Produce quality score.
8. Produce quality state.

---

## Inputs

patient_id
timestamp
signal_name
signal_value
signal_quality

---

## Outputs

quality_score
quality_state
missing_flag
outlier_flag
coverage_score

---

## States

good
acceptable
poor
insufficient_evidence

---

## Known Limits

Does not determine clinical validity.

Only assesses data quality.