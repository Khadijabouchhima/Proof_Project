# Personalization engine (RQ1)


## Purpose

Learn a robust patient-specific baseline from historical observations while avoiding overfitting when limited patient history is available.

Code: `src/biovance/personalization/` · config: `configs/personalization.yaml` · tests: `tests/test_personalization.py`

## Pipeline
1. **Normalize** the upstream table (aliases in the YAML; context labels mapped to rest/active/sleep).
2. **Eligibility**: value present, quality >= 0.5 (unknown quality = ineligible), context allowed (only if `context_mode: rest_only`)
Default threshold = 0.5
Tunable through calibration only..
3. **Daily aggregation**: median of eligible observations per day; fewer than `min_obs_per_day` -> missing (kept in the grid).
4. **Frozen baseline** on days [0, baseline_days): median and 1.4826*MAD; needs >= min_valid_days valid days, else cold start.
5. **Shrinkage** toward a population prior fitted on TRAIN patients only.
6. **Score** days after the window: z = orientation * (x - center) / scale. Baseline window observations are used exclusively for baseline estimation and are excluded from downstream deviation scoring.
## What is learned
Nothing per person beyond robust statistics. The only fitted objects are the population prior (pooled mean/SD,
between-person variance tau^2 of centers, typical within-person scale) and the tuned n0 (calibration patients).

## Ablation modes (one controlled change each)
P0 `population` -> P1 `center` -> P2 `center_scale` -> P3 `shrunk`. All modes score exactly the same days.

## Input fields
Essential:
patient_id
timestamp
signal_value
signal_name
signal_quality


## Outputs

For each patient and signal:

- baseline_center
- baseline_scale
- baseline_confidence
- baseline_mode (population / center / center_scale / shrunk)

For scored days:

- z_score
- eligibility_flag
- quality_flag


## Known limits
- Day-to-day noise is autocorrelated, so the effective baseline sample size is smaller than the number of days;
  the standard error of the center is slightly underestimated.
- Baseline contamination (drift already inside the window) cannot be detected here; the simulator avoids it by design.
- Frozen baseline only; an adaptive baseline is on the cut list.
