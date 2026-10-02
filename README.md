# BIOVANCE (certAIn)

Personalized, context-aware, uncertainty-aware AIoT early warning of emerging
hypertension risk. **Research prototype. Not a diagnostic device.**

Built for the IEEE EMBS Tunisia BIOVANCE challenge (six research questions, 10 points each).

## Status

| Step | Content | State |
|---|---|---|
| 1 | Skeleton, definitions v2.1 (global / simulator / real-checks configs), event logic + tests | done |
| 1b | Personalization engine (RQ1) + tests | done |
| 2 | Simulator + corruptions | next |
| 3 | Splits, past-only baselines, leakage tests | todo |
| 4 | Alarm-metric harness (lead time, false alarms) | todo |
| 5 | M0 / M1 (RQ1) | todo |
| 6+ | M2-M5, robustness, uncertainty, paper, slides | todo |

## Quick start

```bash
pip install -r requirements.txt
make test
```

## Design rules

1. Every scientific constant lives in `configs/` and is read through `biovance.config`.
2. Subject-level splits only; baselines and features use data up to time t only.
3. The simulator's `truth` table is never a model input.
4. Thresholds are tuned on calibration subjects only; models are compared at matched false-alarm burden.
5. Claims on simulated data are methodological, not clinical.

See `docs/definitions.md` for the frozen definitions.
