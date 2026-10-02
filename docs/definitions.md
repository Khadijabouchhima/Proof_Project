# Definitions v2.1

Source of truth: `configs/definitions.yaml` (global), `configs/simulator.yaml`, `configs/real_checks.yaml`.
Executable versions: `src/biovance/events.py` (unit-tested in `tests/test_events.py`).

## Scope
- **Global** (`definitions.yaml`): horizon, reference thresholds, warning rules, states, false-alarm budgets, statistics.
- **Simulator benchmark** (`simulator.yaml`): clean clinic-process `T_ref`, 28-day baseline, subject-level splits, seeds.
- **Real checks** (`real_checks.yaml`): no onset labels exist, so no `T_ref`, no horizon, no supervised risk model; only label-free stages run.

## Reference event T_ref
Readings come from the subject's **reference sequence** (a clean clinic-visit process in the simulator truth table).
A reading qualifies if SBP >= 130 OR DBP >= 80 (sensitivity: 140/90).

**Consecutive** means *adjacent in the reference sequence*: the next reading after a qualifying reading must also qualify.
A non-qualifying reading between two qualifying ones breaks the run. The gap between the two adjacent readings must be
<= `maximum_gap_days` (7). `T_ref` is the day of the second reading.
(The same rule is used in `find_reference_event`; edge cases are tested.)

## Time points
`T_onset` (simulator ground truth, hidden from models) · `T_w` (first reliable warning) · `T_ref` (benchmark reference event).

## Warning outcome partition (every warning episode falls in exactly one class)
| Condition | Outcome | Counts as |
|---|---|---|
| subject has no reference event | FALSE_ALARM_NO_EVENT | false alarm |
| T_w < T_onset | FALSE_ALARM_PREMATURE | false alarm (even if the event follows soon) |
| T_onset <= T_w <= T_ref - 7 | USEFUL | event caught early |
| T_ref - 7 < T_w < T_ref | LATE_MISS | miss, **not** a false alarm |

Days at or after `T_ref` are excluded from evaluation. An event is caught if at least one episode is USEFUL.
This resolves the ambiguity in Spec v2.0 sections 6-7: a premature warning is always a false alarm.
`min_lead_days` has one authoritative value (`task.min_lead_days`).

## Warning episode
Risk >= tau for k = 3 consecutive eligible days (+ quality and evidence gates); 14-day refractory period.
False alarms per subject-month = false-alarm episodes / total subject-months observed
(non-drifters: all days; drifters: days before `T_ref`).

## Operating point
Thresholds chosen on calibration subjects at a fixed false-alarm budget (primary 0.3 per subject-month);
all models compared at that budget. Hyperparameters (e.g. shrinkage n0) live in `models.yaml`, not here.

## Real-data stages
Label-free only: quality, baseline, context, trajectory, detectors, evidence state. A supervised risk model needs `T_ref`
labels, which exist only in simulation. Dryad ambulatory BP never uses the reference rule (thresholds differ from clinic BP).

Design values (H = 30, k = 3, 7-day lead) are choices, not literature values; sensitivity checks justify them.
