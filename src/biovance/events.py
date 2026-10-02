"""Executable definitions of the reference event and warning outcomes.

These two functions pin down the wording of the spec so that it cannot be misread:
  * "consecutive" = ADJACENT readings in the subject's reference sequence
  * every warning episode falls into exactly one outcome class
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence

import numpy as np

from biovance.config import Reference


class Outcome(str, Enum):
    USEFUL = "USEFUL"                          # T_onset <= T_w <= T_ref - min_lead
    LATE_MISS = "LATE_MISS"                    # T_ref - min_lead < T_w < T_ref: too late, NOT a false alarm
    FALSE_ALARM_PREMATURE = "FALSE_ALARM_PREMATURE"      # drifter, T_w < T_onset
    FALSE_ALARM_NO_EVENT = "FALSE_ALARM_NO_EVENT"        # subject never has a reference event


def find_reference_event(days: Sequence[float], sbp: Sequence[float], dbp: Sequence[float],
                         ref: Reference) -> Optional[int]:
    """Return T_ref: the day of the last reading of the first run of `consecutive_readings`
    ADJACENT qualifying readings, where each gap between adjacent readings is <= maximum_gap_days.

    A non-qualifying reading between two qualifying ones breaks the run
    (adjacent in the reference sequence, not 'the next qualifying ones').
    """
    days = np.asarray(days, dtype=float)
    q = (np.asarray(sbp) >= ref.sbp) | (np.asarray(dbp) >= ref.dbp)
    m = ref.consecutive_readings
    for k in range(m - 1, len(days)):
        window = slice(k - m + 1, k + 1)
        gaps = np.diff(days[window])
        if q[window].all() and (gaps <= ref.maximum_gap_days).all():
            return int(days[k])
    return None


def classify_warning(t_w: float, t_onset: Optional[float], t_ref: Optional[float],
                     min_lead_days: int) -> Outcome:
    """Classify ONE warning episode. t_onset/t_ref are None for subjects without an event.

    Days at or after T_ref are excluded from evaluation, so t_w >= t_ref is an error.
    """
    if t_ref is None:
        return Outcome.FALSE_ALARM_NO_EVENT
    if t_onset is None:
        raise ValueError("a subject with a reference event must have an onset")
    if t_w >= t_ref:
        raise ValueError("warnings at/after T_ref are excluded from evaluation")
    if t_w < t_onset:
        return Outcome.FALSE_ALARM_PREMATURE
    if t_w <= t_ref - min_lead_days:
        return Outcome.USEFUL
    return Outcome.LATE_MISS


def is_false_alarm(outcome: Outcome) -> bool:
    return outcome in (Outcome.FALSE_ALARM_PREMATURE, Outcome.FALSE_ALARM_NO_EVENT)


def event_caught(outcomes: Sequence[Outcome]) -> bool:
    """An event is caught early if at least one warning episode is USEFUL."""
    return any(o is Outcome.USEFUL for o in outcomes)
