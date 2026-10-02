import pytest

from biovance.config import load_definitions
from biovance.events import Outcome, classify_warning, event_caught, find_reference_event, is_false_alarm

REF = load_definitions().reference   # 130/80, 2 adjacent readings, gap <= 7 days
MIN_LEAD = 7


def test_two_adjacent_qualifying_readings_confirm_on_second_day():
    assert find_reference_event([4, 7, 11], [120, 131, 133], [70, 70, 70], REF) == 11


def test_non_qualifying_reading_between_breaks_the_run():
    # 131, 120, 133: the two qualifying readings are not adjacent
    assert find_reference_event([4, 7, 11], [131, 120, 133], [70, 70, 70], REF) is None


def test_gap_over_seven_days_breaks_the_run():
    assert find_reference_event([4, 12], [131, 133], [70, 70], REF) is None   # gap = 8
    assert find_reference_event([4, 11], [131, 133], [70, 70], REF) == 11     # gap = 7 is allowed


def test_dbp_alone_qualifies_and_single_reading_does_not():
    assert find_reference_event([4, 7], [118, 118], [80, 81], REF) == 7
    assert find_reference_event([4, 7], [131, 118], [70, 70], REF) is None


def test_no_event():
    assert find_reference_event([4, 7, 11], [118, 119, 121], [70, 71, 72], REF) is None


@pytest.mark.parametrize("t_w,expected", [
    (49, Outcome.FALSE_ALARM_PREMATURE),   # before onset (50)
    (50, Outcome.USEFUL),                  # exactly at onset
    (93, Outcome.USEFUL),                  # T_ref - 7 = 93: still useful
    (94, Outcome.LATE_MISS),               # too late to count, not a false alarm
    (99, Outcome.LATE_MISS),
])
def test_drifter_partition(t_w, expected):
    assert classify_warning(t_w, t_onset=50, t_ref=100, min_lead_days=MIN_LEAD) is expected


def test_premature_warning_followed_by_event_is_still_a_false_alarm():
    # the ambiguity in spec v2.0 section 7: premature is ALWAYS a false alarm
    out = classify_warning(40, t_onset=50, t_ref=60, min_lead_days=MIN_LEAD)
    assert out is Outcome.FALSE_ALARM_PREMATURE and is_false_alarm(out)


def test_non_event_subject_any_warning_is_false_alarm():
    assert classify_warning(80, None, None, MIN_LEAD) is Outcome.FALSE_ALARM_NO_EVENT


def test_late_miss_is_not_a_false_alarm_and_does_not_catch():
    out = classify_warning(95, 50, 100, MIN_LEAD)
    assert not is_false_alarm(out) and not event_caught([out])


def test_event_caught_if_any_episode_useful():
    outs = [classify_warning(40, 50, 100, MIN_LEAD), classify_warning(70, 50, 100, MIN_LEAD)]
    assert event_caught(outs) and sum(is_false_alarm(o) for o in outs) == 1


def test_warning_at_or_after_t_ref_is_rejected():
    with pytest.raises(ValueError, match="excluded"):
        classify_warning(100, 50, 100, MIN_LEAD)
