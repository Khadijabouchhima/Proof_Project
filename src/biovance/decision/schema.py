from __future__ import annotations

from enum import Enum


class DecisionState(str, Enum):
    WARN = "WARN"
    MONITOR = "MONITOR"
    ABSTAIN = "ABSTAIN"


class DecisionCode(str, Enum):
    WARN_STRONG_FUSION = "WARN_STRONG_FUSION"

    WARN_FUSION_WITH_RISK = (
        "WARN_FUSION_WITH_RISK"
    )

    MONITOR_NO_ESCALATION = (
        "MONITOR_NO_ESCALATION"
    )

    ABSTAIN_HIGH_UNCERTAINTY = (
        "ABSTAIN_HIGH_UNCERTAINTY"
    )

    ABSTAIN_INSUFFICIENT_EVIDENCE = (
        "ABSTAIN_INSUFFICIENT_EVIDENCE"
    )

    ABSTAIN_FUSION_UNAVAILABLE = (
        "ABSTAIN_FUSION_UNAVAILABLE"
    )


DECISION_OUTPUT_COLUMNS = [
    "decision_state",
    "decision_code",
    "decision_reason_codes",
    "decision_explanation",
    "decision_version",
]