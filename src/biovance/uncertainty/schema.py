from __future__ import annotations

from enum import Enum


class UncertaintyLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class UncertaintyCode(str, Enum):
    OK = "OK"
    PARTIAL_EVIDENCE = "PARTIAL_EVIDENCE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


UNCERTAINTY_OUTPUT_COLUMNS = [
    "certainty_score",
    "uncertainty_score",
    "uncertainty_level",
    "uncertainty_code",
    "source_coverage",
    "coherence_score",
    "uncertainty_reason_codes",
    "uncertainty_explanation",
    "uncertainty_version",
]