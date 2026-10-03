from __future__ import annotations

from enum import Enum


class RiskCode(str, Enum):
    OK = "OK"
    LOW_INPUT_COMPLETENESS = "LOW_INPUT_COMPLETENESS"


RISK_OUTPUT_COLUMNS = [
    "risk_probability",
    "risk_log_odds",
    "risk_input_completeness",
    "risk_code",
    "risk_model_version",
    "risk_reason_codes",
    "risk_explanation",
]