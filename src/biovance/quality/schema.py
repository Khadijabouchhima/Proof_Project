from dataclasses import dataclass

@dataclass
class QualityInput:
    patient_id: str
    timestamp: str
    signal_name: str
    signal_value: float
    signal_quality: float | None

@dataclass
class QualityResult:
    quality_score: float
    quality_state: str
    missing_flag: bool
    outlier_flag: bool