from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import ExplainabilityConfig
from .schema import (
    EXPLAINABILITY_OUTPUT_COLUMNS,
    validate_explainability_input,
)


class ExplainabilityEngine:
    """
    BioVance Explainability Engine v1.

    Question answered:
        "Why did BioVance produce this WARN / MONITOR / ABSTAIN
        decision?"

    Important:
        This engine does NOT recompute or alter the Decision.

        Decision state and decision code are treated as immutable
        upstream facts.

        Explainability only translates already-computed evidence into
        deterministic structured text.
    """

    def __init__(
        self,
        config: ExplainabilityConfig,
    ) -> None:
        self.config = config

    # ========================================================
    # Public API
    # ========================================================

    def explain(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce one explanation for every upstream Decision row.

        Row count and row order are preserved.
        """

        validate_explainability_input(
            df
        )

        rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            rows.append(
                self._explain_row(
                    row
                )
            )

        result = pd.DataFrame(
            rows,
            index=df.index,
        )

        result = result[
            list(
                EXPLAINABILITY_OUTPUT_COLUMNS
            )
        ]

        return result.reset_index(
            drop=True
        )

    # ========================================================
    # Single-row explanation
    # ========================================================

    def _explain_row(
        self,
        row: pd.Series,
    ) -> dict[str, Any]:

        decision_state = self._text(
            row.get(
                "decision_state"
            )
        )

        decision_code = self._text(
            row.get(
                "decision_code"
            )
        )

        decision_reasons = self._parse_codes(
            row.get(
                "decision_reason_codes"
            )
        )

        uncertainty_reasons = self._parse_codes(
            row.get(
                "uncertainty_reason_codes"
            )
        )

        supporting_signals = self._parse_list(
            row.get(
                "supporting_signals"
            )
        )

        opposing_signals = self._parse_list(
            row.get(
                "opposing_signals"
            )
        )

        primary_reason = self._primary_reason(
            decision_state=decision_state,
            decision_code=decision_code,
            decision_reasons=decision_reasons,
        )

        supporting_evidence = self._supporting_evidence(
            row=row,
            supporting_signals=supporting_signals,
        )

        limiting_factors = self._limiting_factors(
            row=row,
            decision_state=decision_state,
            decision_code=decision_code,
            decision_reasons=decision_reasons,
            uncertainty_reasons=uncertainty_reasons,
            opposing_signals=opposing_signals,
        )

        risk_context = self._risk_context(
            row
        )

        confidence_statement = (
            self._confidence_statement(
                row
            )
        )

        explanation_summary = (
            self._summary(
                row=row,
                decision_state=decision_state,
                decision_code=decision_code,
                supporting_signals=supporting_signals,
                opposing_signals=opposing_signals,
            )
        )

        return {
            "explanation_summary": (
                explanation_summary
            ),
            "primary_reason": (
                primary_reason
            ),
            "supporting_evidence": (
                supporting_evidence
            ),
            "limiting_factors": (
                limiting_factors
            ),
            "risk_context": (
                risk_context
            ),
            "confidence_statement": (
                confidence_statement
            ),
            "explainability_version": (
                self.config.version
            ),
        }

    # ========================================================
    # Summary
    # ========================================================

    def _summary(
        self,
        row: pd.Series,
        decision_state: str,
        decision_code: str,
        supporting_signals: list[str],
        opposing_signals: list[str],
    ) -> str:

        if decision_state == "WARN":
            return self._warn_summary(
                row=row,
                decision_code=decision_code,
                supporting_signals=supporting_signals,
            )

        if decision_state == "MONITOR":
            return self._monitor_summary(
                row=row,
                supporting_signals=supporting_signals,
                opposing_signals=opposing_signals,
            )

        if decision_state == "ABSTAIN":
            return self._abstain_summary(
                row=row,
                decision_code=decision_code,
            )

        return (
            "BioVance produced a decision, but no supported "
            "explanation template is available."
        )

    def _warn_summary(
        self,
        row: pd.Series,
        decision_code: str,
        supporting_signals: list[str],
    ) -> str:

        fusion_state = self._text(
            row.get(
                "fusion_state"
            )
        )

        signal_phrase = (
            self._signal_phrase(
                supporting_signals
            )
        )

        if (
            decision_code
            == "WARN_FUSION_WITH_RISK"
        ):
            if signal_phrase:
                return (
                    "BioVance issued WARN because current "
                    "physiological evidence showed multimodal "
                    f"support involving {signal_phrase}, and "
                    "background cardiovascular risk provided "
                    "additional context. Uncertainty was low "
                    "enough for the upstream Decision Engine "
                    "to permit escalation."
                )

            return (
                "BioVance issued WARN because current "
                "multimodal physiological evidence was "
                "supported by background cardiovascular risk, "
                "with uncertainty low enough for escalation."
            )

        if (
            decision_code
            == "WARN_STRONG_FUSION"
        ):
            if signal_phrase:
                return (
                    "BioVance issued WARN because current "
                    "physiological evidence showed strong "
                    f"multimodal agreement involving "
                    f"{signal_phrase}. Uncertainty was low "
                    "enough for the upstream Decision Engine "
                    "to permit escalation."
                )

            return (
                "BioVance issued WARN because current "
                "physiological evidence showed strong "
                "multimodal agreement and uncertainty was "
                "low enough for escalation."
            )

        if (
            fusion_state
            == "MULTIMODAL_CONSENSUS"
        ):
            return (
                "BioVance issued WARN after strong agreement "
                "across multiple physiological signals."
            )

        return (
            "BioVance issued WARN according to the upstream "
            "Decision Engine using the available current "
            "physiological evidence."
        )

    def _monitor_summary(
        self,
        row: pd.Series,
        supporting_signals: list[str],
        opposing_signals: list[str],
    ) -> str:

        fusion_state = self._text(
            row.get(
                "fusion_state"
            )
        )

        uncertainty_score = self._number(
            row.get(
                "uncertainty_score"
            )
        )

        if (
            fusion_state
            == "CONFLICTING_EVIDENCE"
        ):
            return (
                "BioVance selected MONITOR because the current "
                "physiological signals contain conflicting "
                "evidence rather than consistent multimodal "
                "support for escalation."
            )

        if (
            fusion_state
            == "SINGLE_SIGNAL_SUPPORT"
        ):
            return (
                "BioVance selected MONITOR because concerning "
                "evidence was limited to a single physiological "
                "signal and did not meet the multimodal "
                "requirements for warning."
            )

        if (
            fusion_state
            == "MULTISIGNAL_SAME_FAMILY"
        ):
            return (
                "BioVance selected MONITOR because multiple "
                "signals supported concern, but the support "
                "came from the same physiological family "
                "rather than independent modalities."
            )

        if (
            fusion_state
            == "MULTIMODAL_SUPPORT"
        ):
            return (
                "BioVance selected MONITOR because multimodal "
                "physiological support was present but did not "
                "meet the upstream warning criteria."
            )

        if (
            fusion_state
            == "MULTIMODAL_CONSENSUS"
        ):
            if (
                uncertainty_score
                is not None
            ):
                return (
                    "BioVance selected MONITOR despite strong "
                    "multimodal physiological agreement because "
                    "the upstream uncertainty gate did not "
                    "permit warning escalation."
                )

            return (
                "BioVance selected MONITOR despite multimodal "
                "agreement because the complete upstream "
                "warning criteria were not satisfied."
            )

        if (
            fusion_state
            == "NO_SUPPORT"
        ):
            return (
                "BioVance selected MONITOR because the "
                "currently available physiological evidence "
                "did not provide sufficient concerning support "
                "for warning escalation."
            )

        if opposing_signals:
            return (
                "BioVance selected MONITOR because the current "
                "evidence did not satisfy the upstream warning "
                "criteria and some physiological evidence "
                "opposed escalation."
            )

        if supporting_signals:
            return (
                "BioVance selected MONITOR because some "
                "concerning physiological evidence was present "
                "but the complete warning criteria were not met."
            )

        return (
            "BioVance selected MONITOR because the available "
            "evidence did not meet the upstream criteria for "
            "warning escalation."
        )

    def _abstain_summary(
        self,
        row: pd.Series,
        decision_code: str,
    ) -> str:

        if (
            decision_code
            == "ABSTAIN_FUSION_UNAVAILABLE"
        ):
            risk_probability = self._number(
                row.get(
                    "risk_probability"
                )
            )

            if (
                risk_probability
                is not None
            ):
                return (
                    "BioVance abstained because current "
                    "physiological Fusion evidence was "
                    "unavailable. Background cardiovascular "
                    "risk was available, but background risk "
                    "alone cannot produce a current warning."
                )

            return (
                "BioVance abstained because current "
                "physiological Fusion evidence was unavailable."
            )

        if (
            decision_code
            == "ABSTAIN_HIGH_UNCERTAINTY"
        ):
            return (
                "BioVance abstained because uncertainty in the "
                "currently available evidence was too high for "
                "a reliable warning decision."
            )

        if (
            decision_code
            == "ABSTAIN_INSUFFICIENT_EVIDENCE"
        ):
            return (
                "BioVance abstained because the available "
                "evidence was insufficient to support a "
                "reliable warning decision."
            )

        return (
            "BioVance abstained because the upstream Decision "
            "Engine determined that the available evidence was "
            "not sufficient for warning or monitoring."
        )

    # ========================================================
    # Primary reason
    # ========================================================

    def _primary_reason(
        self,
        decision_state: str,
        decision_code: str,
        decision_reasons: list[str],
    ) -> str:

        mapping = {
            "WARN_STRONG_FUSION": (
                "Strong multimodal physiological evidence"
            ),
            "WARN_FUSION_WITH_RISK": (
                "Multimodal physiological evidence strengthened "
                "by background cardiovascular risk"
            ),
            "MONITOR_NO_ESCALATION": (
                "Warning escalation criteria were not met"
            ),
            "ABSTAIN_HIGH_UNCERTAINTY": (
                "Evidence uncertainty was too high"
            ),
            "ABSTAIN_FUSION_UNAVAILABLE": (
                "Current physiological Fusion evidence was unavailable"
            ),
            "ABSTAIN_INSUFFICIENT_EVIDENCE": (
                "Available evidence was insufficient"
            ),
        }

        if decision_code in mapping:
            return mapping[
                decision_code
            ]

        if decision_reasons:
            return (
                self._humanize_code(
                    decision_reasons[0]
                )
            )

        return (
            f"Upstream Decision Engine selected "
            f"{decision_state}"
        )

    # ========================================================
    # Supporting evidence
    # ========================================================

    def _supporting_evidence(
        self,
        row: pd.Series,
        supporting_signals: list[str],
    ) -> str:

        parts: list[str] = []

        fusion_state = self._text(
            row.get(
                "fusion_state"
            )
        )

        fusion_score = self._number(
            row.get(
                "fusion_score"
            )
        )

        if fusion_state:
            text = (
                "Fusion state was "
                + self._humanize_code(
                    fusion_state
                )
            )

            if fusion_score is not None:
                text += (
                    " with score "
                    + self._fmt_score(
                        fusion_score
                    )
                )

            parts.append(
                text + "."
            )

        if supporting_signals:
            parts.append(
                "Supporting physiological signals: "
                + self._signal_phrase(
                    supporting_signals
                )
                + "."
            )

        available_count = self._number(
            row.get(
                "available_signal_count"
            )
        )

        supporting_count = self._number(
            row.get(
                "supporting_signal_count"
            )
        )

        if (
            available_count is not None
            and supporting_count is not None
        ):
            parts.append(
                f"{int(supporting_count)} of "
                f"{int(available_count)} available signals "
                "supported concern."
            )

        if not parts:
            return (
                "No current physiological supporting evidence "
                "was available."
            )

        return " ".join(
            parts
        )

    # ========================================================
    # Limiting factors
    # ========================================================

    def _limiting_factors(
        self,
        row: pd.Series,
        decision_state: str,
        decision_code: str,
        decision_reasons: list[str],
        uncertainty_reasons: list[str],
        opposing_signals: list[str],
    ) -> str:

        factors: list[str] = []

        if (
            "FUSION_UNAVAILABLE"
            in decision_reasons
            or
            "FUSION_UNAVAILABLE"
            in uncertainty_reasons
            or
            decision_code
            == "ABSTAIN_FUSION_UNAVAILABLE"
        ):
            factors.append(
                "current physiological Fusion evidence was unavailable"
            )

        if (
            "HIGH_UNCERTAINTY"
            in decision_reasons
            or
            decision_code
            == "ABSTAIN_HIGH_UNCERTAINTY"
        ):
            factors.append(
                "evidence uncertainty was high"
            )

        if (
            "INSUFFICIENT_EVIDENCE"
            in decision_reasons
            or
            decision_code
            == "ABSTAIN_INSUFFICIENT_EVIDENCE"
        ):
            factors.append(
                "available evidence was insufficient"
            )

        if (
            "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
            in decision_reasons
            or
            "CONFLICTING_PHYSIOLOGICAL_EVIDENCE"
            in uncertainty_reasons
        ):
            factors.append(
                "physiological evidence was conflicting"
            )

        if (
            "RISK_UNAVAILABLE"
            in uncertainty_reasons
        ):
            factors.append(
                "background risk information was unavailable"
            )

        if opposing_signals:
            factors.append(
                "opposing physiological signals included "
                + self._signal_phrase(
                    opposing_signals
                )
            )

        if (
            decision_state
            == "MONITOR"
            and not factors
        ):
            factors.append(
                "the complete upstream warning criteria "
                "were not satisfied"
            )

        if not factors:
            return (
                "No additional limiting factor was recorded "
                "by the upstream engines."
            )

        return (
            "; ".join(
                factors
            )
            + "."
        )

    # ========================================================
    # Risk
    # ========================================================

    def _risk_context(
        self,
        row: pd.Series,
    ) -> str:

        risk_probability = self._number(
            row.get(
                "risk_probability"
            )
        )

        completeness = self._number(
            row.get(
                "risk_input_completeness"
            )
        )

        if risk_probability is None:
            return (
                "Background cardiovascular risk was unavailable "
                "and did not contribute to this decision."
            )

        probability_text = self._fmt_probability(
            risk_probability
        )

        if completeness is not None:
            return (
                "The available Framingham model estimate was "
                f"{probability_text} for 10-year CHD risk, "
                "with input completeness "
                f"{self._fmt_score(completeness)}. "
                "This is background context and is not a "
                "short-term deterioration probability."
            )

        return (
            "The available Framingham model estimate was "
            f"{probability_text} for 10-year CHD risk. "
            "This is background context and is not a "
            "short-term deterioration probability."
        )

    # ========================================================
    # Confidence / uncertainty
    # ========================================================

    def _confidence_statement(
        self,
        row: pd.Series,
    ) -> str:

        uncertainty = self._number(
            row.get(
                "uncertainty_score"
            )
        )

        level = self._text(
            row.get(
                "uncertainty_level"
            )
        )

        fusion_confidence = self._number(
            row.get(
                "fusion_confidence"
            )
        )

        parts: list[str] = []

        if uncertainty is not None:
            if level:
                parts.append(
                    "Evidence uncertainty was "
                    f"{self._fmt_score(uncertainty)} "
                    f"({level.lower()})."
                )
            else:
                parts.append(
                    "Evidence uncertainty was "
                    f"{self._fmt_score(uncertainty)}."
                )

        if fusion_confidence is not None:
            parts.append(
                "Fusion evidence confidence was "
                f"{self._fmt_score(fusion_confidence)}."
            )

        if not parts:
            return (
                "No uncertainty summary was available."
            )

        return " ".join(
            parts
        )

    # ========================================================
    # Formatting utilities
    # ========================================================

    def _signal_phrase(
        self,
        signals: list[str],
    ) -> str:

        labels = [
            self.config.signal_labels.get(
                signal,
                signal,
            )
            for signal in signals
        ]

        if not labels:
            return ""

        if len(labels) == 1:
            return labels[0]

        if len(labels) == 2:
            return (
                labels[0]
                + " and "
                + labels[1]
            )

        return (
            ", ".join(
                labels[:-1]
            )
            + ", and "
            + labels[-1]
        )

    def _parse_list(
        self,
        value: Any,
    ) -> list[str]:

        if self._is_missing(
            value
        ):
            return []

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            return [
                str(item)
                for item in value
                if not self._is_missing(item)
            ]

        text = str(
            value
        ).strip()

        if not text:
            return []

        try:
            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                list,
            ):
                return [
                    str(item)
                    for item in parsed
                    if not self._is_missing(item)
                ]
        except (
            json.JSONDecodeError,
            TypeError,
        ):
            pass

        if "|" in text:
            return [
                item.strip()
                for item in text.split("|")
                if item.strip()
            ]

        if "," in text:
            return [
                item.strip()
                for item in text.split(",")
                if item.strip()
            ]

        return [
            text
        ]

    def _parse_codes(
        self,
        value: Any,
    ) -> list[str]:

        if self._is_missing(
            value
        ):
            return []

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        text = str(
            value
        ).strip()

        if not text:
            return []

        return [
            code.strip()
            for code in text.split("|")
            if code.strip()
        ]

    @staticmethod
    def _humanize_code(
        value: str,
    ) -> str:
        return (
            str(value)
            .strip()
            .replace(
                "_",
                " ",
            )
            .lower()
        )

    @staticmethod
    def _is_missing(
        value: Any,
    ) -> bool:

        if value is None:
            return True

        try:
            missing = pd.isna(
                value
            )

            if isinstance(
                missing,
                (
                    bool,
                    np.bool_,
                ),
            ):
                return bool(
                    missing
                )

        except (
            TypeError,
            ValueError,
        ):
            pass

        return False

    def _text(
        self,
        value: Any,
    ) -> str:

        if self._is_missing(
            value
        ):
            return ""

        return str(
            value
        ).strip()

    def _number(
        self,
        value: Any,
    ) -> float | None:

        if self._is_missing(
            value
        ):
            return None

        try:
            number = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

        if not np.isfinite(
            number
        ):
            return None

        return number

    def _fmt_score(
        self,
        value: float,
    ) -> str:

        decimals = (
            self.config
            .display
            .score_decimals
        )

        return f"{value:.{decimals}f}"

    def _fmt_probability(
        self,
        value: float,
    ) -> str:

        decimals = (
            self.config
            .display
            .probability_decimals
        )

        return f"{value:.{decimals}f}"