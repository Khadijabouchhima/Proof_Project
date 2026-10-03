from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import RiskConfig
from .model import build_risk_model
from .schema import RiskCode


class RiskEngine:
    """
    BioVance Risk Engine v1.

    Purpose
    -------
    Estimate background 10-year CHD risk using the
    Framingham TenYearCHD outcome.

    This engine does NOT:
    - issue WARN / MONITOR / ABSTAIN decisions
    - detect short-term deterioration
    - use temporal wearable trajectories
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.model = build_risk_model(config)
        self._is_fitted = False

    def _validate_features(
        self,
        df: pd.DataFrame,
    ) -> None:
        missing_columns = [
            col
            for col in self.config.features
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required Risk Engine columns: "
                + ", ".join(missing_columns)
            )

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "RiskEngine":
        """
        Fit Risk v1.

        The dataframe must contain all configured predictors
        and the TenYearCHD target.
        """

        self._validate_features(df)

        target = self.config.target

        if target not in df.columns:
            raise ValueError(
                f"Missing target column: {target}"
            )

        if df[target].isna().any():
            raise ValueError(
                f"Target column {target} contains missing values."
            )

        target_values = set(
            df[target].astype(int).unique()
        )

        if not target_values.issubset({0, 1}):
            raise ValueError(
                f"{target} must be binary 0/1."
            )

        X = df[self.config.features]
        y = df[target].astype(int)

        self.model.fit(X, y)

        self._is_fitted = True

        return self

    def predict(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce background-risk outputs for each row.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "RiskEngine must be fitted before prediction."
            )

        self._validate_features(df)

        X = df[self.config.features].copy()

        probability = self.model.predict_proba(X)[:, 1]

        eps = 1e-12
        p_safe = np.clip(
            probability,
            eps,
            1 - eps,
        )

        log_odds = np.log(
            p_safe / (1 - p_safe)
        )

        completeness = (
            X.notna().sum(axis=1)
            / len(self.config.features)
        )

        codes = np.where(
            completeness
            < self.config.low_completeness_threshold,
            RiskCode.LOW_INPUT_COMPLETENESS.value,
            RiskCode.OK.value,
        )

        reason_codes = []

        explanations = []

        for i in range(len(X)):
            missing = [
                feature
                for feature in self.config.features
                if pd.isna(X.iloc[i][feature])
            ]

            reasons = []

            if missing:
                reasons.append(
                    "IMPUTED_MISSING_INPUTS"
                )

            if (
                completeness.iloc[i]
                < self.config.low_completeness_threshold
            ):
                reasons.append(
                    "LOW_INPUT_COMPLETENESS"
                )

            if not reasons:
                reasons.append(
                    "COMPLETE_OR_MOSTLY_COMPLETE_INPUT"
                )

            reason_codes.append(
                "|".join(reasons)
            )

            if missing:
                explanation = (
                    "Background 10-year CHD risk estimated "
                    "with the BioVance logistic risk model. "
                    f"Missing predictors imputed: "
                    f"{', '.join(missing)}."
                )
            else:
                explanation = (
                    "Background 10-year CHD risk estimated "
                    "with the BioVance logistic risk model "
                    "using all configured predictors."
                )

            explanations.append(
                explanation
            )

        result = pd.DataFrame(
            {
                "risk_probability": probability,
                "risk_log_odds": log_odds,
                "risk_input_completeness": completeness.to_numpy(),
                "risk_code": codes,
                "risk_model_version": self.config.model_version,
                "risk_reason_codes": reason_codes,
                "risk_explanation": explanations,
            },
            index=df.index,
        )

        return result
    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the fitted Risk Engine to disk.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save RiskEngine before fitting."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "config": self.config,
            "model": self.model,
            "is_fitted": self._is_fitted,
        }

        joblib.dump(
            payload,
            path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "RiskEngine":
        """
        Load a previously fitted Risk Engine.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Risk model artifact not found: {path}"
            )

        payload = joblib.load(path)

        engine = cls(
            payload["config"]
        )

        engine.model = payload["model"]
        engine._is_fitted = bool(
            payload["is_fitted"]
        )

        return engine