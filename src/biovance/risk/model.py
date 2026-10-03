from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RiskConfig


def build_risk_model(config: RiskConfig) -> Pipeline:
    """
    Build the BioVance Risk v1 model.

    Continuous predictors:
        median imputation -> standardization

    Binary predictors:
        most-frequent imputation

    Final estimator:
        L2-regularized logistic regression
    """

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
        ]
    )

    continuous_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "binary",
                binary_pipeline,
                config.binary_features,
            ),
            (
                "continuous",
                continuous_pipeline,
                config.continuous_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # l1_ratio=0 corresponds to L2 regularization in
    # current sklearn versions without using the
    # deprecated explicit penalty="l2" argument.
    classifier = LogisticRegression(
        l1_ratio=0,
        C=config.C,
        solver="lbfgs",
        max_iter=config.max_iter,
        random_state=config.random_state,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("classifier", classifier),
        ]
    )