from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from biovance.risk import (
    RiskEngine,
    load_risk_config,
)


DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "framingham"
    / "FraminghamHeartStudy.csv"
)

MODEL_PATH = (
    ROOT
    / "models"
    / "risk"
    / "risk_v1_logistic_framingham.joblib"
)


def main() -> None:
    print("Reading:")
    print(DATA_PATH)

    df = pd.read_csv(
        DATA_PATH
    )

    config = load_risk_config()

    target = config.target

    # ------------------------------------------------------
    # Reproduce the same original 70 / 15 / 15 partition.
    #
    # The final TEST set stays untouched.
    # After model selection is finished, we are allowed to
    # combine TRAIN + CALIBRATION for the deployable model.
    # ------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df[target],
    )

    calibration_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df[target],
    )

    development_df = pd.concat(
        [
            train_df,
            calibration_df,
        ],
        ignore_index=True,
    )

    print("\nRows:")
    print(
        "Development:",
        len(development_df),
    )
    print(
        "Reserved test:",
        len(test_df),
    )

    print(
        "\nDevelopment prevalence:",
        round(
            float(
                development_df[
                    target
                ].mean()
            ),
            4,
        ),
    )

    # ------------------------------------------------------
    # Fit final Risk v1 artifact on development data only.
    #
    # IMPORTANT:
    # Reserved test rows are NOT included.
    # ------------------------------------------------------

    engine = RiskEngine(
        config
    ).fit(
        development_df
    )

    engine.save(
        MODEL_PATH
    )

    print("\nSaved Risk v1:")
    print(MODEL_PATH)

    # ------------------------------------------------------
    # Verify immediately that the artifact reloads.
    # ------------------------------------------------------

    loaded = RiskEngine.load(
        MODEL_PATH
    )

    example = test_df.iloc[
        [0]
    ]

    prediction = loaded.predict(
        example
    )

    print(
        "\nReload check:"
    )

    print(
        prediction[
            [
                "risk_probability",
                "risk_input_completeness",
                "risk_code",
                "risk_model_version",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The saved model was trained on "
        "train + calibration data."
    )

    print(
        "The previously reserved test set "
        "was not used for model fitting."
    )


if __name__ == "__main__":
    main()