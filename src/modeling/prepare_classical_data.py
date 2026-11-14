from __future__ import annotations

import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    USEFUL_SENSOR_COLUMNS,
)
from src.features.classical_features import (
    get_temporal_feature_columns,
)


TRAIN_TEMPORAL_FEATURE_FILE = (
    PROCESSED_DATA_DIR
    / "fd001_development_train_temporal_features.csv"
)

VALIDATION_TEMPORAL_FEATURE_FILE = (
    PROCESSED_DATA_DIR
    / "fd001_development_validation_temporal_features.csv"
)


def load_temporal_feature_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load the already-generated development train and
    development validation temporal feature tables.
    """

    train_df = pd.read_csv(
        TRAIN_TEMPORAL_FEATURE_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_TEMPORAL_FEATURE_FILE
    )

    return train_df, validation_df


def build_modeling_matrices(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Build fair current-state and temporal modeling
    matrices from the same retained engine-cycle rows.
    """

    current_feature_columns = list(
        USEFUL_SENSOR_COLUMNS
    )

    temporal_feature_columns = (
        get_temporal_feature_columns(
            train_df,
            USEFUL_SENSOR_COLUMNS,
        )
    )

    full_temporal_feature_columns = [
        *current_feature_columns,
        *temporal_feature_columns,
    ]

    x_train_current = train_df[
        current_feature_columns
    ].copy()

    x_validation_current = validation_df[
        current_feature_columns
    ].copy()

    x_train_temporal = train_df[
        full_temporal_feature_columns
    ].copy()

    x_validation_temporal = validation_df[
        full_temporal_feature_columns
    ].copy()

    y_train = train_df[
        "rul"
    ].copy()

    y_validation = validation_df[
        "rul"
    ].copy()

    return {
        "x_train_current": x_train_current,
        "x_validation_current": (
            x_validation_current
        ),
        "x_train_temporal": x_train_temporal,
        "x_validation_temporal": (
            x_validation_temporal
        ),
        "y_train": y_train,
        "y_validation": y_validation,
    }


def validate_modeling_matrices(
    matrices: dict[
        str,
        pd.DataFrame | pd.Series,
    ],
) -> None:
    """
    Validate that current and temporal models use
    exactly the same observations.
    """

    x_train_current = matrices[
        "x_train_current"
    ]

    x_train_temporal = matrices[
        "x_train_temporal"
    ]

    x_validation_current = matrices[
        "x_validation_current"
    ]

    x_validation_temporal = matrices[
        "x_validation_temporal"
    ]

    y_train = matrices[
        "y_train"
    ]

    y_validation = matrices[
        "y_validation"
    ]

    if len(x_train_current) != len(
        x_train_temporal
    ):
        raise ValueError(
            "Training current and temporal row "
            "counts do not match."
        )

    if len(x_validation_current) != len(
        x_validation_temporal
    ):
        raise ValueError(
            "Validation current and temporal row "
            "counts do not match."
        )

    if len(x_train_current) != len(y_train):
        raise ValueError(
            "Training predictors and target row "
            "counts do not match."
        )

    if len(x_validation_current) != len(
        y_validation
    ):
        raise ValueError(
            "Validation predictors and target row "
            "counts do not match."
        )

    if (
        x_train_current.isna()
        .sum()
        .sum()
        != 0
    ):
        raise ValueError(
            "Current training features contain missing values."
        )

    if (
        x_train_temporal.isna()
        .sum()
        .sum()
        != 0
    ):
        raise ValueError(
            "Temporal training features contain "
            "missing values."
        )

    if (
        x_validation_current.isna()
        .sum()
        .sum()
        != 0
    ):
        raise ValueError(
            "Current validation features contain "
            "missing values."
        )

    if (
        x_validation_temporal.isna()
        .sum()
        .sum()
        != 0
    ):
        raise ValueError(
            "Temporal validation features contain "
            "missing values."
        )


def main() -> None:
    """
    Build and validate the fair Day 3 classical
    modeling datasets.
    """

    (
        train_df,
        validation_df,
    ) = load_temporal_feature_tables()

    matrices = build_modeling_matrices(
        train_df,
        validation_df,
    )

    validate_modeling_matrices(
        matrices
    )

    print(
        "\n======================================"
    )
    print(
        "FAIR MODELING DATASETS"
    )
    print(
        "======================================"
    )

    print(
        "\nCurrent-state training shape:"
    )
    print(
        matrices[
            "x_train_current"
        ].shape
    )

    print(
        "\nTemporal training shape:"
    )
    print(
        matrices[
            "x_train_temporal"
        ].shape
    )

    print(
        "\nCurrent-state validation shape:"
    )
    print(
        matrices[
            "x_validation_current"
        ].shape
    )

    print(
        "\nTemporal validation shape:"
    )
    print(
        matrices[
            "x_validation_temporal"
        ].shape
    )

    print(
        "\nTraining target shape:"
    )
    print(
        matrices[
            "y_train"
        ].shape
    )

    print(
        "\nValidation target shape:"
    )
    print(
        matrices[
            "y_validation"
        ].shape
    )

    print(
        "\nCurrent-state predictor count:"
    )
    print(
        matrices[
            "x_train_current"
        ].shape[1]
    )

    print(
        "\nTemporal predictor count:"
    )
    print(
        matrices[
            "x_train_temporal"
        ].shape[1]
    )

    print(
        "\nMissing-value checks:"
    )
    print(
        "PASSED"
    )

    print(
        "\nFair row-count checks:"
    )
    print(
        "PASSED"
    )


if __name__ == "__main__":
    main()