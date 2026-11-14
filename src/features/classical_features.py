from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from src.config import (
    PROCESSED_DATA_DIR,
    USEFUL_SENSOR_COLUMNS,
)
from src.data.load_data import load_train_data
from src.data.split_data import get_development_split
from src.features.rul import add_training_rul


DEFAULT_LAG_STEPS = (1, 5)
DEFAULT_ROLLING_WINDOWS = (5,)
DEFAULT_CHANGE_PERIODS = (5,)
DEFAULT_TREND_WINDOWS = (5,)


def validate_temporal_input(
    dataframe: pd.DataFrame,
    sensor_columns: Sequence[str],
) -> None:
    """
    Validate columns required for temporal feature
    engineering.
    """
    required_columns = {
        "engine_id",
        "cycle",
        *sensor_columns,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )


def add_lag_features(
    dataframe: pd.DataFrame,
    sensor_columns: Sequence[str],
    lag_steps: Sequence[int] = DEFAULT_LAG_STEPS,
) -> pd.DataFrame:
    """
    Add previous values from the same engine.
    """
    result_df = dataframe.copy()

    for sensor_column in sensor_columns:
        grouped_sensor = result_df.groupby(
            "engine_id",
            sort=False,
        )[sensor_column]

        for lag in lag_steps:
            result_df[
                f"{sensor_column}_lag_{lag}"
            ] = grouped_sensor.shift(lag)

    return result_df


def add_rolling_features(
    dataframe: pd.DataFrame,
    sensor_columns: Sequence[str],
    rolling_windows: Sequence[int] = (
        DEFAULT_ROLLING_WINDOWS
    ),
) -> pd.DataFrame:
    """
    Add trailing rolling mean and standard deviation.
    """
    result_df = dataframe.copy()

    for sensor_column in sensor_columns:
        grouped_sensor = result_df.groupby(
            "engine_id",
            sort=False,
        )[sensor_column]

        for window in rolling_windows:
            result_df[
                f"{sensor_column}_rolling_mean_{window}"
            ] = grouped_sensor.transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )

            result_df[
                f"{sensor_column}_rolling_std_{window}"
            ] = grouped_sensor.transform(
                lambda series: series.rolling(
                    window=window,
                    min_periods=window,
                ).std()
            )

    return result_df


def add_change_features(
    dataframe: pd.DataFrame,
    sensor_columns: Sequence[str],
    change_periods: Sequence[int] = (
        DEFAULT_CHANGE_PERIODS
    ),
) -> pd.DataFrame:
    """
    Add current value minus a previous value.

    change_5 = sensor(t) - sensor(t-5)
    """
    result_df = dataframe.copy()

    for sensor_column in sensor_columns:
        grouped_sensor = result_df.groupby(
            "engine_id",
            sort=False,
        )[sensor_column]

        for period in change_periods:
            result_df[
                f"{sensor_column}_change_{period}"
            ] = grouped_sensor.diff(
                periods=period
            )

    return result_df


def add_trend_features(
    dataframe: pd.DataFrame,
    sensor_columns: Sequence[str],
    trend_windows: Sequence[int] = (
        DEFAULT_TREND_WINDOWS
    ),
) -> pd.DataFrame:
    """
    Add an efficient endpoint-based trend.

    trend_5 =
    (sensor(t) - sensor(t-4)) / 4
    """
    result_df = dataframe.copy()

    for sensor_column in sensor_columns:
        grouped_sensor = result_df.groupby(
            "engine_id",
            sort=False,
        )[sensor_column]

        for window in trend_windows:
            if window < 2:
                raise ValueError(
                    "Trend window must be >= 2."
                )

            intervals = window - 1

            result_df[
                f"{sensor_column}_trend_{window}"
            ] = (
                grouped_sensor.diff(
                    periods=intervals
                )
                / intervals
            )

    return result_df


def get_temporal_feature_columns(
    dataframe: pd.DataFrame,
    sensor_columns: Sequence[str],
) -> list[str]:
    """
    Return engineered temporal columns belonging to
    the selected sensors.
    """
    temporal_columns = []

    for sensor_column in sensor_columns:
        prefix = f"{sensor_column}_"

        temporal_columns.extend(
            [
                column
                for column in dataframe.columns
                if (
                    column.startswith(prefix)
                    and column != sensor_column
                )
            ]
        )

    return temporal_columns


def drop_incomplete_history_rows(
    dataframe: pd.DataFrame,
    temporal_columns: Sequence[str],
) -> pd.DataFrame:
    """
    Keep only rows with complete temporal history.
    """
    return (
        dataframe
        .dropna(
            subset=list(temporal_columns)
        )
        .reset_index(drop=True)
    )


def validate_engine_boundaries(
    dataframe: pd.DataFrame,
    sensor_columns: Sequence[str],
) -> None:
    """
    Confirm that the literal first row of every engine
    contains no temporal history from another engine.
    """

    first_cycle_mask = (
        dataframe["cycle"]
        ==
        dataframe.groupby("engine_id")[
            "cycle"
        ].transform("min")
    )

    first_cycle_rows = dataframe.loc[
        first_cycle_mask
    ].copy()

    columns_to_check = []

    for sensor_column in sensor_columns:
        columns_to_check.extend(
            [
                f"{sensor_column}_lag_1",
                f"{sensor_column}_lag_5",
                f"{sensor_column}_rolling_mean_5",
                f"{sensor_column}_rolling_std_5",
                f"{sensor_column}_change_5",
                f"{sensor_column}_trend_5",
            ]
        )

    available_columns = [
        column
        for column in columns_to_check
        if column in first_cycle_rows.columns
    ]

    invalid_mask = (
        first_cycle_rows[
            available_columns
        ]
        .notna()
        .any(axis=1)
    )

    if invalid_mask.any():
        print(
            "\nEngine-boundary failure details:"
        )

        print(
            first_cycle_rows.loc[
                invalid_mask,
                [
                    "engine_id",
                    "cycle",
                    *available_columns,
                ],
            ].to_string(index=False)
        )

        raise ValueError(
            "Engine-boundary validation failed."
        )

    print(
        "\nEngine-boundary check:"
    )
    print(
        "All first engine rows have empty "
        "temporal history."
    )

def build_classical_features(
    dataframe: pd.DataFrame,
    sensor_columns: Sequence[str],
    drop_incomplete_history: bool = True,
) -> pd.DataFrame:
    """
    Build leakage-safe classical temporal features.
    """
    validate_temporal_input(
        dataframe,
        sensor_columns,
    )

    feature_df = (
        dataframe
        .sort_values(
            ["engine_id", "cycle"]
        )
        .reset_index(drop=True)
    )

    feature_df = add_lag_features(
        feature_df,
        sensor_columns,
    )

    feature_df = add_rolling_features(
        feature_df,
        sensor_columns,
    )

    feature_df = add_change_features(
        feature_df,
        sensor_columns,
    )

    feature_df = add_trend_features(
        feature_df,
        sensor_columns,
    )

    validate_engine_boundaries(
        feature_df,
        sensor_columns,
    )

    if drop_incomplete_history:
        temporal_columns = (
            get_temporal_feature_columns(
                feature_df,
                sensor_columns,
            )
        )

        feature_df = (
            drop_incomplete_history_rows(
                feature_df,
                temporal_columns,
            )
        )

    return feature_df


def main() -> None:
    """
    Run final Day 2 temporal feature validation.
    """
    train_df = load_train_data()

    train_with_rul_df = add_training_rul(
        train_df
    )

    (
        development_train_df,
        development_validation_df,
        _split_table_df,
    ) = get_development_split(
        train_with_rul_df
    )

    # ---------------------------------------
    # BUILD TEMPORAL FEATURES
    # ---------------------------------------

    raw_feature_df = build_classical_features(
        development_train_df,
        USEFUL_SENSOR_COLUMNS,
        drop_incomplete_history=False,
    )

    final_feature_df = build_classical_features(
        development_train_df,
        USEFUL_SENSOR_COLUMNS,
        drop_incomplete_history=True,
    )

    validation_feature_df = build_classical_features(
    development_validation_df,
    USEFUL_SENSOR_COLUMNS,
    drop_incomplete_history=True,
)

    temporal_columns = (
    get_temporal_feature_columns(
        raw_feature_df,
        USEFUL_SENSOR_COLUMNS,
    )
)
    
    temporal_feature_output_path = (
    PROCESSED_DATA_DIR
    / "fd001_development_train_temporal_features.csv"
   )


    validation_temporal_feature_output_path = (
    PROCESSED_DATA_DIR
    / "fd001_development_validation_temporal_features.csv"
)

    final_feature_df.to_csv(
    temporal_feature_output_path,
    index=False,
   )

    validation_feature_df.to_csv(
    validation_temporal_feature_output_path,
    index=False,
)
    print(
    "\nDevelopment temporal feature table saved to:"
)
    print(
    temporal_feature_output_path
   )

    print(
    "\nDevelopment-validation temporal feature "
    "table saved to:"
)
    print(
    validation_temporal_feature_output_path
)
    
    print(
    "\nFinal temporal feature table shape:"
 )
    print(
    final_feature_df.shape
   )

    print(
    "\nDevelopment-validation temporal table shape:"
)
    print(
    validation_feature_df.shape
)

    # ---------------------------------------
    # SHOW REAL TEMPORAL FEATURE EXAMPLE
    # ---------------------------------------

    print(
        "\n======================================"
    )

    print(
        "REAL TEMPORAL FEATURE VALUES"
    )

    print(
        "======================================"
    )

    example_columns = [
        "engine_id",
        "cycle",
        "sensor_11",
        "sensor_11_lag_1",
        "sensor_11_lag_5",
        "sensor_11_rolling_mean_5",
        "sensor_11_rolling_std_5",
        "sensor_11_change_5",
        "sensor_11_trend_5",
        "rul",
    ]

    example_engine_id = int(
        final_feature_df[
            "engine_id"
        ].iloc[0]
    )

    example_engine_df = final_feature_df[
        final_feature_df["engine_id"]
        == example_engine_id
    ]

    print(
        f"\nEngine {example_engine_id} "
        "first 10 retained cycles:"
    )

    print(
        example_engine_df[
            example_columns
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    # ---------------------------------------
    # VALIDATION SUMMARY
    # ---------------------------------------

    print(
        "\n======================================"
    )

    print(
        "TEMPORAL FEATURE VALIDATION"
    )

    print(
        "======================================"
    )

    print(
        "\nDevelopment-training rows before "
        "dropping incomplete history:"
    )

    print(
        raw_feature_df.shape[0]
    )

    print(
        "\nDevelopment-training rows after "
        "dropping incomplete history:"
    )

    print(
        final_feature_df.shape[0]
    )

    print(
        "\nRows removed:"
    )

    print(
        raw_feature_df.shape[0]
        - final_feature_df.shape[0]
    )

    print(
        "\nNumber of temporal feature columns:"
    )

    print(
        len(temporal_columns)
    )

    print(
        "\nRemaining temporal missing values:"
    )

    print(
        final_feature_df[
            temporal_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    first_kept_cycles = (
        final_feature_df
        .groupby(
            "engine_id"
        )["cycle"]
        .min()
    )

    print(
        "\nMinimum retained cycle per engine:"
    )

    print(
        first_kept_cycles
        .value_counts()
        .sort_index()
    )

    train_engine_ids = set(
        development_train_df[
            "engine_id"
        ].unique()
    )

    validation_engine_ids = set(
        development_validation_df[
            "engine_id"
        ].unique()
    )


    print(
    "\n======================================"
)

  
    print(
        "\nTrain/validation engine overlap:"
    )

    print(
        train_engine_ids
        & validation_engine_ids
    )

    print(
        "\nEngine-boundary validation:"
    )

    print(
        "PASSED"
    )



    # ---------------------------------------
    # DEVELOPMENT-VALIDATION CHECK
    # ---------------------------------------

    print(
        "\n======================================"
    )

    print(
        "DEVELOPMENT-VALIDATION TEMPORAL CHECK"
    )

    print(
        "======================================"
    )

    print(
        "\nValidation rows after dropping "
        "incomplete history:"
    )

    print(
        validation_feature_df.shape[0]
    )

    print(
        "\nValidation engines retained:"
    )

    print(
        validation_feature_df[
            "engine_id"
        ].nunique()
    )

    validation_first_kept_cycles = (
        validation_feature_df
        .groupby(
            "engine_id"
        )["cycle"]
        .min()
    )

    print(
        "\nMinimum retained cycle per "
        "validation engine:"
    )

    print(
        validation_first_kept_cycles
        .value_counts()
        .sort_index()
    )

    print(
        "\nRemaining validation temporal "
        "missing values:"
    )

    print(
        validation_feature_df[
            temporal_columns
        ]
        .isna()
        .sum()
        .sum()
    )

if __name__ == "__main__":
    main()