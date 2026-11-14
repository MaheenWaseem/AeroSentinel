from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from src.config import USEFUL_SENSOR_COLUMNS
from src.modeling.prepare_classical_data import (
    build_modeling_matrices,
    load_temporal_feature_tables,
    validate_modeling_matrices,
)
from src.modeling.train_classical_models import (
    OUTPUT_DIR,
    build_models,
)


TREE_FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR / "rf_tree_feature_importance.csv"
)

PERMUTATION_IMPORTANCE_FILE = (
    OUTPUT_DIR / "rf_permutation_importance.csv"
)

SENSOR_IMPORTANCE_FILE = (
    OUTPUT_DIR / "rf_importance_by_sensor.csv"
)

FEATURE_FAMILY_IMPORTANCE_FILE = (
    OUTPUT_DIR / "rf_importance_by_feature_family.csv"
)

LIFECYCLE_IMPORTANCE_FILE = (
    OUTPUT_DIR / "rf_lifecycle_importance.csv"
)

TOP_FEATURE_IMPORTANCE_PLOT = (
    OUTPUT_DIR / "rf_top_feature_importance.png"
)

GROUPED_SENSOR_IMPORTANCE_PLOT = (
    OUTPUT_DIR / "rf_grouped_sensor_importance.png"
)

FEATURE_FAMILY_IMPORTANCE_PLOT = (
    OUTPUT_DIR / "rf_feature_family_importance.png"
)


RANDOM_STATE = 42

GLOBAL_PERMUTATION_REPEATS = 5

GROUP_PERMUTATION_REPEATS = 3

LIFECYCLE_PERMUTATION_REPEATS = 3

TOP_FEATURE_COUNT = 8

TOP_SENSOR_COUNT = 5

EXPECTED_VALIDATION_MAE = 23.0152


FEATURE_FAMILY_ORDER = [
    "current_sensor",
    "lag_1",
    "lag_5",
    "rolling_mean_5",
    "rolling_std_5",
    "change_5",
    "trend_5",
    "cycle",
]


def identify_sensor_group(
    feature_name: str,
) -> str:
    """
    maps a current or engineered feature back to its
    original sensor.

    Cycle remains its own group.
    """

    if feature_name == "cycle":
        return "cycle"

    for sensor_column in USEFUL_SENSOR_COLUMNS:
        if (
            feature_name == sensor_column
            or feature_name.startswith(
                f"{sensor_column}_"
            )
        ):
            return sensor_column

    raise ValueError(
        "Could not map feature to sensor group: "
        f"{feature_name}"
    )


def identify_feature_family(
    feature_name: str,
) -> str:
    """
    Maps a predictor to its temporal-information family.
    """

    if feature_name == "cycle":
        return "cycle"

    if feature_name in USEFUL_SENSOR_COLUMNS:
        return "current_sensor"

    suffix_to_family = {
        "_lag_1": "lag_1",
        "_lag_5": "lag_5",
        "_rolling_mean_5": "rolling_mean_5",
        "_rolling_std_5": "rolling_std_5",
        "_change_5": "change_5",
        "_trend_5": "trend_5",
    }

    for suffix, family_name in (
        suffix_to_family.items()
    ):
        if feature_name.endswith(suffix):
            return family_name

    raise ValueError(
        "Could not map feature to feature family: "
        f"{feature_name}"
    )


def build_explainability_matrices() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """
    Reconstructing the exact 99-predictor temporal+cycle
    development datasets used by the selected RF.
    """

    train_df, validation_df = (
        load_temporal_feature_tables()
    )

    matrices = build_modeling_matrices(
        train_df,
        validation_df,
    )

    validate_modeling_matrices(
        matrices
    )

    x_train = (
        matrices["x_train_temporal"]
        .copy()
    )

    x_validation = (
        matrices["x_validation_temporal"]
        .copy()
    )

    y_train = matrices["y_train"].copy()

    y_validation = (
        matrices["y_validation"]
        .copy()
    )

    if (
        not x_train.index.equals(
            train_df.index
        )
        or not y_train.index.equals(
            train_df.index
        )
    ):
        raise ValueError(
            "Training feature/target metadata alignment failed."
        )

    if (
        not x_validation.index.equals(
            validation_df.index
        )
        or not y_validation.index.equals(
            validation_df.index
        )
    ):
        raise ValueError(
            "Validation feature/target metadata "
            "alignment failed."
        )

    x_train.insert(
        0,
        "cycle",
        train_df["cycle"],
    )

    x_validation.insert(
        0,
        "cycle",
        validation_df["cycle"],
    )

    if (
        x_train.shape[1] != 99
        or x_validation.shape[1] != 99
    ):
        raise ValueError(
            "Expected exactly 99 temporal+cycle predictors."
        )

    if (
        x_train.isna().any().any()
        or x_validation.isna().any().any()
    ):
        raise ValueError(
            "Explainability matrices contain missing predictor values."
        )

    if not np.allclose(
        y_validation.to_numpy(),
        validation_df["rul"].to_numpy(),
    ):
        raise ValueError(
            "Validation RUL target alignment failed."
        )

    return (
        x_train,
        x_validation,
        y_train,
        y_validation,
        validation_df,
    )


def validate_feature_mapping(
    feature_columns: list[str],
) -> None:
    """
    Confirms that the 99 predictors map exactly to the
    expected sensor and temporal-feature structure.
    """

    sensor_counts: dict[str, int] = {}

    for feature_name in feature_columns:
        sensor_name = identify_sensor_group(
            feature_name
        )

        sensor_counts[sensor_name] = (
            sensor_counts.get(
                sensor_name,
                0,
            )
            + 1
        )

    for sensor_column in USEFUL_SENSOR_COLUMNS:
        actual_count = sensor_counts.get(
            sensor_column,
            0,
        )

        if actual_count != 7:
            raise ValueError(
                f"{sensor_column} should have 7 "
                f"predictors but has {actual_count}."
            )

    if sensor_counts.get("cycle", 0) != 1:
        raise ValueError(
            "Expected exactly one cycle predictor."
        )

    family_counts: dict[str, int] = {}

    for feature_name in feature_columns:
        family_name = identify_feature_family(
            feature_name
        )

        family_counts[family_name] = (
            family_counts.get(
                family_name,
                0,
            )
            + 1
        )

    expected_family_counts = {
        "current_sensor": 14,
        "lag_1": 14,
        "lag_5": 14,
        "rolling_mean_5": 14,
        "rolling_std_5": 14,
        "change_5": 14,
        "trend_5": 14,
        "cycle": 1,
    }

    if family_counts != expected_family_counts:
        raise ValueError(
            "Feature-family mapping does not match the expected 99-predictor structure.\n"
            f"Found: {family_counts}"
        )


def build_locked_random_forest():
    """
    Reconstruct the exact selected
    random_forest_depth12 model.
    """

    reference_model = (
        build_models()["random_forest"]
    )

    model = clone(reference_model)

    model.set_params(
        max_depth=12
    )

    expected_parameters = {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    }

    actual_parameters = (
        model.get_params()
    )

    for (
        parameter_name,
        expected_value,
    ) in expected_parameters.items():

        actual_value = actual_parameters[
            parameter_name
        ]

        if actual_value != expected_value:
            raise ValueError(
                "Locked RF configuration mismatch: "
                f"{parameter_name} expected "
                f"{expected_value}, found "
                f"{actual_value}."
            )

    return model


def calculate_tree_importance(
    model,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Rank all 99 predictors using Random Forest
    impurity-based feature importance.
    """

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": (
                model.feature_importances_
            ),
        }
    )

    importance_df["sensor"] = (
        importance_df["feature"].map(
            identify_sensor_group
        )
    )

    importance_df["feature_family"] = (
        importance_df["feature"].map(
            identify_feature_family
        )
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df["rank"] = (
        np.arange(
            1,
            len(importance_df) + 1,
        )
    )

    return importance_df[
        [
            "feature",
            "sensor",
            "feature_family",
            "importance",
            "rank",
        ]
    ]


def calculate_individual_permutation_importance(
    model,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    baseline_mae: float,
) -> pd.DataFrame:

    result = permutation_importance(
        estimator=model,
        X=x_validation,
        y=y_validation,
        scoring="neg_mean_absolute_error",
        n_repeats=GLOBAL_PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,

        # The RF itself already parallelizes prediction.
        # Avoid nested maximum parallelism here.
        n_jobs=1,
    )

    permutation_df = pd.DataFrame(
        {
            "feature": (
                x_validation.columns
            ),
            "mean_importance": (
                result.importances_mean
            ),
            "std_importance": (
                result.importances_std
            ),
        }
    )

    permutation_df["sensor"] = (
        permutation_df["feature"].map(
            identify_sensor_group
        )
    )

    permutation_df["feature_family"] = (
        permutation_df["feature"].map(
            identify_feature_family
        )
    )

    permutation_df["baseline_mae"] = (
        baseline_mae
    )

    permutation_df[
        "mean_permuted_mae"
    ] = (
        baseline_mae
        + permutation_df[
            "mean_importance"
        ]
    )

    permutation_df = (
        permutation_df
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    permutation_df["rank"] = (
        np.arange(
            1,
            len(permutation_df) + 1,
        )
    )

    return permutation_df[
        [
            "feature",
            "sensor",
            "feature_family",
            "baseline_mae",
            "mean_permuted_mae",
            "mean_importance",
            "std_importance",
            "rank",
        ]
    ]


def build_sensor_groups(
    feature_columns: list[str],
) -> dict[str, list[str]]:
    

    groups: dict[str, list[str]] = {
        sensor_column: []
        for sensor_column in USEFUL_SENSOR_COLUMNS
    }

    groups["cycle"] = []

    for feature_name in feature_columns:
        sensor_name = identify_sensor_group(
            feature_name
        )

        groups[sensor_name].append(
            feature_name
        )

    return groups


def build_feature_family_groups(
    feature_columns: list[str],
) -> dict[str, list[str]]:
    """
    Groups predictors according to the type of temporal
    information they represent.
    """

    groups: dict[str, list[str]] = {
        family_name: []
        for family_name in FEATURE_FAMILY_ORDER
    }

    for feature_name in feature_columns:
        family_name = identify_feature_family(
            feature_name
        )

        groups[family_name].append(
            feature_name
        )

    return groups


def calculate_group_permutation_importance(
    model,
    x_data: pd.DataFrame,
    y_data: pd.Series,
    groups: dict[str, list[str]],
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:

    
    baseline_predictions = model.predict(
        x_data
    )

    baseline_mae = mean_absolute_error(
        y_data,
        baseline_predictions,
    )

    rng = np.random.default_rng(
        random_state
    )

    permutation_indices = [
        rng.permutation(len(x_data))
        for _ in range(n_repeats)
    ]

    result_rows = []

    for group_name, group_columns in (
        groups.items()
    ):
        if not group_columns:
            raise ValueError(
                f"Group {group_name} has no columns."
            )

        mae_increases = []
        permuted_maes = []

        for indices in permutation_indices:
            x_permuted = x_data.copy()

            x_permuted.loc[
                :,
                group_columns,
            ] = (
                x_data.loc[
                    :,
                    group_columns,
                ]
                .iloc[indices]
                .to_numpy()
            )

            predictions = model.predict(
                x_permuted
            )

            permuted_mae = (
                mean_absolute_error(
                    y_data,
                    predictions,
                )
            )

            permuted_maes.append(
                permuted_mae
            )

            mae_increases.append(
                permuted_mae
                - baseline_mae
            )

        result_rows.append(
            {
                "group": group_name,
                "baseline_mae": (
                    baseline_mae
                ),
                "mean_permuted_mae": (
                    float(
                        np.mean(
                            permuted_maes
                        )
                    )
                ),
                "mean_importance": (
                    float(
                        np.mean(
                            mae_increases
                        )
                    )
                ),
                "std_importance": (
                    float(
                        np.std(
                            mae_increases
                        )
                    )
                ),
            }
        )

    result_df = pd.DataFrame(
        result_rows
    )

    result_df = (
        result_df
        .sort_values(
            "mean_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result_df["permutation_rank"] = (
        np.arange(
            1,
            len(result_df) + 1,
        )
    )

    return result_df


def calculate_sensor_importance(
    tree_importance_df: pd.DataFrame,
    model,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """
    Combine aggregated tree importance with a direct
    grouped permutation test for each sensor family.
    """

    tree_group_df = (
        tree_importance_df
        .groupby(
            "sensor",
            as_index=False,
        )["importance"]
        .sum()
        .rename(
            columns={
                "importance": (
                    "tree_importance"
                )
            }
        )
    )

    tree_group_df = (
        tree_group_df
        .sort_values(
            "tree_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    tree_group_df["tree_rank"] = (
        np.arange(
            1,
            len(tree_group_df) + 1,
        )
    )

    sensor_groups = build_sensor_groups(
        list(x_validation.columns)
    )

    permutation_df = (
        calculate_group_permutation_importance(
            model=model,
            x_data=x_validation,
            y_data=y_validation,
            groups=sensor_groups,
            n_repeats=GROUP_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
        )
    )

    permutation_df = (
        permutation_df.rename(
            columns={
                "group": "sensor",
                "mean_importance": (
                    "group_permutation_mae_increase"
                ),
                "std_importance": (
                    "group_permutation_std"
                ),
            }
        )
    )

    result_df = tree_group_df.merge(
        permutation_df,
        on="sensor",
        how="inner",
    )

    return result_df[
        [
            "sensor",
            "tree_importance",
            "tree_rank",
            "baseline_mae",
            "mean_permuted_mae",
            "group_permutation_mae_increase",
            "group_permutation_std",
            "permutation_rank",
        ]
    ]


def calculate_feature_family_importance(
    tree_importance_df: pd.DataFrame,
    model,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:
    """
    Measure importance by type of information:
    current, lag, rolling, change, trend, and cycle.
    """

    tree_group_df = (
        tree_importance_df
        .groupby(
            "feature_family",
            as_index=False,
        )["importance"]
        .sum()
        .rename(
            columns={
                "importance": (
                    "tree_importance"
                )
            }
        )
    )

    tree_group_df = (
        tree_group_df
        .sort_values(
            "tree_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    tree_group_df["tree_rank"] = (
        np.arange(
            1,
            len(tree_group_df) + 1,
        )
    )

    family_groups = (
        build_feature_family_groups(
            list(x_validation.columns)
        )
    )

    permutation_df = (
        calculate_group_permutation_importance(
            model=model,
            x_data=x_validation,
            y_data=y_validation,
            groups=family_groups,
            n_repeats=GROUP_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
        )
    )

    permutation_df = (
        permutation_df.rename(
            columns={
                "group": "feature_family",
                "mean_importance": (
                    "group_permutation_mae_increase"
                ),
                "std_importance": (
                    "group_permutation_std"
                ),
            }
        )
    )

    result_df = tree_group_df.merge(
        permutation_df,
        on="feature_family",
        how="inner",
    )

    return result_df[
        [
            "feature_family",
            "tree_importance",
            "tree_rank",
            "baseline_mae",
            "mean_permuted_mae",
            "group_permutation_mae_increase",
            "group_permutation_std",
            "permutation_rank",
        ]
    ]


def build_lifecycle_masks(
    y_validation: pd.Series,
) -> dict[str, pd.Series]:
    

    return {
        "near_failure_0_30": (
            y_validation <= 30
        ),
        "medium_31_100": (
            (y_validation >= 31)
            & (y_validation <= 100)
        ),
        "high_>100": (
            y_validation > 100
        ),
    }


def calculate_lifecycle_importance(
    model,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    permutation_importance_df: pd.DataFrame,
    sensor_importance_df: pd.DataFrame,
) -> pd.DataFrame:


    top_features = (
        permutation_importance_df[
            "feature"
        ]
        .head(TOP_FEATURE_COUNT)
        .tolist()
    )

    top_sensors = (
        sensor_importance_df.loc[
            sensor_importance_df[
                "sensor"
            ] != "cycle"
        ]
        .sort_values(
            "group_permutation_mae_increase",
            ascending=False,
        )["sensor"]
        .head(TOP_SENSOR_COUNT)
        .tolist()
    )

    all_sensor_groups = build_sensor_groups(
        list(x_validation.columns)
    )

    masks = build_lifecycle_masks(
        y_validation
    )

    lifecycle_tables = []

    for band_number, (
        band_name,
        band_mask,
    ) in enumerate(
        masks.items(),
        start=1,
    ):
        x_band = (
            x_validation.loc[
                band_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        y_band = (
            y_validation.loc[
                band_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        if len(x_band) == 0:
            raise ValueError(
                f"No rows found for {band_name}."
            )

        feature_groups = {
            feature_name: [
                feature_name
            ]
            for feature_name in top_features
        }

        feature_result = (
            calculate_group_permutation_importance(
                model=model,
                x_data=x_band,
                y_data=y_band,
                groups=feature_groups,
                n_repeats=(
                    LIFECYCLE_PERMUTATION_REPEATS
                ),
                random_state=(
                    RANDOM_STATE
                    + band_number
                ),
            )
        )

        feature_result = (
            feature_result.rename(
                columns={
                    "group": "item",
                }
            )
        )

        feature_result[
            "importance_type"
        ] = "feature"

        feature_result[
            "rul_band"
        ] = band_name

        feature_result[
            "row_count"
        ] = len(x_band)

        sensor_groups = {
            sensor_name: (
                all_sensor_groups[
                    sensor_name
                ]
            )
            for sensor_name in top_sensors
        }

        sensor_result = (
            calculate_group_permutation_importance(
                model=model,
                x_data=x_band,
                y_data=y_band,
                groups=sensor_groups,
                n_repeats=(
                    LIFECYCLE_PERMUTATION_REPEATS
                ),
                random_state=(
                    RANDOM_STATE
                    + 100
                    + band_number
                ),
            )
        )

        sensor_result = (
            sensor_result.rename(
                columns={
                    "group": "item",
                }
            )
        )

        sensor_result[
            "importance_type"
        ] = "sensor_family"

        sensor_result[
            "rul_band"
        ] = band_name

        sensor_result[
            "row_count"
        ] = len(x_band)

        lifecycle_tables.extend(
            [
                feature_result,
                sensor_result,
            ]
        )

    result_df = pd.concat(
        lifecycle_tables,
        ignore_index=True,
    )

    result_df = result_df.rename(
        columns={
            "permutation_rank": (
                "rank_within_band_and_type"
            )
        }
    )

    return result_df[
        [
            "rul_band",
            "row_count",
            "importance_type",
            "item",
            "baseline_mae",
            "mean_permuted_mae",
            "mean_importance",
            "std_importance",
            "rank_within_band_and_type",
        ]
    ]


def save_top_feature_plot(
    tree_importance_df: pd.DataFrame,
) -> None:

   

    plot_df = (
        tree_importance_df
        .head(20)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["importance"],
    )

    ax.set_xlabel(
        "Impurity-based feature importance"
    )

    ax.set_ylabel(
        "Predictor"
    )

    ax.set_title(
        "Random Forest: Top 20 Individual Features"
    )

    fig.tight_layout()

    fig.savefig(
        TOP_FEATURE_IMPORTANCE_PLOT,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_sensor_plot(
    sensor_importance_df: pd.DataFrame,
) -> None:



    plot_df = (
        sensor_importance_df
        .sort_values(
            "tree_importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        plot_df["sensor"],
        plot_df["tree_importance"],
    )

    ax.set_xlabel(
        "Aggregated impurity-based importance"
    )

    ax.set_ylabel(
        "Sensor family"
    )

    ax.set_title(
        "Random Forest: Importance by Sensor Family"
    )

    fig.tight_layout()

    fig.savefig(
        GROUPED_SENSOR_IMPORTANCE_PLOT,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_feature_family_plot(
    family_importance_df: pd.DataFrame,
) -> None:



    plot_df = (
        family_importance_df
        .sort_values(
            "tree_importance",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_df["feature_family"],
        plot_df["tree_importance"],
    )

    ax.set_xlabel(
        "Aggregated impurity-based importance"
    )

    ax.set_ylabel(
        "Feature family"
    )

    ax.set_title(
        "Random Forest: Importance by Feature Family"
    )

    fig.tight_layout()

    fig.savefig(
        FEATURE_FAMILY_IMPORTANCE_PLOT,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    """
    Explains the locked random_forest_depth12 model
    using development data only.
    """

    print(
        "\n======================================"
    )

    print(
        "FD001 RANDOM FOREST EXPLAINABILITY"
    )

    print(
        "======================================"
    )

    (
        x_train,
        x_validation,
        y_train,
        y_validation,
        validation_df,
    ) = build_explainability_matrices()

    validate_feature_mapping(
        list(x_train.columns)
    )

    print(
        "\nTraining shape:"
    )
    print(
        x_train.shape
    )

    print(
        "\nValidation shape:"
    )
    print(
        x_validation.shape
    )

    print(
        "\nValidation engines:"
    )
    print(
        validation_df[
            "engine_id"
        ].nunique()
    )

    print(
        "\nPredictor structure check:"
    )
    print(
        "PASSED — 14 sensors x 7 representations + cycle = 99 predictors"
    )

    model = build_locked_random_forest()

    print(
        "\nLocked RF configuration:"
    )

    print(
        "n_estimators:",
        model.n_estimators,
    )

    print(
        "max_depth:",
        model.max_depth,
    )

    print(
        "min_samples_leaf:",
        model.min_samples_leaf,
    )

    print(
        "random_state:",
        model.random_state,
    )

    print(
        "\nFitting exact locked RF for "
        "explainability."
    )

    model.fit(
        x_train,
        y_train,
    )

    baseline_predictions = model.predict(
        x_validation
    )

    baseline_mae = mean_absolute_error(
        y_validation,
        baseline_predictions,
    )

    print(
        "\nReconstructed validation MAE:"
    )

    print(
        f"{baseline_mae:.4f}"
    )

    if not np.isclose(
        baseline_mae,
        EXPECTED_VALIDATION_MAE,
        rtol=0.0,
        atol=0.01,
    ):
        raise ValueError(
            "Reconstructed RF does not match the "
            "locked development benchmark closely "
            f"Expected about "
            f"{EXPECTED_VALIDATION_MAE:.4f}, "
            f"found {baseline_mae:.4f}."
        )

    print(
        "\nLocked benchmark reproduction:"
    )
    print(
        "PASSED"
    )

    # ---------------------------------------
    # A. TREE FEATURE IMPORTANCE
    # ---------------------------------------

    tree_importance_df = (
        calculate_tree_importance(
            model=model,
            feature_columns=list(
                x_train.columns
            ),
        )
    )

    # ---------------------------------------
    # D. INDIVIDUAL PERMUTATION IMPORTANCE
    # ---------------------------------------

    print(
        "\nCalculating validation permutation importance"
    )

    permutation_importance_df = (
        calculate_individual_permutation_importance(
            model=model,
            x_validation=x_validation,
            y_validation=y_validation,
            baseline_mae=baseline_mae,
        )
    )

    # ---------------------------------------
    # B. SENSOR-FAMILY IMPORTANCE
    # ---------------------------------------

    print(
        "\nCalculating grouped sensor importance."
    )

    sensor_importance_df = (
        calculate_sensor_importance(
            tree_importance_df=(
                tree_importance_df
            ),
            model=model,
            x_validation=x_validation,
            y_validation=y_validation,
        )
    )

    # ---------------------------------------
    # C. FEATURE-FAMILY IMPORTANCE
    # ---------------------------------------

    print(
        "\nCalculating grouped feature-family importance..."
    )

    family_importance_df = (
        calculate_feature_family_importance(
            tree_importance_df=(
                tree_importance_df
            ),
            model=model,
            x_validation=x_validation,
            y_validation=y_validation,
        )
    )

    # ---------------------------------------
    # E. TARGETED LIFECYCLE IMPORTANCE
    # ---------------------------------------

    print(
        "\nCalculating targeted lifecycle importance..."
    )

    lifecycle_importance_df = (
        calculate_lifecycle_importance(
            model=model,
            x_validation=x_validation,
            y_validation=y_validation,
            permutation_importance_df=(
                permutation_importance_df
            ),
            sensor_importance_df=(
                sensor_importance_df
            ),
        )
    )

    # ---------------------------------------
    # SAVE TABLES
    # ---------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tree_importance_df.to_csv(
        TREE_FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    permutation_importance_df.to_csv(
        PERMUTATION_IMPORTANCE_FILE,
        index=False,
    )

    sensor_importance_df.to_csv(
        SENSOR_IMPORTANCE_FILE,
        index=False,
    )

    family_importance_df.to_csv(
        FEATURE_FAMILY_IMPORTANCE_FILE,
        index=False,
    )

    lifecycle_importance_df.to_csv(
        LIFECYCLE_IMPORTANCE_FILE,
        index=False,
    )

    # ---------------------------------------
    # SAVE PLOTS
    # ---------------------------------------

    save_top_feature_plot(
        tree_importance_df
    )

    save_sensor_plot(
        sensor_importance_df
    )

    save_feature_family_plot(
        family_importance_df
    )

    # ---------------------------------------
    # RESULTS
    # ---------------------------------------

    print(
        "\n======================================"
    )

    print(
        "TOP TREE FEATURES"
    )

    print(
        "======================================"
    )

    print(
        tree_importance_df
        .head(15)
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n======================================"
    )

    print(
        "TOP PERMUTATION FEATURES"
    )

    print(
        "======================================"
    )

    print(
        permutation_importance_df
        .head(15)
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n======================================"
    )

    print(
        "IMPORTANCE BY SENSOR FAMILY"
    )

    print(
        "======================================"
    )

    print(
        sensor_importance_df
        .sort_values(
            "group_permutation_mae_increase",
            ascending=False,
        )
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n======================================"
    )

    print(
        "IMPORTANCE BY FEATURE FAMILY"
    )

    print(
        "======================================"
    )

    print(
        family_importance_df
        .sort_values(
            "group_permutation_mae_increase",
            ascending=False,
        )
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n======================================"
    )

    print(
        "TARGETED LIFECYCLE IMPORTANCE"
    )

    print(
        "======================================"
    )

    print(
        lifecycle_importance_df
        .round(6)
        .to_string(index=False)
    )

    print(
        "\n======================================"
    )


    for output_file in [
        TREE_FEATURE_IMPORTANCE_FILE,
        PERMUTATION_IMPORTANCE_FILE,
        SENSOR_IMPORTANCE_FILE,
        FEATURE_FAMILY_IMPORTANCE_FILE,
        LIFECYCLE_IMPORTANCE_FILE,
        TOP_FEATURE_IMPORTANCE_PLOT,
        GROUPED_SENSOR_IMPORTANCE_PLOT,
        FEATURE_FAMILY_IMPORTANCE_PLOT,
    ]:
        print(
            output_file
        )


if __name__ == "__main__":
    main()