from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.modeling.prepare_classical_data import (
    build_modeling_matrices,
    load_temporal_feature_tables,
    validate_modeling_matrices,
)
from src.modeling.train_classical_models import (
    CLASSICAL_COMPARISON_FILE,
    OUTPUT_DIR,
    build_models,
    evaluate_model,
)


REFINEMENT_COMPARISON_FILE = (
    OUTPUT_DIR / "classical_refinement_comparison.csv"
)

REFINEMENT_PREDICTIONS_FILE = (
    OUTPUT_DIR / "classical_refinement_predictions.csv"
)

REFINEMENT_ENGINE_METRICS_FILE = (
    OUTPUT_DIR / "classical_refinement_engine_metrics.csv"
)

METRIC_COLUMNS = [
    "mae",
    "rmse",
    "mean_engine_mae",
    "median_engine_mae",
    "mean_engine_rmse",
]


def load_day3_reference() -> pd.Series:
    """
    Load the saved Day 3 temporal Random Forest metrics.
    """

    comparison_df = pd.read_csv(
        CLASSICAL_COMPARISON_FILE
    )

    reference_df = comparison_df.loc[
        (comparison_df["model"] == "random_forest")
        & (comparison_df["feature_set"] == "temporal")
    ]

    if len(reference_df) != 1:
        raise ValueError(
            "Expected exactly one Day 3 temporal "
            "Random Forest reference row."
        )

    return reference_df.iloc[0]


def verify_day3_reference(
    metrics_row: dict[str, float | str],
    saved_reference: pd.Series,
) -> None:
    """
    Check that the rerun matches the saved Day 3
    reference, allowing small numerical differences.
    """

    for metric_name in METRIC_COLUMNS:
        saved_value = float(
            saved_reference[metric_name]
        )
        rerun_value = float(
            metrics_row[metric_name]
        )

        if not np.isclose(
            saved_value,
            rerun_value,
            rtol=0.0,
            atol=0.0001,
        ):
            raise ValueError(
                f"Day 3 reference mismatch for {metric_name}: "
                f"saved={saved_value:.6f}, "
                f"rerun={rerun_value:.6f}. "
                "Check the data, model settings, and "
                "software environment before continuing."
            )


def main() -> None:
    """
    Compare temporal Random Forest with and without
    cycle using the fixed development datasets.
    """

    saved_reference = load_day3_reference()

    train_df, validation_df = (
        load_temporal_feature_tables()
    )

    matrices = build_modeling_matrices(
        train_df,
        validation_df,
    )

    validate_modeling_matrices(matrices)

    x_train_temporal = matrices["x_train_temporal"]
    x_validation_temporal = (
        matrices["x_validation_temporal"]
    )

    y_train = matrices["y_train"]
    y_validation = matrices["y_validation"]

    # Confirm predictor, target, and metadata alignment.
    for split_name, features, target, metadata in [
        (
            "training",
            x_train_temporal,
            y_train,
            train_df,
        ),
        (
            "validation",
            x_validation_temporal,
            y_validation,
            validation_df,
        ),
    ]:
        if (
            not features.index.equals(metadata.index)
            or not features.index.equals(target.index)
        ):
            raise ValueError(
                f"Row alignment failed for {split_name}."
            )

        if features.shape[1] != 98:
            raise ValueError(
                f"Expected 98 temporal predictors for "
                f"{split_name}, found {features.shape[1]}."
            )

        if any(
            column in features.columns
            for column in ["engine_id", "cycle", "rul"]
        ):
            raise ValueError(
                "The reference predictors must exclude "
                "engine_id, cycle, and rul."
            )

        if metadata["cycle"].isna().any():
            raise ValueError(
                f"Missing cycle values in {split_name}."
            )

        if target.isna().any():
            raise ValueError(
                f"Missing RUL targets in {split_name}."
            )

    # Add cycle to copies of the existing temporal matrices.
    x_train_temporal_cycle = x_train_temporal.copy()
    x_validation_temporal_cycle = (
        x_validation_temporal.copy()
    )

    x_train_temporal_cycle.insert(
        0,
        "cycle",
        train_df["cycle"],
    )

    x_validation_temporal_cycle.insert(
        0,
        "cycle",
        validation_df["cycle"],
    )

    print("\nDAY 4 FEATURE CHECKS")
    print(
        "Temporal training shape:",
        x_train_temporal.shape,
    )
    print(
        "Temporal + cycle training shape:",
        x_train_temporal_cycle.shape,
    )
    print(
        "Temporal validation shape:",
        x_validation_temporal.shape,
    )
    print(
        "Temporal + cycle validation shape:",
        x_validation_temporal_cycle.shape,
    )

    reference_model = build_models()["random_forest"]

    experiments = [
        (
            "temporal",
            x_train_temporal,
            x_validation_temporal,
        ),
        (
            "temporal_cycle",
            x_train_temporal_cycle,
            x_validation_temporal_cycle,
        ),
    ]

    metrics_rows = []
    prediction_tables = []
    engine_metric_tables = []

    for (
        feature_set_name,
        x_train,
        x_validation,
    ) in experiments:

        print(
            f"\nRunning random_forest - {feature_set_name}",
            flush=True,
        )

        # Each experiment receives a fresh, unfitted model
        # with the exact same hyperparameters.
        model = clone(reference_model)

        (
            metrics_row,
            prediction_df,
            engine_metrics_df,
        ) = evaluate_model(
            model_name="random_forest",
            feature_set_name=feature_set_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            validation_df=validation_df,
        )

        if feature_set_name == "temporal":
            verify_day3_reference(
                metrics_row,
                saved_reference,
            )
            print("Day 3 reference reproduction: PASSED")

        metrics_row["predictor_count"] = (
            x_train.shape[1]
        )

        metrics_rows.append(metrics_row)
        prediction_tables.append(prediction_df)
        engine_metric_tables.append(engine_metrics_df)

        print(
            f"Validation MAE: {metrics_row['mae']:.4f}"
        )
        print(
            f"Validation RMSE: {metrics_row['rmse']:.4f}"
        )

    comparison_df = pd.DataFrame(metrics_rows)

    all_predictions_df = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    all_engine_metrics_df = pd.concat(
        engine_metric_tables,
        ignore_index=True,
    )

    results_by_feature = comparison_df.set_index(
        "feature_set"
    )

    reference_mae = float(
        results_by_feature.loc["temporal", "mae"]
    )
    cycle_mae = float(
        results_by_feature.loc["temporal_cycle", "mae"]
    )

    mae_improvement = reference_mae - cycle_mae
    mae_improvement_percent = (
        mae_improvement / reference_mae * 100
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        REFINEMENT_COMPARISON_FILE,
        index=False,
    )

    all_predictions_df.to_csv(
        REFINEMENT_PREDICTIONS_FILE,
        index=False,
    )

    all_engine_metrics_df.to_csv(
        REFINEMENT_ENGINE_METRICS_FILE,
        index=False,
    )

    print("\nCONTROLLED CYCLE COMPARISON")
    print(
        comparison_df.round(4).to_string(index=False)
    )

    print(
        "\nMAE improvement from adding cycle:",
        f"{mae_improvement:.4f}",
    )
    print(
        "MAE improvement percent:",
        f"{mae_improvement_percent:.4f}%",
    )
    
    print("Positive improvement means lower MAE.")

    print(
        "\nTotal prediction rows:",
        len(all_predictions_df),
    )
    print(
        "Total engine-metric rows:",
        len(all_engine_metrics_df),
    )

    print("\nSaved:")
    for output_file in [
        REFINEMENT_COMPARISON_FILE,
        REFINEMENT_PREDICTIONS_FILE,
        REFINEMENT_ENGINE_METRICS_FILE,
    ]:
        print(output_file)


def run_complexity_comparison() -> None:
    """
    Run two controlled RF complexity experiments.
    """

    comparison_df = pd.read_csv(
        REFINEMENT_COMPARISON_FILE
    )
    predictions_df = pd.read_csv(
        REFINEMENT_PREDICTIONS_FILE
    )
    engine_metrics_df = pd.read_csv(
        REFINEMENT_ENGINE_METRICS_FILE
    )
    
    reference_tables = []

    for table in [
        comparison_df,
        predictions_df,
        engine_metrics_df,
    ]:
        reference_mask = (
            (table["model"] == "random_forest")
            & table["feature_set"].isin(
                ["temporal", "temporal_cycle"]
            )
        )

        reference_tables.append(
            table.loc[reference_mask].copy()
        )

    (
        comparison_df,
        predictions_df,
        engine_metrics_df,
    ) = reference_tables

    for feature_set_name in [
        "temporal",
        "temporal_cycle",
    ]:
        reference_rows = comparison_df.loc[
            comparison_df["feature_set"]
            == feature_set_name
        ]

        if len(reference_rows) != 1:
            raise ValueError(
                "Expected one saved Part 1B comparison "
                f"row for {feature_set_name}."
            )

    train_df, validation_df = (
        load_temporal_feature_tables()
    )

    matrices = build_modeling_matrices(
        train_df,
        validation_df,
    )

    validate_modeling_matrices(matrices)

    x_train = matrices["x_train_temporal"].copy()
    x_validation = (
        matrices["x_validation_temporal"].copy()
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
            "Expected 99 predictors including cycle."
        )

    if (
        x_train.isna().any().any()
        or x_validation.isna().any().any()
    ):
        raise ValueError(
            "Complexity experiment predictors "
            "contain missing values."
        )

    reference_model = build_models()["random_forest"]

    candidates = [
        (
            "random_forest_leaf4",
            {"min_samples_leaf": 4},
        ),
        (
            "random_forest_depth12",
            {"max_depth": 12},
        ),
    ]

    new_metrics_rows = []
    new_prediction_tables = []
    new_engine_metric_tables = []

    print("\nCONTROLLED RF COMPLEXITY EXPERIMENTS")
    print("Training shape:", x_train.shape)
    print("Validation shape:", x_validation.shape)

    for model_name, parameter_changes in candidates:
        model = clone(reference_model)
        model.set_params(**parameter_changes)

        print(
            f"\nRunning {model_name} - temporal_cycle",
            flush=True,
        )
        print("Parameter change:", parameter_changes)

        (
            metrics_row,
            prediction_df,
            candidate_engine_metrics_df,
        ) = evaluate_model(
            model_name=model_name,
            feature_set_name="temporal_cycle",
            model=model,
            x_train=x_train,
            y_train=matrices["y_train"],
            x_validation=x_validation,
            y_validation=matrices["y_validation"],
            validation_df=validation_df,
        )

        metrics_row["predictor_count"] = (
            x_train.shape[1]
        )

        new_metrics_rows.append(metrics_row)
        new_prediction_tables.append(prediction_df)
        new_engine_metric_tables.append(
            candidate_engine_metrics_df
        )

        print(
            f"Validation MAE: {metrics_row['mae']:.4f}"
        )
        print(
            "Mean engine MAE:",
            f"{metrics_row['mean_engine_mae']:.4f}",
        )

    combined_comparison_df = pd.concat(
        [
            comparison_df,
            pd.DataFrame(new_metrics_rows),
        ],
        ignore_index=True,
    )

    combined_predictions_df = pd.concat(
        [predictions_df, *new_prediction_tables],
        ignore_index=True,
    )

    combined_engine_metrics_df = pd.concat(
        [
            engine_metrics_df,
            *new_engine_metric_tables,
        ],
        ignore_index=True,
    )

    # Record settings for each experiment in the CSV.
    parameter_lookup = {
        "random_forest": ("None", 2),
        "random_forest_leaf4": ("None", 4),
        "random_forest_depth12": ("12", 2),
    }

    combined_comparison_df["n_estimators"] = (
        reference_model.n_estimators
    )
    combined_comparison_df["random_state"] = (
        reference_model.random_state
    )
    combined_comparison_df["max_depth"] = (
        combined_comparison_df["model"].map(
            lambda name: parameter_lookup[name][0]
        )
    )
    combined_comparison_df["min_samples_leaf"] = (
        combined_comparison_df["model"].map(
            lambda name: parameter_lookup[name][1]
        )
    )

    combined_comparison_df.to_csv(
        REFINEMENT_COMPARISON_FILE,
        index=False,
    )
    combined_predictions_df.to_csv(
        REFINEMENT_PREDICTIONS_FILE,
        index=False,
    )
    combined_engine_metrics_df.to_csv(
        REFINEMENT_ENGINE_METRICS_FILE,
        index=False,
    )

    print("\nALL DAY 4 REFINEMENT RESULTS")
    print(
        combined_comparison_df.sort_values(
            ["mae", "mean_engine_mae"]
        )
        .round(4)
        .to_string(index=False)
    )

    print(
        "\nTotal experiments:",
        len(combined_comparison_df),
    )
    print(
        "Total prediction rows:",
        len(combined_predictions_df),
    )
    print(
        "Total engine-metric rows:",
        len(combined_engine_metrics_df),
    )

    print("\nUpdated:")
    print(REFINEMENT_COMPARISON_FILE)
    print(REFINEMENT_PREDICTIONS_FILE)
    print(REFINEMENT_ENGINE_METRICS_FILE)

    

if __name__ == "__main__":
    run_complexity_comparison()