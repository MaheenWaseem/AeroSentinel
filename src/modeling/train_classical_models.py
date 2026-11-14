from __future__ import annotations
from pathlib import Path
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.rul_metrics import (
    build_prediction_table,
    calculate_engine_metrics,
    calculate_regression_metrics,
    summarize_engine_metrics,
)
from src.modeling.prepare_classical_data import (
    build_modeling_matrices,
    load_temporal_feature_tables,
    validate_modeling_matrices,
)


RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs")

CLASSICAL_COMPARISON_FILE = (
    OUTPUT_DIR
    / "classical_model_comparison.csv"
)

CLASSICAL_PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "classical_predictions.csv"
)

CLASSICAL_ENGINE_METRICS_FILE = (
    OUTPUT_DIR
    / "classical_engine_metrics.csv"
)



def build_models() -> dict[str, object]:
    """
    Build the classical regression models used
    in the controlled comparison.
    """

    ridge_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "regressor",
                Ridge(
                    alpha=1.0,
                ),
            ),
        ]
    )

    random_forest_model = (
        RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    hist_gradient_boosting_model = (
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        )
    )

    return {
        "ridge": ridge_model,
        "random_forest": (
            random_forest_model
        ),
        "hist_gradient_boosting": (
            hist_gradient_boosting_model
        ),
    }


def evaluate_model(
    model_name: str,
    feature_set_name: str,
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_df: pd.DataFrame,
) -> tuple[
    dict[str, float | str],
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Fit one model, predict development-validation
    RUL, and calculate row-level and engine-level
    metrics.
    """

    model.fit(
        x_train,
        y_train,
    )

    validation_predictions = (
        model.predict(
            x_validation
        )
    )

    overall_metrics = (
        calculate_regression_metrics(
            y_validation,
            validation_predictions,
        )
    )

    prediction_df = (
        build_prediction_table(
            metadata_df=validation_df,
            y_true=y_validation,
            y_pred=validation_predictions,
        )
    )

    prediction_df[
        "model"
    ] = model_name

    prediction_df[
        "feature_set"
    ] = feature_set_name

    engine_metrics_df = (
        calculate_engine_metrics(
            prediction_df
        )
    )

    engine_metrics_df[
        "model"
    ] = model_name

    engine_metrics_df[
        "feature_set"
    ] = feature_set_name

    engine_summary = (
        summarize_engine_metrics(
            engine_metrics_df
        )
    )

    metrics_row = {
        "model": model_name,
        "feature_set": feature_set_name,
        "mae": overall_metrics["mae"],
        "rmse": overall_metrics["rmse"],
        "mean_engine_mae": (
            engine_summary[
                "mean_engine_mae"
            ]
        ),
        "median_engine_mae": (
            engine_summary[
                "median_engine_mae"
            ]
        ),
        "mean_engine_rmse": (
            engine_summary[
                "mean_engine_rmse"
            ]
        ),
    }

    return (
        metrics_row,
        prediction_df,
        engine_metrics_df,
    )


def main() -> None:
    """
    Run the six controlled current-state versus
    temporal classical RUL experiments.
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

    models = build_models()

    experiments = [
        (
            "ridge",
            "current",
            models["ridge"],
            matrices["x_train_current"],
            matrices[
                "x_validation_current"
            ],
        ),
        (
            "ridge",
            "temporal",
            models["ridge"],
            matrices[
                "x_train_temporal"
            ],
            matrices[
                "x_validation_temporal"
            ],
        ),
        (
            "random_forest",
            "current",
            models[
                "random_forest"
            ],
            matrices["x_train_current"],
            matrices[
                "x_validation_current"
            ],
        ),
        (
            "random_forest",
            "temporal",
            models[
                "random_forest"
            ],
            matrices[
                "x_train_temporal"
            ],
            matrices[
                "x_validation_temporal"
            ],
        ),
        (
            "hist_gradient_boosting",
            "current",
            models[
                "hist_gradient_boosting"
            ],
            matrices["x_train_current"],
            matrices[
                "x_validation_current"
            ],
        ),
        (
            "hist_gradient_boosting",
            "temporal",
            models[
                "hist_gradient_boosting"
            ],
            matrices[
                "x_train_temporal"
            ],
            matrices[
                "x_validation_temporal"
            ],
        ),
    ]

    metrics_rows = []

    prediction_tables = []

    engine_metric_tables = []

    for (
        model_name,
        feature_set_name,
        model,
        x_train,
        x_validation,
    ) in experiments:

        print(
            "\nRunning:"
        )
        print(
            model_name,
            "-",
            feature_set_name,
        )

        (
            metrics_row,
            prediction_df,
            engine_metrics_df,
        ) = evaluate_model(
            model_name=model_name,
            feature_set_name=(
                feature_set_name
            ),
            model=model,
            x_train=x_train,
            y_train=matrices[
                "y_train"
            ],
            x_validation=x_validation,
            y_validation=matrices[
                "y_validation"
            ],
            validation_df=validation_df,
        )

        metrics_rows.append(
            metrics_row
        )

        prediction_tables.append(
            prediction_df
        )

        engine_metric_tables.append(
            engine_metrics_df
        )

    comparison_df = pd.DataFrame(
        metrics_rows
    )

    all_predictions_df = pd.concat(
        prediction_tables,
        ignore_index=True,
    )

    all_engine_metrics_df = pd.concat(
        engine_metric_tables,
        ignore_index=True,
    ) 

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        CLASSICAL_COMPARISON_FILE,
        index=False,
    )

    all_predictions_df.to_csv(
        CLASSICAL_PREDICTIONS_FILE,
        index=False,
    )

    all_engine_metrics_df.to_csv(
        CLASSICAL_ENGINE_METRICS_FILE,
        index=False,
    )

    print(
        "\n======================================"
    )

    print(
        "CLASSICAL MODEL COMPARISON"
    )

    print(
        "======================================"
    )

    print(
        comparison_df
        .sort_values(
            "mae"
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nTotal prediction rows:"
    )

    print(
        all_predictions_df.shape
    )

    print(
        "\nTotal engine-metric rows:"
    )

    print(
        all_engine_metrics_df.shape
    )


    print(
        "\nClassical comparison saved to:"
    )
    print(
        CLASSICAL_COMPARISON_FILE
    )

    print(
        "\nClassical predictions saved to:"
    )
    print(
        CLASSICAL_PREDICTIONS_FILE
    )

    print(
        "\nClassical engine metrics saved to:"
    )
    print(
        CLASSICAL_ENGINE_METRICS_FILE
    )


if __name__ == "__main__":
    main()