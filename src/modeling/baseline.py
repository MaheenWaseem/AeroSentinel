from __future__ import annotations

import numpy as np

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


def main() -> None:
    print("BASELINE MAIN IS RUNNING")
    """
    Evaluate a simple training-mean RUL baseline
    on development validation data.
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

    y_train = matrices[
        "y_train"
    ]

    y_validation = matrices[
        "y_validation"
    ]

    mean_training_rul = float(
        y_train.mean()
    )

    baseline_predictions = np.full(
        shape=len(y_validation),
        fill_value=mean_training_rul,
    )

    overall_metrics = (
        calculate_regression_metrics(
            y_validation,
            baseline_predictions,
        )
    )

    prediction_df = (
        build_prediction_table(
            metadata_df=validation_df,
            y_true=y_validation,
            y_pred=baseline_predictions,
        )
    )

    engine_metrics_df = (
        calculate_engine_metrics(
            prediction_df
        )
    )

    engine_summary = (
        summarize_engine_metrics(
            engine_metrics_df
        )
    )

    print(
        "\n======================================"
    )
    print(
        "TRAINING-MEAN RUL BASELINE"
    )
    print(
        "======================================"
    )

    print(
        "\nMean training RUL:"
    )
    print(
        round(
            mean_training_rul,
            4,
        )
    )

    print(
        "\nValidation MAE:"
    )
    print(
        round(
            overall_metrics["mae"],
            4,
        )
    )

    print(
        "\nValidation RMSE:"
    )
    print(
        round(
            overall_metrics["rmse"],
            4,
        )
    )

    print(
        "\nMean engine MAE:"
    )
    print(
        round(
            engine_summary[
                "mean_engine_mae"
            ],
            4,
        )
    )

    print(
        "\nMedian engine MAE:"
    )
    print(
        round(
            engine_summary[
                "median_engine_mae"
            ],
            4,
        )
    )

    print(
        "\nMean engine RMSE:"
    )
    print(
        round(
            engine_summary[
                "mean_engine_rmse"
            ],
            4,
        )
    )

    print(
        "\nWorst 5 validation engines by MAE:"
    )

    print(
        engine_metrics_df
        .sort_values(
            "engine_mae",
            ascending=False,
        )
        .head(5)
        .round(4)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()