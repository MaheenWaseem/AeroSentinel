from __future__ import annotations

from pathlib import Path

import pandas as pd


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

FEATURE_SET_EFFECT_FILE = (
    OUTPUT_DIR
    / "classical_feature_set_effect.csv"
)

ENGINE_IMPROVEMENT_FILE = (
    OUTPUT_DIR
    / "classical_engine_improvement.csv"
)

ERROR_DIRECTION_FILE = (
    OUTPUT_DIR
    / "classical_error_direction.csv"
)


def load_classical_results() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load the Day 3 classical comparison,
    predictions, and per-engine metrics.
    """

    comparison_df = pd.read_csv(
        CLASSICAL_COMPARISON_FILE
    )

    predictions_df = pd.read_csv(
        CLASSICAL_PREDICTIONS_FILE
    )

    engine_metrics_df = pd.read_csv(
        CLASSICAL_ENGINE_METRICS_FILE
    )

    return (
        comparison_df,
        predictions_df,
        engine_metrics_df,
    )


def calculate_feature_set_effect(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare current-state and temporal MAE
    for each model.
    """

    current_df = (
        comparison_df[
            comparison_df[
                "feature_set"
            ] == "current"
        ]
        [
            [
                "model",
                "mae",
                "rmse",
                "mean_engine_mae",
            ]
        ]
        .rename(
            columns={
                "mae": "current_mae",
                "rmse": "current_rmse",
                "mean_engine_mae": (
                    "current_mean_engine_mae"
                ),
            }
        )
    )

    temporal_df = (
        comparison_df[
            comparison_df[
                "feature_set"
            ] == "temporal"
        ]
        [
            [
                "model",
                "mae",
                "rmse",
                "mean_engine_mae",
            ]
        ]
        .rename(
            columns={
                "mae": "temporal_mae",
                "rmse": "temporal_rmse",
                "mean_engine_mae": (
                    "temporal_mean_engine_mae"
                ),
            }
        )
    )

    effect_df = current_df.merge(
        temporal_df,
        on="model",
        how="inner",
    )

    effect_df[
        "mae_improvement"
    ] = (
        effect_df["current_mae"]
        - effect_df["temporal_mae"]
    )

    effect_df[
        "mae_improvement_percent"
    ] = (
        effect_df[
            "mae_improvement"
        ]
        / effect_df[
            "current_mae"
        ]
        * 100
    )

    effect_df[
        "rmse_improvement"
    ] = (
        effect_df["current_rmse"]
        - effect_df["temporal_rmse"]
    )

    return effect_df.sort_values(
        "mae_improvement",
        ascending=False,
    )


def calculate_engine_improvement(
    engine_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate whether temporal history improved
    MAE for each individual engine and model.
    """

    current_df = (
        engine_metrics_df[
            engine_metrics_df[
                "feature_set"
            ] == "current"
        ]
        [
            [
                "engine_id",
                "model",
                "engine_mae",
            ]
        ]
        .rename(
            columns={
                "engine_mae": (
                    "current_engine_mae"
                )
            }
        )
    )

    temporal_df = (
        engine_metrics_df[
            engine_metrics_df[
                "feature_set"
            ] == "temporal"
        ]
        [
            [
                "engine_id",
                "model",
                "engine_mae",
            ]
        ]
        .rename(
            columns={
                "engine_mae": (
                    "temporal_engine_mae"
                )
            }
        )
    )

    improvement_df = current_df.merge(
        temporal_df,
        on=[
            "engine_id",
            "model",
        ],
        how="inner",
    )

    improvement_df[
        "engine_mae_improvement"
    ] = (
        improvement_df[
            "current_engine_mae"
        ]
        - improvement_df[
            "temporal_engine_mae"
        ]
    )

    improvement_df[
        "temporal_helped"
    ] = (
        improvement_df[
            "engine_mae_improvement"
        ]
        > 0
    )

    return improvement_df


def calculate_error_direction(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize signed prediction error for each
    model and feature set.
    """

    rows = []

    for (
        model_name,
        feature_set_name,
    ), group_df in predictions_df.groupby(
        [
            "model",
            "feature_set",
        ]
    ):

        mean_error = (
            group_df[
                "error"
            ]
            .mean()
        )

        overprediction_fraction = (
            group_df[
                "error"
            ]
            .gt(0)
            .mean()
        )

        underprediction_fraction = (
            group_df[
                "error"
            ]
            .lt(0)
            .mean()
        )

        rows.append(
            {
                "model": model_name,
                "feature_set": (
                    feature_set_name
                ),
                "mean_signed_error": (
                    mean_error
                ),
                "overprediction_fraction": (
                    overprediction_fraction
                ),
                "underprediction_fraction": (
                    underprediction_fraction
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def select_classical_champion(
    comparison_df: pd.DataFrame,
) -> pd.Series:
    """
    Select the development classical champion
    using lowest validation MAE.
    """

    champion_row = (
        comparison_df
        .sort_values(
            [
                "mae",
                "mean_engine_mae",
            ]
        )
        .iloc[0]
    )

    return champion_row


def main() -> None:
    (
        comparison_df,
        predictions_df,
        engine_metrics_df,
    ) = load_classical_results()

    feature_effect_df = (
        calculate_feature_set_effect(
            comparison_df
        )
    )

    engine_improvement_df = (
        calculate_engine_improvement(
            engine_metrics_df
        )
    )

    error_direction_df = (
        calculate_error_direction(
            predictions_df
        )
    )

    champion = (
        select_classical_champion(
            comparison_df
        )
    )

    champion_model = champion[
        "model"
    ]

    champion_feature_set = champion[
        "feature_set"
    ]

    champion_engine_metrics_df = (
        engine_metrics_df[
            (
                engine_metrics_df[
                    "model"
                ]
                == champion_model
            )
            &
            (
                engine_metrics_df[
                    "feature_set"
                ]
                == champion_feature_set
            )
        ]
    )

    worst_engines_df = (
        champion_engine_metrics_df
        .sort_values(
            "engine_mae",
            ascending=False,
        )
        .head(5)
    )

    champion_improvement_df = (
        engine_improvement_df[
            engine_improvement_df[
                "model"
            ]
            == champion_model
        ]
    )

    engines_helped = int(
        champion_improvement_df[
            "temporal_helped"
        ]
        .sum()
    )

    total_engines = len(
        champion_improvement_df
    )

    feature_effect_df.to_csv(
        FEATURE_SET_EFFECT_FILE,
        index=False,
    )

    engine_improvement_df.to_csv(
        ENGINE_IMPROVEMENT_FILE,
        index=False,
    )

    error_direction_df.to_csv(
        ERROR_DIRECTION_FILE,
        index=False,
    )

    print(
        "\n======================================"
    )
    print(
        "CURRENT VS TEMPORAL EFFECT"
    )
    print(
        "======================================"
    )

    print(
        feature_effect_df
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n======================================"
    )
    print(
        "TEMPORAL CONSISTENCY"
    )
    print(
        "======================================"
    )

    print(
        "\nChampion-model engines where "
        "temporal features improved MAE:"
    )

    print(
        f"{engines_helped} / {total_engines}"
    )

    print(
        "\n======================================"
    )
    print(
        "ERROR DIRECTION"
    )
    print(
        "======================================"
    )

    print(
        error_direction_df
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n======================================"
    )
    print(
        "WORST CHAMPION ENGINES"
    )
    print(
        "======================================"
    )

    print(
        worst_engines_df
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\n======================================"
    )
    print(
        "DEVELOPMENT CLASSICAL CHAMPION"
    )
    print(
        "======================================"
    )

    print(
        "\nModel:"
    )
    print(
        champion_model
    )

    print(
        "\nFeature set:"
    )
    print(
        champion_feature_set
    )

    print(
        "\nValidation MAE:"
    )
    print(
        round(
            champion["mae"],
            4,
        )
    )

    print(
        "\nValidation RMSE:"
    )
    print(
        round(
            champion["rmse"],
            4,
        )
    )

    print(
        "\nMean engine MAE:"
    )
    print(
        round(
            champion[
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
            champion[
                "median_engine_mae"
            ],
            4,
        )
    )

    print(
        "\nAnalysis files saved to:"
    )

    print(
        FEATURE_SET_EFFECT_FILE
    )

    print(
        ENGINE_IMPROVEMENT_FILE
    )

    print(
        ERROR_DIRECTION_FILE
    )


if __name__ == "__main__":
    main()