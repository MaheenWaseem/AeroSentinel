from __future__ import annotations

import json

import pandas as pd

from src.evaluation.analyze_refinement_errors import (
    ENGINE_COMPARISON_FILE,
    ENGINE_SUMMARY_FILE,
    NEAR_FAILURE_FILE,
    RUL_SLICES_FILE,
)
from src.modeling.prepare_classical_data import (
    build_modeling_matrices,
    load_temporal_feature_tables,
    validate_modeling_matrices,
)
from src.modeling.refine_classical_models import (
    REFINEMENT_COMPARISON_FILE,
    REFINEMENT_PREDICTIONS_FILE,
)
from src.modeling.train_classical_models import (
    OUTPUT_DIR,
    build_models,
)


BENCHMARK_SPEC_FILE = (
    OUTPUT_DIR / "classical_benchmark_specification.json"
)

BENCHMARK_PREDICTIONS_FILE = (
    OUTPUT_DIR / "classical_benchmark_predictions.csv"
)

BENCHMARK_ENGINE_METRICS_FILE = (
    OUTPUT_DIR / "classical_benchmark_engine_metrics.csv"
)

BENCHMARK_RUL_SLICES_FILE = (
    OUTPUT_DIR / "classical_benchmark_rul_slices.csv"
)

SELECTED_MODEL = "random_forest_depth12"
SELECTED_FEATURE_SET = "temporal_cycle"


def select_benchmark_rows(
    table: pd.DataFrame,
) -> pd.DataFrame:

    return table.loc[
        (table["model"] == SELECTED_MODEL)
        & (table["feature_set"] == SELECTED_FEATURE_SET)
    ].copy()


def single_record(table: pd.DataFrame) -> dict:

    if len(table) != 1:
        raise ValueError(
            "Expected exactly one benchmark summary row."
        )

    return json.loads(
        table.to_json(orient="records")
    )[0]


def main() -> None:

    comparison_df = pd.read_csv(
        REFINEMENT_COMPARISON_FILE
    )

    ranked_df = comparison_df.sort_values(
        ["mae", "mean_engine_mae"]
    )

    winner = ranked_df.iloc[0]

    if (
        winner["model"] != SELECTED_MODEL
        or winner["feature_set"] != SELECTED_FEATURE_SET
    ):
        raise ValueError(
            "Saved results no longer select the reviewed "
            "depth-12 temporal-cycle benchmark."
        )

    selected_metrics_df = select_benchmark_rows(
        comparison_df
    )

    predictions_df = select_benchmark_rows(
        pd.read_csv(REFINEMENT_PREDICTIONS_FILE)
    )

    engine_metrics_df = select_benchmark_rows(
        pd.read_csv(ENGINE_COMPARISON_FILE)
    )

    rul_slices_df = select_benchmark_rows(
        pd.read_csv(RUL_SLICES_FILE)
    )

    near_failure_df = select_benchmark_rows(
        pd.read_csv(NEAR_FAILURE_FILE)
    )

    engine_summary_df = select_benchmark_rows(
        pd.read_csv(ENGINE_SUMMARY_FILE)
    )

    train_df, validation_df = (
        load_temporal_feature_tables()
    )

    matrices = build_modeling_matrices(
        train_df,
        validation_df,
    )

    validate_modeling_matrices(matrices)

    feature_columns = [
        "cycle",
        *matrices["x_train_temporal"].columns.tolist(),
    ]

    if len(feature_columns) != 99:
        raise ValueError(
            "Expected exactly 99 benchmark predictors."
        )

    if len(set(feature_columns)) != 99:
        raise ValueError("Duplicate benchmark predictors.")

    if {"engine_id", "rul"} & set(feature_columns):
        raise ValueError(
            "Identifier or target found in predictors."
        )

    if predictions_df.duplicated(
        ["engine_id", "cycle"]
    ).any():
        raise ValueError(
            "Duplicate benchmark prediction observations."
        )

    saved_targets = (
        predictions_df.set_index(["engine_id", "cycle"])
        ["actual_rul"]
        .sort_index()
    )

    expected_targets = (
        validation_df.set_index(["engine_id", "cycle"])
        ["rul"]
        .sort_index()
    )

    if (
        not saved_targets.index.equals(expected_targets.index)
        or not (
            saved_targets.to_numpy()
            == expected_targets.to_numpy()
        ).all()
    ):
        raise ValueError(
            "Saved benchmark predictions do not match "
            "the current validation observations and targets."
        )

    if (
        len(engine_metrics_df) != 20
        or engine_metrics_df["engine_id"].nunique() != 20
        or len(rul_slices_df) != 3
    ):
        raise ValueError(
            "Incomplete benchmark engine or RUL-slice evidence."
        )

    # Build the configuration without fitting it.
    model = build_models()["random_forest"]
    model.set_params(max_depth=12)

    specification = {
        "benchmark_name": (
            "Refined Development Classical Champion"
        ),
        "dataset": "NASA C-MAPSS FD001",
        "model": SELECTED_MODEL,
        "estimator": "RandomForestRegressor",
        "feature_set": SELECTED_FEATURE_SET,
        "selection_rule": [
            "Lowest development-validation MAE",
            "Lowest mean engine MAE if MAE is tied",
        ],
        "model_parameters": model.get_params(),
        "predictor_count": len(feature_columns),
        "feature_columns": feature_columns,
        "development_data": {
            "train_engine_count": int(
                train_df["engine_id"].nunique()
            ),
            "validation_engine_count": int(
                validation_df["engine_id"].nunique()
            ),
            "train_row_count": len(train_df),
            "validation_row_count": len(validation_df),
        },
        "validation_metrics": single_record(
            selected_metrics_df
        ),
        "near_failure_metrics": single_record(
            near_failure_df
        ),
        "engine_comparison_summary": single_record(
            engine_summary_df
        ),
        "diagnostic_thresholds": {
            "near_failure_max_actual_rul": 30,
            "medium_rul_max_actual_rul": 100,
            "large_overprediction_error_gt": 20,
            "meaning": (
                "Project diagnostic thresholds, "
                "not maintenance limits."
            ),
        },
        "limitations": [
            "Selected on one fixed development split.",
            "Small MAE advantage over unrestricted RF with cycle.",
            "Mean engine MAE is worse than original temporal RF.",
            "Twelve engines improved and eight worsened.",
            "Engines 91 and 45 deteriorated substantially.",
            "Near-failure errors retain positive bias.",
            "Cycle may encode dataset-specific lifetime patterns.",
            "Official test performance has not been evaluated yet.",
        ],
        "fitted_model_saved": False,
        "artifacts": {
            "predictions": str(BENCHMARK_PREDICTIONS_FILE),
            "engine_metrics": str(BENCHMARK_ENGINE_METRICS_FILE),
            "rul_slices": str(BENCHMARK_RUL_SLICES_FILE),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(
        BENCHMARK_PREDICTIONS_FILE,
        index=False,
    )
    engine_metrics_df.to_csv(
        BENCHMARK_ENGINE_METRICS_FILE,
        index=False,
    )
    rul_slices_df.to_csv(
        BENCHMARK_RUL_SLICES_FILE,
        index=False,
    )

    BENCHMARK_SPEC_FILE.write_text(
        json.dumps(specification, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nCLASSICAL BENCHMARK FROZEN")
    print("Model:", SELECTED_MODEL)
    print("Feature set:", SELECTED_FEATURE_SET)
    print("Predictor count:", len(feature_columns))
    print(
        "Validation MAE:",
        f"{winner['mae']:.4f}",
    )
    print(
        "Validation RMSE:",
        f"{winner['rmse']:.4f}",
    )
    print("Prediction rows:", len(predictions_df))
    print("Engine rows:", len(engine_metrics_df))
    print("RUL slice rows:", len(rul_slices_df))

    print("\nSaved:")
    print(BENCHMARK_SPEC_FILE)
    print(BENCHMARK_PREDICTIONS_FILE)
    print(BENCHMARK_ENGINE_METRICS_FILE)
    print(BENCHMARK_RUL_SLICES_FILE)

    print("\nFitted model saved: No")
    print("Official test evaluated by this module: No")


if __name__ == "__main__":
    main()