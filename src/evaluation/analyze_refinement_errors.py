from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling.refine_classical_models import (
    REFINEMENT_PREDICTIONS_FILE,
)
from src.modeling.train_classical_models import (
    OUTPUT_DIR,
)


RUL_SLICES_FILE = (
    OUTPUT_DIR / "classical_refinement_rul_slices.csv"
)

NEAR_FAILURE_FILE = (
    OUTPUT_DIR / "classical_refinement_near_failure.csv"
)

ENGINE_COMPARISON_FILE = (
    OUTPUT_DIR / "classical_refinement_engine_comparison.csv"
)

ENGINE_SUMMARY_FILE = (
    OUTPUT_DIR / "classical_refinement_engine_summary.csv"
)

EXPERIMENT_COLUMNS = ["model", "feature_set"]

OVERPREDICTION_THRESHOLD = 20.0


def summarize_errors(
    prediction_df: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Summarize errors for one experiment or RUL slice.
    """

    errors = prediction_df["error"]
    positive_errors = errors.loc[errors > 0]

    return {
        "row_count": len(prediction_df),
        "engine_count": (
            prediction_df["engine_id"].nunique()
        ),
        "mae": float(errors.abs().mean()),
        "rmse": float(
            np.sqrt(errors.pow(2).mean())
        ),
        "mean_signed_error": float(errors.mean()),
        "overprediction_fraction": float(
            errors.gt(0).mean()
        ),
        "mean_overprediction_when_over": float(
            positive_errors.mean()
        ),
        "overprediction_gt20_fraction": float(
            errors.gt(OVERPREDICTION_THRESHOLD).mean()
        ),
        "p95_absolute_error": float(
            errors.abs().quantile(0.95)
        ),
    }


def main() -> None:
   
    predictions_df = pd.read_csv(
        REFINEMENT_PREDICTIONS_FILE
    )

    required_columns = [
        "model",
        "feature_set",
        "engine_id",
        "cycle",
        "actual_rul",
        "predicted_rul",
        "error",
    ]

    if predictions_df[required_columns].isna().any().any():
        raise ValueError(
            "Required prediction columns contain "
            "missing values."
        )

    row_key = [
        "model",
        "feature_set",
        "engine_id",
        "cycle",
    ]

    if predictions_df.duplicated(row_key).any():
        raise ValueError(
            "Duplicate experiment engine-cycle rows."
        )

    if predictions_df["actual_rul"].lt(0).any():
        raise ValueError("Actual RUL cannot be negative.")

    if not np.allclose(
        predictions_df["error"],
        predictions_df["predicted_rul"]
        - predictions_df["actual_rul"],
    ):
        raise ValueError(
            "Saved errors do not match predicted "
            "minus actual RUL."
        )

    baseline_mask = (
        (predictions_df["model"] == "random_forest")
        & (predictions_df["feature_set"] == "temporal")
    )

    baseline_df = predictions_df.loc[
        baseline_mask
    ].copy()

    if baseline_df.empty:
        raise ValueError(
            "Original temporal RF reference is missing."
        )

    # Verify identical observations and targets.
    reference_targets = (
        baseline_df.set_index(["engine_id", "cycle"])
        ["actual_rul"]
        .sort_index()
    )

    for _, experiment_df in predictions_df.groupby(
        EXPERIMENT_COLUMNS
    ):
        experiment_targets = (
            experiment_df.set_index(["engine_id", "cycle"])
            ["actual_rul"]
            .sort_index()
        )

        if (
            not experiment_targets.index.equals(
                reference_targets.index
            )
            or not np.array_equal(
                experiment_targets.to_numpy(),
                reference_targets.to_numpy(),
            )
        ):
            raise ValueError(
                "Experiments do not share identical "
                "engine-cycle observations and targets."
            )

    predictions_df["rul_band"] = pd.cut(
        predictions_df["actual_rul"],
        bins=[-np.inf, 30, 100, np.inf],
        labels=[
            "near_failure_0_30",
            "medium_rul_31_100",
            "high_rul_above100",
        ],
        right=True,
    )

    slice_rows = []

    for (
        model_name,
        feature_set_name,
        rul_band,
    ), slice_df in predictions_df.groupby(
        [*EXPERIMENT_COLUMNS, "rul_band"],
        observed=True,
        sort=True,
    ):
        slice_rows.append(
            {
                "model": model_name,
                "feature_set": feature_set_name,
                "rul_band": rul_band,
                **summarize_errors(slice_df),
            }
        )

    slices_df = pd.DataFrame(slice_rows)

    near_failure_df = slices_df.loc[
        slices_df["rul_band"] == "near_failure_0_30"
    ].copy()

    predictions_df["absolute_error"] = (
        predictions_df["error"].abs()
    )

    engine_metrics_df = (
        predictions_df.groupby(
            [*EXPERIMENT_COLUMNS, "engine_id"],
            as_index=False,
        )
        .agg(
            row_count=("error", "size"),
            engine_mae=("absolute_error", "mean"),
            mean_signed_error=("error", "mean"),
        )
    )

    reference_engine_metrics = (
        engine_metrics_df.loc[
            (engine_metrics_df["model"] == "random_forest")
            & (engine_metrics_df["feature_set"] == "temporal"),
            ["engine_id", "engine_mae"],
        ]
        .rename(
            columns={
                "engine_mae": "reference_engine_mae"
            }
        )
    )

    engine_comparison_df = engine_metrics_df.merge(
        reference_engine_metrics,
        on="engine_id",
        how="left",
        validate="many_to_one",
    )

    engine_comparison_df["mae_improvement"] = (
        engine_comparison_df["reference_engine_mae"]
        - engine_comparison_df["engine_mae"]
    )

    summary_rows = []

    for (
        model_name,
        feature_set_name,
    ), engine_df in engine_comparison_df.groupby(
        EXPERIMENT_COLUMNS
    ):
        improvement = engine_df["mae_improvement"]

        summary_rows.append(
            {
                "model": model_name,
                "feature_set": feature_set_name,
                "engines_improved": int(
                    improvement.gt(0.000001).sum()
                ),
                "engines_worsened": int(
                    improvement.lt(-0.000001).sum()
                ),
                "engines_unchanged": int(
                    improvement.abs().le(0.000001).sum()
                ),
                "mean_engine_mae": float(
                    engine_df["engine_mae"].mean()
                ),
                "mean_engine_mae_improvement": float(
                    improvement.mean()
                ),
            }
        )

    engine_summary_df = pd.DataFrame(summary_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for table, output_file in [
        (slices_df, RUL_SLICES_FILE),
        (near_failure_df, NEAR_FAILURE_FILE),
        (engine_comparison_df, ENGINE_COMPARISON_FILE),
        (engine_summary_df, ENGINE_SUMMARY_FILE),
    ]:
        table.to_csv(output_file, index=False)

    print("\nRUL SLICE RESULTS")
    print(slices_df.round(4).to_string(index=False))

    print("\nNEAR-FAILURE COMPARISON")
    print(
        near_failure_df.sort_values("mae")
        .round(4)
        .to_string(index=False)
    )

    print("\nENGINE CHANGES VS ORIGINAL TEMPORAL RF")
    print(
        engine_summary_df.round(4).to_string(index=False)
    )

    depth12_engine_df = engine_comparison_df.loc[
        (engine_comparison_df["model"] == "random_forest_depth12")
        & (engine_comparison_df["feature_set"] == "temporal_cycle")
    ].copy()

    print("\nDEPTH12: FIVE WORST ENGINES")
    print(
        depth12_engine_df.nlargest(5, "engine_mae")
        .round(4)
        .to_string(index=False)
    )

    print("\nDEPTH12: FIVE LARGEST DETERIORATIONS")
    print(
        depth12_engine_df.sort_values("mae_improvement")
        .head(5)
        .round(4)
        .to_string(index=False)
    )

    print("\nPREVIOUSLY DIFFICULT ENGINES: DEPTH12")
    print(
        depth12_engine_df.loc[
            depth12_engine_df["engine_id"].isin(
                [84, 1, 31, 46, 45]
            )
        ]
        .sort_values("engine_mae", ascending=False)
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved:")
    for output_file in [
        RUL_SLICES_FILE,
        NEAR_FAILURE_FILE,
        ENGINE_COMPARISON_FILE,
        ENGINE_SUMMARY_FILE,
    ]:
        print(output_file)


if __name__ == "__main__":
    main()