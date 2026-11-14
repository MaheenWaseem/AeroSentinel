from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    OUTPUTS_DIR,
)
from src.data.load_data import (
    load_test_rul,
)
from src.evaluation.rul_metrics import (
    calculate_regression_metrics,
)


UNSCORED_PREDICTIONS_PATH = (
    OUTPUTS_DIR
    / "fd001_official_test_predictions_unscored.csv"
)

SCORED_PREDICTIONS_PATH = (
    OUTPUTS_DIR
    / "fd001_official_test_predictions.csv"
)

OFFICIAL_TEST_METRICS_PATH = (
    OUTPUTS_DIR
    / "fd001_official_test_metrics.csv"
)


DEVELOPMENT_MAE = 17.4595
DEVELOPMENT_RMSE = 25.0458


def load_frozen_predictions() -> pd.DataFrame:

    if not UNSCORED_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Frozen official-test predictions "
            f"were not found: {UNSCORED_PREDICTIONS_PATH}"
        )

    prediction_df = pd.read_csv(
        UNSCORED_PREDICTIONS_PATH
    )

    required_columns = [
        "engine_id",
        "last_observed_cycle",
        "predicted_rul",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in prediction_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Frozen prediction file is missing "
            f"columns: {missing_columns}"
        )

    if len(prediction_df) != 100:
        raise ValueError(
            "Frozen prediction file must contain "
            "exactly 100 rows."
        )

    if (
        prediction_df[
            "engine_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Frozen prediction file contains "
            "duplicate engine IDs."
        )

    expected_engine_ids = list(
        range(
            1,
            101,
        )
    )

    actual_engine_ids = sorted(
        prediction_df[
            "engine_id"
        ]
        .astype(int)
        .tolist()
    )

    if actual_engine_ids != expected_engine_ids:
        raise ValueError(
            "Frozen prediction engine IDs are not "
            "exactly 1 through 100."
        )

    if not np.isfinite(
        prediction_df[
            "predicted_rul"
        ].to_numpy()
    ).all():
        raise ValueError(
            "Frozen predictions contain "
            "non-finite values."
        )

    return prediction_df


def validate_official_rul(
    test_rul_df: pd.DataFrame,
) -> None:

    if len(test_rul_df) != 100:
        raise ValueError(
            "Official RUL file must contain "
            "exactly 100 rows."
        )

    if (
        test_rul_df[
            "engine_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Official RUL file contains "
            "duplicate engine IDs."
        )

    expected_engine_ids = list(
        range(
            1,
            101,
        )
    )

    actual_engine_ids = sorted(
        test_rul_df[
            "engine_id"
        ]
        .astype(int)
        .tolist()
    )

    if actual_engine_ids != expected_engine_ids:
        raise ValueError(
            "Official RUL engine IDs are not "
            "exactly 1 through 100."
        )

    if not np.isfinite(
        test_rul_df[
            "rul"
        ].to_numpy()
    ).all():
        raise ValueError(
            "Official RUL values contain "
            "non-finite values."
        )


def build_scored_table(
    frozen_predictions_df: pd.DataFrame,
    test_rul_df: pd.DataFrame,
) -> pd.DataFrame:

    actual_rul_df = (
        test_rul_df
        .rename(
            columns={
                "rul": "actual_rul"
            }
        )
    )

    scored_df = (
        frozen_predictions_df
        .merge(
            actual_rul_df,
            on="engine_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            "engine_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(scored_df) != 100:
        raise ValueError(
            "Scored official-test table must "
            "contain exactly 100 rows."
        )

    scored_df[
        "error"
    ] = (
        scored_df[
            "predicted_rul"
        ]
        - scored_df[
            "actual_rul"
        ]
    )

    scored_df[
        "absolute_error"
    ] = (
        scored_df[
            "error"
        ]
        .abs()
    )

    scored_df[
        "squared_error"
    ] = (
        scored_df[
            "error"
        ]
        ** 2
    )

    scored_df = scored_df[
        [
            "engine_id",
            "last_observed_cycle",
            "actual_rul",
            "predicted_rul",
            "error",
            "absolute_error",
            "squared_error",
        ]
    ]

    return scored_df


def build_metrics_table(
    scored_df: pd.DataFrame,
) -> pd.DataFrame:

    metrics = calculate_regression_metrics(
        scored_df[
            "actual_rul"
        ].to_numpy(),
        scored_df[
            "predicted_rul"
        ].to_numpy(),
    )

    mean_signed_error = float(
        scored_df[
            "error"
        ].mean()
    )

    median_absolute_error = float(
        scored_df[
            "absolute_error"
        ].median()
    )

    p95_absolute_error = float(
        scored_df[
            "absolute_error"
        ].quantile(
            0.95
        )
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": "cycle_aware_lstm",
                "engine_count": 100,
                "mae": float(
                    metrics[
                        "mae"
                    ]
                ),
                "rmse": float(
                    metrics[
                        "rmse"
                    ]
                ),
                "mean_signed_error": (
                    mean_signed_error
                ),
                "median_absolute_error": (
                    median_absolute_error
                ),
                "p95_absolute_error": (
                    p95_absolute_error
                ),
                "development_mae": (
                    DEVELOPMENT_MAE
                ),
                "development_rmse": (
                    DEVELOPMENT_RMSE
                ),
                "mae_gap_test_minus_development": (
                    float(
                        metrics[
                            "mae"
                        ]
                    )
                    - DEVELOPMENT_MAE
                ),
                "rmse_gap_test_minus_development": (
                    float(
                        metrics[
                            "rmse"
                        ]
                    )
                    - DEVELOPMENT_RMSE
                ),
            }
        ]
    )

    return metrics_df


def main() -> None:

    print(
        "\n======================================"
    )

    print(
        "FD001 OFFICIAL TEST SCORING"
    )

    print(
        "======================================"
    )

    frozen_predictions_df = (
        load_frozen_predictions()
    )

    print(
        "\nFrozen prediction file:"
    )

    print(
        "PASSED"
    )

    print(
        "\nFrozen predictions:"
    )

    print(
        len(
            frozen_predictions_df
        )
    )

    print(
        "\n======================================"
    )

    print(
        "LOADING OFFICIAL TEST RUL"
    )

    print(
        "======================================"
    )

    test_rul_df = load_test_rul()

    validate_official_rul(
        test_rul_df
    )

    print(
        "\nOfficial RUL values loaded:"
    )

    print(
        len(
            test_rul_df
        )
    )

    print(
        "\nOfficial RUL validation:"
    )

    print(
        "PASSED"
    )

    scored_df = build_scored_table(
        frozen_predictions_df,
        test_rul_df,
    )

    print(
        "\nEngine/RUL alignment:"
    )

    print(
        "PASSED"
    )

    metrics_df = build_metrics_table(
        scored_df
    )

    SCORED_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored_df.to_csv(
        SCORED_PREDICTIONS_PATH,
        index=False,
    )

    metrics_df.to_csv(
        OFFICIAL_TEST_METRICS_PATH,
        index=False,
    )

    result = metrics_df.iloc[0]

    print(
        "\n======================================"
    )

    print(
        "OFFICIAL TEST RESULTS"
    )

    print(
        "======================================"
    )

    print(
        "\nMAE:"
    )

    print(
        f"{result['mae']:.4f}"
    )

    print(
        "\nRMSE:"
    )

    print(
        f"{result['rmse']:.4f}"
    )

    print(
        "\nMean signed error:"
    )

    print(
        f"{result['mean_signed_error']:.4f}"
    )

    print(
        "\nMedian absolute error:"
    )

    print(
        f"{result['median_absolute_error']:.4f}"
    )

    print(
        "\nP95 absolute error:"
    )

    print(
        f"{result['p95_absolute_error']:.4f}"
    )

    print(
        "\nDevelopment MAE:"
    )

    print(
        f"{DEVELOPMENT_MAE:.4f}"
    )

    print(
        "\nOfficial test minus development MAE:"
    )

    print(
        f"{result['mae_gap_test_minus_development']:.4f}"
    )

    print(
        "\nDevelopment RMSE:"
    )

    print(
        f"{DEVELOPMENT_RMSE:.4f}"
    )

    print(
        "\nOfficial test minus development RMSE:"
    )

    print(
        f"{result['rmse_gap_test_minus_development']:.4f}"
    )

    print(
        "\nFirst 5 scored predictions:"
    )

    print(
        scored_df.head()
    )

    print(
        "\nScored predictions saved to:"
    )

    print(
        SCORED_PREDICTIONS_PATH
    )

    print(
        "\nOfficial test metrics saved to:"
    )

    print(
        OFFICIAL_TEST_METRICS_PATH
    )

    print(
        "\n======================================"
    )

    print(
        "OFFICIAL TEST CONSUMED"
    )

    print(
        "======================================"
    )


if __name__ == "__main__":
    main()