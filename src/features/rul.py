import pandas as pd

from src.data.load_data import (
    load_train_data,
)


# =========================================================
# TRAINING RUL TARGET
# =========================================================

def add_training_rul(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add remaining useful life (RUL) to each training
    engine-cycle observation.

    RUL =
        final cycle for that engine
        -
        current cycle
    """

    required_columns = {
        "engine_id",
        "cycle",
    }

    missing_columns = (
        required_columns
        - set(train_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training data is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    train_with_rul_df = (
        train_df.copy()
    )

    final_cycle_per_engine = (
        train_with_rul_df
        .groupby(
            "engine_id"
        )["cycle"]
        .transform("max")
    )

    train_with_rul_df[
        "rul"
    ] = (
        final_cycle_per_engine
        -
        train_with_rul_df["cycle"]
    )

    train_with_rul_df[
        "rul"
    ] = (
        train_with_rul_df[
            "rul"
        ]
        .astype(int)
    )

    if (
        train_with_rul_df[
            "rul"
        ] < 0
    ).any():
        raise ValueError(
            "Negative RUL values were produced."
        )

    return train_with_rul_df


# =========================================================
# RUL VALIDATION
# =========================================================

def validate_training_rul(
    train_with_rul_df: pd.DataFrame,
) -> None:
    """
    Validate that every training engine reaches
    RUL = 0 at its final recorded cycle.
    """

    final_engine_rows = (
        train_with_rul_df
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .groupby(
            "engine_id",
            as_index=False,
        )
        .tail(1)
    )

    all_final_rul_zero = (
        final_engine_rows[
            "rul"
        ]
        .eq(0)
        .all()
    )

    if not all_final_rul_zero:
        raise ValueError(
            "RUL validation failed: the final "
            "cycle of every training engine "
            "should have RUL = 0."
        )


# =========================================================
# SCRIPT CHECK
# =========================================================

def main() -> None:

    train_df = (
        load_train_data()
    )

    train_with_rul_df = (
        add_training_rul(
            train_df
        )
    )

    validate_training_rul(
        train_with_rul_df
    )

    print(
        "\n======================================"
    )

    print(
        "TRAINING RUL CHECK"
    )

    print(
        "======================================"
    )

    print(
        "\nTraining shape after adding RUL:"
    )

    print(
        train_with_rul_df.shape
    )

    print(
        "\nMinimum RUL:"
    )

    print(
        train_with_rul_df[
            "rul"
        ].min()
    )

    print(
        "\nMaximum RUL:"
    )

    print(
        train_with_rul_df[
            "rul"
        ].max()
    )

    print(
        "\nFirst 5 rows for Engine 1:"
    )

    print(
        train_with_rul_df[
            train_with_rul_df[
                "engine_id"
            ] == 1
        ][
            [
                "engine_id",
                "cycle",
                "rul",
            ]
        ].head()
    )

    print(
        "\nLast 5 rows for Engine 1:"
    )

    print(
        train_with_rul_df[
            train_with_rul_df[
                "engine_id"
            ] == 1
        ][
            [
                "engine_id",
                "cycle",
                "rul",
            ]
        ].tail()
    )

    print(
        "\nRUL validation passed:"
    )

    print(
        "Every training engine reaches "
        "RUL = 0 at its final cycle."
    )


if __name__ == "__main__":
    main()