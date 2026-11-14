import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
)

from src.config import (
    DEVELOPMENT_SPLIT_FILE,
    RANDOM_STATE,
    VALIDATION_SIZE,
)

from src.data.load_data import (
    load_train_data,
)

from src.features.rul import (
    add_training_rul,
)


# =========================================================
# CREATE ENGINE-LEVEL SPLIT TABLE
# =========================================================

def create_engine_split_table(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a fixed development-training /
    development-validation split at ENGINE level.

    Every engine belongs entirely to one split.
    """

    if "engine_id" not in train_df.columns:
        raise ValueError(
            "Training data must contain "
            "'engine_id'."
        )

    engine_ids_df = pd.DataFrame(
        {
            "engine_id": sorted(
                train_df[
                    "engine_id"
                ].unique()
            )
        }
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    (
        train_engine_indices,
        validation_engine_indices,
    ) = next(
        splitter.split(
            engine_ids_df,
            groups=engine_ids_df[
                "engine_id"
            ],
        )
    )

    development_train_engines = (
        engine_ids_df.iloc[
            train_engine_indices
        ]["engine_id"]
        .tolist()
    )

    development_validation_engines = (
        engine_ids_df.iloc[
            validation_engine_indices
        ]["engine_id"]
        .tolist()
    )

    train_split_df = pd.DataFrame(
        {
            "engine_id":
                development_train_engines,

            "split":
                "train",
        }
    )

    validation_split_df = pd.DataFrame(
        {
            "engine_id":
                development_validation_engines,

            "split":
                "validation",
        }
    )

    split_table_df = pd.concat(
        [
            train_split_df,
            validation_split_df,
        ],
        ignore_index=True,
    )

    split_table_df = (
        split_table_df
        .sort_values(
            "engine_id"
        )
        .reset_index(
            drop=True
        )
    )

    validate_engine_split_table(
        split_table_df,
        train_df,
    )

    return split_table_df


# =========================================================
# VALIDATE SPLIT TABLE
# =========================================================

def validate_engine_split_table(
    split_table_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> None:
    """
    Verify that the split contains every engine exactly
    once and that train/validation groups do not overlap.
    """

    required_columns = {
        "engine_id",
        "split",
    }

    missing_columns = (
        required_columns
        - set(split_table_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Split table is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    if (
        split_table_df[
            "engine_id"
        ]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "An engine appears more than once "
            "in the split table."
        )

    allowed_split_names = {
        "train",
        "validation",
    }

    actual_split_names = set(
        split_table_df[
            "split"
        ].unique()
    )

    if (
        actual_split_names
        != allowed_split_names
    ):
        raise ValueError(
            "Split table must contain exactly "
            "'train' and 'validation'."
        )

    dataset_engine_ids = set(
        train_df[
            "engine_id"
        ].unique()
    )

    split_engine_ids = set(
        split_table_df[
            "engine_id"
        ].unique()
    )

    if (
        dataset_engine_ids
        != split_engine_ids
    ):
        raise ValueError(
            "Split table engines do not match "
            "training dataset engines."
        )

    train_engine_ids = set(
        split_table_df.loc[
            split_table_df[
                "split"
            ] == "train",
            "engine_id",
        ]
    )

    validation_engine_ids = set(
        split_table_df.loc[
            split_table_df[
                "split"
            ] == "validation",
            "engine_id",
        ]
    )

    overlap = (
        train_engine_ids
        &
        validation_engine_ids
    )

    if overlap:
        raise ValueError(
            "Engine leakage detected. "
            "These engines occur in both "
            f"splits: {sorted(overlap)}"
        )


# =========================================================
# SAVE SPLIT TABLE
# =========================================================

def save_engine_split_table(
    split_table_df: pd.DataFrame,
) -> None:
    """
    Save the fixed engine assignments so future models
    use the exact same development split.
    """

    DEVELOPMENT_SPLIT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_table_df.to_csv(
        DEVELOPMENT_SPLIT_FILE,
        index=False,
    )


# =========================================================
# LOAD SAVED SPLIT TABLE
# =========================================================

def load_engine_split_table() -> pd.DataFrame:
    """
    Load the previously saved engine-level split.
    """

    if not DEVELOPMENT_SPLIT_FILE.exists():
        raise FileNotFoundError(
            "Development split file does not "
            "exist yet: "
            f"{DEVELOPMENT_SPLIT_FILE}"
        )

    split_table_df = pd.read_csv(
        DEVELOPMENT_SPLIT_FILE
    )

    split_table_df[
        "engine_id"
    ] = (
        split_table_df[
            "engine_id"
        ]
        .astype(int)
    )

    return split_table_df


# =========================================================
# APPLY SAVED SPLIT TO TRAINING ROWS
# =========================================================

def apply_engine_split(
    train_df: pd.DataFrame,
    split_table_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split all cycle rows according to their engine-level
    assignment.
    """

    validate_engine_split_table(
        split_table_df,
        train_df,
    )

    development_train_engine_ids = set(
        split_table_df.loc[
            split_table_df[
                "split"
            ] == "train",
            "engine_id",
        ]
    )

    development_validation_engine_ids = set(
        split_table_df.loc[
            split_table_df[
                "split"
            ] == "validation",
            "engine_id",
        ]
    )

    development_train_df = (
        train_df[
            train_df[
                "engine_id"
            ].isin(
                development_train_engine_ids
            )
        ]
        .copy()
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    development_validation_df = (
        train_df[
            train_df[
                "engine_id"
            ].isin(
                development_validation_engine_ids
            )
        ]
        .copy()
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        development_train_df,
        development_validation_df,
    )


# =========================================================
# CREATE OR LOAD FIXED DEVELOPMENT SPLIT
# =========================================================

def get_development_split(
    train_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Reuse an existing engine split when available.
    Otherwise create and save it.
    """

    if DEVELOPMENT_SPLIT_FILE.exists():

        split_table_df = (
            load_engine_split_table()
        )

    else:

        split_table_df = (
            create_engine_split_table(
                train_df
            )
        )

        save_engine_split_table(
            split_table_df
        )

    validate_engine_split_table(
        split_table_df,
        train_df,
    )

    (
        development_train_df,
        development_validation_df,
    ) = apply_engine_split(
        train_df,
        split_table_df,
    )

    return (
        development_train_df,
        development_validation_df,
        split_table_df,
    )


# =========================================================
# SCRIPT CHECK
# =========================================================

def main() -> None:

    train_df = load_train_data()

    train_with_rul_df = (
        add_training_rul(
            train_df
        )
    )

    (
        development_train_df,
        development_validation_df,
        split_table_df,
    ) = get_development_split(
        train_with_rul_df
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

    engine_overlap = (
        train_engine_ids
        &
        validation_engine_ids
    )

    print(
        "\n======================================"
    )

    print(
        "FD001 ENGINE-AWARE SPLIT CHECK"
    )

    print(
        "======================================"
    )

    print(
        "\nTotal engines:"
    )

    print(
        train_with_rul_df[
            "engine_id"
        ].nunique()
    )

    print(
        "\nDevelopment-training engines:"
    )

    print(
        len(train_engine_ids)
    )

    print(
        "\nDevelopment-validation engines:"
    )

    print(
        len(validation_engine_ids)
    )

    print(
        "\nDevelopment-training rows:"
    )

    print(
        development_train_df.shape
    )

    print(
        "\nDevelopment-validation rows:"
    )

    print(
        development_validation_df.shape
    )

    print(
        "\nEngine overlap:"
    )

    print(
        engine_overlap
    )

    print(
        "\nFirst 10 engine assignments:"
    )

    print(
        split_table_df.head(10)
    )

    print(
        "\nSplit saved to:"
    )

    print(
        DEVELOPMENT_SPLIT_FILE
    )

    print(
        "\nValidation passed:"
    )

    print(
        "No engine appears in both "
        "development splits."
    )


if __name__ == "__main__":
    main()