from pathlib import Path
import pandas as pd

from src.config import (
    CMAPSS_COLUMNS,
    TEST_FILE,
    TEST_RUL_FILE,
    TRAIN_FILE,
)

 

def _validate_file_exists(
    file_path: Path,
) -> None:
    """
    Make sure a required dataset file exists.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found: "
            f"{file_path}"
        )



def _load_cmapss_feature_file(
    file_path: Path,
) -> pd.DataFrame:
    """
    Load a C-MAPSS train/test feature file and assign
    the canonical project column names.
    """

    _validate_file_exists(file_path)

    dataframe = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
    )

    expected_column_count = len(
        CMAPSS_COLUMNS
    )

    actual_column_count = (
        dataframe.shape[1]
    )

    if (
        actual_column_count
        != expected_column_count
    ):
        raise ValueError(
            "Unexpected number of columns in "
            f"{file_path.name}. "
            f"Expected {expected_column_count}, "
            f"found {actual_column_count}."
        )

    dataframe.columns = CMAPSS_COLUMNS

    dataframe["engine_id"] = (
        dataframe["engine_id"]
        .astype(int)
    )

    dataframe["cycle"] = (
        dataframe["cycle"]
        .astype(int)
    )

    return dataframe


# =========================================================
# PUBLIC TRAIN LOADER
# =========================================================

def load_train_data() -> pd.DataFrame:
    """
    Load the FD001 training trajectories.
    """

    return _load_cmapss_feature_file(
        TRAIN_FILE
    )


# =========================================================
# PUBLIC TEST LOADER
# =========================================================

def load_test_data() -> pd.DataFrame:
   

    return _load_cmapss_feature_file(
        TEST_FILE
    )


# =========================================================
# PUBLIC TEST-RUL LOADER
# =========================================================

def load_test_rul() -> pd.DataFrame:
    

    _validate_file_exists(
        TEST_RUL_FILE
    )

    test_rul_df = pd.read_csv(
        TEST_RUL_FILE,
        sep=r"\s+",
        header=None,
    )

    if test_rul_df.shape[1] != 1:
        raise ValueError(
            "RUL file should contain exactly "
            "one value per row."
        )

    test_rul_df.columns = [
        "rul"
    ]

    test_rul_df.insert(
        0,
        "engine_id",
        range(
            1,
            len(test_rul_df) + 1,
        ),
    )

    test_rul_df["engine_id"] = (
        test_rul_df["engine_id"]
        .astype(int)
    )

    test_rul_df["rul"] = (
        test_rul_df["rul"]
        .astype(int)
    )

    return test_rul_df


# =========================================================
# LOAD COMPLETE FD001 DATA
# =========================================================

def load_fd001_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load training data, test data and official
    test RUL values together.
    """

    train_df = load_train_data()

    test_df = load_test_data()

    test_rul_df = load_test_rul()

    number_of_test_engines = (
        test_df["engine_id"]
        .nunique()
    )

    if (
        len(test_rul_df)
        != number_of_test_engines
    ):
        raise ValueError(
            "Number of official RUL values does "
            "not match the number of test engines."
        )

    return (
        train_df,
        test_df,
        test_rul_df,
    )


# =========================================================
# SCRIPT CHECK
# =========================================================

def main() -> None:

    (
        train_df,
        test_df,
        test_rul_df,
    ) = load_fd001_data()

    print(
        "\n======================================"
    )

    print(
        "C-MAPSS FD001 DATA CHECK"
    )

    print(
        "======================================"
    )

    print(
        "\nTraining shape:"
    )

    print(
        train_df.shape
    )

    print(
        "\nTest shape:"
    )

    print(
        test_df.shape
    )

    print(
        "\nOfficial test RUL shape:"
    )

    print(
        test_rul_df.shape
    )

    print(
        "\nNumber of training engines:"
    )

    print(
        train_df[
            "engine_id"
        ].nunique()
    )

    print(
        "\nNumber of test engines:"
    )

    print(
        test_df[
            "engine_id"
        ].nunique()
    )

    print(
        "\nTraining columns:"
    )

    print(
        train_df.columns.tolist()
    )

    print(
        "\nFirst 5 training rows:"
    )

    print(
        train_df.head()
    )

    print(
        "\nMissing values in training data:"
    )

    print(
        train_df.isna().sum().sum()
    )

    print(
        "\nFirst 5 official test RUL values:"
    )

    print(
        test_rul_df.head()
    )


if __name__ == "__main__":
    main()