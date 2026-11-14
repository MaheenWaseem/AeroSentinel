from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

 

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = (
    DATA_DIR
    / "processed"
)


OUTPUTS_DIR = (
    PROJECT_ROOT
    / "outputs"
)

FIGURES_DIR = (
    OUTPUTS_DIR
    / "figures"
)

METRICS_DIR = (
    OUTPUTS_DIR
    / "metrics"
)

PREDICTIONS_DIR = (
    OUTPUTS_DIR
    / "predictions"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

SAVED_ARTIFACTS_DIR = (
    MODELS_DIR
    / "saved_artifacts"
)


REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)


RANDOM_STATE = 42


VALIDATION_SIZE = 0.20

CMAPSS_SUBSET = "FD001"


CMAPSS_COLUMNS = [
    "engine_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[
        f"sensor_{sensor_number}"
        for sensor_number in range(
            1,
            22,
        )
    ],
]



TRAIN_FILE = (
    RAW_DATA_DIR
    / f"train_{CMAPSS_SUBSET}.txt"
)

TEST_FILE = (
    RAW_DATA_DIR
    / f"test_{CMAPSS_SUBSET}.txt"
)

TEST_RUL_FILE = (
    RAW_DATA_DIR
    / f"RUL_{CMAPSS_SUBSET}.txt"
)


DEVELOPMENT_SPLIT_FILE = (
    PROCESSED_DATA_DIR
    / "fd001_engine_split.csv"
)


USEFUL_SENSOR_COLUMNS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]