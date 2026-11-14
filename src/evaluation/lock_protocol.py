from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.config import (
    CMAPSS_SUBSET,
    OUTPUTS_DIR,
    RANDOM_STATE,
    USEFUL_SENSOR_COLUMNS,
)
from src.modeling.prepare_sequence_data import (
    STRIDE,
    WINDOW_LENGTH,
)
from src.modeling.train_cycle_aware_lstm import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_LAYERS,
    SEED,
)


LOCK_FILE = (
    OUTPUTS_DIR
    / "lock_development_protocol.json"
)


FINAL_SELECTED_EPOCH = 18

EXPECTED_DEVELOPMENT_MAE = 17.4595
EXPECTED_DEVELOPMENT_RMSE = 25.0458

FINAL_MODEL_NAME = "cycle_aware_lstm"


def validate_locked_repository_configuration() -> None:

    if CMAPSS_SUBSET != "FD001":
        raise ValueError(
            "lock expects CMAPSS_SUBSET='FD001'."
        )

    if len(USEFUL_SENSOR_COLUMNS) != 14:
        raise ValueError(
            "Locked sensor set must contain exactly "
            "14 sensors."
        )

    if INPUT_SIZE != len(
        USEFUL_SENSOR_COLUMNS
    ):
        raise ValueError(
            "Model input size does not match the locked sensor count."
        )

    if WINDOW_LENGTH != 30:
        raise ValueError(
            "Locked sequence length must be 30."
        )

    if STRIDE != 1:
        raise ValueError(
            "Locked sequence stride must be 1."
        )

    if INPUT_SIZE != 14:
        raise ValueError(
            "Locked model input size must be 14."
        )

    if HIDDEN_SIZE != 64:
        raise ValueError(
            "Locked hidden size must be 64."
        )

    if NUM_LAYERS != 1:
        raise ValueError(
            "Locked LSTM layer count must be 1."
        )

    if BATCH_SIZE != 64:
        raise ValueError(
            "Locked batch size must be 64."
        )

    if LEARNING_RATE != 0.001:
        raise ValueError(
            "Locked learning rate must be 0.001."
        )

    if SEED != 42:
        raise ValueError(
            "Locked training seed must be 42."
        )

    if RANDOM_STATE != 42:
        raise ValueError(
            "Random state must remain 42."
        )


def validate_development_checkpoint() -> dict:

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Development cycle-aware checkpoint was "
            f"not found: {BEST_MODEL_PATH}"
        )

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location="cpu",
    )

    required_keys = {
        "model_state_dict",
        "input_size",
        "hidden_size",
        "num_layers",
        "validation_mae",
        "epoch",
        "seed",
        "cycle_feature",
    }

    missing_keys = (
        required_keys
        - set(checkpoint.keys())
    )

    if missing_keys:
        raise ValueError(
            "Development checkpoint is missing "
            f"required keys: {sorted(missing_keys)}"
        )

    if checkpoint["input_size"] != INPUT_SIZE:
        raise ValueError(
            "Checkpoint input size does not match "
            "the locked model configuration."
        )

    if checkpoint["hidden_size"] != HIDDEN_SIZE:
        raise ValueError(
            "Checkpoint hidden size does not match "
            "the locked model configuration."
        )

    if checkpoint["num_layers"] != NUM_LAYERS:
        raise ValueError(
            "Checkpoint layer count does not match "
            "the locked model configuration."
        )

    if checkpoint["seed"] != SEED:
        raise ValueError(
            "Checkpoint seed does not match the"
            "locked training seed."
        )

    if checkpoint["epoch"] != FINAL_SELECTED_EPOCH:
        raise ValueError(
            "Preserved development checkpoint is not "
            f"from locked epoch {FINAL_SELECTED_EPOCH}. "
            f"Found epoch {checkpoint['epoch']}."
        )

    return checkpoint


def build_locked_protocol(
    checkpoint: dict,
) -> dict:

    protocol = {
        "lock_metadata": {
            "artifact_name": (
                "locked_development_protocol"
            ),
            "locked_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "development_phase_status": "closed",
            "official_test_ground_truth_used": False,
            "official_test_predictions_generated": False,
        },

        "dataset": {
            "name": "NASA C-MAPSS",
            "subset": CMAPSS_SUBSET,
            "training_engine_count": 100,
            "official_test_engine_count": 100,
            "target_definition": (
                "uncapped_rul_equals_final_training_"
                "cycle_minus_current_cycle"
            ),
            "rul_clipping": False,
        },

        "development_protocol": {
            "development_train_engines": 80,
            "development_validation_engines": 20,
            "engine_overlap": False,
            "official_test_used_for_development": False,
        },

        "sensor_protocol": {
            "sensor_count": len(
                USEFUL_SENSOR_COLUMNS
            ),
            "sensor_order": list(
                USEFUL_SENSOR_COLUMNS
            ),
        },

        "sequence_protocol": {
            "window_length": WINDOW_LENGTH,
            "stride": STRIDE,
            "padding": False,
            "cross_engine_windows": False,
            "future_information": False,
            "sequence_target": (
                "rul_at_final_cycle_of_sequence"
            ),
        },

        "selected_model": {
            "name": FINAL_MODEL_NAME,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "batch_first": True,
            "explicit_cycle_feature": True,
            "cycle_feature_definition": (
                "standardized_ending_cycle"
            ),
            "regression_head": (
                "Linear(hidden_size_plus_1, 1)"
            ),
        },

        "training_configuration": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "loss": "MSELoss",
            "seed": SEED,
            "development_max_epochs": NUM_EPOCHS,
            "selected_development_epoch": (
                FINAL_SELECTED_EPOCH
            ),
            "final_training_epochs": (
                FINAL_SELECTED_EPOCH
            ),
            "official_test_early_stopping": False,
        },

        "preprocessing_protocol": {
            "sensor_scaler": {
                "type": "StandardScaler",
                "fit_data_for_final_model": (
                    "all raw rows from all 100 NASA "
                    "training engines, restricted to "
                    "the locked 14 sensor columns"
                ),
                "fit_before_sequence_construction": True,
                "fit_on_official_test": False,
                "train_test_normalized_independently": False,
            },

            "cycle_scaler": {
                "type": "StandardScaler",
                "feature": "sequence_ending_cycle",
                "fit_data_for_final_model": (
                    "ending-cycle values from the "
                    "full-training sequence metadata"
                ),
                "fit_on_official_test": False,
                "train_test_normalized_independently": False,
            },
        },

        "development_model_selection_evidence": {
            "selected_model": FINAL_MODEL_NAME,
            "best_epoch": int(
                checkpoint["epoch"]
            ),
            "validation_mae": float(
                checkpoint["validation_mae"]
            ),
            "validation_rmse": (
                EXPECTED_DEVELOPMENT_RMSE
            ),
            "reported_development_mae": (
                EXPECTED_DEVELOPMENT_MAE
            ),
            "mean_engine_mae": 16.8561,
            "median_engine_mae": 16.3902,
            "cycle_aware_beats_sensor_only_engines": 11,
            "cycle_aware_beats_random_forest_engines": 13,
        },

        "official_test_protocol": {
            "primary_test_model": FINAL_MODEL_NAME,
            "one_prediction_per_engine": True,
            "use_final_30_observed_cycles": True,
            "test_padding_allowed": False,
            "change_window_length_for_test": False,
            "fit_scalers_on_test": False,
            "load_ground_truth_before_predictions": False,
            "prediction_count_expected": 100,
        },

        "official_test_metrics": {
            "mandatory": [
                "mae",
                "rmse",
            ],
            "additional": [
                "mean_signed_error",
                "median_absolute_error",
                "p95_absolute_error",
            ],
            "nasa_asymmetric_score": (
                "optional_only_if_formula_and_"
                "sign_convention_are_verified"
            ),
        },

        "test_integrity_declaration": {
            "model_selection_reopens_after_test": False,
            "sensor_selection_changes_after_test": False,
            "architecture_changes_after_test": False,
            "hyperparameter_tuning_after_test": False,
            "epoch_selection_after_test": False,
            "preprocessing_changes_after_test": False,
            "official_test_used_as_validation": False,
            "statement": (
                "No locked development decision will "
                "change in response to official-test "
                "performance."
            ),
        },
    }

    return protocol


def save_locked_protocol(
    protocol: dict,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:

        json.dump(
            protocol,
            json_file,
            indent=4,
        )


def main() -> None:

    print(
        "\n======================================"
    )

    print(
        "DEVELOPMENT PROTOCOL LOCK"
    )

    print(
        "======================================"
    )

    validate_locked_repository_configuration()

    print(
        "\nRepository configuration check:"
    )

    print(
        "PASSED"
    )

    checkpoint = (
        validate_development_checkpoint()
    )

    print(
        "\nDevelopment checkpoint check:"
    )

    print(
        "PASSED"
    )

    print(
        "\nLocked development checkpoint epoch:"
    )

    print(
        checkpoint["epoch"]
    )

    print(
        "\nLocked development validation MAE:"
    )

    print(
        f"{checkpoint['validation_mae']:.4f}"
    )

    protocol = build_locked_protocol(
        checkpoint
    )

    save_locked_protocol(
        protocol,
        LOCK_FILE,
    )

    print(
        "\nLocked dataset:"
    )

    print(
        CMAPSS_SUBSET
    )

    print(
        "\nLocked sensors:"
    )

    print(
        USEFUL_SENSOR_COLUMNS
    )

    print(
        "\nLocked sequence length:"
    )

    print(
        WINDOW_LENGTH
    )

    print(
        "\nLocked final model:"
    )

    print(
        FINAL_MODEL_NAME
    )

    print(
        "\nLocked final training epochs:"
    )

    print(
        FINAL_SELECTED_EPOCH
    )

    print(
        "\nLock artifact saved to:"
    )

    print(
        LOCK_FILE
    )

    print(
        "\n======================================"
    )

    print(
        "DEVELOPMENT PROTOCOL LOCKED"
    )

    print(
        "======================================"
    )

if __name__ == "__main__":
    main()