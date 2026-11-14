from __future__ import annotations

import json

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    OUTPUTS_DIR,
    SAVED_ARTIFACTS_DIR,
    USEFUL_SENSOR_COLUMNS,
)
from src.data.load_data import (
    load_train_data,
)
from src.features.rul import (
    add_training_rul,
    validate_training_rul,
)
from src.modeling.cycle_aware_lstm_model import (
    CycleAwareRULLSTM,
)
from src.modeling.cycle_aware_sequence_dataset import (
    CycleAwareRULSequenceDataset,
)
from src.modeling.prepare_sequence_data import (
    STRIDE,
    WINDOW_LENGTH,
    apply_sensor_scaling,
    build_sequences,
    calculate_expected_sequence_count,
    validate_sequence_alignment,
    validate_sequence_data,
)
from src.modeling.train_cycle_aware_lstm import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    INPUT_SIZE,
    LEARNING_RATE,
    NUM_LAYERS,
    SEED,
    set_random_seeds,
    train_one_epoch,
)


LOCK_FILE = (
    OUTPUTS_DIR
    / "lock_development_protocol.json"
)


FINAL_SENSOR_SCALER_PATH = (
    SAVED_ARTIFACTS_DIR
    / "fd001_final_sequence_sensor_scaler.joblib"
)


FINAL_CYCLE_SCALER_PATH = (
    SAVED_ARTIFACTS_DIR
    / "fd001_final_cycle_feature_scaler.joblib"
)


FINAL_MODEL_PATH = (
    SAVED_ARTIFACTS_DIR
    / "fd001_final_cycle_aware_lstm.pt"
)


def load_locked_protocol() -> dict:
    """
    Load and validate the development decisions that
    were frozen before final retraining.
    """

    if not LOCK_FILE.exists():
        raise FileNotFoundError(
            "Development lock artifact was not found: "
            f"{LOCK_FILE}"
        )

    with LOCK_FILE.open(
        "r",
        encoding="utf-8",
    ) as json_file:

        protocol = json.load(
            json_file
        )

    lock_metadata = protocol[
        "lock_metadata"
    ]

    if (
        lock_metadata[
            "development_phase_status"
        ]
        != "closed"
    ):
        raise ValueError(
            "Development phase is not marked closed."
        )

    if lock_metadata[
        "official_test_ground_truth_used"
    ]:
        raise ValueError(
            "Lock artifact indicates that official "
            "test ground truth was already used."
        )

    selected_model = protocol[
        "selected_model"
    ]

    if (
        selected_model["name"]
        != "cycle_aware_lstm"
    ):
        raise ValueError(
            "Locked final model is not the "
            "cycle-aware LSTM."
        )

    if (
        selected_model["input_size"]
        != INPUT_SIZE
    ):
        raise ValueError(
            "Locked input size does not match "
            "the training implementation."
        )

    if (
        selected_model["hidden_size"]
        != HIDDEN_SIZE
    ):
        raise ValueError(
            "Locked hidden size does not match "
            "the training implementation."
        )

    if (
        selected_model["num_layers"]
        != NUM_LAYERS
    ):
        raise ValueError(
            "Locked layer count does not match "
            "the training implementation."
        )

    sequence_protocol = protocol[
        "sequence_protocol"
    ]

    if (
        sequence_protocol["window_length"]
        != WINDOW_LENGTH
    ):
        raise ValueError(
            "Locked window length does not match."
        )

    if (
        sequence_protocol["stride"]
        != STRIDE
    ):
        raise ValueError(
            "Locked stride does not match."
        )

    sensor_protocol = protocol[
        "sensor_protocol"
    ]

    if (
        sensor_protocol["sensor_order"]
        != USEFUL_SENSOR_COLUMNS
    ):
        raise ValueError(
            "Locked sensor order does not match "
            "the repository sensor order."
        )

    training_configuration = protocol[
        "training_configuration"
    ]

    if (
        training_configuration["batch_size"]
        != BATCH_SIZE
    ):
        raise ValueError(
            "Locked batch size does not match."
        )

    if (
        training_configuration[
            "learning_rate"
        ]
        != LEARNING_RATE
    ):
        raise ValueError(
            "Locked learning rate does not match."
        )

    if (
        training_configuration["seed"]
        != SEED
    ):
        raise ValueError(
            "Locked seed does not match."
        )

    return protocol


def fit_final_sensor_scaler(
    training_df,
) -> StandardScaler:
    """
    Fit a NEW sensor scaler using all raw rows from
    all 100 FD001 training engines only.
    """

    scaler = StandardScaler()

    scaler.fit(
        training_df[
            USEFUL_SENSOR_COLUMNS
        ]
    )

    return scaler


def prepare_final_cycle_features(
    sequence_metadata_df,
) -> tuple[
    np.ndarray,
    StandardScaler,
]:
    """
    Fit the final cycle scaler on ending-cycle values
    from all full-training sequences.
    """

    scaler = StandardScaler()

    ending_cycles = (
        sequence_metadata_df[
            ["cycle"]
        ]
        .to_numpy()
    )

    cycle_features = (
        scaler
        .fit_transform(
            ending_cycles
        )
        .astype(
            np.float32
        )
    )

    return (
        cycle_features,
        scaler,
    )


def save_final_artifacts(
    model: CycleAwareRULLSTM,
    sensor_scaler: StandardScaler,
    cycle_scaler: StandardScaler,
    final_epochs: int,
    training_engine_count: int,
    training_sequence_count: int,
    final_training_mse: float,
) -> None:
    """
    Save the final full-training scalers and model
    separately from all development artifacts.
    """

    SAVED_ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        sensor_scaler,
        FINAL_SENSOR_SCALER_PATH,
    )

    joblib.dump(
        cycle_scaler,
        FINAL_CYCLE_SCALER_PATH,
    )

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "epoch": final_epochs,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "cycle_feature": (
            "standardized_ending_cycle"
        ),
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "sensor_columns": list(
            USEFUL_SENSOR_COLUMNS
        ),
        "training_engine_count": (
            training_engine_count
        ),
        "training_sequence_count": (
            training_sequence_count
        ),
        "final_training_mse": (
            final_training_mse
        ),
        "sensor_scaler_path": str(
            FINAL_SENSOR_SCALER_PATH
        ),
        "cycle_scaler_path": str(
            FINAL_CYCLE_SCALER_PATH
        ),
        "development_lock_file": str(
            LOCK_FILE
        ),
        "training_scope": (
            "all_100_fd001_training_engines"
        ),
    }

    torch.save(
        checkpoint,
        FINAL_MODEL_PATH,
    )


def main() -> None:

    print(
        "\n======================================"
    )

    print(
        "FD001 FINAL CYCLE-AWARE LSTM TRAINING"
    )

    print(
        "======================================"
    )

    protocol = load_locked_protocol()

    final_epochs = int(
        protocol[
            "training_configuration"
        ][
            "final_training_epochs"
        ]
    )

    print(
        "\nDevelopment lock check:"
    )

    print(
        "PASSED"
    )

    print(
        "\nLocked final training epochs:"
    )

    print(
        final_epochs
    )

    set_random_seeds(
        SEED
    )

    device = torch.device(
        "cpu"
    )

    # -------------------------------------------------
    # LOAD FULL NASA TRAINING DATA ONLY
    # -------------------------------------------------

    train_df = load_train_data()

    training_engine_count = int(
        train_df[
            "engine_id"
        ].nunique()
    )

    if training_engine_count != 100:
        raise ValueError(
            "Final training must use exactly "
            "100 FD001 training engines."
        )

    print(
        "\nFull training source shape:"
    )

    print(
        train_df.shape
    )

    print(
        "\nTraining engines used:"
    )

    print(
        training_engine_count
    )

    print(
        "\nOfficial test rows used:"
    )

    print(
        0
    )

    # -------------------------------------------------
    # ADD UNCAPPPED TRAINING RUL
    # -------------------------------------------------

    train_with_rul_df = (
        add_training_rul(
            train_df
        )
    )

    validate_training_rul(
        train_with_rul_df
    )

    print(
        "\nTraining RUL validation:"
    )

    print(
        "PASSED"
    )

    # -------------------------------------------------
    # FIT FINAL SENSOR SCALER
    # -------------------------------------------------

    sensor_scaler = (
        fit_final_sensor_scaler(
            train_with_rul_df
        )
    )

    scaled_training_df = (
        apply_sensor_scaling(
            train_with_rul_df,
            sensor_scaler,
        )
    )

    if not np.isfinite(
        scaled_training_df[
            USEFUL_SENSOR_COLUMNS
        ].to_numpy()
    ).all():
        raise ValueError(
            "Scaled training sensors contain "
            "non-finite values."
        )

    print(
        "\nFinal sensor scaler:"
    )

    print(
        "FITTED ON FULL TRAINING DATA ONLY"
    )

    print(
        "\nSensor order:"
    )

    print(
        USEFUL_SENSOR_COLUMNS
    )

    print(
        "\nScaled full-training sensor means:"
    )

    print(
        scaled_training_df[
            USEFUL_SENSOR_COLUMNS
        ]
        .mean()
        .round(4)
    )

    # -------------------------------------------------
    # BUILD FULL-TRAINING SEQUENCES
    # -------------------------------------------------

    (
        x_training_sequences,
        y_training_sequences,
        training_metadata_df,
    ) = build_sequences(
        scaled_training_df
    )

    validate_sequence_data(
        x_training_sequences,
        y_training_sequences,
        training_metadata_df,
    )

    validate_sequence_alignment(
        train_with_rul_df,
        y_training_sequences,
        training_metadata_df,
    )

    expected_sequence_count = (
        calculate_expected_sequence_count(
            train_with_rul_df
        )
    )

    if (
        len(x_training_sequences)
        != expected_sequence_count
    ):
        raise ValueError(
            "Full-training sequence count does "
            "not match the expected count."
        )

    print(
        "\nFull-training sequence shape:"
    )

    print(
        x_training_sequences.shape
    )

    print(
        "\nFull-training target shape:"
    )

    print(
        y_training_sequences.shape
    )

    print(
        "\nExpected sequence count:"
    )

    print(
        expected_sequence_count
    )

    print(
        "\nSequence validation:"
    )

    print(
        "PASSED"
    )

    # -------------------------------------------------
    # FIT FINAL CYCLE SCALER
    # -------------------------------------------------

    (
        training_cycle_features,
        cycle_scaler,
    ) = prepare_final_cycle_features(
        training_metadata_df
    )

    if not np.isfinite(
        training_cycle_features
    ).all():
        raise ValueError(
            "Cycle features contain non-finite values."
        )

    print(
        "\nFinal cycle scaler:"
    )

    print(
        "FITTED ON FULL-TRAINING SEQUENCE "
        "ENDING CYCLES ONLY"
    )

    print(
        "\nTraining cycle feature shape:"
    )

    print(
        training_cycle_features.shape
    )

    print(
        "\nFinal cycle mean:"
    )

    print(
        f"{cycle_scaler.mean_[0]:.4f}"
    )

    print(
        "\nFinal cycle scale:"
    )

    print(
        f"{cycle_scaler.scale_[0]:.4f}"
    )

    # -------------------------------------------------
    # DATASET + DATALOADER
    # -------------------------------------------------

    training_dataset = (
        CycleAwareRULSequenceDataset(
            x_training_sequences,
            training_cycle_features,
            y_training_sequences,
        )
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # -------------------------------------------------
    # FINAL MODEL
    # -------------------------------------------------

    model = CycleAwareRULLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )

    model = model.to(
        device
    )

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(
        "\nFinal model:"
    )

    print(
        model
    )

    print(
        "\nFinal training configuration:"
    )

    print(
        f"Input size: {INPUT_SIZE}"
    )

    print(
        f"Hidden size: {HIDDEN_SIZE}"
    )

    print(
        f"Layers: {NUM_LAYERS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Epochs: {final_epochs}"
    )

    print(
        "\n======================================"
    )

    print(
        "FINAL TRAINING START"
    )

    print(
        "======================================"
    )

    final_training_mse = float(
        "nan"
    )

    for epoch in range(
        1,
        final_epochs + 1,
    ):

        final_training_mse = (
            train_one_epoch(
                model,
                training_loader,
                loss_function,
                optimizer,
                device,
            )
        )

        print(
            f"\nEpoch {epoch:02d}/{final_epochs}"
        )

        print(
            "Training MSE: "
            f"{final_training_mse:.4f}"
        )

    # -------------------------------------------------
    # SAVE FINAL ARTIFACTS
    # -------------------------------------------------

    save_final_artifacts(
        model=model,
        sensor_scaler=sensor_scaler,
        cycle_scaler=cycle_scaler,
        final_epochs=final_epochs,
        training_engine_count=(
            training_engine_count
        ),
        training_sequence_count=len(
            x_training_sequences
        ),
        final_training_mse=(
            final_training_mse
        ),
    )

    print(
        "\n======================================"
    )

    print(
        "FINAL TRAINING COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        "\nEpochs completed:"
    )

    print(
        final_epochs
    )

    print(
        "\nFinal training MSE:"
    )

    print(
        f"{final_training_mse:.4f}"
    )

    print(
        "\nFinal sensor scaler saved to:"
    )

    print(
        FINAL_SENSOR_SCALER_PATH
    )

    print(
        "\nFinal cycle scaler saved to:"
    )

    print(
        FINAL_CYCLE_SCALER_PATH
    )

    print(
        "\nFinal model saved to:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print(
        "\nOfficial test data used:"
    )

    print(
        "NO"
    )


if __name__ == "__main__":
    main()