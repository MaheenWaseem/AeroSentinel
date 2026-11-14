from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.evaluation.rul_metrics import (
    calculate_regression_metrics,
)
from src.modeling.lstm_model import (
    RULLSTM,
)
from src.modeling.sequence_dataset import (
    build_sequence_dataloaders,
)


SEED = 42

BATCH_SIZE = 64
INPUT_SIZE = 14
HIDDEN_SIZE = 64
NUM_LAYERS = 1

LEARNING_RATE = 0.001
NUM_EPOCHS = 20


TRAIN_SEQUENCES_PATH = Path(
    "data/processed/fd001_train_sequences.npy"
)

TRAIN_TARGETS_PATH = Path(
    "data/processed/fd001_train_sequence_targets.npy"
)

VALIDATION_SEQUENCES_PATH = Path(
    "data/processed/fd001_validation_sequences.npy"
)

VALIDATION_TARGETS_PATH = Path(
    "data/processed/fd001_validation_sequence_targets.npy"
)

BEST_MODEL_PATH = Path(
    "models/saved_artifacts/fd001_lstm_best.pt"
)

TRAINING_HISTORY_PATH = Path(
    "outputs/lstm_training_history.csv"
)

MSE_LEARNING_CURVE_PATH = Path(
    "outputs/lstm_mse_learning_curve.png"
)

VALIDATION_METRICS_CURVE_PATH = Path(
    "outputs/lstm_validation_metrics_curve.png"
)


def set_random_seeds(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )


def calculate_average_loss(
    total_loss: float,
    sample_count: int,
) -> float:

    if sample_count == 0:
        raise ValueError(
            "Cannot calculate average loss "
            "with zero samples."
        )

    return (
        total_loss
        / sample_count
    )


def train_one_epoch(
    model: RULLSTM,
    train_loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:

    model.train()

    total_loss = 0.0
    sample_count = 0

    for inputs, targets in train_loader:

        inputs = inputs.to(
            device
        )

        targets = targets.to(
            device
        )

        optimizer.zero_grad()

        predictions = model(
            inputs
        )

        loss = loss_function(
            predictions,
            targets,
        )

        loss.backward()

        optimizer.step()

        batch_size = inputs.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        sample_count += batch_size

    average_loss = calculate_average_loss(
        total_loss,
        sample_count,
    )

    return average_loss


def validate_one_epoch(
    model: RULLSTM,
    validation_loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
]:

    model.eval()

    total_loss = 0.0
    sample_count = 0

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    with torch.no_grad():

        for inputs, targets in validation_loader:

            inputs = inputs.to(
                device
            )

            targets = targets.to(
                device
            )

            predictions = model(
                inputs
            )

            loss = loss_function(
                predictions,
                targets,
            )

            batch_size = inputs.shape[0]

            total_loss += (
                loss.item()
                * batch_size
            )

            sample_count += batch_size

            all_targets.append(
                targets.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

    average_loss = calculate_average_loss(
        total_loss,
        sample_count,
    )

    y_true = np.concatenate(
        all_targets
    )

    y_pred = np.concatenate(
        all_predictions
    )

    return (
        average_loss,
        y_true,
        y_pred,
    )


def save_model_checkpoint(
    model: RULLSTM,
    validation_mae: float,
    epoch: int,
) -> None:

    BEST_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "validation_mae": validation_mae,
        "epoch": epoch,
        "seed": SEED,
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH,
    )


def save_training_history(
    training_history: list[dict[str, float | int]],
) -> None:

    TRAINING_HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "training_mse",
        "validation_mse",
        "validation_mae",
        "validation_rmse",
    ]

    with TRAINING_HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            training_history
        )


def save_learning_curves(
    training_history: list[dict[str, float | int]],
) -> None:

    epochs = [
        int(row["epoch"])
        for row in training_history
    ]

    training_mse = [
        float(row["training_mse"])
        for row in training_history
    ]

    validation_mse = [
        float(row["validation_mse"])
        for row in training_history
    ]

    validation_mae = [
        float(row["validation_mae"])
        for row in training_history
    ]

    validation_rmse = [
        float(row["validation_rmse"])
        for row in training_history
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        training_mse,
        marker="o",
        label="Training MSE",
    )

    plt.plot(
        epochs,
        validation_mse,
        marker="o",
        label="Validation MSE",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Mean Squared Error"
    )

    plt.title(
        "FD001 LSTM Training and Validation MSE"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        MSE_LEARNING_CURVE_PATH,
        dpi=150,
    )

    plt.close()

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        validation_mae,
        marker="o",
        label="Validation MAE",
    )

    plt.plot(
        epochs,
        validation_rmse,
        marker="o",
        label="Validation RMSE",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Error in RUL Cycles"
    )

    plt.title(
        "FD001 LSTM Validation Metrics"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        VALIDATION_METRICS_CURVE_PATH,
        dpi=150,
    )

    plt.close()


def main() -> None:

    print(
        "\n======================================"
    )
    print(
        "FD001 LSTM TRAINING"
    )
    print(
        "======================================"
    )

    set_random_seeds(
        SEED
    )

    device = torch.device(
        "cpu"
    )

    print(
        "\nDevice:"
    )
    print(
        device
    )

    print(
        "\nsequence artifacts"
    )

    x_train_sequences = np.load(
        TRAIN_SEQUENCES_PATH
    )

    y_train_sequences = np.load(
        TRAIN_TARGETS_PATH
    )

    x_validation_sequences = np.load(
        VALIDATION_SEQUENCES_PATH
    )

    y_validation_sequences = np.load(
        VALIDATION_TARGETS_PATH
    )

    print(
        "\nTraining sequence shape:"
    )
    print(
        x_train_sequences.shape
    )

    print(
        "\nValidation sequence shape:"
    )
    print(
        x_validation_sequences.shape
    )

    train_loader, validation_loader = (
        build_sequence_dataloaders(
            x_train_sequences,
            y_train_sequences,
            x_validation_sequences,
            y_validation_sequences,
            batch_size=BATCH_SIZE,
        )
    )

    model = RULLSTM(
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
        "\nModel:"
    )
    print(
        model
    )

    print(
        "\nTraining configuration:"
    )
    print(
        f"Batch size: {BATCH_SIZE}"
    )
    print(
        f"Learning rate: {LEARNING_RATE}"
    )
    print(
        f"Epochs: {NUM_EPOCHS}"
    )
    print(
        f"Seed: {SEED}"
    )

    best_validation_mae = float(
        "inf"
    )

    best_epoch = 0

    training_history: list[
        dict[str, float | int]
    ] = []

    print(
        "\n======================================"
    )
    print(
        "TRAINING START"
    )
    print(
        "======================================"
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        training_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        (
            validation_loss,
            validation_targets,
            validation_predictions,
        ) = validate_one_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )

        validation_metrics = (
            calculate_regression_metrics(
                validation_targets,
                validation_predictions,
            )
        )

        validation_mae = (
            validation_metrics[
                "mae"
            ]
        )

        validation_rmse = (
            validation_metrics[
                "rmse"
            ]
        )

        training_history.append(
            {
                "epoch": epoch,
                "training_mse": training_loss,
                "validation_mse": validation_loss,
                "validation_mae": validation_mae,
                "validation_rmse": validation_rmse,
            }
        )

        save_training_history(
            training_history
        )

        print(
            f"\nEpoch {epoch:02d}/{NUM_EPOCHS}"
        )

        print(
            f"Training MSE: "
            f"{training_loss:.4f}"
        )

        print(
            f"Validation MSE: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation MAE: "
            f"{validation_mae:.4f}"
        )

        print(
            f"Validation RMSE: "
            f"{validation_rmse:.4f}"
        )

        if (
            validation_mae
            < best_validation_mae
        ):

            best_validation_mae = (
                validation_mae
            )

            best_epoch = epoch

            save_model_checkpoint(
                model,
                validation_mae,
                epoch,
            )

            print(
                "Best validation MAE so far."
            )
            print(
                "Model checkpoint saved."
            )

    save_learning_curves(
        training_history
    )

    print(
        "\n======================================"
    )
    print(
        "TRAINING COMPLETE"
    )
    print(
        "======================================"
    )

    print(
        "\nBest epoch:"
    )
    print(
        best_epoch
    )

    print(
        "\nBest validation MAE:"
    )
    print(
        f"{best_validation_mae:.4f}"
    )

    print(
        "\nBest model saved to:"
    )
    print(
        BEST_MODEL_PATH
    )

    print(
        "\nTraining history saved to:"
    )
    print(
        TRAINING_HISTORY_PATH
    )

    print(
        "\nMSE learning curve saved to:"
    )
    print(
        MSE_LEARNING_CURVE_PATH
    )

    print(
        "\nValidation metrics curve saved to:"
    )
    print(
        VALIDATION_METRICS_CURVE_PATH
    )


if __name__ == "__main__":
    main()