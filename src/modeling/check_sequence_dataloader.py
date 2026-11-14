from __future__ import annotations

import numpy as np

from src.modeling.sequence_dataset import (
    build_sequence_dataloaders,
)


TRAIN_SEQUENCES_PATH = (
    "data/processed/fd001_train_sequences.npy"
)

TRAIN_TARGETS_PATH = (
    "data/processed/fd001_train_sequence_targets.npy"
)

VALIDATION_SEQUENCES_PATH = (
    "data/processed/fd001_validation_sequences.npy"
)

VALIDATION_TARGETS_PATH = (
    "data/processed/fd001_validation_sequence_targets.npy"
)


def main() -> None:

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

    train_loader, validation_loader = (
        build_sequence_dataloaders(
            x_train_sequences,
            y_train_sequences,
            x_validation_sequences,
            y_validation_sequences,
            batch_size=64,
        )
    )

    train_inputs, train_targets = next(
        iter(train_loader)
    )

    validation_inputs, validation_targets = next(
        iter(validation_loader)
    )

    print(
        "\n======================================"
    )
    print(
        "SEQUENCE DATALOADER SANITY CHECK"
    )
    print(
        "======================================"
    )

    print(
        "\nTraining batch input shape:"
    )
    print(
        train_inputs.shape
    )

    print(
        "\nTraining batch target shape:"
    )
    print(
        train_targets.shape
    )

    print(
        "\nTraining input dtype:"
    )
    print(
        train_inputs.dtype
    )

    print(
        "\nTraining target dtype:"
    )
    print(
        train_targets.dtype
    )

    print(
        "\nValidation batch input shape:"
    )
    print(
        validation_inputs.shape
    )

    print(
        "\nValidation batch target shape:"
    )
    print(
        validation_targets.shape
    )

    print(
        "\nValidation input dtype:"
    )
    print(
        validation_inputs.dtype
    )

    print(
        "\nValidation target dtype:"
    )
    print(
        validation_targets.dtype
    )

    print(
        "\nFirst training sequence shape:"
    )
    print(
        train_inputs[0].shape
    )

    print(
        "\nFirst training time step shape:"
    )
    print(
        train_inputs[0, 0].shape
    )

    print(
        "\nFirst training target:"
    )
    print(
        train_targets[0]
    )

    print(
        "\nAll sanity checks passed."
    )


if __name__ == "__main__":
    main()