from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
)


class CycleAwareRULSequenceDataset(
    Dataset
):

    def __init__(
        self,
        sensor_sequences: np.ndarray,
        cycle_features: np.ndarray,
        targets: np.ndarray,
    ) -> None:

        if (
            len(sensor_sequences)
            != len(cycle_features)
        ):

            raise ValueError(
                "Sensor sequence count and "
                "cycle feature count do not match."
            )

        if (
            len(sensor_sequences)
            != len(targets)
        ):

            raise ValueError(
                "Sensor sequence count and "
                "target count do not match."
            )

        if sensor_sequences.ndim != 3:

            raise ValueError(
                "Sensor sequences must have shape "
                "(samples, window_length, sensors)."
            )

        if cycle_features.ndim != 2:

            raise ValueError(
                "Cycle features must have shape "
                "(samples, 1)."
            )

        if cycle_features.shape[1] != 1:

            raise ValueError(
                "Cycle features must contain exactly "
                "one feature per sample."
            )

        self.sensor_sequences = (
            torch.tensor(
                sensor_sequences,
                dtype=torch.float32,
            )
        )

        self.cycle_features = (
            torch.tensor(
                cycle_features,
                dtype=torch.float32,
            )
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.float32,
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.sensor_sequences
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        return (
            self.sensor_sequences[index],
            self.cycle_features[index],
            self.targets[index],
        )


def build_cycle_aware_sequence_dataloaders(
    x_train_sequences: np.ndarray,
    train_cycle_features: np.ndarray,
    y_train_sequences: np.ndarray,
    x_validation_sequences: np.ndarray,
    validation_cycle_features: np.ndarray,
    y_validation_sequences: np.ndarray,
    batch_size: int = 64,
) -> tuple[
    DataLoader,
    DataLoader,
]:

    train_dataset = (
        CycleAwareRULSequenceDataset(
            x_train_sequences,
            train_cycle_features,
            y_train_sequences,
        )
    )

    validation_dataset = (
        CycleAwareRULSequenceDataset(
            x_validation_sequences,
            validation_cycle_features,
            y_validation_sequences,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
    )