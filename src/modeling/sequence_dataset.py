from __future__ import annotations

import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    Dataset,
)


class RULSequenceDataset(Dataset):
   

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
    ) -> None:

        if len(sequences) != len(targets):
            raise ValueError(
                "Sequences and targets must "
                "contain the same number of samples."
            )

        self.sequences = torch.from_numpy(
            sequences
        ).float()

        self.targets = torch.from_numpy(
            targets
        ).float()

    def __len__(self) -> int:

        return len(
            self.sequences
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        return (
            self.sequences[index],
            self.targets[index],
        )


def build_sequence_dataloaders(
    x_train_sequences: np.ndarray,
    y_train_sequences: np.ndarray,
    x_validation_sequences: np.ndarray,
    y_validation_sequences: np.ndarray,
    batch_size: int = 64,
) -> tuple[
    DataLoader,
    DataLoader,
]:

    train_dataset = RULSequenceDataset(
        x_train_sequences,
        y_train_sequences,
    )

    validation_dataset = RULSequenceDataset(
        x_validation_sequences,
        y_validation_sequences,
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