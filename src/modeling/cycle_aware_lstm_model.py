from __future__ import annotations

import torch
from torch import nn


class CycleAwareRULLSTM(
    nn.Module
):

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
    ) -> None:

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.regression_head = nn.Linear(
            hidden_size + 1,
            1,
        )

    def forward(
        self,
        sensor_sequences: torch.Tensor,
        cycle_feature: torch.Tensor,
    ) -> torch.Tensor:

        _, (
            hidden_state,
            _,
        ) = self.lstm(
            sensor_sequences
        )

        final_hidden_state = (
            hidden_state[-1]
        )

        if cycle_feature.ndim == 1:

            cycle_feature = (
                cycle_feature.unsqueeze(
                    dim=1
                )
            )

        combined_features = torch.cat(
            [
                final_hidden_state,
                cycle_feature,
            ],
            dim=1,
        )

        predictions = self.regression_head(
            combined_features
        )

        predictions = predictions.squeeze(
            dim=1
        )

        return predictions