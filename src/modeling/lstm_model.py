from __future__ import annotations

import torch
from torch import nn


class RULLSTM(nn.Module):

    def __init__(
        self,
        input_size: int = 14,
        hidden_size: int = 64,
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
            hidden_size,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Expected LSTM input with shape "
                "(batch, time, sensors)."
            )

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, "
                f"but received {x.shape[-1]}."
            )

        _, (
            hidden_state,
            _,
        ) = self.lstm(x)

        final_hidden_state = hidden_state[-1]

        predictions = self.regression_head(
            final_hidden_state
        )

        predictions = predictions.squeeze(-1)

        return predictions


def check_model_shapes() -> None:

    batch_size = 64
    sequence_length = 30
    input_size = 14

    sample_batch = torch.randn(
        batch_size,
        sequence_length,
        input_size,
    )

    model = RULLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=1,
    )

    print(
        "\n======================================"
    )
    print(
        "LSTM MODEL CHECK"
    )
    print(
        "======================================"
    )

    print(
        "\nModel architecture:"
    )
    print(
        model
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\nTotal parameters:"
    )
    print(
        total_parameters
    )

    print(
        "\nTrainable parameters:"
    )
    print(
        trainable_parameters
    )

    predictions = model(
        sample_batch
    )

    print(
        "\nInput shape:"
    )
    print(
        sample_batch.shape
    )

    print(
        "\nPrediction shape:"
    )
    print(
        predictions.shape
    )

    print(
        "\nExpected prediction shape:"
    )
    print(
        torch.Size([batch_size])
    )

    if predictions.shape != torch.Size(
        [batch_size]
    ):
        raise ValueError(
            "Unexpected prediction shape."
        )

    print(
        "\nLSTM model shape check passed."
    )


if __name__ == "__main__":
    check_model_shapes()