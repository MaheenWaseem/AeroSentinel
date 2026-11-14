from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.config import (
    SAVED_ARTIFACTS_DIR,
    USEFUL_SENSOR_COLUMNS,
)
from src.modeling.cycle_aware_lstm_model import (
    CycleAwareRULLSTM,
)
from src.modeling.prepare_sequence_data import (
    STRIDE,
    WINDOW_LENGTH,
    apply_sensor_scaling,
)


LOGGER = logging.getLogger(__name__)


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


class RULInferenceService:

    def __init__(
        self,
    ) -> None:

        self.device = torch.device(
            "cpu"
        )

        self._validate_artifact_files()

        self.sensor_scaler = joblib.load(
            FINAL_SENSOR_SCALER_PATH
        )

        self.cycle_scaler = joblib.load(
            FINAL_CYCLE_SCALER_PATH
        )

        self.checkpoint = torch.load(
            FINAL_MODEL_PATH,
            map_location=self.device,
        )

        self._validate_scalers()

        self._validate_checkpoint()

        self.model = self._build_model()

        LOGGER.info(
            "Final FD001 inference artifacts loaded successfully."
        )

    def _validate_artifact_files(
        self,
    ) -> None:

        required_paths = [
            FINAL_SENSOR_SCALER_PATH,
            FINAL_CYCLE_SCALER_PATH,
            FINAL_MODEL_PATH,
        ]

        missing_paths = [
            path
            for path in required_paths
            if not path.exists()
        ]

        if missing_paths:
            raise FileNotFoundError(
                "Required final inference artifacts "
                f"were not found: {missing_paths}"
            )

    def _validate_scalers(
        self,
    ) -> None:

        if not isinstance(
            self.sensor_scaler,
            StandardScaler,
        ):
            raise TypeError(
                "Final sensor scaler is not a "
                "StandardScaler."
            )

        if not isinstance(
            self.cycle_scaler,
            StandardScaler,
        ):
            raise TypeError(
                "Final cycle scaler is not a "
                "StandardScaler."
            )

        if not hasattr(
            self.sensor_scaler,
            "n_features_in_",
        ):
            raise ValueError(
                "Final sensor scaler is not fitted."
            )

        if not hasattr(
            self.cycle_scaler,
            "n_features_in_",
        ):
            raise ValueError(
                "Final cycle scaler is not fitted."
            )

        if (
            self.sensor_scaler.n_features_in_
            != len(USEFUL_SENSOR_COLUMNS)
        ):
            raise ValueError(
                "Final sensor scaler feature count "
                "does not match the locked sensor count."
            )

        if self.cycle_scaler.n_features_in_ != 1:
            raise ValueError(
                "Final cycle scaler must expect "
                "exactly one feature."
            )

        if not hasattr(
            self.sensor_scaler,
            "feature_names_in_",
        ):
            raise ValueError(
                "Final sensor scaler does not contain "
                "fitted sensor names."
            )

        fitted_sensor_names = (
            self.sensor_scaler
            .feature_names_in_
            .tolist()
        )

        if (
            fitted_sensor_names
            != USEFUL_SENSOR_COLUMNS
        ):
            raise ValueError(
                "Final sensor scaler feature order "
                "does not match the locked sensor order."
            )

        if not np.isfinite(
            self.sensor_scaler.mean_
        ).all():
            raise ValueError(
                "Final sensor scaler contains "
                "non-finite means."
            )

        if not np.isfinite(
            self.sensor_scaler.scale_
        ).all():
            raise ValueError(
                "Final sensor scaler contains "
                "non-finite scales."
            )

        if not np.isfinite(
            self.cycle_scaler.mean_
        ).all():
            raise ValueError(
                "Final cycle scaler contains "
                "non-finite means."
            )

        if not np.isfinite(
            self.cycle_scaler.scale_
        ).all():
            raise ValueError(
                "Final cycle scaler contains "
                "non-finite scales."
            )

        if (
            self.sensor_scaler.scale_
            == 0
        ).any():
            raise ValueError(
                "Final sensor scaler contains "
                "a zero scale."
            )

        if (
            self.cycle_scaler.scale_
            == 0
        ).any():
            raise ValueError(
                "Final cycle scaler contains "
                "a zero scale."
            )

    def _validate_checkpoint(
        self,
    ) -> None:

        if not isinstance(
            self.checkpoint,
            dict,
        ):
            raise TypeError(
                "Final model checkpoint must be "
                "a dictionary."
            )

        required_keys = {
            "model_state_dict",
            "input_size",
            "hidden_size",
            "num_layers",
            "epoch",
            "sensor_columns",
            "window_length",
            "stride",
            "cycle_feature",
            "training_engine_count",
        }

        missing_keys = (
            required_keys
            - set(
                self.checkpoint.keys()
            )
        )

        if missing_keys:
            raise ValueError(
                "Final model checkpoint is missing "
                f"required keys: {sorted(missing_keys)}"
            )

        if (
            self.checkpoint["input_size"]
            != len(USEFUL_SENSOR_COLUMNS)
        ):
            raise ValueError(
                "Checkpoint input size does not match "
                "the locked sensor count."
            )

        if (
            self.checkpoint["sensor_columns"]
            != USEFUL_SENSOR_COLUMNS
        ):
            raise ValueError(
                "Checkpoint sensor order does not "
                "match the locked sensor order."
            )

        if (
            self.checkpoint["window_length"]
            != WINDOW_LENGTH
        ):
            raise ValueError(
                "Checkpoint window length does not "
                "match the inference window length."
            )

        if (
            self.checkpoint["stride"]
            != STRIDE
        ):
            raise ValueError(
                "Checkpoint stride does not match "
                "the locked sequence protocol."
            )

        if (
            self.checkpoint["cycle_feature"]
            != "standardized_ending_cycle"
        ):
            raise ValueError(
                "Checkpoint cycle feature does not "
                "match the expected inference feature."
            )

        if (
            self.checkpoint[
                "training_engine_count"
            ]
            != 100
        ):
            raise ValueError(
                "Final model checkpoint was not "
                "trained on exactly 100 engines."
            )

        if (
            not isinstance(
                self.checkpoint["hidden_size"],
                int,
            )
            or self.checkpoint[
                "hidden_size"
            ] <= 0
        ):
            raise ValueError(
                "Checkpoint hidden size is invalid."
            )

        if (
            not isinstance(
                self.checkpoint["num_layers"],
                int,
            )
            or self.checkpoint[
                "num_layers"
            ] <= 0
        ):
            raise ValueError(
                "Checkpoint layer count is invalid."
            )

        if (
            not isinstance(
                self.checkpoint["epoch"],
                int,
            )
            or self.checkpoint["epoch"] <= 0
        ):
            raise ValueError(
                "Checkpoint epoch value is invalid."
            )

    def _build_model(
        self,
    ) -> CycleAwareRULLSTM:

        model = CycleAwareRULLSTM(
            input_size=self.checkpoint[
                "input_size"
            ],
            hidden_size=self.checkpoint[
                "hidden_size"
            ],
            num_layers=self.checkpoint[
                "num_layers"
            ],
        )

        try:
            model.load_state_dict(
                self.checkpoint[
                    "model_state_dict"
                ]
            )
        except RuntimeError as error:
            raise ValueError(
                "Final model state dictionary is "
                "incompatible with the reconstructed "
                "cycle-aware LSTM."
            ) from error

        model = model.to(
            self.device
        )

        model.eval()

        return model

    def _validate_engine_history(
        self,
        engine_history: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(
            engine_history,
            pd.DataFrame,
        ):
            raise TypeError(
                "Engine history must be provided as "
                "a pandas DataFrame."
            )

        if engine_history.empty:
            raise ValueError(
                "Engine history cannot be empty."
            )

        required_columns = [
            "engine_id",
            "cycle",
            *USEFUL_SENSOR_COLUMNS,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column
            not in engine_history.columns
        ]

        if missing_columns:
            raise ValueError(
                "Engine history is missing required "
                f"columns: {missing_columns}"
            )

        validated_df = (
            engine_history[
                required_columns
            ]
            .copy()
        )

        if validated_df.isna().any().any():
            missing_value_columns = (
                validated_df
                .columns[
                    validated_df
                    .isna()
                    .any()
                ]
                .tolist()
            )

            raise ValueError(
                "Engine history contains missing "
                "values in columns: "
                f"{missing_value_columns}"
            )

        non_numeric_columns = [
            column
            for column
            in required_columns
            if not pd.api.types.is_numeric_dtype(
                validated_df[column]
            )
        ]

        if non_numeric_columns:
            raise ValueError(
                "Required inference columns must be "
                "numeric. Invalid columns: "
                f"{non_numeric_columns}"
            )

        required_values = (
            validated_df[
                required_columns
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if not np.isfinite(
            required_values
        ).all():
            raise ValueError(
                "Engine history contains non-finite "
                "values."
            )

        unique_engine_ids = (
            validated_df[
                "engine_id"
            ]
            .unique()
        )

        if len(unique_engine_ids) != 1:
            raise ValueError(
                "One inference request must contain "
                "history for exactly one engine."
            )

        engine_id_value = float(
            unique_engine_ids[0]
        )

        if not engine_id_value.is_integer():
            raise ValueError(
                "Engine ID must be an integer."
            )

        engine_id = int(
            engine_id_value
        )

        if engine_id <= 0:
            raise ValueError(
                "Engine ID must be positive."
            )

        cycle_values = (
            validated_df[
                "cycle"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if not np.equal(
            cycle_values,
            np.floor(
                cycle_values
            ),
        ).all():
            raise ValueError(
                "Cycle values must be integers."
            )

        if (
            cycle_values
            <= 0
        ).any():
            raise ValueError(
                "Cycle values must be positive."
            )

        if validated_df[
            "cycle"
        ].duplicated().any():
            duplicate_cycles = (
                validated_df.loc[
                    validated_df[
                        "cycle"
                    ].duplicated(
                        keep=False
                    ),
                    "cycle",
                ]
                .astype(int)
                .tolist()
            )

            raise ValueError(
                "Engine history contains duplicate "
                f"cycle values: {duplicate_cycles}"
            )

        if len(validated_df) < WINDOW_LENGTH:
            raise ValueError(
                "Engine history must contain at least "
                f"{WINDOW_LENGTH} observations. "
                f"Received {len(validated_df)}."
            )

        validated_df[
            "engine_id"
        ] = validated_df[
            "engine_id"
        ].astype(int)

        validated_df[
            "cycle"
        ] = validated_df[
            "cycle"
        ].astype(int)

        validated_df = (
            validated_df
            .sort_values(
                "cycle"
            )
            .reset_index(
                drop=True
            )
        )

        return validated_df

    def _prepare_model_inputs(
        self,
        validated_df: pd.DataFrame,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        int,
    ]:

        scaled_df = apply_sensor_scaling(
            validated_df,
            self.sensor_scaler,
        )

        final_window_df = (
            scaled_df
            .tail(
                WINDOW_LENGTH
            )
            .copy()
        )

        sensor_sequence = (
            final_window_df[
                USEFUL_SENSOR_COLUMNS
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        expected_sequence_shape = (
            WINDOW_LENGTH,
            len(
                USEFUL_SENSOR_COLUMNS
            ),
        )

        if (
            sensor_sequence.shape
            != expected_sequence_shape
        ):
            raise ValueError(
                "Prepared sensor sequence has an "
                "unexpected shape. "
                f"Expected {expected_sequence_shape}, "
                f"received {sensor_sequence.shape}."
            )

        if not np.isfinite(
            sensor_sequence
        ).all():
            raise ValueError(
                "Prepared sensor sequence contains "
                "non-finite values."
            )

        engine_id = int(
            validated_df[
                "engine_id"
            ].iloc[-1]
        )

        last_observed_cycle = int(
            validated_df[
                "cycle"
            ].iloc[-1]
        )

        ending_cycle_array = np.asarray(
            [
                [
                    last_observed_cycle
                ]
            ],
            dtype=np.float64,
        )

        cycle_feature = (
            self.cycle_scaler
            .transform(
                ending_cycle_array
            )
            .astype(
                np.float32
            )
        )

        if cycle_feature.shape != (
            1,
            1,
        ):
            raise ValueError(
                "Prepared cycle feature has an "
                "unexpected shape."
            )

        if not np.isfinite(
            cycle_feature
        ).all():
            raise ValueError(
                "Prepared cycle feature contains "
                "non-finite values."
            )

        sensor_tensor = (
            torch.from_numpy(
                sensor_sequence
            )
            .unsqueeze(
                dim=0
            )
            .to(
                self.device
            )
        )

        cycle_tensor = (
            torch.from_numpy(
                cycle_feature
            )
            .to(
                self.device
            )
        )

        if sensor_tensor.shape != (
            1,
            WINDOW_LENGTH,
            len(
                USEFUL_SENSOR_COLUMNS
            ),
        ):
            raise ValueError(
                "Sensor tensor has an unexpected "
                "inference shape."
            )

        if cycle_tensor.shape != (
            1,
            1,
        ):
            raise ValueError(
                "Cycle tensor has an unexpected "
                "inference shape."
            )

        return (
            sensor_tensor,
            cycle_tensor,
            engine_id,
            last_observed_cycle,
        )

    def predict(
        self,
        engine_history: pd.DataFrame,
    ) -> dict[
        str,
        int | float,
    ]:

        validated_df = (
            self._validate_engine_history(
                engine_history
            )
        )

        (
            sensor_tensor,
            cycle_tensor,
            engine_id,
            last_observed_cycle,
        ) = self._prepare_model_inputs(
            validated_df
        )

        self.model.eval()

        with torch.no_grad():

            prediction_tensor = self.model(
                sensor_tensor,
                cycle_tensor,
            )

        if prediction_tensor.shape != (
            1,
        ):
            raise ValueError(
                "Model returned an unexpected "
                "prediction shape."
            )

        if not torch.isfinite(
            prediction_tensor
        ).all().item():
            raise ValueError(
                "Model returned a non-finite "
                "RUL prediction."
            )

        predicted_rul = float(
            prediction_tensor.item()
        )

        LOGGER.info(
            "Generated RUL prediction for "
            "engine %s at cycle %s.",
            engine_id,
            last_observed_cycle,
        )

        return {
            "engine_id": engine_id,
            "last_observed_cycle":
                last_observed_cycle,
            "predicted_rul":
                predicted_rul,
        }


def main() -> None:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(levelname)s: %(message)s"
        ),
    )

    service = RULInferenceService()

    print(
        "\n======================================"
    )

    print(
        "FD001 RUL INFERENCE SERVICE CHECK"
    )

    print(
        "======================================"
    )

    print(
        "\nArtifact loading:"
    )

    print(
        "PASSED"
    )

    print(
        "\nDevice:"
    )

    print(
        service.device
    )

    print(
        "\nWindow length:"
    )

    print(
        WINDOW_LENGTH
    )

    print(
        "\nSensor count:"
    )

    print(
        len(
            USEFUL_SENSOR_COLUMNS
        )
    )

    print(
        "\nFinal training epochs:"
    )

    print(
        service.checkpoint[
            "epoch"
        ]
    )

    print(
        "\nFinal training engines:"
    )

    print(
        service.checkpoint[
            "training_engine_count"
        ]
    )

    print(
        "\nModel:"
    )

    print(
        service.model.__class__.__name__
    )



if __name__ == "__main__":
    main()