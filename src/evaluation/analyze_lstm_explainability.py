from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import USEFUL_SENSOR_COLUMNS
from src.evaluation.analyze_cycle_aware_lstm import (
    CYCLE_SCALER_PATH,
    MODEL_PATH,
    VALIDATION_METADATA_PATH,
    VALIDATION_SEQUENCES_PATH,
    VALIDATION_TARGETS_PATH,
    generate_predictions,
)
from src.modeling.cycle_aware_lstm_model import (
    CycleAwareRULLSTM,
)
from src.modeling.cycle_aware_sequence_dataset import (
    CycleAwareRULSequenceDataset,
)


OUTPUT_DIR = Path("outputs")

SENSOR_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "cycle_aware_lstm_sensor_importance.csv"
)

LIFECYCLE_SENSOR_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "cycle_aware_lstm_lifecycle_sensor_importance.csv"
)

CYCLE_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "cycle_aware_lstm_cycle_permutation_importance.csv"
)

SENSOR_IMPORTANCE_PLOT = (
    OUTPUT_DIR
    / "cycle_aware_lstm_sensor_importance.png"
)

LIFECYCLE_SENSOR_IMPORTANCE_PLOT = (
    OUTPUT_DIR
    / "cycle_aware_lstm_lifecycle_sensor_importance.png"
)


RANDOM_STATE = 42

GLOBAL_SENSOR_REPEATS = 5

LIFECYCLE_SENSOR_REPEATS = 3

GLOBAL_CYCLE_REPEATS = 5

LIFECYCLE_CYCLE_REPEATS = 3

EXPECTED_VALIDATION_MAE = 17.4595


def calculate_mae(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    Calculate mean absolute error in RUL cycles.
    """

    return float(
        np.mean(
            np.abs(
                targets
                - predictions
            )
        )
    )


def build_prediction_dataset(
    sensor_sequences: np.ndarray,
    cycle_features: np.ndarray,
    targets: np.ndarray,
) -> CycleAwareRULSequenceDataset:
    """
    Build the existing cycle-aware validation
    dataset interface.
    """

    return CycleAwareRULSequenceDataset(
        sensor_sequences=sensor_sequences,
        cycle_features=cycle_features,
        targets=targets,
    )


def predict_arrays(
    model: CycleAwareRULLSTM,
    sensor_sequences: np.ndarray,
    cycle_features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Generate predictions using the project's existing
    evaluation path.
    """

    dataset = build_prediction_dataset(
        sensor_sequences=sensor_sequences,
        cycle_features=cycle_features,
        targets=targets,
    )

    (
        returned_targets,
        predictions,
    ) = generate_predictions(
        model=model,
        dataset=dataset,
        device=device,
    )

    if not np.allclose(
        returned_targets,
        targets,
    ):
        raise ValueError(
            "Prediction dataset targets became "
            "misaligned."
        )

    return predictions


def load_explainability_components() -> tuple[
    CycleAwareRULLSTM,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    torch.device,
    dict,
]:
    """
    Load the exact saved cycle-aware LSTM and its
    development-validation inputs.
    """

    device = torch.device(
        "cpu"
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model = CycleAwareRULLSTM(
        input_size=checkpoint[
            "input_size"
        ],
        hidden_size=checkpoint[
            "hidden_size"
        ],
        num_layers=checkpoint[
            "num_layers"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    sensor_sequences = np.load(
        VALIDATION_SEQUENCES_PATH
    )

    targets = np.load(
        VALIDATION_TARGETS_PATH
    )

    metadata_df = pd.read_csv(
        VALIDATION_METADATA_PATH
    )

    if sensor_sequences.ndim != 3:
        raise ValueError(
            "Validation sequences must have shape "
            "(samples, window, sensors)."
        )

    if (
        sensor_sequences.shape[2]
        != len(USEFUL_SENSOR_COLUMNS)
    ):
        raise ValueError(
            "Validation sensor count does not match "
            "USEFUL_SENSOR_COLUMNS."
        )

    if (
        checkpoint["input_size"]
        != len(USEFUL_SENSOR_COLUMNS)
    ):
        raise ValueError(
            "Checkpoint input size does not match "
            "the canonical sensor list."
        )

    if (
        len(sensor_sequences)
        != len(targets)
        or len(sensor_sequences)
        != len(metadata_df)
    ):
        raise ValueError(
            "Validation sequence, target, and "
            "metadata counts do not match."
        )

    if not np.allclose(
        metadata_df[
            "rul"
        ].to_numpy(),
        targets,
    ):
        raise ValueError(
            "Validation metadata RUL values do not "
            "match sequence targets."
        )

    cycle_scaler = joblib.load(
        CYCLE_SCALER_PATH
    )

    cycle_features = (
        cycle_scaler.transform(
            metadata_df[
                ["cycle"]
            ].to_numpy()
        )
        .astype(
            np.float32
        )
    )

    if cycle_features.shape != (
        len(sensor_sequences),
        1,
    ):
        raise ValueError(
            "Cycle feature shape is incorrect."
        )

    return (
        model,
        sensor_sequences,
        cycle_features,
        targets,
        metadata_df,
        device,
        checkpoint,
    )


def create_permutation_indices(
    sample_count: int,
    n_repeats: int,
    random_state: int,
) -> list[np.ndarray]:
    """
    Create reproducible row permutations.

    The same set of row permutations can be used
    across sensors so their importance estimates are
    compared under similar random perturbations.
    """

    rng = np.random.default_rng(
        random_state
    )

    return [
        rng.permutation(
            sample_count
        )
        for _ in range(
            n_repeats
        )
    ]


def calculate_sensor_permutation_importance(
    model: CycleAwareRULLSTM,
    sensor_sequences: np.ndarray,
    cycle_features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Permutes one complete 30-step sensor trajectory
    across samples while preserving the trajectory's
    internal temporal order.
    """

    baseline_predictions = predict_arrays(
        model=model,
        sensor_sequences=sensor_sequences,
        cycle_features=cycle_features,
        targets=targets,
        device=device,
    )

    baseline_mae = calculate_mae(
        targets,
        baseline_predictions,
    )

    permutation_indices = (
        create_permutation_indices(
            sample_count=len(
                sensor_sequences
            ),
            n_repeats=n_repeats,
            random_state=random_state,
        )
    )

    result_rows = []

    for (
        sensor_index,
        sensor_name,
    ) in enumerate(
        USEFUL_SENSOR_COLUMNS
    ):

        permuted_maes = []

        for indices in (
            permutation_indices
        ):

            permuted_sequences = (
                sensor_sequences.copy()
            )

            # Replace the COMPLETE trajectory for
            # one sensor with another sample's
            # internally ordered trajectory.
            permuted_sequences[
                :,
                :,
                sensor_index,
            ] = (
                sensor_sequences[
                    indices,
                    :,
                    sensor_index,
                ]
            )

            predictions = predict_arrays(
                model=model,
                sensor_sequences=(
                    permuted_sequences
                ),
                cycle_features=(
                    cycle_features
                ),
                targets=targets,
                device=device,
            )

            permuted_mae = calculate_mae(
                targets,
                predictions,
            )

            permuted_maes.append(
                permuted_mae
            )

        mae_increases = (
            np.asarray(
                permuted_maes,
                dtype=float,
            )
            - baseline_mae
        )

        result_rows.append(
            {
                "sensor": sensor_name,
                "sensor_index": (
                    sensor_index
                ),
                "baseline_mae": (
                    baseline_mae
                ),
                "mean_permuted_mae": (
                    float(
                        np.mean(
                            permuted_maes
                        )
                    )
                ),
                "mean_mae_increase": (
                    float(
                        np.mean(
                            mae_increases
                        )
                    )
                ),
                "std_mae_increase": (
                    float(
                        np.std(
                            mae_increases
                        )
                    )
                ),
            }
        )

    result_df = pd.DataFrame(
        result_rows
    )

    result_df = (
        result_df
        .sort_values(
            "mean_mae_increase",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result_df["rank"] = (
        np.arange(
            1,
            len(result_df) + 1,
        )
    )

    return result_df[
        [
            "sensor",
            "sensor_index",
            "baseline_mae",
            "mean_permuted_mae",
            "mean_mae_increase",
            "std_mae_increase",
            "rank",
        ]
    ]


def calculate_cycle_permutation_importance(
    model: CycleAwareRULLSTM,
    sensor_sequences: np.ndarray,
    cycle_features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    n_repeats: int,
    random_state: int,
) -> dict[str, float]:
    """
    Keep the trained model and sensor sequences fixed,
    but permute the standardized ending-cycle feature
    across samples.
    """

    baseline_predictions = predict_arrays(
        model=model,
        sensor_sequences=sensor_sequences,
        cycle_features=cycle_features,
        targets=targets,
        device=device,
    )

    baseline_mae = calculate_mae(
        targets,
        baseline_predictions,
    )

    permutation_indices = (
        create_permutation_indices(
            sample_count=len(
                cycle_features
            ),
            n_repeats=n_repeats,
            random_state=random_state,
        )
    )

    permuted_maes = []

    for indices in (
        permutation_indices
    ):

        permuted_cycles = (
            cycle_features[
                indices
            ]
            .copy()
        )

        predictions = predict_arrays(
            model=model,
            sensor_sequences=(
                sensor_sequences
            ),
            cycle_features=(
                permuted_cycles
            ),
            targets=targets,
            device=device,
        )

        permuted_mae = calculate_mae(
            targets,
            predictions,
        )

        permuted_maes.append(
            permuted_mae
        )

    mae_increases = (
        np.asarray(
            permuted_maes,
            dtype=float,
        )
        - baseline_mae
    )

    return {
        "baseline_mae": (
            baseline_mae
        ),
        "mean_permuted_mae": (
            float(
                np.mean(
                    permuted_maes
                )
            )
        ),
        "mean_mae_increase": (
            float(
                np.mean(
                    mae_increases
                )
            )
        ),
        "std_mae_increase": (
            float(
                np.std(
                    mae_increases
                )
            )
        ),
    }


def build_lifecycle_masks(
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Build the same established RUL lifecycle bands
    used throughout the project.
    """

    return {
        "near_failure_0_30": (
            targets <= 30
        ),
        "medium_rul_31_100": (
            (targets > 30)
            & (targets <= 100)
        ),
        "high_rul_above_100": (
            targets > 100
        ),
    }


def calculate_lifecycle_sensor_importance(
    model: CycleAwareRULLSTM,
    sensor_sequences: np.ndarray,
    cycle_features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """
    Repeat sensor-trajectory permutation separately
    within each lifecycle stage.

    Permutations remain inside each RUL band rather
    than transferring trajectories between lifecycle
    stages.
    """

    lifecycle_masks = (
        build_lifecycle_masks(
            targets
        )
    )

    lifecycle_tables = []

    for (
        band_number,
        (
            band_name,
            band_mask,
        ),
    ) in enumerate(
        lifecycle_masks.items(),
        start=1,
    ):

        band_sequences = (
            sensor_sequences[
                band_mask
            ]
            .copy()
        )

        band_cycles = (
            cycle_features[
                band_mask
            ]
            .copy()
        )

        band_targets = (
            targets[
                band_mask
            ]
            .copy()
        )

        if len(
            band_sequences
        ) == 0:
            raise ValueError(
                "Lifecycle band contains no "
                f"samples: {band_name}"
            )

        band_importance_df = (
            calculate_sensor_permutation_importance(
                model=model,
                sensor_sequences=(
                    band_sequences
                ),
                cycle_features=(
                    band_cycles
                ),
                targets=(
                    band_targets
                ),
                device=device,
                n_repeats=(
                    LIFECYCLE_SENSOR_REPEATS
                ),
                random_state=(
                    RANDOM_STATE
                    + band_number
                ),
            )
        )

        band_importance_df.insert(
            0,
            "rul_band",
            band_name,
        )

        band_importance_df.insert(
            1,
            "row_count",
            len(
                band_sequences
            ),
        )

        lifecycle_tables.append(
            band_importance_df
        )

    return pd.concat(
        lifecycle_tables,
        ignore_index=True,
    )


def calculate_all_cycle_importance(
    model: CycleAwareRULLSTM,
    sensor_sequences: np.ndarray,
    cycle_features: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """
    Measure cycle reliance globally and within the
    established lifecycle bands.
    """

    result_rows = []

    global_result = (
        calculate_cycle_permutation_importance(
            model=model,
            sensor_sequences=(
                sensor_sequences
            ),
            cycle_features=(
                cycle_features
            ),
            targets=targets,
            device=device,
            n_repeats=(
                GLOBAL_CYCLE_REPEATS
            ),
            random_state=RANDOM_STATE,
        )
    )

    result_rows.append(
        {
            "scope": "global",
            "row_count": int(
                len(targets)
            ),
            **global_result,
        }
    )

    lifecycle_masks = (
        build_lifecycle_masks(
            targets
        )
    )

    for (
        band_number,
        (
            band_name,
            band_mask,
        ),
    ) in enumerate(
        lifecycle_masks.items(),
        start=1,
    ):

        band_result = (
            calculate_cycle_permutation_importance(
                model=model,
                sensor_sequences=(
                    sensor_sequences[
                        band_mask
                    ]
                ),
                cycle_features=(
                    cycle_features[
                        band_mask
                    ]
                ),
                targets=(
                    targets[
                        band_mask
                    ]
                ),
                device=device,
                n_repeats=(
                    LIFECYCLE_CYCLE_REPEATS
                ),
                random_state=(
                    RANDOM_STATE
                    + 100
                    + band_number
                ),
            )
        )

        result_rows.append(
            {
                "scope": band_name,
                "row_count": int(
                    band_mask.sum()
                ),
                **band_result,
            }
        )

    return pd.DataFrame(
        result_rows
    )


def save_sensor_importance_plot(
    sensor_importance_df: pd.DataFrame,
) -> None:
    """
    Save global sensor-channel permutation
    importance.
    """

    plot_df = (
        sensor_importance_df
        .sort_values(
            "mean_mae_increase",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    ax.barh(
        plot_df[
            "sensor"
        ],
        plot_df[
            "mean_mae_increase"
        ],
        xerr=plot_df[
            "std_mae_increase"
        ],
    )

    ax.axvline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Increase in validation MAE "
        "after trajectory permutation"
    )

    ax.set_ylabel(
        "Sensor"
    )

    ax.set_title(
        "Cycle-Aware LSTM: "
        "Sensor Reliance"
    )

    fig.tight_layout()

    fig.savefig(
        SENSOR_IMPORTANCE_PLOT,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_lifecycle_sensor_plot(
    lifecycle_df: pd.DataFrame,
) -> None:
    """
    Save lifecycle-specific sensor permutation
    importance.
    """

    pivot_df = (
        lifecycle_df
        .pivot(
            index="sensor",
            columns="rul_band",
            values="mean_mae_increase",
        )
    )

    # Order sensors by strongest importance observed
    # in any lifecycle stage.
    sensor_order = (
        pivot_df
        .max(axis=1)
        .sort_values(
            ascending=False
        )
        .index
    )

    pivot_df = (
        pivot_df.loc[
            sensor_order
        ]
    )

    ax = pivot_df.plot(
        kind="bar",
        figsize=(12, 7),
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Sensor"
    )

    ax.set_ylabel(
        "Increase in MAE after "
        "trajectory permutation"
    )

    ax.set_title(
        "Cycle-Aware LSTM: "
        "Sensor Reliance by Lifecycle Stage"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        LIFECYCLE_SENSOR_IMPORTANCE_PLOT,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main() -> None:
    """
    Explain the already-trained cycle-aware LSTM
    using development-validation perturbation only.
    """

    print(
        "\n======================================"
    )

    print(
        "FD001 CYCLE-AWARE LSTM EXPLAINABILITY"
    )

    print(
        "======================================"
    )

    (
        model,
        sensor_sequences,
        cycle_features,
        targets,
        metadata_df,
        device,
        checkpoint,
    ) = load_explainability_components()

    print(
        "\nLoaded checkpoint:"
    )

    print(
        "Epoch:",
        checkpoint[
            "epoch"
        ],
    )

    print(
        "Saved validation MAE:",
        f"{checkpoint['validation_mae']:.4f}",
    )

    print(
        "\nValidation sequence shape:"
    )

    print(
        sensor_sequences.shape
    )

    print(
        "\nCycle feature shape:"
    )

    print(
        cycle_features.shape
    )

    print(
        "\nValidation engines:"
    )

    print(
        metadata_df[
            "engine_id"
        ].nunique()
    )

    print(
        "\nSensor order:"
    )

    for (
        sensor_index,
        sensor_name,
    ) in enumerate(
        USEFUL_SENSOR_COLUMNS
    ):
        print(
            f"{sensor_index}: "
            f"{sensor_name}"
        )

    # ---------------------------------------
    # BASELINE REPRODUCTION
    # ---------------------------------------

    baseline_predictions = predict_arrays(
        model=model,
        sensor_sequences=sensor_sequences,
        cycle_features=cycle_features,
        targets=targets,
        device=device,
    )

    baseline_mae = calculate_mae(
        targets,
        baseline_predictions,
    )

    print(
        "\nReconstructed validation MAE:"
    )

    print(
        f"{baseline_mae:.4f}"
    )

    if not np.isclose(
        baseline_mae,
        EXPECTED_VALIDATION_MAE,
        rtol=0.0,
        atol=0.01,
    ):
        raise ValueError(
            "Cycle-aware checkpoint reproduction "
            "does not match the established "
            "development result. "
            f"Expected approximately "
            f"{EXPECTED_VALIDATION_MAE:.4f}, "
            f"found {baseline_mae:.4f}."
        )

    print(
        "\nCheckpoint reproduction:"
    )

    print(
        "PASSED"
    )

    # ---------------------------------------
    # GLOBAL SENSOR PERMUTATION
    # ---------------------------------------

    print(
        "\nCalculating global sensor-trajectory "
        "permutation importance..."
    )

    sensor_importance_df = (
        calculate_sensor_permutation_importance(
            model=model,
            sensor_sequences=(
                sensor_sequences
            ),
            cycle_features=(
                cycle_features
            ),
            targets=targets,
            device=device,
            n_repeats=(
                GLOBAL_SENSOR_REPEATS
            ),
            random_state=(
                RANDOM_STATE
            ),
        )
    )

    # ---------------------------------------
    # LIFECYCLE SENSOR PERMUTATION
    # ---------------------------------------

    print(
        "\nCalculating lifecycle-specific "
        "sensor importance..."
    )

    lifecycle_sensor_df = (
        calculate_lifecycle_sensor_importance(
            model=model,
            sensor_sequences=(
                sensor_sequences
            ),
            cycle_features=(
                cycle_features
            ),
            targets=targets,
            device=device,
        )
    )

    # ---------------------------------------
    # CYCLE PERMUTATION
    # ---------------------------------------

    print(
        "\nCalculating cycle permutation "
        "importance..."
    )

    cycle_importance_df = (
        calculate_all_cycle_importance(
            model=model,
            sensor_sequences=(
                sensor_sequences
            ),
            cycle_features=(
                cycle_features
            ),
            targets=targets,
            device=device,
        )
    )

    # ---------------------------------------
    # SAVE
    # ---------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sensor_importance_df.to_csv(
        SENSOR_IMPORTANCE_FILE,
        index=False,
    )

    lifecycle_sensor_df.to_csv(
        LIFECYCLE_SENSOR_IMPORTANCE_FILE,
        index=False,
    )

    cycle_importance_df.to_csv(
        CYCLE_IMPORTANCE_FILE,
        index=False,
    )

    save_sensor_importance_plot(
        sensor_importance_df
    )

    save_lifecycle_sensor_plot(
        lifecycle_sensor_df
    )

    # ---------------------------------------
    # PRINT RESULTS
    # ---------------------------------------

    print(
        "\n======================================"
    )

    print(
        "GLOBAL SENSOR IMPORTANCE"
    )

    print(
        "======================================"
    )

    print(
        sensor_importance_df
        .round(6)
        .to_string(
            index=False
        )
    )

    print(
        "\n======================================"
    )

    print(
        "LIFECYCLE SENSOR IMPORTANCE"
    )

    print(
        "======================================"
    )

    for band_name in (
        lifecycle_sensor_df[
            "rul_band"
        ].unique()
    ):

        print(
            f"\n{band_name}:"
        )

        band_df = (
            lifecycle_sensor_df.loc[
                lifecycle_sensor_df[
                    "rul_band"
                ]
                == band_name
            ]
            .sort_values(
                "rank"
            )
        )

        print(
            band_df
            .round(6)
            .to_string(
                index=False
            )
        )

    print(
        "\n======================================"
    )

    print(
        "CYCLE PERMUTATION IMPORTANCE"
    )

    print(
        "======================================"
    )

    print(
        cycle_importance_df
        .round(6)
        .to_string(
            index=False
        )
    )

    print(
        "\n======================================"
    )

    print(
        "METHOD INTERPRETATION"
    )

    print(
        "======================================"
    )

    print(
        "\nSensor permutation:"
    )

    print(
        "A complete 30-step sensor trajectory "
        "was transferred between validation "
        "samples while preserving its internal "
        "time order."
    )

    print(
        "\nCycle permutation:"
    )

    print(
        "The trained model and sensor sequences "
        "were kept fixed while standardized "
        "ending-cycle values were shuffled."
    )

    print(
        "\nImportant limitation:"
    )

    print(
        "These perturbations measure trained-model "
        "reliance. They do not establish causal "
        "sensor importance and can create synthetic "
        "cross-sensor combinations."
    )

    print(
        "\nSaved:"
    )

    for output_file in [
        SENSOR_IMPORTANCE_FILE,
        LIFECYCLE_SENSOR_IMPORTANCE_FILE,
        CYCLE_IMPORTANCE_FILE,
        SENSOR_IMPORTANCE_PLOT,
        LIFECYCLE_SENSOR_IMPORTANCE_PLOT,
    ]:
        print(
            output_file
        )


if __name__ == "__main__":
    main()