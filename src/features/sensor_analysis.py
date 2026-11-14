from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    CMAPSS_COLUMNS,
    FIGURES_DIR,
    METRICS_DIR,
)
from src.data.load_data import load_train_data
from src.data.split_data import get_development_split
from src.features.rul import add_training_rul


NEAR_CONSTANT_DOMINANT_FRACTION = 0.995


def get_sensor_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return the canonical sensor columns that are present
    in the dataframe.
    """
    sensor_columns = [
        column
        for column in CMAPSS_COLUMNS
        if (
            column.startswith("sensor_")
            and column in dataframe.columns
        )
    ]

    if not sensor_columns:
        raise ValueError(
            "No sensor columns were found."
        )

    return sensor_columns


def safe_correlation(
    feature_series: pd.Series,
    target_series: pd.Series,
) -> float:
    """
    Calculate Pearson correlation when the feature has
    enough variation to make correlation meaningful.
    """
    if feature_series.nunique(dropna=False) <= 1:
        return np.nan

    correlation = feature_series.corr(
        target_series
    )

    return float(correlation)


def build_sensor_screening_table(
    development_train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build sensor-level descriptive statistics using only
    development-training engines.

    This table is for screening and interpretation.
    It is not an automatic sensor-removal rule.
    """
    sensor_columns = get_sensor_columns(
        development_train_df
    )

    rows: list[dict[str, object]] = []

    for sensor_column in sensor_columns:
        sensor_series = development_train_df[
            sensor_column
        ]

        value_counts = sensor_series.value_counts(
            normalize=True,
            dropna=False,
        )

        dominant_value_fraction = float(
            value_counts.iloc[0]
        )

        unique_values = int(
            sensor_series.nunique(
                dropna=False
            )
        )

        is_constant = unique_values == 1

        near_constant_candidate = (
            dominant_value_fraction
            >= NEAR_CONSTANT_DOMINANT_FRACTION
        )

        sensor_minimum = float(
            sensor_series.min()
        )

        sensor_maximum = float(
            sensor_series.max()
        )

        rows.append(
            {
                "sensor": sensor_column,
                "mean": float(
                    sensor_series.mean()
                ),
                "standard_deviation": float(
                    sensor_series.std()
                ),
                "variance": float(
                    sensor_series.var()
                ),
                "minimum": sensor_minimum,
                "maximum": sensor_maximum,
                "range": (
                    sensor_maximum
                    - sensor_minimum
                ),
                "unique_values": unique_values,
                "dominant_value_fraction": (
                    dominant_value_fraction
                ),
                "constant": is_constant,
                "near_constant_candidate": (
                    near_constant_candidate
                ),
                "cycle_correlation": (
                    safe_correlation(
                        sensor_series,
                        development_train_df[
                            "cycle"
                        ],
                    )
                ),
                "rul_correlation": (
                    safe_correlation(
                        sensor_series,
                        development_train_df[
                            "rul"
                        ],
                    )
                ),
            }
        )

    screening_df = pd.DataFrame(rows)

    screening_df[
        "absolute_rul_correlation"
    ] = screening_df[
        "rul_correlation"
    ].abs()

    return screening_df


def choose_representative_engines(
    development_train_df: pd.DataFrame,
) -> list[tuple[str, int, int]]:
    """
    Select three development-training engines from
    approximately the 20th, 50th, and 80th percentiles
    of observed engine lifespan.
    """
    engine_lifespans = (
        development_train_df.groupby(
            "engine_id"
        )["cycle"]
        .max()
        .sort_values()
    )

    percentile_specs = [
        ("shorter-life", 0.20),
        ("medium-life", 0.50),
        ("longer-life", 0.80),
    ]

    selected_engines: list[
        tuple[str, int, int]
    ] = []

    used_engine_ids: set[int] = set()

    for label, percentile in percentile_specs:
        target_lifespan = float(
            engine_lifespans.quantile(
                percentile
            )
        )

        distance_from_target = (
            engine_lifespans
            - target_lifespan
        ).abs().sort_values()

        chosen_engine_id: int | None = None

        for engine_id in (
            distance_from_target.index
        ):
            engine_id = int(engine_id)

            if engine_id not in used_engine_ids:
                chosen_engine_id = engine_id
                break

        if chosen_engine_id is None:
            raise ValueError(
                "Could not select three unique "
                "representative engines."
            )

        used_engine_ids.add(
            chosen_engine_id
        )

        chosen_lifespan = int(
            engine_lifespans.loc[
                chosen_engine_id
            ]
        )

        selected_engines.append(
            (
                label,
                chosen_engine_id,
                chosen_lifespan,
            )
        )

    return selected_engines


def choose_diagnostic_sensors(
    screening_df: pd.DataFrame,
) -> list[tuple[str, str]]:
    """
    Choose a small diagnostic set for plotting.

    The purpose is interpretation, not final automatic
    feature selection.
    """
    choices: list[tuple[str, str]] = []

    constant_candidates = (
        screening_df[
            screening_df["constant"]
        ]
        .sort_values(
            "dominant_value_fraction",
            ascending=False,
        )
    )

    if not constant_candidates.empty:
        stable_sensor = str(
            constant_candidates.iloc[0][
                "sensor"
            ]
        )

        choices.append(
            (
                stable_sensor,
                "constant/stable example",
            )
        )
    else:
        stable_sensor = str(
            screening_df.sort_values(
                "dominant_value_fraction",
                ascending=False,
            ).iloc[0]["sensor"]
        )

        choices.append(
            (
                stable_sensor,
                "lowest-information candidate",
            )
        )

    non_constant_df = screening_df[
        ~screening_df["constant"]
    ].copy()

    correlation_available_df = (
        non_constant_df.dropna(
            subset=[
                "absolute_rul_correlation"
            ]
        )
    )

    if not correlation_available_df.empty:
        strongest_sensor = str(
            correlation_available_df.sort_values(
                "absolute_rul_correlation",
                ascending=False,
            ).iloc[0]["sensor"]
        )

        if strongest_sensor not in {
            sensor
            for sensor, _ in choices
        }:
            choices.append(
                (
                    strongest_sensor,
                    "strong RUL-correlation example",
                )
            )

        weakest_sensor = str(
            correlation_available_df.sort_values(
                "absolute_rul_correlation",
                ascending=True,
            ).iloc[0]["sensor"]
        )

        if weakest_sensor not in {
            sensor
            for sensor, _ in choices
        }:
            choices.append(
                (
                    weakest_sensor,
                    "weak linear-correlation example",
                )
            )

    return choices[:3]


def save_sensor_trajectory_plots(
    development_train_df: pd.DataFrame,
    representative_engines: list[
        tuple[str, int, int]
    ],
    diagnostic_sensors: list[
        tuple[str, str]
    ],
) -> list[Path]:
    """
    Plot selected sensor trajectories for three
    representative development-training engines.
    """
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_paths: list[Path] = []

    for sensor_column, reason in (
        diagnostic_sensors
    ):
        plt.figure(
            figsize=(9, 5)
        )

        for (
            life_label,
            engine_id,
            lifespan,
        ) in representative_engines:
            engine_df = (
                development_train_df[
                    development_train_df[
                        "engine_id"
                    ]
                    == engine_id
                ]
                .sort_values("cycle")
            )

            plt.plot(
                engine_df["cycle"],
                engine_df[sensor_column],
                label=(
                    f"{life_label}: "
                    f"engine {engine_id} "
                    f"({lifespan} cycles)"
                ),
            )

        plt.xlabel("Cycle")
        plt.ylabel(sensor_column)

        plt.title(
            f"{sensor_column} trajectory\n"
            f"{reason}"
        )

        plt.legend()
        plt.tight_layout()

        output_path = (
            FIGURES_DIR
            / (
                "sensor_trajectory_"
                f"{sensor_column}.png"
            )
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        saved_paths.append(
            output_path
        )

    return saved_paths


def print_screening_results(
    screening_df: pd.DataFrame,
    representative_engines: list[
        tuple[str, int, int]
    ],
    diagnostic_sensors: list[
        tuple[str, str]
    ],
) -> None:
    """
    Print the most useful Part 1 diagnostic results.
    """
    display_columns = [
        "sensor",
        "variance",
        "range",
        "unique_values",
        "dominant_value_fraction",
        "constant",
        "near_constant_candidate",
        "cycle_correlation",
        "rul_correlation",
    ]

    print(
        "\n======================================"
    )
    print(
        "FD001 SENSOR SCREENING"
    )
    print(
        "DEVELOPMENT-TRAINING ENGINES ONLY"
    )
    print(
        "======================================"
    )

    print(
        "\nAll sensor statistics:"
    )

    print(
        screening_df[
            display_columns
        ]
        .round(6)
        .to_string(index=False)
    )

    constant_sensors = screening_df.loc[
        screening_df["constant"],
        "sensor",
    ].tolist()

    near_constant_sensors = (
        screening_df.loc[
            screening_df[
                "near_constant_candidate"
            ],
            "sensor",
        ].tolist()
    )

    print(
        "\nConstant sensors:"
    )
    print(
        constant_sensors
    )

    print(
        "\nNear-constant candidates:"
    )
    print(
        near_constant_sensors
    )

    print(
        "\nSensors ranked by absolute "
        "RUL correlation:"
    )

    correlation_table = (
        screening_df[
            [
                "sensor",
                "rul_correlation",
                "absolute_rul_correlation",
            ]
        ]
        .dropna()
        .sort_values(
            "absolute_rul_correlation",
            ascending=False,
        )
    )

    print(
        correlation_table
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nRepresentative engines:"
    )

    for (
        label,
        engine_id,
        lifespan,
    ) in representative_engines:
        print(
            f"{label}: "
            f"Engine {engine_id} "
            f"({lifespan} cycles)"
        )

    print(
        "\nDiagnostic sensors plotted:"
    )

    for (
        sensor_column,
        reason,
    ) in diagnostic_sensors:
        print(
            f"{sensor_column}: {reason}"
        )


def main() -> None:
    """
    Run Day 2 Part 1 sensor and degradation
    investigation.
    """
    train_df = load_train_data()

    train_with_rul_df = add_training_rul(
        train_df
    )

    (
        development_train_df,
        development_validation_df,
        _split_table_df,
    ) = get_development_split(
        train_with_rul_df
    )

    train_engine_ids = set(
        development_train_df[
            "engine_id"
        ].unique()
    )

    validation_engine_ids = set(
        development_validation_df[
            "engine_id"
        ].unique()
    )

    engine_overlap = (
        train_engine_ids
        & validation_engine_ids
    )

    if engine_overlap:
        raise ValueError(
            "Development train/validation "
            "engine overlap detected."
        )

    print(
        "\nDevelopment-training shape:"
    )
    print(
        development_train_df.shape
    )

    print(
        "\nDevelopment-validation shape:"
    )
    print(
        development_validation_df.shape
    )

    print(
        "\nEngine overlap:"
    )
    print(
        engine_overlap
    )

    screening_df = (
        build_sensor_screening_table(
            development_train_df
        )
    )

    representative_engines = (
        choose_representative_engines(
            development_train_df
        )
    )

    diagnostic_sensors = (
        choose_diagnostic_sensors(
            screening_df
        )
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    screening_output_path = (
        METRICS_DIR
        / "sensor_variability.csv"
    )

    screening_df.to_csv(
        screening_output_path,
        index=False,
    )

    figure_paths = (
        save_sensor_trajectory_plots(
            development_train_df,
            representative_engines,
            diagnostic_sensors,
        )
    )

    print_screening_results(
        screening_df,
        representative_engines,
        diagnostic_sensors,
    )

    print(
        "\nSensor screening table saved to:"
    )
    print(
        screening_output_path
    )

    print(
        "\nTrajectory figures saved to:"
    )

    for figure_path in figure_paths:
        print(
            figure_path
        )


if __name__ == "__main__":
    main()