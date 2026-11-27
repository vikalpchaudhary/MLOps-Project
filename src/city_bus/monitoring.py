"""Stage 4 monitoring for the MP187 Kochi City Bus Delay project.

Run from the repository root after fetching the DVC data and confirming that
models/model.joblib (or the path in params.yaml) contains the v1 champion:

    python -m src.city_bus.monitoring

The script computes PSI for the six numeric predictor features and v1 MAE for
North, South, and East on the complete later.csv dataset. It deliberately does
not retrain a model, change the MLflow champion alias, or make holdout
predictions; those tasks are reserved for the teammate handling the second
half of Stage 4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

TRAIN_PATH = Path("data/raw/train.csv")
LATER_PATH = Path("data/raw/later.csv")
OUTPUT_PATH = Path("reports/monitoring_results.json")

NUMERIC_FEATURES = [
    "route_km",
    "stops",
    "passengers",
    "hour",
    "temp_c",
    "driver_exp_yr",
]
CATEGORICAL_FEATURES = ["weather", "depot_zone"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPECTED_DEPOTS = ("North", "South", "East")


def _as_finite_numeric(values: pd.Series, column_name: str) -> np.ndarray:
    """Convert a series to finite floats, rejecting an entirely empty column."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    array = numeric.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError(f"Column '{column_name}' has no finite numeric values.")
    return array


def calculate_psi(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI using quantile cut-points from the reference dataset.

    The training/reference data defines the bin boundaries. The same fixed
    boundaries are then applied to the later/current data, so the percentages
    are comparable. Infinite outer boundaries keep new extreme values inside
    the first or last bin. A small epsilon prevents log(0) when a bin is empty.
    """
    if bins < 1:
        raise ValueError("bins must be at least 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than 0")

    reference_values = _as_finite_numeric(reference, "reference")
    current_values = _as_finite_numeric(current, "current")

    # Use the internal reference quantiles as fixed cut-points. np.unique
    # handles tied quantiles (which are common in small or discrete datasets).
    quantile_levels = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    internal_edges = (
        np.quantile(reference_values, quantile_levels)
        if quantile_levels.size
        else np.array([], dtype=float)
    )
    edges = np.unique(
        np.concatenate((np.array([-np.inf]), internal_edges, np.array([np.inf])))
    )

    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    reference_counts = reference_counts.astype(float)
    current_counts = current_counts.astype(float)

    reference_pct = (reference_counts + epsilon) / (
        reference_counts.sum() + epsilon * len(reference_counts)
    )
    current_pct = (current_counts + epsilon) / (
        current_counts.sum() + epsilon * len(current_counts)
    )

    psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))
    return float(psi)


def calculate_monitoring_metrics(
    cleaned_train: pd.DataFrame,
    cleaned_later: pd.DataFrame,
    v1_model: Any,
    bins: int = 10,
) -> dict[str, Any]:
    """Return feature PSI and per-depot MAE for the existing v1 model."""
    required_columns = set(FEATURE_COLUMNS + ["delay_min", "depot_zone"])
    missing_train = required_columns.difference(cleaned_train.columns)
    missing_later = required_columns.difference(cleaned_later.columns)
    if missing_train:
        raise ValueError(f"Training data is missing columns: {sorted(missing_train)}")
    if missing_later:
        raise ValueError(f"Later data is missing columns: {sorted(missing_later)}")

    psi_by_feature = {
        column: calculate_psi(cleaned_train[column], cleaned_later[column], bins=bins)
        for column in NUMERIC_FEATURES
    }
    most_drifted_feature = max(psi_by_feature, key=psi_by_feature.get)

    predictions = np.asarray(v1_model.predict(cleaned_later[FEATURE_COLUMNS]), dtype=float)
    if predictions.ndim != 1 or len(predictions) != len(cleaned_later):
        raise ValueError("The v1 model must return exactly one prediction per later.csv row.")

    scored_later = cleaned_later[["depot_zone", "delay_min"]].copy()
    scored_later["_prediction"] = predictions

    mae_by_group: dict[str, float] = {}
    for depot in EXPECTED_DEPOTS:
        group = scored_later.loc[scored_later["depot_zone"] == depot]
        if group.empty:
            raise ValueError(
                f"No rows found for depot '{depot}' in later.csv; expected all three depots."
            )
        mae_by_group[depot] = float(
            mean_absolute_error(group["delay_min"], group["_prediction"])
        )

    worst_served_group = max(mae_by_group, key=mae_by_group.get)

    return {
        "psi": psi_by_feature,
        "most_drifted_feature": most_drifted_feature,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served_group,
    }


def run_monitoring() -> dict[str, Any]:
    """Load this project's own data and v1 artifact, then run the monitoring."""
    # Local imports keep the pure metric functions importable in unit tests
    # without forcing tests to provide repository data/config files.
    from .config import load_params
    from .data import clean_data

    params = load_params()
    train_path = Path(params.get("train_path", TRAIN_PATH))
    later_path = LATER_PATH
    model_path = Path(params.get("model_path", "models/model.joblib"))

    for required_path in (train_path, later_path, model_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}. Fetch the project data "
                "with DVC and make sure the v1 model artifact exists before running."
            )

    # The training file contains known data-quality issues and needs the full
    # training clean-up. later.csv is already clean; only drop the leakage column.
    cleaned_train = clean_data(pd.read_csv(train_path), is_train=True)
    cleaned_later = clean_data(pd.read_csv(later_path), is_train=False)
    v1_model = joblib.load(model_path)

    results = calculate_monitoring_metrics(cleaned_train, cleaned_later, v1_model, bins=10)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nSaved monitoring metrics to: {OUTPUT_PATH}")
    print("Copy these four result fields into reports/findings.yaml -> monitoring:")
    print("  psi, most_drifted_feature, mae_by_group, worst_served_group")
    return results


if __name__ == "__main__":
    run_monitoring()
