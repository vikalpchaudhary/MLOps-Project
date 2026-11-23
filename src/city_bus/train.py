"""Run a tracked ridge regression experiment from the repository root."""

import argparse
import hashlib
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_params
from .data import clean_data


def main(alpha=None):
    params = load_params()
    if alpha is None:
        alpha = params["ridge_experiment_alpha"]
    train_path = Path(params["train_path"])
    data = clean_data(pd.read_csv(train_path))
    validation_start = params["validation_start"]
    train = data.loc[data["date"] < validation_start]
    validation = data.loc[data["date"] >= validation_start]
    numeric = ["route_km", "stops", "passengers", "hour", "temp_c", "driver_exp_yr"]
    categorical = ["weather", "depot_zone"]
    features = numeric + categorical
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = make_pipeline(preprocessing, Ridge(alpha=alpha))
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(params["experiment_name"])
    with mlflow.start_run(run_name=f"ridge-alpha-{alpha:g}") as run:
        mlflow.log_params({
            "model": "Ridge",
            "alpha": alpha,
            "validation_start": validation_start,
            "data_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        })
        model.fit(train[features], train["delay_min"])
        predictions = model.predict(validation[features])
        mae = mean_absolute_error(validation["delay_min"], predictions)
        mlflow.log_metric("validation_mae", mae)
        print(f"{run.info.run_id}: validation MAE = {mae:.4f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, help="Override the configured ridge regularization strength")
    main(parser.parse_args().alpha)
