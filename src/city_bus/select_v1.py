"""Compare candidate regressors and register the first production model."""

import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import load_params
from .data import clean_data


def main():
    params = load_params()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(params["experiment_name"])
    client = MlflowClient()
    name = params["registered_model_name"]
    if client.search_model_versions(f"name='{name}'"):
        raise SystemExit("Version 1 already exists. Use the retraining workflow for later versions.")

    train_path = Path(params["train_path"])
    data = clean_data(pd.read_csv(train_path))
    numeric = ["route_km", "stops", "passengers", "hour", "temp_c", "driver_exp_yr"]
    categorical = ["weather", "depot_zone"]
    x = data[numeric + categorical]
    y = data["delay_min"]
    cutoff = params["validation_start"]
    train = data["date"] < cutoff
    digest = hashlib.sha256(train_path.read_bytes()).hexdigest()
    results = []
    candidates = params["candidates"]
    for estimator in (
        Ridge(**candidates["ridge"]),
        HuberRegressor(**candidates["huber"]),
        GradientBoostingRegressor(**candidates["gradient_boosting"]),
    ):
        transform = ColumnTransformer([
            ("numeric", StandardScaler(), numeric),
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ])
        pipeline = make_pipeline(transform, estimator)
        with mlflow.start_run(run_name=type(estimator).__name__) as run:
            mlflow.log_params(estimator.get_params())
            mlflow.log_params(dict(model=type(estimator).__name__, data_sha256=digest, validation_start=cutoff))
            pipeline.fit(x[train], y[train])
            mae = mean_absolute_error(y[~train], pipeline.predict(x[~train]))
            mlflow.log_metric("validation_mae", mae)
            results.append((mae, run.info.run_id, pipeline))
            print(type(estimator).__name__, mae)

    _, winner, model = min(results, key=lambda result: result[0])
    model.fit(x, y)
    sample = x.iloc[:10]
    with mlflow.start_run(run_id=winner):
        mlflow.log_param("final_fit_rows", len(x))
        info = mlflow.sklearn.log_model(model, "model", input_example=sample)
    version = mlflow.register_model(info.model_uri, name)
    restored = mlflow.sklearn.load_model(f"models:/{name}/{version.version}")
    np.testing.assert_allclose(restored.predict(sample), model.predict(sample))
    path = Path(params["model_path"])
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(sample), restored.predict(sample))
    client.set_registered_model_alias(name, "champion", version.version)
    assert client.get_model_version_by_alias(name, "champion").run_id == winner
    print("Champion", name, version.version, winner)


if __name__ == "__main__":
    main()
