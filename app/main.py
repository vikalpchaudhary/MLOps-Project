"""FastAPI prediction service for MP187 — City Bus Delay (Kochi)."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.city_bus.config import load_params

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ORDER = [
    "route_km",
    "stops",
    "passengers",
    "hour",
    "temp_c",
    "driver_exp_yr",
    "weather",
    "depot_zone",
]


class JsonFormatter(logging.Formatter):
    """Format log records as a single JSON line for stdout."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        return json.dumps(payload)


class StdoutStreamHandler(logging.StreamHandler):
    """StreamHandler that always emits to the active sys.stdout."""

    def __init__(self) -> None:
        super().__init__(sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def _configure_json_logger() -> logging.Logger:
    logger = logging.getLogger("mp187.api")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = StdoutStreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


logger = _configure_json_logger()


class TripFeatures(BaseModel):
    """Request schema for POST /predict with client validation rules."""

    route_km: float = Field(..., description="Length of the bus route in km")
    stops: int = Field(..., description="Number of stops on the route")
    passengers: int = Field(..., description="Passengers on board at trip start")
    hour: int = Field(..., description="Trip start hour (24-hour clock)")
    temp_c: float = Field(
        ...,
        le=70.0,
        description="Air temperature in deg C (values above 70 are impossible)",
    )
    driver_exp_yr: float = Field(
        ...,
        ge=0.0,
        description="Driver experience in years (values below 0 are invalid)",
    )
    weather: str = Field(..., description="Weather at trip start (clear, rain, fog)")
    depot_zone: str = Field(..., description="Depot zone (North, South, East)")


class PredictionResponse(BaseModel):
    """Response payload returned by POST /predict."""

    prediction: float


@lru_cache(maxsize=1)
def load_model() -> Any:
    """Load the champion model artifact configured in params.yaml."""
    params_path = REPO_ROOT / "params.yaml"
    if params_path.exists():
        params = load_params() if Path("params.yaml").exists() else None
    else:
        params = None

    rel_model_path = (
        params["model_path"]
        if params and "model_path" in params
        else "models/model.joblib"
    )
    model_path = REPO_ROOT / rel_model_path
    return joblib.load(model_path)


app = FastAPI(
    title="MP187 Kochi City Bus Delay API",
    description="Predict bus arrival delay (delay_min) at trip start for Kochi City Bus Service.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness and readiness probe endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripFeatures) -> PredictionResponse:
    """Predict trip delay in minutes and emit one structured JSON log line to stdout."""
    start = time.perf_counter()
    model = load_model()

    row = pd.DataFrame([trip.model_dump()], columns=FEATURE_ORDER)
    predicted_delay = float(model.predict(row)[0])
    latency_ms = round((time.perf_counter() - start) * 1000.0, 3)

    logger.info(
        "prediction_served",
        extra={
            "extra_fields": {
                "event": "prediction",
                "depot_zone": trip.depot_zone,
                "weather": trip.weather,
                "route_km": trip.route_km,
                "temp_c": trip.temp_c,
                "prediction": round(predicted_delay, 4),
                "latency_ms": latency_ms,
            }
        },
    )

    return PredictionResponse(prediction=round(predicted_delay, 4))
