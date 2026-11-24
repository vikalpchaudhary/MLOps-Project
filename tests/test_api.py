"""API, validation, and logging tests for Stage 3."""

import json
import math

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_TRIP = {
    "route_km": 15.5,
    "stops": 14,
    "passengers": 45,
    "hour": 16,
    "temp_c": 31.2,
    "driver_exp_yr": 8.0,
    "weather": "rain",
    "depot_zone": "North",
}


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_finite_prediction():
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))
    assert math.isfinite(body["prediction"])


def test_predict_rejects_temp_above_70():
    bad_payload = {**VALID_TRIP, "temp_c": 70.1}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_negative_driver_experience():
    bad_payload = {**VALID_TRIP, "driver_exp_yr": -0.5}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_accepts_exact_boundary_values():
    boundary_payload = {**VALID_TRIP, "temp_c": 70.0, "driver_exp_yr": 0.0}
    response = client.post("/predict", json=boundary_payload)
    assert response.status_code == 200
    assert math.isfinite(response.json()["prediction"])


def test_predict_rejects_missing_required_field():
    incomplete = {k: v for k, v in VALID_TRIP.items() if k != "weather"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_logs_valid_json_line_to_stdout(capsys):
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert lines, "Expected at least one JSON log line on stdout"
    record = json.loads(lines[-1])
    assert record["event"] == "prediction"
    assert record["level"] == "INFO"
    assert "prediction" in record
    assert "latency_ms" in record
