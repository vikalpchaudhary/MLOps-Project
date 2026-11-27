"""Unit tests for Stage 4 PSI and depot-level MAE calculations."""

import numpy as np
import pandas as pd

from src.city_bus.monitoring import (
    FEATURE_COLUMNS,
    calculate_monitoring_metrics,
    calculate_psi,
)


class ZeroPredictionModel:
    """Small deterministic stand-in so the tests need no model artifact."""

    def predict(self, frame):
        return np.zeros(len(frame), dtype=float)


def test_psi_is_zero_for_identical_distributions():
    values = pd.Series(np.arange(100, dtype=float))
    assert abs(calculate_psi(values, values.copy(), bins=10)) < 1e-10


def test_psi_increases_when_later_distribution_shifts():
    reference = pd.Series(np.linspace(0, 10, 200))
    current = pd.Series(np.linspace(20, 30, 200))
    assert calculate_psi(reference, current, bins=10) > 0.5


def test_monitoring_returns_six_psi_values_and_three_depot_maes():
    rng = np.random.default_rng(46)
    rows = []
    for depot, delay in (("North", 4.0), ("South", 7.0), ("East", 2.0)):
        for _ in range(20):
            rows.append(
                {
                    "route_km": rng.uniform(2, 20),
                    "stops": int(rng.integers(3, 20)),
                    "passengers": int(rng.integers(5, 60)),
                    "hour": int(rng.integers(5, 23)),
                    "temp_c": rng.uniform(20, 35),
                    "driver_exp_yr": rng.uniform(1, 20),
                    "weather": "clear",
                    "depot_zone": depot,
                    "delay_min": delay,
                }
            )

    later = pd.DataFrame(rows)
    train = later.copy()
    results = calculate_monitoring_metrics(train, later, ZeroPredictionModel(), bins=10)

    assert set(results["psi"]) == {
        "route_km", "stops", "passengers", "hour", "temp_c", "driver_exp_yr"
    }
    assert set(results["mae_by_group"]) == {"North", "South", "East"}
    assert results["worst_served_group"] == "South"
    assert results["mae_by_group"]["South"] == 7.0
    assert len(FEATURE_COLUMNS) == 8
