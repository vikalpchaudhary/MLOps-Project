from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.city_bus.data import clean_data


def test_training_repairs_and_deduplicates_records():
    raw = pd.DataFrame({
        "depot_zone": ["North", "North", "South"],
        "driver_exp_yr": [36., 36., 8.],
        "delay_min": [120., 120., 18.],
        "delay_logged_by_conductor": [12., 12., 18.],
    })
    expected = pd.DataFrame({
        "depot_zone": ["North", "South"],
        "driver_exp_yr": [3., 8.],
        "delay_min": [12., 18.],
    })
    pd.testing.assert_frame_equal(clean_data(raw).reset_index(drop=True), expected)


def test_clean_later_trips_keep_units_targets_and_row_count():
    later = pd.DataFrame({
        "depot_zone": ["North", "North"],
        "driver_exp_yr": [4., 4.],
        "delay_min": [55., 55.],
        "delay_logged_by_conductor": [54., 54.],
    })
    expected = later.drop(columns="delay_logged_by_conductor")
    pd.testing.assert_frame_equal(clean_data(later, is_train=False), expected)


def test_saved_model_predicts_from_trip_start_fields():
    trips = pd.DataFrame({
        "route_km": [12., 22.], "stops": [9, 14], "passengers": [25, 40],
        "hour": [8, 17], "temp_c": [29., 32.], "driver_exp_yr": [3., 7.],
        "weather": ["clear", "rain"], "depot_zone": ["North", "East"],
    })
    model = joblib.load(Path(__file__).resolve().parents[1] / "models" / "model.joblib")
    predictions = model.predict(trips)
    assert predictions.shape == (len(trips),)
    assert np.isfinite(predictions).all()
