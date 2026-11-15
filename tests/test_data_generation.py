"""Unit tests for synthetic data generation module."""
import pytest
import pandas as pd
import numpy as np
import config
from src.data_generation.generator import generate_building_data, generate_all_building_data

def test_generate_building_data():
    b_id = "building_A"
    profile = config.BUILDINGS[b_id]
    df = generate_building_data(b_id, profile, num_days=5, samples_per_day=24)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5 * 24
    
    expected_cols = [
        "timestamp", "building_id", "energy_consumption", "voltage",
        "current", "power", "temperature", "occupancy", "is_anomaly", "anomaly_type"
    ]
    for col in expected_cols:
        assert col in df.columns
        
    assert (df["is_anomaly"].isin([0, 1])).all()
    assert df["energy_consumption"].min() > 0
    assert df["voltage"].mean() > 200

def test_non_iid_profiles():
    # Verify building profiles have distinct average loads
    df_a = generate_building_data("building_A", config.BUILDINGS["building_A"], num_days=10)
    df_b = generate_building_data("building_B", config.BUILDINGS["building_B"], num_days=10)
    df_d = generate_building_data("building_D", config.BUILDINGS["building_D"], num_days=10)
    
    # Lab (D) should have significantly higher baseline than Academic (B)
    assert df_d["energy_consumption"].mean() > df_b["energy_consumption"].mean()

def test_anomaly_injection():
    df = generate_building_data("building_C", config.BUILDINGS["building_C"], num_days=20, anomaly_rate=0.1)
    anom_count = df["is_anomaly"].sum()
    assert anom_count > 0
    
    # Check that anomaly types are properly labeled
    anom_types = set(df[df["is_anomaly"] == 1]["anomaly_type"])
    assert len(anom_types) > 1
