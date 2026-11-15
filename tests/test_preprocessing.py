"""Unit tests for preprocessing and feature engineering pipeline."""
import pytest
import numpy as np
import pandas as pd
import config
from src.data_generation.generator import generate_building_data
from src.preprocessing.pipeline import (
    engineer_features,
    preprocess_dataframe,
    load_and_preprocess_building_data,
    create_pytorch_dataloaders,
    FeatureScaler
)

def test_engineer_features():
    df = generate_building_data("building_A", config.BUILDINGS["building_A"], num_days=3)
    df_feat = engineer_features(df)
    
    assert "hour_sin" in df_feat.columns
    assert "hour_cos" in df_feat.columns
    assert "day_sin" in df_feat.columns
    assert "day_cos" in df_feat.columns
    
    # Check sin/cos ranges
    assert df_feat["hour_sin"].between(-1.0, 1.0).all()
    assert df_feat["hour_cos"].between(-1.0, 1.0).all()

def test_feature_scaler():
    data = np.random.randn(100, 5) * 50 + 100
    scaler = FeatureScaler()
    scaled = scaler.fit_transform(data)
    
    np.testing.assert_almost_equal(scaled.mean(axis=0), np.zeros(5), decimal=1)
    np.testing.assert_almost_equal(scaled.std(axis=0), np.ones(5), decimal=1)

def test_pytorch_dataloaders():
    X_train = np.random.randn(128, len(config.FEATURE_COLUMNS)).astype(np.float32)
    train_loader, _, _ = create_pytorch_dataloaders(X_train, batch_size=32)
    
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (32, len(config.FEATURE_COLUMNS))
    assert batch_y.shape == (32, len(config.FEATURE_COLUMNS))
