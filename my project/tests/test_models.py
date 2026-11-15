"""Unit tests for EnergyAutoencoder model and threshold calibration."""
import pytest
import torch
import numpy as np
import config
from src.anomaly_detection.models import (
    EnergyAutoencoder,
    compute_anomaly_threshold,
    predict_anomalies
)

def test_energy_autoencoder_architecture():
    input_dim = len(config.FEATURE_COLUMNS)
    model = EnergyAutoencoder(input_dim=input_dim, latent_dim=8, hidden_dims=[32, 16])
    
    dummy_input = torch.randn(16, input_dim)
    output = model(dummy_input)
    
    assert output.shape == (16, input_dim)
    assert model.count_parameters() > 0
    assert model.get_model_size_bytes() > 0

def test_weights_serialization():
    model1 = EnergyAutoencoder()
    model2 = EnergyAutoencoder()
    
    # Verify different initial weights
    weights1 = model1.get_weights()
    model2.set_weights(weights1)
    
    for k in weights1:
        assert torch.equal(model1.state_dict()[k], model2.state_dict()[k])

def test_anomaly_prediction():
    model = EnergyAutoencoder()
    X = np.random.randn(50, len(config.FEATURE_COLUMNS)).astype(np.float32)
    threshold = 1.0
    
    preds, errors = predict_anomalies(model, X, threshold)
    assert len(preds) == 50
    assert len(errors) == 50
    assert set(preds).issubset({0, 1})
    assert (errors >= 0).all()
