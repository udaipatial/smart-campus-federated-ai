"""
Deep Autoencoder Architecture for Energy Consumption Anomaly Detection.
Supports weight serialization for Federated Averaging (FedAvg) and exact byte-level parameter accounting.
"""

import copy
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import config

class EnergyAutoencoder(nn.Module):
    """
    Symmetric Deep Autoencoder designed for multi-sensor campus edge anomaly detection.
    Compresses input feature vector into a low-dimensional latent bottleneck and reconstructs it.
    Anomalies manifest as elevated reconstruction errors (MSE).
    """
    def __init__(
        self,
        input_dim: int = len(config.FEATURE_COLUMNS),
        latent_dim: int = config.FEDERATED_CONFIG["latent_dim"],
        hidden_dims: List[int] = config.FEDERATED_CONFIG["hidden_dims"]
    ):
        super(EnergyAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build Encoder
        encoder_layers = []
        in_d = input_dim
        for h_d in hidden_dims:
            encoder_layers.append(nn.Linear(in_d, h_d))
            encoder_layers.append(nn.BatchNorm1d(h_d))
            encoder_layers.append(nn.LeakyReLU(0.1))
            in_d = h_d
        encoder_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build Decoder
        decoder_layers = []
        in_d = latent_dim
        for h_d in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_d, h_d))
            decoder_layers.append(nn.BatchNorm1d(h_d))
            decoder_layers.append(nn.LeakyReLU(0.1))
            in_d = h_d
        decoder_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
        
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Returns deep copy of state_dict for federated aggregation."""
        return copy.deepcopy(self.state_dict())
        
    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Loads state_dict into the model."""
        self.load_state_dict(weights)
        
    def count_parameters(self) -> int:
        """Returns total number of trainable and non-trainable model parameters."""
        return sum(p.numel() for p in self.parameters())
        
    def get_model_size_bytes(self) -> int:
        """
        Calculates exact size in bytes of the model state_dict (FP32 tensors = 4 bytes/element).
        """
        total_bytes = 0
        for tensor in self.state_dict().values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes

    def compute_reconstruction_error(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        Computes sample-wise Mean Squared Error (MSE) reconstruction loss for input matrix X.
        """
        self.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_x = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
                recon_x = self.forward(batch_x)
                mse = torch.mean((batch_x - recon_x) ** 2, dim=1).cpu().numpy()
                errors.extend(mse)
        return np.array(errors, dtype=np.float32)

def compute_anomaly_threshold(
    model: EnergyAutoencoder,
    X_val: np.ndarray,
    y_val: np.ndarray = None,
    percentile: float = config.ANOMALY_PERCENTILE_THRESHOLD
) -> float:
    """
    Computes optimal anomaly decision threshold based on reconstruction errors on validation data.
    If ground truth labels are provided, finds the threshold that maximizes validation F1 score;
    otherwise uses the specified percentile.
    """
    val_errors = model.compute_reconstruction_error(X_val)
    
    if y_val is not None and len(np.unique(y_val)) > 1:
        # Calibrate optimal F1 threshold across candidate percentiles
        best_f1 = -1.0
        best_threshold = float(np.percentile(val_errors, percentile))
        
        candidates = np.percentile(val_errors, np.linspace(80, 99.5, 50))
        for cand in candidates:
            preds = (val_errors > cand).astype(int)
            tp = np.sum((preds == 1) & (y_val == 1))
            fp = np.sum((preds == 1) & (y_val == 0))
            fn = np.sum((preds == 0) & (y_val == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = cand
                
        return float(best_threshold)
    else:
        return float(np.percentile(val_errors, percentile))

def predict_anomalies(
    model: EnergyAutoencoder,
    X: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict binary anomaly labels and raw anomaly scores (MSE).
    Returns (predictions, anomaly_scores).
    """
    errors = model.compute_reconstruction_error(X)
    predictions = (errors > threshold).astype(int)
    return predictions, errors
