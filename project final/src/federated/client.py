"""
Simulated Edge Client representing a Smart Campus Building.
Performs strictly local on-device neural training without transmitting raw energy data.
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.anomaly_detection.models import EnergyAutoencoder, compute_anomaly_threshold, predict_anomalies
from src.preprocessing.pipeline import load_and_preprocess_building_data
import config

class FederatedClient:
    """
    Edge computing node attached to a smart campus building.
    Manages local data partitioning, edge model training, and privacy preservation.
    """
    def __init__(
        self,
        building_id: str,
        learning_rate: float = config.FEDERATED_CONFIG["learning_rate"],
        batch_size: int = config.FEDERATED_CONFIG["batch_size"]
    ):
        self.client_id = building_id
        self.building_name = config.BUILDINGS.get(building_id, {}).get("name", building_id)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Load local building dataset (STRICTLY PRIVATE TO THIS CLIENT)
        self._data = load_and_preprocess_building_data(building_id)
        self.num_train_samples = len(self._data["train"]["X"])
        
        # Instantiate local model instance
        self.model = EnergyAutoencoder()
        self.criterion = nn.MSELoss()
        
        # Local threshold for anomaly detection
        self.threshold = None
        
        # Setup local training dataloader
        train_x = torch.tensor(self._data["train"]["X"], dtype=torch.float32)
        train_ds = TensorDataset(train_x, train_x)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
    def get_num_samples(self) -> int:
        """Returns the number of local training samples."""
        return self.num_train_samples

    def train(
        self, 
        global_weights: Dict[str, torch.Tensor], 
        epochs: int = config.FEDERATED_CONFIG["local_epochs"]
    ) -> Dict[str, Any]:
        """
        Performs local edge model training starting from global weights.
        Returns model parameters and telemetry (NO RAW DATA).
        """
        # Load global model parameters into local model
        self.model.set_weights(global_weights)
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        start_time = time.perf_counter()
        epoch_losses = []
        
        for epoch in range(epochs):
            batch_losses = []
            for batch_x, _ in self.train_loader:
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(np.mean(batch_losses))
            
        elapsed_time = time.perf_counter() - start_time
        
        # Calculate local model size for verification
        model_size_bytes = self.model.get_model_size_bytes()
        
        # Calibrate edge anomaly threshold on local validation partition
        self.threshold = compute_anomaly_threshold(
            self.model,
            self._data["val"]["X"],
            self._data["val"]["y"]
        )
        
        # Return strictly model weights and telemetry
        return {
            "client_id": self.client_id,
            "weights": self.model.get_weights(),
            "num_samples": self.num_train_samples,
            "train_loss": float(np.mean(epoch_losses)),
            "train_time_sec": elapsed_time,
            "model_size_bytes": model_size_bytes,
            "threshold": self.threshold
        }

    def evaluate_local(self, weights: Dict[str, torch.Tensor] = None) -> Dict[str, float]:
        """
        Evaluates local model (or given global weights) on local test data.
        """
        if weights is not None:
            self.model.set_weights(weights)
            
        if self.threshold is None:
            self.threshold = compute_anomaly_threshold(
                self.model,
                self._data["val"]["X"],
                self._data["val"]["y"]
            )
            
        X_test = self._data["test"]["X"]
        y_test = self._data["test"]["y"]
        
        start_eval = time.perf_counter()
        preds, errors = predict_anomalies(self.model, X_test, self.threshold)
        eval_latency = (time.perf_counter() - start_eval) / len(X_test)
        
        tp = np.sum((preds == 1) & (y_test == 1))
        fp = np.sum((preds == 1) & (y_test == 0))
        fn = np.sum((preds == 0) & (y_test == 1))
        tn = np.sum((preds == 0) & (y_test == 0))
        
        accuracy = (tp + tn) / len(y_test) if len(y_test) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "client_id": self.client_id,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "test_samples": len(y_test),
            "anomaly_samples": int(np.sum(y_test == 1)),
            "detected_anomalies": int(np.sum(preds == 1)),
            "inference_latency_ms": eval_latency * 1000.0
        }
        
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns local test features and labels for benchmark aggregation."""
        return self._data["test"]["X"], self._data["test"]["y"]
        
    def get_val_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns local validation features and labels."""
        return self._data["val"]["X"], self._data["val"]["y"]
