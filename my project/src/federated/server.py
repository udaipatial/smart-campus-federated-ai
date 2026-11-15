"""
Central Federated Server implementing Federated Averaging (FedAvg).
Aggregates local neural weights, tracks precise byte-level communication overhead,
and manages global model distribution.
"""

import copy
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from src.anomaly_detection.models import EnergyAutoencoder, compute_anomaly_threshold, predict_anomalies
import config

class FederatedServer:
    """
    Central server orchestrating the federated aggregation round cycles.
    """
    def __init__(self):
        self.global_model = EnergyAutoencoder()
        self.current_round = 0
        self.global_threshold = None
        
        # Communication tracking
        self.model_size_bytes = self.global_model.get_model_size_bytes()
        self.total_bytes_uplink = 0      # Client -> Server
        self.total_bytes_downlink = 0    # Server -> Client
        self.round_history = []
        
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Returns the current global model weights."""
        return self.global_model.get_weights()
        
    def aggregate_fedavg(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Federated Averaging (FedAvg) algorithm:
        w_{global} = sum_{k=1}^K (n_k / N) * w_k
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation.")
            
        start_agg_time = time.perf_counter()
        
        total_samples = sum(update["num_samples"] for update in client_updates)
        num_participating = len(client_updates)
        
        # Initialize empty state_dict with zero tensors matching model architecture
        first_weights = client_updates[0]["weights"]
        averaged_weights = {}
        for key in first_weights.keys():
            averaged_weights[key] = torch.zeros_like(first_weights[key], dtype=torch.float32)
            
        # Weighted parameter aggregation
        for update in client_updates:
            w_k = update["weights"]
            n_k = update["num_samples"]
            weight_factor = n_k / total_samples
            
            for key in averaged_weights.keys():
                # Convert to float for safe numerical accumulation
                averaged_weights[key] += w_k[key].float() * weight_factor
                
        # Cast back to original types if needed (e.g. LongTensor for num_batches_tracked)
        for key in averaged_weights.keys():
            if first_weights[key].dtype in (torch.int64, torch.long):
                averaged_weights[key] = averaged_weights[key].long()
                
        # Update server global model
        self.global_model.set_weights(averaged_weights)
        aggregation_latency = time.perf_counter() - start_agg_time
        
        # Calculate Communication Overhead for this round
        # Uplink: All participating clients send their model weights to server
        round_uplink_bytes = num_participating * self.model_size_bytes
        # Downlink: Server distributes global weights to all participating clients
        round_downlink_bytes = num_participating * self.model_size_bytes
        round_total_bytes = round_uplink_bytes + round_downlink_bytes
        
        self.total_bytes_uplink += round_uplink_bytes
        self.total_bytes_downlink += round_downlink_bytes
        self.current_round += 1
        
        round_telemetry = {
            "round": self.current_round,
            "participating_clients": num_participating,
            "total_samples": total_samples,
            "aggregation_time_sec": aggregation_latency,
            "round_uplink_bytes": round_uplink_bytes,
            "round_downlink_bytes": round_downlink_bytes,
            "round_total_bytes": round_total_bytes,
            "cumulative_bytes": self.total_bytes_uplink + self.total_bytes_downlink,
            "cumulative_mb": (self.total_bytes_uplink + self.total_bytes_downlink) / (1024 * 1024)
        }
        
        return round_telemetry

    def evaluate_global_model(
        self,
        test_partitions: List[Tuple[np.ndarray, np.ndarray]],
        val_partitions: List[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Evaluates the aggregated global model across all edge test sets.
        """
        # Calibrate global threshold across pooled validation sets if available
        if val_partitions:
            pooled_val_x = np.vstack([vx for vx, _ in val_partitions])
            pooled_val_y = np.concatenate([vy for _, vy in val_partitions])
            self.global_threshold = compute_anomaly_threshold(
                self.global_model,
                pooled_val_x,
                pooled_val_y
            )
            
        pooled_test_x = np.vstack([tx for tx, _ in test_partitions])
        pooled_test_y = np.concatenate([ty for _, ty in test_partitions])
        
        start_inf = time.perf_counter()
        preds, errors = predict_anomalies(self.global_model, pooled_test_x, self.global_threshold)
        inf_latency_per_sample = (time.perf_counter() - start_inf) / len(pooled_test_x)
        
        tp = np.sum((preds == 1) & (pooled_test_y == 1))
        fp = np.sum((preds == 1) & (pooled_test_y == 0))
        fn = np.sum((preds == 0) & (pooled_test_y == 1))
        tn = np.sum((preds == 0) & (pooled_test_y == 0))
        
        accuracy = (tp + tn) / len(pooled_test_y) if len(pooled_test_y) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "inference_latency_ms": inf_latency_per_sample * 1000.0,
            "threshold": self.global_threshold,
            "total_test_samples": len(pooled_test_y),
            "total_ground_truth_anomalies": int(np.sum(pooled_test_y == 1)),
            "total_detected_anomalies": int(np.sum(preds == 1))
        }

    def save_model(self, file_path=config.MODELS_DIR / "federated_global.pt"):
        """Saves global model weights and metadata."""
        torch.save({
            "model_state_dict": self.global_model.state_dict(),
            "threshold": self.global_threshold,
            "current_round": self.current_round,
            "model_size_bytes": self.model_size_bytes,
            "total_bytes_communicated": self.total_bytes_uplink + self.total_bytes_downlink
        }, file_path)
