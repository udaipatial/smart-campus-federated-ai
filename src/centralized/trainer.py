"""
Centralized Machine Learning Baseline Trainer.
Simulates traditional centralized AI pipeline where all raw campus building data
is transmitted to a single cloud server for monolithic model training.
"""

import time
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.anomaly_detection.models import EnergyAutoencoder, compute_anomaly_threshold, predict_anomalies
from src.preprocessing.pipeline import load_and_preprocess_building_data
import config

def run_centralized_training(
    epochs: int = config.CENTRALIZED_CONFIG["epochs"],
    batch_size: int = config.CENTRALIZED_CONFIG["batch_size"],
    lr: float = config.CENTRALIZED_CONFIG["learning_rate"]
) -> Dict[str, Any]:
    """
    Trains an anomaly detection Autoencoder centrally on aggregated raw data
    from all buildings.
    """
    print("==================================================")
    print("STARTING CENTRALIZED BASELINE TRAINING")
    print("==================================================")
    
    # 1. Pool all raw building datasets onto central server
    all_train_x = []
    all_val_x = []
    all_val_y = []
    all_test_x = []
    all_test_y = []
    
    total_raw_bytes_transmitted = 0
    
    for b_id in config.BUILDINGS.keys():
        b_data = load_and_preprocess_building_data(b_id)
        all_train_x.append(b_data["train"]["X"])
        all_val_x.append(b_data["val"]["X"])
        all_val_y.append(b_data["val"]["y"])
        all_test_x.append(b_data["test"]["X"])
        all_test_y.append(b_data["test"]["y"])
        
        # Calculate raw data communication cost (CSV file size transmitted over network)
        raw_file = config.DATA_DIR / f"{b_id}.csv"
        if raw_file.exists():
            total_raw_bytes_transmitted += os.path.getsize(raw_file)
            
    X_train_pool = np.vstack(all_train_x)
    X_val_pool = np.vstack(all_val_x)
    y_val_pool = np.concatenate(all_val_y)
    X_test_pool = np.vstack(all_test_x)
    y_test_pool = np.concatenate(all_test_y)
    
    print(f"Aggregated Centralized Dataset: {len(X_train_pool)} train samples, {len(X_val_pool)} val samples, {len(X_test_pool)} test samples.")
    print(f"Raw data uploaded to central server: {total_raw_bytes_transmitted:,} bytes ({total_raw_bytes_transmitted / (1024*1024):.2f} MB)")
    
    # Setup PyTorch DataLoader
    train_tensor = torch.tensor(X_train_pool, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
    
    # 2. Instantiate and train Centralized Autoencoder
    model = EnergyAutoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    start_train_time = time.perf_counter()
    loss_history = []
    
    for ep in range(epochs):
        model.train()
        batch_losses = []
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_loss = np.mean(batch_losses)
        loss_history.append(avg_loss)
        
        if (ep + 1) % 15 == 0 or ep == 0:
            print(f"Centralized Epoch {ep+1:02d}/{epochs:02d} | Training Loss (MSE): {avg_loss:.4f}")
            
    total_train_duration = time.perf_counter() - start_train_time
    
    # 3. Calibrate Anomaly Threshold on Pooled Validation Set
    threshold = compute_anomaly_threshold(model, X_val_pool, y_val_pool)
    
    # 4. Evaluate Centralized Model on Pooled Test Set
    start_inf = time.perf_counter()
    preds, errors = predict_anomalies(model, X_test_pool, threshold)
    inf_latency_per_sample = (time.perf_counter() - start_inf) / len(X_test_pool)
    
    tp = np.sum((preds == 1) & (y_test_pool == 1))
    fp = np.sum((preds == 1) & (y_test_pool == 0))
    fn = np.sum((preds == 0) & (y_test_pool == 1))
    tn = np.sum((preds == 0) & (y_test_pool == 0))
    
    accuracy = (tp + tn) / len(y_test_pool) if len(y_test_pool) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 5. Save Centralized Model
    model_save_path = config.MODELS_DIR / "centralized.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
        "epochs": epochs,
        "f1": f1,
        "accuracy": accuracy
    }, model_save_path)
    
    print("\nCentralized Baseline Training Complete!")
    print(f"Accuracy: {accuracy*100:.2f}% | Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1-Score: {f1*100:.2f}%")
    print(f"Total Training Time: {total_train_duration:.2f} seconds | Inference Latency: {inf_latency_per_sample*1000:.3f} ms/sample\n")
    
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "training_time_sec": total_train_duration,
        "inference_latency_ms": inf_latency_per_sample * 1000.0,
        "communication_bytes": total_raw_bytes_transmitted,
        "communication_mb": total_raw_bytes_transmitted / (1024 * 1024),
        "threshold": threshold,
        "total_test_samples": len(y_test_pool),
        "total_ground_truth_anomalies": int(np.sum(y_test_pool == 1)),
        "total_detected_anomalies": int(np.sum(preds == 1)),
        "loss_history": loss_history
    }
    
    return results

if __name__ == "__main__":
    run_centralized_training()
