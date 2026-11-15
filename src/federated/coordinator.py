"""
Federated Learning Coordinator.
Orchestrates multi-round federated training across all simulated edge clients.
"""

import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .client import FederatedClient
from .server import FederatedServer
import config

def run_federated_learning(
    num_rounds: int = config.FEDERATED_CONFIG["num_rounds"],
    local_epochs: int = config.FEDERATED_CONFIG["local_epochs"]
) -> Dict[str, Any]:
    """
    Executes end-to-end Federated Averaging training across all campus edge nodes.
    """
    print("==================================================")
    print("STARTING FEDERATED LEARNING PIPELINE (FedAvg)")
    print("==================================================")
    
    # 1. Initialize Edge Clients (one per building)
    clients: List[FederatedClient] = []
    for b_id in config.BUILDINGS.keys():
        client = FederatedClient(b_id)
        clients.append(client)
        print(f"Edge Node initialized: {b_id} ({client.building_name}) with {client.get_num_samples()} local samples.")
        
    # 2. Initialize Central Federated Server
    server = FederatedServer()
    print(f"Federated Server initialized. Model size: {server.model_size_bytes:,} bytes ({server.model_size_bytes/1024:.2f} KB).")
    print(f"Total participating clients: {len(clients)}")
    print(f"Federated rounds: {num_rounds}, Local epochs per round: {local_epochs}\n")
    
    # Pre-extract validation and test splits for evaluation
    val_partitions = [c.get_val_data() for c in clients]
    test_partitions = [c.get_test_data() for c in clients]
    
    round_records = []
    total_training_time = 0.0
    
    start_fl_time = time.perf_counter()
    
    for r in range(1, num_rounds + 1):
        round_start = time.perf_counter()
        
        # Server broadcasts current global model parameters to clients
        current_global_weights = server.get_global_weights()
        
        # Clients perform strictly local on-device training in parallel/sequence
        client_updates = []
        client_train_times = []
        client_losses = []
        
        for client in clients:
            update = client.train(current_global_weights, epochs=local_epochs)
            client_updates.append(update)
            client_train_times.append(update["train_time_sec"])
            client_losses.append(update["train_loss"])
            
        # In edge simulation, parallel round time is bounded by slowest node;
        # total compute work is sum of edge training
        parallel_client_time = max(client_train_times)
        
        # Server performs Federated Averaging (FedAvg)
        telemetry = server.aggregate_fedavg(client_updates)
        
        round_duration = parallel_client_time + telemetry["aggregation_time_sec"]
        total_training_time += round_duration
        
        # Evaluate global model performance on global test set
        eval_metrics = server.evaluate_global_model(test_partitions, val_partitions)
        
        record = {
            "round": r,
            "mean_local_loss": float(np.mean(client_losses)),
            "accuracy": eval_metrics["accuracy"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1": eval_metrics["f1"],
            "round_duration_sec": round_duration,
            "aggregation_time_sec": telemetry["aggregation_time_sec"],
            "round_comm_kb": telemetry["round_total_bytes"] / 1024.0,
            "cumulative_comm_mb": telemetry["cumulative_mb"],
            "inference_latency_ms": eval_metrics["inference_latency_ms"]
        }
        round_records.append(record)
        
        print(f"Round {r:02d}/{num_rounds:02d} | Loss: {record['mean_local_loss']:.4f} | "
              f"Acc: {record['accuracy']*100:.2f}% | F1: {record['f1']*100:.2f}% | "
              f"Comm: {record['cumulative_comm_mb']:.3f} MB | Time: {round_duration:.2f}s")
              
    fl_elapsed_wall_clock = time.perf_counter() - start_fl_time
    
    # Save the final aggregated global model
    server.save_model()
    
    # Final comprehensive evaluation per edge client
    per_client_eval = {}
    for client in clients:
        per_client_eval[client.client_id] = client.evaluate_local(server.get_global_weights())
        
    final_global_eval = server.evaluate_global_model(test_partitions, val_partitions)
    
    history_df = pd.DataFrame(round_records)
    history_csv = config.RESULTS_DIR / "federated_history.csv"
    history_df.to_csv(history_csv, index=False)
    
    print("\nFederated Learning Completed Successfully!")
    print(f"Final Global F1-Score: {final_global_eval['f1']*100:.2f}%")
    print(f"Total Cumulative Communication Overhead: {server.round_history[-1]['cumulative_mb'] if server.round_history else telemetry['cumulative_mb']:.3f} MB")
    print(f"Total FL Training Time: {total_training_time:.2f} seconds.\n")
    
    return {
        "final_metrics": final_global_eval,
        "round_history": round_records,
        "history_df": history_df,
        "per_client_eval": per_client_eval,
        "total_training_time_sec": total_training_time,
        "wall_clock_sec": fl_elapsed_wall_clock,
        "total_comm_bytes": server.total_bytes_uplink + server.total_bytes_downlink,
        "total_comm_mb": telemetry["cumulative_mb"],
        "server": server,
        "clients": clients
    }

if __name__ == "__main__":
    run_federated_learning()
