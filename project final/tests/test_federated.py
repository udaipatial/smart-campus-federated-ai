"""Unit tests for Federated Learning edge client and server aggregation."""
import pytest
import torch
import numpy as np
import config
from src.data_generation.generator import generate_all_building_data
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer

@pytest.fixture(scope="module")
def setup_datasets():
    generate_all_building_data()

def test_federated_client_local_training(setup_datasets):
    client = FederatedClient("building_A")
    assert client.get_num_samples() > 0
    
    server = FederatedServer()
    initial_weights = server.get_global_weights()
    
    update = client.train(initial_weights, epochs=2)
    
    assert update["client_id"] == "building_A"
    assert "weights" in update
    assert update["train_loss"] >= 0
    assert update["train_time_sec"] > 0
    assert update["model_size_bytes"] > 0

def test_fedavg_aggregation(setup_datasets):
    server = FederatedServer()
    client_a = FederatedClient("building_A")
    client_b = FederatedClient("building_B")
    
    global_w = server.get_global_weights()
    update_a = client_a.train(global_w, epochs=1)
    update_b = client_b.train(global_w, epochs=1)
    
    telemetry = server.aggregate_fedavg([update_a, update_b])
    
    assert telemetry["round"] == 1
    assert telemetry["participating_clients"] == 2
    assert telemetry["round_total_bytes"] > 0
    assert telemetry["cumulative_mb"] > 0
    assert server.current_round == 1
