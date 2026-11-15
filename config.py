# Project Configuration for Federated Smart Campus Energy Anomaly Detection

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Ensure required directories exist (safely handled for read-only environments like Vercel)
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# Simulation & Data Settings
RANDOM_SEED = 42
NUM_DAYS = 60                       # 60 days of simulation
SAMPLES_PER_DAY = 24                # Hourly readings (1,440 hours per building)
ANOMALY_PERCENTAGE = 0.06           # ~6% injected anomalies

# Building Profiles (Non-IID Characteristics)
BUILDINGS = {
    "building_A": {
        "name": "Hostel & Residential Complex",
        "type": "residential",
        "base_load": 280.0,
        "peak_multiplier": 2.2,
        "peak_hours": [6, 7, 8, 19, 20, 21, 22, 23],  # Morning & Evening/Night
        "occupancy_base": 850,
        "voltage_nominal": 230.0,
        "description": "High evening/night power demand with distinct diurnal residential patterns."
    },
    "building_B": {
        "name": "Academic & Classroom Block",
        "type": "academic",
        "base_load": 150.0,
        "peak_multiplier": 3.0,
        "peak_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],  # Daytime lectures
        "occupancy_base": 1200,
        "voltage_nominal": 230.0,
        "description": "High daytime lecture load (8 AM - 5 PM), minimal night baseline."
    },
    "building_C": {
        "name": "Central Campus Library",
        "type": "library",
        "base_load": 180.0,
        "peak_multiplier": 2.4,
        "peak_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  # 9 AM - 10 PM
        "occupancy_base": 600,
        "voltage_nominal": 230.0,
        "description": "Sustained long daytime and evening study hours with constant HVAC/lighting."
    },
    "building_D": {
        "name": "Advanced Research Laboratories & Data Center",
        "type": "laboratory",
        "base_load": 420.0,
        "peak_multiplier": 1.9,
        "peak_hours": [10, 11, 12, 14, 15, 16, 17, 18],  # Heavy equipment & continuous server load
        "occupancy_base": 300,
        "voltage_nominal": 230.0,
        "description": "High continuous baseline from servers & cleanrooms, with sporadic intense computing spikes."
    }
}

# Features for Model Training
FEATURE_COLUMNS = [
    "energy_consumption",
    "voltage",
    "current",
    "power",
    "temperature",
    "occupancy",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]

# Federated Learning Hyperparameters
FEDERATED_CONFIG = {
    "num_rounds": 15,
    "local_epochs": 4,
    "batch_size": 32,
    "learning_rate": 0.003,
    "latent_dim": 8,
    "hidden_dims": [32, 16],
    "client_fraction": 1.0          # 100% of available buildings participate per round
}

# Centralized Learning Hyperparameters
CENTRALIZED_CONFIG = {
    "epochs": 60,                   # Equivalent total training budget (15 rounds * 4 local epochs)
    "batch_size": 64,
    "learning_rate": 0.003,
    "latent_dim": 8,
    "hidden_dims": [32, 16]
}

# Anomaly Detection Evaluation Threshold
ANOMALY_PERCENTILE_THRESHOLD = 93.5  # Percentile of normal reconstruction error
