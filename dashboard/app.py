"""
FastAPI Dashboard Backend for Federated Smart Campus Energy Monitor.
Serves interactive APIs and web user interface.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure repository root is on sys.path for robust resolution in all environments (local, Vercel serverless, etc.)
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

import config

app = FastAPI(
    title="Federated Smart Campus Energy Anomaly Monitor",
    description="Privacy-Preserving Edge-AI Anomaly Detection Dashboard",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and figures safely
STATIC_DIR = Path(__file__).resolve().parent / "static"
try:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if config.FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(config.FIGURES_DIR)), name="figures")

@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Federated Energy Dashboard API running. Please build static UI."}

@app.get("/background.gif")
async def get_background_gif():
    bg_file = STATIC_DIR / "background.gif"
    if bg_file.exists():
        return FileResponse(bg_file)
    raise HTTPException(status_code=404, detail="Background GIF not found")

@app.get("/api/summary")
async def get_summary():
    """Returns the empirical benchmark comparison between Centralized and Federated systems."""
    summary_file = config.RESULTS_DIR / "summary_metrics.json"
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Summary metrics not found. Run pipeline first.")
    with open(summary_file, "r") as f:
        data = json.load(f)
    return data

@app.get("/api/buildings")
async def get_buildings():
    """Returns high-level status, current power, and anomaly alert state for all campus edge nodes."""
    buildings_data = []
    
    for b_id, profile in config.BUILDINGS.items():
        csv_path = config.DATA_DIR / f"{b_id}.csv"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        last_row = df.iloc[-1].to_dict()
        recent_24h = df.iloc[-24:]
        
        total_anomalies = int(df["is_anomaly"].sum())
        total_samples = len(df)
        training_samples = int(len(df[df["is_anomaly"] == 0]))
        current_status = "ANOMALY" if int(last_row.get("is_anomaly", 0)) == 1 else "NORMAL"
        
        buildings_data.append({
            "id": b_id,
            "name": profile["name"],
            "type": profile["type"],
            "description": profile["description"],
            "current_power_kwh": float(last_row.get("energy_consumption", 0)),
            "current_voltage": float(last_row.get("voltage", 230)),
            "current_occupancy": int(last_row.get("occupancy", 0)),
            "status": current_status,
            "anomaly_type": last_row.get("anomaly_type", "normal"),
            "total_anomalies_recorded": total_anomalies,
            "anomaly_rate_pct": round((total_anomalies / total_samples) * 100, 2),
            "total_samples": total_samples,
            "training_samples": training_samples,
            "total_energy_kwh": round(float(df["energy_consumption"].sum()), 2),
            "recent_sparkline": recent_24h["energy_consumption"].tolist()
        })
        
    return buildings_data

@app.get("/api/building/{building_id}/telemetry")
async def get_building_telemetry(building_id: str, limit: int = 150):
    """Returns recent physical time-series measurements for a specific edge building."""
    csv_path = config.DATA_DIR / f"{building_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail=f"Data for {building_id} not found.")
        
    df = pd.read_csv(csv_path).tail(limit)
    return {
        "building_id": building_id,
        "timestamps": df["timestamp"].tolist(),
        "energy_consumption": df["energy_consumption"].tolist(),
        "voltage": df["voltage"].tolist(),
        "current": df["current"].tolist(),
        "occupancy": df["occupancy"].tolist(),
        "is_anomaly": df["is_anomaly"].tolist(),
        "anomaly_types": df["anomaly_type"].tolist()
    }

@app.get("/api/federated-rounds")
async def get_federated_rounds():
    """Returns convergence telemetry across all federated communication rounds."""
    csv_path = config.RESULTS_DIR / "federated_history.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Federated history not found.")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

@app.get("/api/anomalies")
async def get_anomalies_feed(limit: int = 50):
    """Returns list of recent anomalies detected across the entire campus."""
    all_anomalies = []
    for b_id, profile in config.BUILDINGS.items():
        csv_path = config.DATA_DIR / f"{b_id}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        anom_df = df[df["is_anomaly"] == 1].copy()
        for _, row in anom_df.iterrows():
            all_anomalies.append({
                "building_id": b_id,
                "building_name": profile["name"],
                "timestamp": row["timestamp"],
                "energy_kwh": row["energy_consumption"],
                "voltage": row["voltage"],
                "anomaly_type": row["anomaly_type"],
                "severity": "CRITICAL" if row["energy_consumption"] > 1000 or row["anomaly_type"] == "sudden_spike" else "WARNING"
            })
            
    # Sort descending by timestamp
    all_anomalies = sorted(all_anomalies, key=lambda x: x["timestamp"], reverse=True)[:limit]
    return all_anomalies

@app.get("/api/privacy-audit")
async def get_privacy_audit():
    """Returns privacy verification proofs and data locality audit logs."""
    audit_path = config.RESULTS_DIR / "privacy_audit.json"
    if not audit_path.exists():
        raise HTTPException(status_code=404, detail="Privacy audit not found.")
    with open(audit_path, "r") as f:
        return json.load(f)

import copy
from datetime import datetime, timedelta
import torch
from src.anomaly_detection.models import EnergyAutoencoder, predict_anomalies
from src.preprocessing.pipeline import engineer_features, FeatureScaler, preprocess_dataframe

# Realtime In-Memory Streaming State
class RealtimeCampusStream:
    def __init__(self):
        self.simulated_time = datetime.now()
        self.trained_model = None
        self.threshold = 0.5
        self.scalers = {}
        self.live_history = {b_id: [] for b_id in config.BUILDINGS.keys()}
        self.pending_injections = {}
        self._init_models()
        
    def _init_models(self):
        # Load global federated model checkpoint if available
        model_path = config.MODELS_DIR / "federated_global.pt"
        self.trained_model = EnergyAutoencoder()
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self.trained_model.load_state_dict(checkpoint["model_state_dict"])
            self.threshold = checkpoint.get("threshold", 0.45)
        self.trained_model.eval()
        
        # Fit scaler on training portion of existing data
        for b_id in config.BUILDINGS.keys():
            csv_path = config.DATA_DIR / f"{b_id}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df_feat = engineer_features(df)
                normal_feat = df_feat[df_feat["is_anomaly"] == 0][config.FEATURE_COLUMNS].values.astype(np.float32)
                scaler = FeatureScaler()
                scaler.fit(normal_feat)
                self.scalers[b_id] = scaler

    def generate_live_tick(self, b_id: str, inject_type: str = None) -> Dict[str, Any]:
        """Simulates the next real-time sensory reading and runs edge on-device inference."""
        self.simulated_time += timedelta(minutes=15)
        profile = config.BUILDINGS[b_id]
        hour = self.simulated_time.hour
        is_weekend = self.simulated_time.weekday() >= 5
        
        # Base environmental signals
        temp_base = 22.0 + 8.0 * np.sin((hour - 8) * np.pi / 12) + np.random.normal(0, 1.0)
        
        # Base physical power
        peak_hours = set(profile["peak_hours"])
        b_type = profile["type"]
        base_load = profile["base_load"]
        peak_mult = profile["peak_multiplier"]
        
        if b_type == "residential":
            occ_factor = 0.85 if hour in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7] else 0.35
        elif b_type == "academic":
            occ_factor = 0.85 if (8 <= hour <= 17 and not is_weekend) else 0.05
        elif b_type == "library":
            occ_factor = 0.75 if (9 <= hour <= 21) else 0.08
        else:
            occ_factor = 0.70 if (9 <= hour <= 18) else 0.45
            
        occupancy = max(5, int(profile["occupancy_base"] * occ_factor + np.random.normal(0, 8)))
        
        if hour in peak_hours and (not is_weekend or b_type == "residential"):
            mult = peak_mult * np.random.uniform(0.9, 1.1)
        else:
            mult = 1.0 + np.random.normal(0, 0.1)
            
        power_kw = max(25.0, base_load * mult + occupancy * 0.12)
        voltage = float(np.random.normal(230.0, 2.0))
        current = float((power_kw * 1000) / (voltage * 0.92))
        
        is_anomaly_injected = False
        anomaly_type = "normal"
        
        # Check for queued injection
        inj = inject_type or self.pending_injections.pop(b_id, None)
        if inj == "spike":
            power_kw *= np.random.uniform(3.5, 4.8)
            current *= 3.8
            is_anomaly_injected = True
            anomaly_type = "sudden_spike"
        elif inj == "drop":
            power_kw = np.random.uniform(5.0, 15.0)
            current = np.random.uniform(1.0, 5.0)
            is_anomaly_injected = True
            anomaly_type = "sudden_drop"
        elif inj == "night_load":
            power_kw *= 3.0
            current *= 2.8
            is_anomaly_injected = True
            anomaly_type = "unusual_night_load"
        elif inj == "voltage_sag":
            voltage = np.random.uniform(160.0, 175.0)
            current *= 1.5
            is_anomaly_injected = True
            anomaly_type = "voltage_sag"
            
        # Create single row dataframe for real-time edge processing
        sample_df = pd.DataFrame([{
            "timestamp": self.simulated_time.strftime("%Y-%m-%d %H:%M:%S"),
            "energy_consumption": round(power_kw, 2),
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power": round(power_kw, 2),
            "temperature": round(temp_base, 2),
            "occupancy": int(occupancy)
        }])
        
        # Preprocess using edge scaler
        sample_feat = engineer_features(sample_df)
        X_raw = sample_feat[config.FEATURE_COLUMNS].values.astype(np.float32)
        scaler = self.scalers.get(b_id)
        X_scaled = scaler.transform(X_raw) if scaler else X_raw
        
        # Real-time On-Device Neural Edge Inference
        start_inf = datetime.now()
        preds, errors = predict_anomalies(self.trained_model, X_scaled, self.threshold)
        inf_latency_us = (datetime.now() - start_inf).total_seconds() * 1_000_000 # microseconds
        
        detected_as_anomaly = bool(preds[0] == 1)
        
        reading = {
            "building_id": b_id,
            "building_name": profile["name"],
            "timestamp": self.simulated_time.strftime("%H:%M:%S"),
            "full_timestamp": self.simulated_time.strftime("%Y-%m-%d %H:%M:%S"),
            "energy_consumption": round(float(power_kw), 2),
            "voltage": round(float(voltage), 2),
            "current": round(float(current), 2),
            "power": round(float(power_kw), 2),
            "temperature": round(float(temp_base), 2),
            "occupancy": int(occupancy),
            "reconstruction_error": round(float(errors[0]), 5),
            "threshold": round(float(self.threshold), 5),
            "is_detected_anomaly": detected_as_anomaly,
            "ground_truth_anomaly": is_anomaly_injected,
            "anomaly_type": anomaly_type if is_anomaly_injected else ("detected_anomaly" if detected_as_anomaly else "normal"),
            "inference_latency_us": round(inf_latency_us, 2)
        }
        
        self.live_history[b_id].append(reading)
        if len(self.live_history[b_id]) > 200:
            self.live_history[b_id].pop(0)
            
        return reading

realtime_stream = RealtimeCampusStream()

@app.get("/api/stream/tick")
async def stream_tick(building_id: str = "all", inject: str = None):
    """
    Simulates real-time incoming smart meter tick and executes live on-device inference.
    """
    if building_id == "all":
        results = {}
        for b_id in config.BUILDINGS.keys():
            results[b_id] = realtime_stream.generate_live_tick(b_id, inject_type=inject if inject else None)
        return results
    else:
        if building_id not in config.BUILDINGS:
            raise HTTPException(status_code=404, detail="Building not found")
        return realtime_stream.generate_live_tick(building_id, inject_type=inject)

@app.post("/api/stream/inject")
async def inject_live_anomaly(building_id: str, anomaly_type: str):
    """
    Queues an instantaneous physical anomaly on a specific edge node for live demonstration.
    """
    if building_id not in config.BUILDINGS:
        raise HTTPException(status_code=404, detail="Building not found")
    realtime_stream.pending_injections[building_id] = anomaly_type
    return {
        "status": "QUEUED",
        "message": f"Queued {anomaly_type} anomaly injection for {building_id}.",
        "target_building": building_id
    }

@app.get("/api/figures-list")
async def get_figures_list():
    """Lists all available publication-ready figures."""
    figures = []
    for fig_file in sorted(config.FIGURES_DIR.glob("*.png")):
        figures.append({
            "name": fig_file.stem.replace("_", " ").title(),
            "filename": fig_file.name,
            "url": f"/figures/{fig_file.name}"
        })
    return figures

