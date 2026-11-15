"""
Synthetic Smart-Campus Energy Data Generator
Generates realistic, Non-IID physical energy profiles for campus edge nodes with
injected multi-type energy anomalies.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import config

def generate_building_data(
    building_id: str,
    profile: dict,
    num_days: int = config.NUM_DAYS,
    samples_per_day: int = config.SAMPLES_PER_DAY,
    anomaly_rate: float = config.ANOMALY_PERCENTAGE,
    random_seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate synthetic time series energy readings for a specific building.
    Creates non-IID diurnal behaviors and injects realistic anomalies.
    """
    np.random.seed(random_seed + hash(building_id) % 10000)
    
    total_samples = num_days * samples_per_day
    start_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(total_samples)]
    
    records = []
    base_load = profile["base_load"]
    peak_mult = profile["peak_multiplier"]
    peak_hours = set(profile["peak_hours"])
    occupancy_base = profile["occupancy_base"]
    voltage_nominal = profile["voltage_nominal"]
    b_type = profile["type"]
    
    # Generate baseline physical signals
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.weekday()
        is_weekend = day_of_week >= 5
        
        # Diurnal temperature oscillation (coldest at 5 AM, hottest at 2 PM)
        temp_base = 22.0 + 8.0 * np.sin((hour - 8) * np.pi / 12) + np.random.normal(0, 1.2)
        
        # Occupancy modeling based on building type and time
        if b_type == "residential":
            # Hostel: high morning & night, moderate weekend
            if hour in [22, 23, 0, 1, 2, 3, 4, 5, 6, 7]:
                occ_factor = 0.85 + 0.15 * np.random.uniform(0, 1)
            elif hour in [18, 19, 20, 21]:
                occ_factor = 0.95 + 0.05 * np.random.uniform(0, 1)
            else:
                occ_factor = 0.35 if not is_weekend else 0.70
        elif b_type == "academic":
            # Academic: strictly daytime weekdays
            if not is_weekend and (8 <= hour <= 17):
                occ_factor = 0.80 + 0.20 * np.sin((hour - 8) * np.pi / 9)
            elif is_weekend:
                occ_factor = 0.05
            else:
                occ_factor = 0.02
        elif b_type == "library":
            # Library: 9 AM to 10 PM
            if 9 <= hour <= 21:
                occ_factor = 0.70 + 0.25 * np.random.uniform(0, 1)
            else:
                occ_factor = 0.08
        else: # laboratory
            # Lab: continuous baseline, higher during day
            if 9 <= hour <= 18:
                occ_factor = 0.75 + 0.20 * np.random.uniform(0, 1)
            else:
                occ_factor = 0.45
                
        occupancy = max(5, int(occupancy_base * occ_factor + np.random.normal(0, 10)))
        
        # Base power consumption (kW)
        if hour in peak_hours and (not is_weekend or b_type == "residential"):
            power_factor_mult = peak_mult * (0.85 + 0.3 * np.random.uniform(0, 1))
        else:
            power_factor_mult = 1.0 + 0.15 * np.random.normal(0, 0.5)
            
        power_kw = max(30.0, base_load * power_factor_mult + (occupancy * 0.12) + (temp_base > 26) * 45.0)
        # Add realistic micro-fluctuations
        power_kw += np.random.normal(0, power_kw * 0.03)
        
        # Energy consumption for the interval (kWh)
        energy_kwh = power_kw * 1.0 # for 1 hour interval
        
        # Voltage with slight power-grid jitter
        voltage = float(np.random.normal(voltage_nominal, 2.5))
        
        # Current I = P / V (with power factor ~0.92)
        power_factor = 0.92
        current = float((power_kw * 1000) / (voltage * power_factor))
        
        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "building_id": building_id,
            "energy_consumption": round(energy_kwh, 2),
            "voltage": round(voltage, 2),
            "current": round(current, 2),
            "power": round(power_kw, 2),
            "temperature": round(temp_base, 2),
            "occupancy": int(occupancy),
            "is_anomaly": 0,
            "anomaly_type": "normal"
        })
        
    df = pd.DataFrame(records)
    
    # Inject Realistic Multi-Type Anomalies
    start_margin = min(20, max(1, total_samples // 10))
    end_margin = max(start_margin + 2, total_samples - start_margin)
    candidate_range = list(range(start_margin, end_margin))
    
    num_anomalies = min(int(total_samples * anomaly_rate), len(candidate_range))
    if num_anomalies > 0:
        anomaly_indices = np.random.choice(candidate_range, size=num_anomalies, replace=False)
    else:
        anomaly_indices = []
    
    # 5 Anomaly types to distribute
    anomaly_types = [
        "sudden_spike",
        "sudden_drop",
        "unusual_night_load",
        "voltage_sag_surge",
        "sustained_high_load"
    ]
    
    for idx, anomaly_idx in enumerate(anomaly_indices):
        atype = anomaly_types[idx % len(anomaly_types)]
        
        if atype == "sudden_spike":
            # 1. Extreme power surge (e.g. short circuit, heavy inductive load spike, meter anomaly)
            df.loc[anomaly_idx, "energy_consumption"] *= np.random.uniform(3.0, 5.0)
            df.loc[anomaly_idx, "power"] = df.loc[anomaly_idx, "energy_consumption"]
            df.loc[anomaly_idx, "current"] *= np.random.uniform(3.0, 4.8)
            df.loc[anomaly_idx, "is_anomaly"] = 1
            df.loc[anomaly_idx, "anomaly_type"] = "sudden_spike"
            
        elif atype == "sudden_drop":
            # 2. Complete or partial power outage / sensor disconnect
            df.loc[anomaly_idx, "energy_consumption"] = np.random.uniform(5.0, 25.0)
            df.loc[anomaly_idx, "power"] = df.loc[anomaly_idx, "energy_consumption"]
            df.loc[anomaly_idx, "current"] = np.random.uniform(1.0, 8.0)
            df.loc[anomaly_idx, "is_anomaly"] = 1
            df.loc[anomaly_idx, "anomaly_type"] = "sudden_drop"
            
        elif atype == "unusual_night_load":
            # 3. Off-hours equipment running (e.g. HVAC stuck on, lights left active at 2 AM)
            df.loc[anomaly_idx, "energy_consumption"] *= np.random.uniform(2.2, 3.5)
            df.loc[anomaly_idx, "power"] = df.loc[anomaly_idx, "energy_consumption"]
            df.loc[anomaly_idx, "current"] *= np.random.uniform(2.0, 3.2)
            df.loc[anomaly_idx, "is_anomaly"] = 1
            df.loc[anomaly_idx, "anomaly_type"] = "unusual_night_load"
            
        elif atype == "voltage_sag_surge":
            # 4. Severe grid voltage abnormality (sag down to 160-185V or surge to 265-285V)
            if np.random.rand() > 0.5:
                df.loc[anomaly_idx, "voltage"] = np.random.uniform(160.0, 185.0) # Sag -> current increases
                df.loc[anomaly_idx, "current"] *= 1.45
            else:
                df.loc[anomaly_idx, "voltage"] = np.random.uniform(265.0, 288.0) # Surge
            df.loc[anomaly_idx, "is_anomaly"] = 1
            df.loc[anomaly_idx, "anomaly_type"] = "voltage_sag_surge"
            
        elif atype == "sustained_high_load":
            # 5. Sustained high load across 3-5 consecutive hours
            duration = np.random.randint(3, 6)
            for d in range(duration):
                if anomaly_idx + d < len(df):
                    df.loc[anomaly_idx + d, "energy_consumption"] *= np.random.uniform(2.0, 2.8)
                    df.loc[anomaly_idx + d, "power"] = df.loc[anomaly_idx + d, "energy_consumption"]
                    df.loc[anomaly_idx + d, "current"] *= np.random.uniform(1.9, 2.6)
                    df.loc[anomaly_idx + d, "is_anomaly"] = 1
                    df.loc[anomaly_idx + d, "anomaly_type"] = "sustained_high_load"

    return df

def generate_all_building_data(output_dir=config.DATA_DIR) -> dict:
    """
    Generate and save distinct local datasets for all campus buildings.
    """
    datasets = {}
    print("==================================================")
    print("GENERATING SYNTHETIC SMART CAMPUS DATASETS")
    print("==================================================")
    
    for b_id, profile in config.BUILDINGS.items():
        df = generate_building_data(b_id, profile)
        file_path = output_dir / f"{b_id}.csv"
        df.to_csv(file_path, index=False)
        datasets[b_id] = df
        
        anomaly_count = df["is_anomaly"].sum()
        pct = (anomaly_count / len(df)) * 100
        print(f"-> Generated {b_id} ({profile['name']}): {len(df)} samples, {anomaly_count} anomalies ({pct:.1f}%) -> {file_path}")
        
    print("All building datasets successfully generated.\n")
    return datasets

if __name__ == "__main__":
    generate_all_building_data()
