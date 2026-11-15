"""
Master Orchestration Script for Federated Edge-AI Smart Campus Energy Monitor.
Executes the complete experimental pipeline end-to-end:
1. Synthetic Non-IID Data Generation
2. Centralized Model Baseline Training
3. Federated Edge-AI Training (FedAvg)
4. Cryptographic Privacy Audit & Verification
5. Empirical Comparison Table Generation
6. Publication-Quality Figure Generation
7. Optional Dashboard Web Server Launch
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.data_generation.generator import generate_all_building_data
from src.centralized.trainer import run_centralized_training
from src.federated.coordinator import run_federated_learning
from src.utils.privacy_audit import generate_privacy_audit_report
from src.evaluation.metrics import generate_comparison_summary
from src.evaluation.visualizer import generate_all_publication_figures

def run_pipeline(serve_dashboard: bool = False, host: str = "0.0.0.0", port: int = 8000):
    start_total_time = time.perf_counter()
    
    print("\n" + "="*70)
    print("FEDERATED EDGE-AI FRAMEWORK FOR SMART CAMPUS ENERGY ANOMALY DETECTION")
    print("="*70 + "\n")
    
    # STEP 1: Generate Synthetic Non-IID Campus Energy Data
    print("[STEP 1/6] Synthesizing Multi-Building Non-IID Energy Telemetry...")
    building_datasets = generate_all_building_data(config.DATA_DIR)
    
    # STEP 2: Centralized Baseline Training
    print("[STEP 2/6] Executing Centralized Machine Learning Baseline...")
    centralized_results = run_centralized_training()
    
    # STEP 3: Federated Learning (FedAvg) Simulation
    print("[STEP 3/6] Executing Federated Edge-AI (FedAvg) Multi-Round Training...")
    federated_results = run_federated_learning()
    
    # STEP 4: Privacy Audit & Data Isolation Verification
    print("[STEP 4/6] Conducting Cryptographic Privacy Audit & Data Containment Proof...")
    privacy_report = generate_privacy_audit_report()
    print("-> Privacy Audit Report saved to results/privacy_audit.json")
    
    # STEP 5: Synthesize Evaluation Metrics & Comparison Table
    print("[STEP 5/6] Synthesizing Empirical Results & Generating Comparison Tables...")
    summary = generate_comparison_summary(centralized_results, federated_results)
    print("-> Summary metrics saved to results/summary_metrics.json")
    print("-> Comparison table saved to results/comparison_table.csv & results/comparison_table.md")
    
    # STEP 6: Generate High-Resolution Publication Figures
    print("[STEP 6/6] Generating Publication-Quality Research Figures (300 DPI)...")
    generate_all_publication_figures(
        building_datasets=building_datasets,
        model=federated_results["server"].global_model,
        threshold=federated_results["final_metrics"]["threshold"],
        summary=summary,
        history_df=federated_results["history_df"]
    )
    
    total_elapsed = time.perf_counter() - start_total_time
    print("="*70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print(f"Total Execution Time: {total_elapsed:.2f} seconds")
    print("="*70)
    
    # Print Terminal Comparison Summary Table
    print("\nEMPIRICAL RESULTS SUMMARY:")
    print("-" * 65)
    print(f"{'Metric':<25} | {'Centralized':<16} | {'Federated (FedAvg)':<18}")
    print("-" * 65)
    for m_name, vals in summary["metrics_comparison"].items():
        u = vals["Unit"] if vals["Unit"] != "Category" else ""
        c_val = f"{vals['Centralized']} {u}".strip()
        f_val = f"{vals['Federated']} {u}".strip()
        print(f"{m_name:<25} | {c_val:<16} | {f_val:<18}")
    print("-" * 65 + "\n")
    
    if serve_dashboard:
        start_dashboard_server(host=host, port=port)

def start_dashboard_server(host: str = "0.0.0.0", port: int = 8000):
    """Starts FastAPI Uvicorn web server for interactive dashboard."""
    import uvicorn
    print("\n" + "="*70)
    print(f"STARTING INTERACTIVE DASHBOARD SERVER AT: http://{host}:{port}")
    print("Press CTRL+C to stop the server.")
    print("="*70 + "\n")
    uvicorn.run("dashboard.app:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Smart Campus Energy Anomaly Detection")
    parser.add_argument("--serve", action="store_true", help="Launch interactive web dashboard after training")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP for dashboard server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for dashboard server (default: 8000)")
    args = parser.parse_args()
    
    run_pipeline(serve_dashboard=args.serve, host=args.host, port=args.port)
