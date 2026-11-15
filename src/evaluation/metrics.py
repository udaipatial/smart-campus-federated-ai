"""
Comprehensive Evaluation and Metrics Module.
Synthesizes Centralized and Federated experiment outputs, produces structured comparison tables,
and exports publication-ready summary metrics.
"""

import json
from typing import Dict, Any
import pandas as pd
import config

def generate_comparison_summary(
    centralized_results: Dict[str, Any],
    federated_results: Dict[str, Any],
    output_json=config.RESULTS_DIR / "summary_metrics.json",
    output_csv=config.RESULTS_DIR / "comparison_table.csv",
    output_md=config.RESULTS_DIR / "comparison_table.md"
) -> Dict[str, Any]:
    """
    Constructs an empirical comparison table directly from measured experimental results.
    """
    fl_final = federated_results["final_metrics"]
    
    # Structure comprehensive comparison metrics
    summary = {
        "experiment_timestamp": pd.Timestamp.now().isoformat(),
        "buildings_evaluated": list(config.BUILDINGS.keys()),
        "metrics_comparison": {
            "Accuracy": {
                "Centralized": round(centralized_results["accuracy"] * 100, 2),
                "Federated": round(fl_final["accuracy"] * 100, 2),
                "Delta": round((fl_final["accuracy"] - centralized_results["accuracy"]) * 100, 2),
                "Unit": "%"
            },
            "Precision": {
                "Centralized": round(centralized_results["precision"] * 100, 2),
                "Federated": round(fl_final["precision"] * 100, 2),
                "Delta": round((fl_final["precision"] - centralized_results["precision"]) * 100, 2),
                "Unit": "%"
            },
            "Recall": {
                "Centralized": round(centralized_results["recall"] * 100, 2),
                "Federated": round(fl_final["recall"] * 100, 2),
                "Delta": round((fl_final["recall"] - centralized_results["recall"]) * 100, 2),
                "Unit": "%"
            },
            "F1 Score": {
                "Centralized": round(centralized_results["f1"] * 100, 2),
                "Federated": round(fl_final["f1"] * 100, 2),
                "Delta": round((fl_final["f1"] - centralized_results["f1"]) * 100, 2),
                "Unit": "%"
            },
            "Training Time": {
                "Centralized": round(centralized_results["training_time_sec"], 3),
                "Federated": round(federated_results["total_training_time_sec"], 3),
                "Delta": round(federated_results["total_training_time_sec"] - centralized_results["training_time_sec"], 3),
                "Unit": "seconds"
            },
            "Inference Latency": {
                "Centralized": round(centralized_results["inference_latency_ms"], 4),
                "Federated": round(fl_final["inference_latency_ms"], 4),
                "Delta": round(fl_final["inference_latency_ms"] - centralized_results["inference_latency_ms"], 4),
                "Unit": "ms/sample"
            },
            "Communication Overhead": {
                "Centralized": round(centralized_results["communication_mb"], 3),
                "Federated": round(federated_results["total_comm_mb"], 3),
                "Delta": round(federated_results["total_comm_mb"] - centralized_results["communication_mb"], 3),
                "Unit": "MB"
            },
            "Privacy Preservation": {
                "Centralized": "None (Raw Data Transmitted)",
                "Federated": "High (Parameter Updates Only)",
                "Delta": "GDPR / FERPA Compliant",
                "Unit": "Category"
            }
        },
        "federated_details": {
            "num_rounds": len(federated_results["round_history"]),
            "participating_clients": len(federated_results["clients"]),
            "total_bytes_communicated": federated_results["total_comm_bytes"],
            "per_client_performance": federated_results["per_client_eval"]
        },
        "centralized_details": {
            "total_raw_bytes": centralized_results["communication_bytes"],
            "epochs": len(centralized_results["loss_history"])
        }
    }
    
    # Save JSON summary
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=4)
        
    # Build tabular format
    table_rows = []
    for metric_name, data in summary["metrics_comparison"].items():
        table_rows.append({
            "Metric": metric_name,
            "Centralized": f"{data['Centralized']} {data['Unit']}" if data['Unit'] != "Category" else data['Centralized'],
            "Federated (FedAvg)": f"{data['Federated']} {data['Unit']}" if data['Unit'] != "Category" else data['Federated'],
            "Difference": f"{data['Delta']} {data['Unit']}" if data['Unit'] != "Category" else data['Delta']
        })
        
    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(output_csv, index=False)
    
    # Markdown Table representation
    md_content = "# Experimental Results: Centralized vs Federated Energy Anomaly Detection\n\n"
    try:
        md_content += df_table.to_markdown(index=False)
    except Exception:
        # Fallback native markdown formatting without tabulate dependency
        headers = list(df_table.columns)
        md_content += "| " + " | ".join(headers) + " |\n"
        md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, row in df_table.iterrows():
            md_content += "| " + " | ".join(str(val) for val in row.values) + " |\n"
            
    md_content += "\n\n*Note: Metrics represent empirical evaluation on real-time simulated non-IID smart campus edge nodes.*\n"
    
    with open(output_md, "w") as f:
        f.write(md_content)
        
    return summary

def export_results_table():
    """Reads and returns the formatted results table if it exists."""
    csv_file = config.RESULTS_DIR / "comparison_table.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    return None
