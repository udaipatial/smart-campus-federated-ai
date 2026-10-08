"""
Publication-Quality Visualization Generator.
Produces high-resolution figures (300 DPI) and interactive graphs for the research paper.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from src.anomaly_detection.models import EnergyAutoencoder, predict_anomalies
from src.preprocessing.pipeline import load_and_preprocess_building_data
import config

# Set elegant academic plotting aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "figure.autolayout": True
})

def plot_energy_consumption_over_time(building_datasets: dict, output_path=config.FIGURES_DIR / "1_energy_consumption_over_time.png"):
    """
    Plots temporal energy profiles of all 4 non-IID buildings with highlighted anomalies.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    colors = ["#2563EB", "#059669", "#D97706", "#7C3AED"]
    
    for i, (b_id, df) in enumerate(building_datasets.items()):
        ax = axes[i]
        b_name = config.BUILDINGS.get(b_id, {}).get("name", b_id)
        
        # Take first 300 samples for clean readable visualization
        sub_df = df.iloc[:300].copy()
        
        normal_mask = sub_df["is_anomaly"] == 0
        anomaly_mask = sub_df["is_anomaly"] == 1
        
        ax.plot(sub_df.index, sub_df["energy_consumption"], label="Normal Energy (kWh)", color=colors[i], alpha=0.85, linewidth=1.5)
        
        if anomaly_mask.sum() > 0:
            ax.scatter(
                sub_df.index[anomaly_mask], 
                sub_df.loc[anomaly_mask, "energy_consumption"], 
                color="#DC2626", 
                s=45, 
                zorder=5, 
                label="Injected Anomaly",
                marker="^"
            )
            
        ax.set_title(f"{b_id.upper()}: {b_name}", fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel("Power (kWh)")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    axes[-1].set_xlabel("Time (Hourly Timeline Index)")
    fig.suptitle("Smart Campus Non-IID Energy Consumption & Anomaly Profiles", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved: {output_path}")

def plot_anomaly_detection_reconstruction(
    model: EnergyAutoencoder, 
    threshold: float,
    building_id: str = "building_A",
    output_path=config.FIGURES_DIR / "2_anomaly_detection_visualization.png"
):
    """
    Visualizes Autoencoder reconstruction errors (MSE) vs the calibrated decision threshold.
    """
    data = load_and_preprocess_building_data(building_id)
    X_test = data["test"]["X"]
    y_test = data["test"]["y"]
    
    preds, errors = predict_anomalies(model, X_test, threshold)
    
    plt.figure(figsize=(13, 6))
    indices = np.arange(len(errors))
    
    # Plot reconstruction error
    plt.plot(indices, errors, label="Autoencoder Reconstruction Error (MSE)", color="#3B82F6", linewidth=1.4)
    plt.axhline(y=threshold, color="#EF4444", linestyle="--", linewidth=2.0, label=f"Decision Threshold (\u03c4 = {threshold:.4f})")
    
    # Highlight true anomalies
    true_anomaly_idx = indices[y_test == 1]
    if len(true_anomaly_idx) > 0:
        plt.scatter(true_anomaly_idx, errors[true_anomaly_idx], color="#DC2626", s=60, label="Ground Truth Anomaly", zorder=5, marker="o")
        
    # Highlight false positives if any
    fp_idx = indices[(preds == 1) & (y_test == 0)]
    if len(fp_idx) > 0:
        plt.scatter(fp_idx, errors[fp_idx], color="#F59E0B", s=40, label="False Positive", zorder=4, marker="x")
        
    plt.title(f"Energy Anomaly Detection via Deep Reconstruction Loss ({building_id.upper()})", fontsize=14, fontweight="bold")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Reconstruction Error (MSE)")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved: {output_path}")

def plot_centralized_vs_federated_metrics(
    summary: dict,
    output_f1=config.FIGURES_DIR / "3_centralized_vs_federated_f1.png",
    output_acc=config.FIGURES_DIR / "4_centralized_vs_federated_accuracy.png"
):
    """
    Bar chart comparisons of F1, Precision, Recall, and Accuracy.
    """
    metrics_comp = summary["metrics_comparison"]
    
    # 1. Classification Metrics Comparison (Precision, Recall, F1)
    metrics_list = ["Precision", "Recall", "F1 Score"]
    cent_vals = [metrics_comp[m]["Centralized"] for m in metrics_list]
    fed_vals = [metrics_comp[m]["Federated"] for m in metrics_list]
    
    x = np.arange(len(metrics_list))
    width = 0.32
    
    fig, ax = plt.subplots(figsize=(9, 6))
    rects1 = ax.bar(x - width/2, cent_vals, width, label="Centralized Baseline", color="#3B82F6", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, fed_vals, width, label="Federated Learning (FedAvg)", color="#10B981", edgecolor="black", alpha=0.9)
    
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.set_title("Centralized vs Federated Learning: Detection Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Add labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
                        
    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(output_f1, dpi=300)
    plt.close()
    print(f"-> Saved: {output_f1}")
    
    # 2. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(7, 6))
    acc_cent = metrics_comp["Accuracy"]["Centralized"]
    acc_fed = metrics_comp["Accuracy"]["Federated"]
    
    bars = ax.bar(["Centralized Cloud", "Federated Edge (FedAvg)"], [acc_cent, acc_fed], color=["#6366F1", "#14B8A6"], width=0.45, edgecolor="black")
    ax.set_ylabel("Overall Accuracy (%)", fontweight="bold")
    ax.set_title("Overall Model Classification Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
        
    plt.tight_layout()
    plt.savefig(output_acc, dpi=300)
    plt.close()
    print(f"-> Saved: {output_acc}")

def plot_federated_training_rounds(history_df: pd.DataFrame, output_path=config.FIGURES_DIR / "5_federated_training_rounds.png"):
    """
    Convergence curves of Loss and F1-score across FedAvg rounds.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    
    rounds = history_df["round"]
    loss = history_df["mean_local_loss"]
    f1 = history_df["f1"] * 100
    acc = history_df["accuracy"] * 100
    
    color = "#EF4444"
    ax1.set_xlabel("Federated Communication Round", fontweight="bold")
    ax1.set_ylabel("Mean Reconstruction Loss (MSE)", color=color, fontweight="bold")
    line1 = ax1.plot(rounds, loss, color=color, marker="o", linewidth=2.0, label="Reconstruction Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color2 = "#2563EB"
    ax2.set_ylabel("Global Model F1 Score (%)", color=color2, fontweight="bold")
    line2 = ax2.plot(rounds, f1, color=color2, marker="s", linewidth=2.0, label="Global F1-Score (%)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 105)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)
    
    plt.title("Federated Learning Convergence Dynamics across Rounds", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved: {output_path}")

def plot_communication_overhead(
    history_df: pd.DataFrame,
    centralized_raw_mb: float,
    output_path=config.FIGURES_DIR / "6_communication_overhead.png"
):
    """
    Cumulative communication overhead (MB) progression vs Centralized data transmission.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    rounds = history_df["round"]
    cum_mb = history_df["cumulative_comm_mb"]
    
    ax.plot(rounds, cum_mb, marker="^", color="#8B5CF6", linewidth=2.2, label="Federated Learning (Weight Vectors Only)")
    ax.axhline(y=centralized_raw_mb, color="#DC2626", linestyle="--", linewidth=2.0, label=f"Centralized (Raw Building Telemetry: {centralized_raw_mb:.2f} MB)")
    
    ax.set_xlabel("Federated Round", fontweight="bold")
    ax.set_ylabel("Cumulative Transmitted Data (MB)", fontweight="bold")
    ax.set_title("Network Communication Overhead: Federated vs Centralized Ingestion", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Annotation
    final_fl_mb = cum_mb.iloc[-1]
    ax.annotate(f"Final FL Overhead: {final_fl_mb:.3f} MB", 
                xy=(rounds.iloc[-1], final_fl_mb),
                xytext=(rounds.iloc[-1] - 4, final_fl_mb + 0.05),
                arrowprops=dict(arrowstyle="->", color="#8B5CF6"),
                fontweight="bold")
                
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved: {output_path}")

def plot_latency_and_runtime(
    summary: dict,
    output_path=config.FIGURES_DIR / "7_training_latency_comparison.png"
):
    """
    Training runtime and inference latency comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics = summary["metrics_comparison"]
    
    # 1. Total Training Runtime
    train_times = [metrics["Training Time"]["Centralized"], metrics["Training Time"]["Federated"]]
    bars1 = ax1.bar(["Centralized", "Federated (FedAvg)"], train_times, color=["#3B82F6", "#10B981"], width=0.45, edgecolor="black")
    ax1.set_ylabel("Total Time (seconds)", fontweight="bold")
    ax1.set_title("End-to-End Training Runtime", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2., h + 0.1, f"{h:.2f}s", ha="center", va="bottom", fontweight="bold")
        
    # 2. Per-Sample Inference Latency
    inf_times = [metrics["Inference Latency"]["Centralized"], metrics["Inference Latency"]["Federated"]]
    bars2 = ax2.bar(["Centralized", "Federated Edge"], inf_times, color=["#6366F1", "#EC4899"], width=0.45, edgecolor="black")
    ax2.set_ylabel("Latency (ms per sample)", fontweight="bold")
    ax2.set_title("Edge Inference Latency", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + 0.005, f"{h:.4f} ms", ha="center", va="bottom", fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved: {output_path}")

def plot_system_architecture(output_path=config.FIGURES_DIR / "8_system_architecture.png"):
    """
    Generates a clear visual schematic contrasting Centralized vs Federated Architectures.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left: Centralized Paradigm
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("CENTRALIZED PARADIGM\n(High Privacy Risk / Large Ingestion)", fontsize=13, fontweight="bold", color="#DC2626")
    
    # Client boxes
    buildings = ["Building A\n(Hostel)", "Building B\n(Academic)", "Building C\n(Library)", "Building D\n(Lab)"]
    for i, b in enumerate(buildings):
        y_pos = 8 - i * 2.2
        rect = mpatches.FancyBboxPatch((0.5, y_pos), 2.8, 1.4, boxstyle="round,pad=0.2", facecolor="#FEE2E2", edgecolor="#DC2626", linewidth=2)
        ax1.add_patch(rect)
        ax1.text(1.9, y_pos + 0.7, b, ha="center", va="center", fontsize=9, fontweight="bold")
        
        # Arrows pointing to central server with raw data label
        ax1.annotate("Raw Data \u2192", xy=(6.5, 4.5), xytext=(3.5, y_pos + 0.7),
                     arrowprops=dict(facecolor="#DC2626", edgecolor="#DC2626", width=2, headwidth=7),
                     fontsize=8, color="#DC2626", fontweight="bold")
                     
    # Central Server box
    server_rect = mpatches.FancyBboxPatch((6.5, 3.0), 3.0, 3.2, boxstyle="round,pad=0.2", facecolor="#E0E7FF", edgecolor="#4338CA", linewidth=2)
    ax1.add_patch(server_rect)
    ax1.text(8.0, 4.6, "Central Cloud Server\n\n- Pooled Raw Data\n- Central Model\n[!] Privacy Exposure", ha="center", va="center", fontsize=9, fontweight="bold")

    # Right: Federated Edge-AI Paradigm
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("FEDERATED EDGE-AI PARADIGM\n(Privacy-Preserving FedAvg)", fontsize=13, fontweight="bold", color="#059669")
    
    for i, b in enumerate(buildings):
        y_pos = 8 - i * 2.2
        rect = mpatches.FancyBboxPatch((0.5, y_pos), 2.8, 1.4, boxstyle="round,pad=0.2", facecolor="#D1FAE5", edgecolor="#059669", linewidth=2)
        ax2.add_patch(rect)
        ax2.text(1.9, y_pos + 0.7, f"{b}\n[Local Training]", ha="center", va="center", fontsize=8.5, fontweight="bold")
        
        # Bidirectional arrows for weights only
        ax2.annotate("", xy=(6.5, 4.8), xytext=(3.5, y_pos + 0.8),
                     arrowprops=dict(arrowstyle="<->", color="#059669", lw=2))
        ax2.text(4.8, (y_pos + 0.8 + 4.8)/2, "delta-w (Weights)", fontsize=7.5, color="#059669", fontweight="bold", ha="center")
        
    # Fed Server box
    fed_server_rect = mpatches.FancyBboxPatch((6.5, 3.0), 3.0, 3.2, boxstyle="round,pad=0.2", facecolor="#FEF3C7", edgecolor="#D97706", linewidth=2)
    ax2.add_patch(fed_server_rect)
    ax2.text(8.0, 4.6, "Federated Server\n\n- FedAvg Aggregation\n- Zero Raw Data Seen\n[OK] 100% Edge Privacy", ha="center", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved: {output_path}")

def generate_all_publication_figures(
    building_datasets: dict,
    model: EnergyAutoencoder,
    threshold: float,
    summary: dict,
    history_df: pd.DataFrame
):
    """Generates the full suite of research figures."""
    print("==================================================")
    print("GENERATING PUBLICATION-QUALITY RESEARCH FIGURES")
    print("==================================================")
    plot_energy_consumption_over_time(building_datasets)
    plot_anomaly_detection_reconstruction(model, threshold)
    plot_centralized_vs_federated_metrics(summary)
    plot_federated_training_rounds(history_df)
    plot_communication_overhead(history_df, summary["metrics_comparison"]["Communication Overhead"]["Centralized"])
    plot_latency_and_runtime(summary)
    plot_system_architecture()
    print("All research figures generated in results/figures/.\n")
