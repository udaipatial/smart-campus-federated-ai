"""Unit tests for evaluation metrics and privacy audit."""
import pytest
import config
from src.utils.privacy_audit import generate_privacy_audit_report
from src.evaluation.metrics import generate_comparison_summary

def test_privacy_audit_generation():
    report = generate_privacy_audit_report()
    assert "privacy_guarantees" in report
    assert "comparison_matrix" in report
    assert report["comparison_matrix"]["Federated"]["raw_data_leaves_edge"] is False
    assert report["comparison_matrix"]["Centralized"]["raw_data_leaves_edge"] is True

def test_metrics_summary_structure():
    mock_cent = {
        "accuracy": 0.95,
        "precision": 0.90,
        "recall": 0.88,
        "f1": 0.89,
        "training_time_sec": 4.5,
        "inference_latency_ms": 0.05,
        "communication_bytes": 1024000,
        "communication_mb": 0.976,
        "loss_history": [0.5, 0.2]
    }
    mock_fed = {
        "final_metrics": {
            "accuracy": 0.945,
            "precision": 0.89,
            "recall": 0.87,
            "f1": 0.88,
            "inference_latency_ms": 0.052,
            "threshold": 0.75
        },
        "round_history": [{"round": 1}],
        "clients": [1, 2, 3, 4],
        "total_training_time_sec": 6.2,
        "total_comm_bytes": 150000,
        "total_comm_mb": 0.143,
        "per_client_eval": {}
    }
    
    summary = generate_comparison_summary(mock_cent, mock_fed)
    assert "metrics_comparison" in summary
    assert "Accuracy" in summary["metrics_comparison"]
    assert "F1 Score" in summary["metrics_comparison"]
    assert "Communication Overhead" in summary["metrics_comparison"]
