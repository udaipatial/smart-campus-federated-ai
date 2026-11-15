"""
Isolation Forest Baseline Anomaly Detector (Scikit-Learn).
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import config

class IsolationForestDetector:
    """
    Standard Tree-Based Isolation Forest for Anomaly Detection.
    """
    def __init__(
        self,
        contamination: float = config.ANOMALY_PERCENTAGE,
        random_state: int = config.RANDOM_SEED
    ):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.fitted = False

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Isolation forest returns -1 for anomalies and 1 for normal
        raw_preds = self.model.predict(X)
        return (raw_preds == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # Returns opposite of anomaly score (lower is more anomalous)
        return -self.model.score_samples(X)
