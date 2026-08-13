"""
model.py
--------
Thin wrapper around scikit-learn's IsolationForest, chosen for the prototype
because it:
    * needs no labeled anomalies (logs are almost never pre-labeled)
    * scales roughly linearly with rows, so it stays fast on a laptop
    * gives a continuous anomaly score, not just a binary flag, which is
      what lets us rank and threshold results in the UI

See docs/architecture.md for why IsolationForest was chosen over
Local Outlier Factor and an autoencoder for this prototype.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DEFAULT_CONTAMINATION = 0.05  # assume ~5% of events are anomalous by default
RANDOM_STATE = 42


class AnomalyDetector:
    def __init__(self, contamination: float = DEFAULT_CONTAMINATION, n_estimators: int = 200):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "AnomalyDetector":
        self.model.fit(X)
        self._fitted = True
        return self

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with:
            anomaly_score : higher = more anomalous (0-100 scale, easier to read in a UI)
            is_anomaly    : boolean flag from the model's own decision boundary
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")

        raw_scores = self.model.decision_function(X)  # higher = more normal
        predictions = self.model.predict(X)  # 1 = normal, -1 = anomaly

        # Flip and rescale decision_function output to an intuitive 0-100
        # "anomaly score" where higher always means more suspicious.
        inverted = -raw_scores
        min_v, max_v = inverted.min(), inverted.max()
        if max_v - min_v < 1e-9:
            normalized = np.zeros_like(inverted)
        else:
            normalized = (inverted - min_v) / (max_v - min_v) * 100

        return pd.DataFrame(
            {
                "anomaly_score": np.round(normalized, 2),
                "is_anomaly": predictions == -1,
            },
            index=X.index,
        )

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path, contamination: float = DEFAULT_CONTAMINATION) -> "AnomalyDetector":
        detector = cls(contamination=contamination)
        detector.model = joblib.load(path)
        detector._fitted = True
        return detector


def run_detection(feature_df: pd.DataFrame, contamination: float = DEFAULT_CONTAMINATION) -> pd.DataFrame:
    """Convenience one-shot function used by the API layer."""
    detector = AnomalyDetector(contamination=contamination)
    detector.fit(feature_df)
    return detector.score(feature_df)
