import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import features, model, parser


def _sample_dataframe(n=120, n_anomalies=6):
    """Builds a small parsed-log-shaped DataFrame with an obvious anomaly cluster."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2026-06-01", periods=n, freq="30s")
    levels = rng.choice(["INFO", "INFO", "INFO", "WARN"], size=n)
    sources = rng.choice(["api", "auth", "db"], size=n)
    messages = [f"request completed in {rng.integers(10, 200)}ms" for _ in range(n)]

    df = pd.DataFrame({"timestamp": timestamps, "level": levels, "source": sources, "message": messages})

    anomaly_idx = df.index[-n_anomalies:]
    df.loc[anomaly_idx, "level"] = "CRITICAL"
    df.loc[anomaly_idx, "message"] = "unauthorized access attempt to /admin/config SQL injection detected"
    df.loc[anomaly_idx, "source"] = "auth"
    return df


def test_anomaly_detector_fit_score_shapes():
    df = _sample_dataframe()
    feature_df, *_ = features.build_feature_matrix(df)
    detector = model.AnomalyDetector(contamination=0.05)
    detector.fit(feature_df)
    scores = detector.score(feature_df)

    assert len(scores) == len(df)
    assert set(scores.columns) == {"anomaly_score", "is_anomaly"}
    assert scores["anomaly_score"].between(0, 100).all()


def test_run_detection_flags_at_least_one_anomaly():
    df = _sample_dataframe()
    feature_df, *_ = features.build_feature_matrix(df)
    scores = model.run_detection(feature_df, contamination=0.1)
    assert scores["is_anomaly"].sum() >= 1


def test_score_raises_if_not_fitted():
    detector = model.AnomalyDetector()
    df = _sample_dataframe(n=10, n_anomalies=1)
    feature_df, *_ = features.build_feature_matrix(df)
    with pytest.raises(RuntimeError):
        detector.score(feature_df)


def test_save_and_load_roundtrip(tmp_path):
    df = _sample_dataframe()
    feature_df, *_ = features.build_feature_matrix(df)
    detector = model.AnomalyDetector(contamination=0.05).fit(feature_df)

    path = tmp_path / "model.joblib"
    detector.save(path)
    assert path.exists()

    loaded = model.AnomalyDetector.load(path)
    scores = loaded.score(feature_df)
    assert len(scores) == len(df)
