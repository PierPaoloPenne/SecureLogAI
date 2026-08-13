"""
features.py
------------
Converts a parsed log DataFrame (timestamp, level, source, message) into a
numeric feature matrix suitable for an unsupervised anomaly detector.

Feature groups
    1. Temporal      - hour of day, day of week, weekend flag, seconds since
                        the previous event from the same source (burst detection)
    2. Categorical    - one-hot-ish encodings of log level and source frequency
    3. Text           - TF-IDF over the message text, reduced with TruncatedSVD
                        so it stays cheap to compute on a laptop
    4. Volume         - rolling event count per source in a sliding window,
                        which is what catches brute-force / flooding patterns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

LEVEL_SEVERITY = {
    "DEBUG": 0,
    "INFO": 1,
    "WARN": 2,
    "WARNING": 2,
    "ERROR": 3,
    "CRITICAL": 4,
    "FATAL": 4,
}

TEXT_SVD_COMPONENTS = 8
ROLLING_WINDOW = "60s"


def _temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    out = pd.DataFrame(index=df.index)
    out["hour"] = ts.dt.hour.fillna(-1)
    out["day_of_week"] = ts.dt.dayofweek.fillna(-1)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int).fillna(0)

    # Seconds since the previous event from the same source — small values
    # repeated many times in a row are the signature of a flood / brute force.
    df_sorted = df.sort_values("timestamp")
    gap = (
        df_sorted.groupby("source")["timestamp"]
        .diff()
        .dt.total_seconds()
    )
    out["seconds_since_prev_same_source"] = gap.reindex(df.index).fillna(gap.median() if gap.notna().any() else 0)
    return out


def _severity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["severity"] = df["level"].map(LEVEL_SEVERITY).fillna(1)
    return out


def _volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling count of events per source within a trailing time window."""
    out = pd.Series(0.0, index=df.index, name="events_per_min_same_source")
    if df["timestamp"].notna().sum() == 0:
        return out.to_frame()

    for source, group in df.groupby("source"):
        g = group.dropna(subset=["timestamp"]).sort_values("timestamp")
        if g.empty:
            continue
        counts = (
            g.set_index("timestamp")
            .assign(_one=1)["_one"]
            .rolling(ROLLING_WINDOW)
            .sum()
        )
        out.loc[g.index] = counts.values
    return out.to_frame()


def _text_features(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer, TruncatedSVD]:
    messages = df["message"].fillna("").astype(str)
    n_docs = len(messages)
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
    tfidf = vectorizer.fit_transform(messages if n_docs > 0 else [""])

    n_components = min(TEXT_SVD_COMPONENTS, max(1, tfidf.shape[1] - 1), max(1, n_docs - 1))
    n_components = max(1, n_components)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf)

    cols = [f"text_svd_{i}" for i in range(reduced.shape[1])]
    return pd.DataFrame(reduced, columns=cols, index=df.index), vectorizer, svd


def build_feature_matrix(df: pd.DataFrame):
    """
    Returns (feature_df, scaler, vectorizer, svd) so the same transformers
    can be reused at inference time if the model is persisted.
    """
    df = df.copy()
    df["message_length"] = df["message"].fillna("").astype(str).str.len()

    temporal = _temporal_features(df)
    severity = _severity_features(df)
    volume = _volume_features(df)
    text_df, vectorizer, svd = _text_features(df)

    numeric = pd.concat(
        [temporal, severity, volume, df[["message_length"]], text_df],
        axis=1,
    ).fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)
    feature_df = pd.DataFrame(scaled, columns=numeric.columns, index=df.index)

    return feature_df, scaler, vectorizer, svd
