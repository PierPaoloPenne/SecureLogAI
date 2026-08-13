"""
utils.py
--------
Shared helpers: input validation, logging setup, and plot generation.
Kept separate from main.py so they can be unit-tested without spinning up
the web server.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display server required
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# --------------------------------------------------------------------------
# Security-relevant upload constraints (see README "Security" section)
# --------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".log", ".txt", ".csv"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB — generous for a prototype, small enough to block abuse
MAX_ROWS = 200_000  # protects the IsolationForest / TF-IDF step from a memory-exhaustion upload


class UploadValidationError(ValueError):
    pass


def validate_upload(filename: str, file_bytes: bytes) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file type '{ext}'. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    if len(file_bytes) == 0:
        raise UploadValidationError("Uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise UploadValidationError(
            f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit for this prototype."
        )


def enforce_row_limit(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > MAX_ROWS:
        logging.getLogger("securelog").warning(
            "Uploaded log has %s rows; truncating to %s for this prototype.", len(df), MAX_ROWS
        )
        return df.iloc[:MAX_ROWS].copy()
    return df


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("securelog")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def redact_message(message: str, max_len: int = 200) -> str:
    """
    Defensive truncation so an overly long/crafted log message can't blow up
    the HTML results table. This is NOT a substitute for real PII scrubbing —
    see README "Security & Privacy" for the production-grade recommendation.
    """
    if message is None:
        return ""
    message = str(message)
    return message if len(message) <= max_len else message[:max_len] + "…"


def generate_static_plot(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Matplotlib PNG for the outputs/ folder and for offline reports."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    normal = df[~df["is_anomaly"]]
    anomalous = df[df["is_anomaly"]]

    ax.scatter(normal.index, normal["anomaly_score"], s=10, alpha=0.5, label="Normal", color="#4C72B0")
    ax.scatter(anomalous.index, anomalous["anomaly_score"], s=28, alpha=0.9, label="Anomaly", color="#C44E52")
    ax.set_xlabel("Log event index")
    ax.set_ylabel("Anomaly score (0-100)")
    ax.set_title("SecureLog AI — Anomaly Scores per Log Event")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_interactive_plot_html(df: pd.DataFrame) -> str:
    """Plotly HTML fragment embedded directly into results.html."""
    plot_df = df.reset_index().rename(columns={"index": "event_index"})
    fig = px.scatter(
        plot_df,
        x="event_index",
        y="anomaly_score",
        color="is_anomaly",
        color_discrete_map={True: "#C44E52", False: "#4C72B0"},
        hover_data=["level", "source", "message"] if "message" in plot_df.columns else None,
        title="Anomaly Scores per Log Event",
        labels={"is_anomaly": "Anomaly"},
    )
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=50, b=40))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
