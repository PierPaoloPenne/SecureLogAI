"""
generate_sample_logs.py
------------------------
Generates a small synthetic log dataset for demoing and testing the
pipeline, since real production logs aren't available for an academic
prototype and shouldn't be committed to a public repo anyway (see
docs/architecture.md "Why synthetic data").

Produces:
    data/sample_logs.csv      - raw-looking structured log export
    data/processed_logs.csv   - the same data after app/parser.py + app/features.py

Run with:  python data/generate_sample_logs.py
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)

HERE = Path(__file__).resolve().parent

SOURCES = ["auth-service", "api-gateway", "payment-service", "db-primary", "web-frontend"]
NORMAL_MESSAGES = [
    "user {uid} logged in",
    "user {uid} logged out",
    "GET /api/v1/orders status=200",
    "GET /api/v1/products status=200",
    "cache hit for key session:{uid}",
    "scheduled health check passed",
    "connection pool at 42% capacity",
    "request completed in {ms}ms",
]
ANOMALOUS_MESSAGES = [
    "failed login attempt for user {uid} — invalid password",
    "failed login attempt for user {uid} — invalid password",
    "failed login attempt for user {uid} — invalid password",
    "SQL syntax error near 'UNION SELECT' in query",
    "unauthorized access attempt to /admin/config",
    "connection pool exhausted, 100% capacity",
    "repeated 401 responses from IP 203.0.113.{oct}",
    "unexpected process termination, signal=SIGSEGV",
    "certificate validation failed for upstream host",
]


def build_dataset(n_normal: int = 480, n_burst_anomalies: int = 14, n_scattered_anomalies: int = 10) -> pd.DataFrame:
    start = datetime(2026, 6, 1, 0, 0, 0)
    rows = []

    # Normal traffic spread across 24 hours
    for i in range(n_normal):
        ts = start + timedelta(seconds=int(np.random.exponential(180) * i / 4))
        template = random.choice(NORMAL_MESSAGES)
        msg = template.format(uid=random.randint(1000, 9999), ms=random.randint(10, 300))
        level = "INFO" if random.random() > 0.05 else "WARN"
        rows.append(
            {"timestamp": ts, "level": level, "source": random.choice(SOURCES), "message": msg}
        )

    # A brute-force burst: many failed logins in a tight time window from one source
    burst_start = start + timedelta(hours=3, minutes=12)
    for i in range(n_burst_anomalies):
        ts = burst_start + timedelta(seconds=i * 2)
        msg = random.choice(ANOMALOUS_MESSAGES[:3]).format(uid=4471)
        rows.append({"timestamp": ts, "level": "ERROR", "source": "auth-service", "message": msg})

    # Scattered one-off anomalies through the rest of the day
    for i in range(n_scattered_anomalies):
        ts = start + timedelta(hours=random.uniform(0, 23), minutes=random.uniform(0, 59))
        msg = random.choice(ANOMALOUS_MESSAGES[3:]).format(oct=random.randint(1, 254))
        rows.append(
            {"timestamp": ts, "level": random.choice(["ERROR", "CRITICAL"]), "source": random.choice(SOURCES), "message": msg}
        )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE.parent))
    from app import features, model, parser  # noqa: E402

    df = build_dataset()
    raw_path = HERE / "sample_logs.csv"
    df.to_csv(raw_path, index=False)
    print(f"Wrote {len(df)} rows to {raw_path}")

    # Run it through the real pipeline once so processed_logs.csv reflects
    # actual output shape, not a hand-written mock.
    parsed = parser.parse_csv_log(raw_path.read_bytes())
    feature_df, *_ = features.build_feature_matrix(parsed.df)
    scores = model.run_detection(feature_df)
    processed = pd.concat([parsed.df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    processed = processed.sort_values("anomaly_score", ascending=False)

    processed_path = HERE / "processed_logs.csv"
    processed.to_csv(processed_path, index=False)
    print(f"Wrote {len(processed)} rows to {processed_path}")
    print(f"Flagged {int(processed['is_anomaly'].sum())} anomalies")
