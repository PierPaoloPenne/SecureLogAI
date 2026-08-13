# Architecture

This document goes one level deeper than the README's system-design summary:
component responsibilities, the request sequence, the model-selection
reasoning, and scalability notes for anyone extending the prototype.

## 1. Component overview

| Component | File | Responsibility | Depends on |
|---|---|---|---|
| Parser | `app/parser.py` | Detects log format, converts raw text/CSV into a structured DataFrame (`timestamp`, `level`, `source`, `message`) | pandas |
| Feature builder | `app/features.py` | Converts the structured DataFrame into a numeric, scaled feature matrix | pandas, scikit-learn |
| Anomaly model | `app/model.py` | Fits Isolation Forest and produces a 0-100 anomaly score + boolean flag per event | scikit-learn, joblib |
| Utilities | `app/utils.py` | Upload validation, row-count limits, logging, static/interactive plot generation | matplotlib, plotly |
| Web app | `app/main.py` | FastAPI routes tying the above together, template rendering, output persistence | FastAPI, Jinja2 |
| Templates | `app/templates/*.html` | Upload form and results dashboard | — |

Each module is independently unit-testable and has no hidden dependency on
FastAPI request/response objects, which is what makes it possible to reuse
the exact same functions in `notebooks/anomaly_experiments.ipynb`.

## 2. Request sequence — `POST /analyze`

```
User            FastAPI (main.py)     parser        features        model         utils
 |  upload file         |                 |               |             |             |
 |--------------------->|                 |               |             |             |
 |                      |--validate-------------------------------------------------->|
 |                      |<--ok/err----------------------------------------------------|
 |                      |--parse_log_file->|              |             |             |
 |                      |<--ParseResult----|              |             |             |
 |                      |--enforce_row_limit----------------------------------------->|
 |                      |--build_feature_matrix---------->|             |             |
 |                      |<--feature_df, scaler, ...-------|             |             |
 |                      |--run_detection------------------------------->|             |
 |                      |<--scores (anomaly_score, is_anomaly)----------|             |
 |                      |--redact_message per row------------------------------------>|
 |                      |--generate_static_plot / interactive plot------------------->|
 |                      |--write outputs/anomaly_results.csv, anomaly_plot.png        |
 |<--results.html-------|                 |               |             |             |
```

If validation fails, the pipeline short-circuits and re-renders `index.html`
with an inline error rather than a generic 500 — this matters for a tool
whose users are actively triaging something, where a confusing failure is
worse than a slow one.

## 3. Feature design rationale

The feature matrix combines four signal types, each targeting a different
kind of anomaly:

1. **Temporal** (`hour`, `day_of_week`, `is_weekend`,
   `seconds_since_prev_same_source`) — catches things that are *timed*
   strangely: activity at 3 a.m. from a service that's normally quiet then,
   or a burst of events from the same source far faster than its usual
   cadence (the classic brute-force signature).
2. **Severity** (`severity`, mapped from log level) — a blunt but useful
   signal; an unusual severity for a given source is often anomalous even
   before looking at the message text.
3. **Volume** (`events_per_min_same_source`, a rolling 60-second count) —
   directly targets flooding/DoS-shaped patterns that a single-event view
   would miss entirely.
4. **Text** (TF-IDF over `message`, reduced to 8 components via
   TruncatedSVD) — captures *what* is being said. TF-IDF was chosen over a
   full embedding model because it needs no pretrained weights, runs
   instantly on CPU, and stays interpretable (you can trace a flagged event
   back to the vocabulary driving its score) — an explicit trade-off for a
   transparent, dependency-light prototype over a heavier NLP pipeline.

All numeric features are combined and passed through `StandardScaler` so
that Isolation Forest's tree splits aren't dominated by whichever feature
happens to have the largest raw numeric range.

## 4. Model selection: Isolation Forest vs. alternatives

| Model | Needs labels? | Scores new events without refit? | Notes |
|---|---|---|---|
| **Isolation Forest** (chosen) | No | Yes | Tree-based partitioning; anomalies are isolated in fewer splits. Good default for mixed numeric/text-derived features at this scale. |
| Local Outlier Factor | No | No (density is computed relative to the fitted set) | Explored in the notebook; strong at finding *local* density anomalies, but its design point is scoring a fixed dataset, which is a poor fit for a "score whatever gets uploaded next" service. |
| Autoencoder (neural net) | No (but needs more data to train well) | Yes | Considered and rejected for this prototype: needs a deep learning framework, meaningfully more training data than a demo log file provides, and loses the interpretability TF-IDF gives us. Worth revisiting if the project moves toward continuous, high-volume ingestion. |
| Rule-based / keyword matching | No (hand-authored) | Yes | The status quo this project is explicitly positioned against — see README "Problem statement". Fast and explainable, but brittle to novel failure modes and prone to the alert-fatigue problem cited in the market research. |

## 5. Data & privacy notes

### Why synthetic data
Real production logs weren't used for the sample dataset in this repo for
two reasons: (1) real logs from any live system are very likely to contain
internal hostnames, IPs, user identifiers, or other sensitive data that
shouldn't be committed to a public academic repository, and (2) a synthetic
dataset with deliberately injected anomalies (a brute-force burst plus
scattered one-off anomalies — see `data/generate_sample_logs.py`) gives a
known ground truth to sanity-check the model against, which real
unlabeled logs don't.

### If real logs are used
Anyone pointing this at real log data should treat `outputs/` as sensitive:
it's an unfiltered copy of the input messages, ranked. The README's
Security section lists what's still missing for that use case (redaction
before storage, encryption at rest, access control).

## 6. Scalability notes (beyond this prototype's scope)

The current design assumes "one file, one synchronous request." Moving
toward continuous ingestion would need:
- A streaming/batched fit-and-score loop instead of fitting Isolation
  Forest fresh on every upload (e.g., periodic retraining on a rolling
  window, with `AnomalyDetector.save`/`.load` already in place to support
  this).
- A real datastore instead of `outputs/*.csv`, since CSV rewrite-on-every-
  request doesn't scale past a single-user demo.
- Async/background processing (FastAPI supports this natively) so a large
  upload doesn't block the request thread.
- Horizontal scaling of the TF-IDF + Isolation Forest step, which is
  currently single-process.

These are called out explicitly rather than implemented, in line with the
project's financial-constraint goal of a $0, laptop-runnable prototype
(see README §4).
