# SecureLog AI — ML-Powered Log Anomaly Detector

An NLP/ML prototype that ingests raw application, server, or security logs and
flags the events most likely to represent something abnormal — a brute-force
login burst, a malformed request, a service failure — without needing
hand-written rules or a labeled training set.

> **Status:** working academic prototype. See [Limitations & Scope](#limitations--scope)
> for what is and isn't production-ready.

---

## Table of contents

1. [Problem statement](#1-problem-statement)
2. [Market research](#2-market-research)
3. [Project plan & timeline](#3-project-plan--timeline)
4. [Cost analysis & financial constraints](#4-cost-analysis--financial-constraints)
5. [System design](#5-system-design)
6. [Tech stack](#6-tech-stack)
7. [Development approach](#7-development-approach)
8. [Security](#8-security)
9. [Getting started](#9-getting-started)
10. [Testing](#10-testing)
11. [Repository layout](#11-repository-layout)
12. [Limitations & scope](#12-limitations--scope)
13. [Team](#13-team)

---

## 1. Problem statement

Modern systems generate far more log data than any human team can read line
by line. Most of it is routine — health checks, successful requests, normal
logins — and the events that matter (an intrusion attempt, a cascading
failure, a misconfiguration) are buried in the noise. Traditional log tools
lean on **static rules and keyword alerts** ("flag any line containing
`ERROR`"), which are brittle: they miss novel failure modes and they
over-trigger on noisy-but-harmless log sources.

SecureLog AI takes a different approach: treat anomaly detection as an
**unsupervised learning problem**. Instead of asking "does this line match a
known bad pattern?", the system asks "how different does this event look
from the normal behavior of this system?" — using temporal patterns
(bursts, odd hours), severity, and the text of the message itself.

**Target users for this prototype:**
- Small engineering/security teams without a SIEM budget
- Students and researchers who want a transparent, inspectable anomaly
  detection pipeline rather than a vendor black box
- Anyone triaging a one-off incident and needing to skim a large log file
  quickly for the events worth reading first

## 2. Market research

Log management and anomaly detection is a large and actively growing space,
which is exactly why a lightweight, transparent, open-source-stack
alternative has a niche:

- Multiple industry market-research firms put the global **log management
  software market in the low-to-mid single-digit billions of USD as of
  2026**, with consistent forecasts of roughly **7-16% CAGR** through the
  early-to-mid 2030s as log volumes keep growing. Estimates vary noticeably
  between research firms (a sign to treat any single number as directional,
  not precise), but the direction — steady, strong growth — is consistent
  across sources.
- The broader **AI/ML-driven anomaly detection market** is estimated in the
  ~$5 billion range in 2025-2026, also forecast to grow at a **double-digit
  CAGR** over the next decade, driven by finance, telecom, and digital
  commerce adoption.
- Reports consistently cite **AI-driven log analytics as a fast-growing
  segment within log management**, with a majority of newly released
  commercial platforms now embedding some form of automated anomaly
  detection, and vendors reporting meaningful reductions in mean-time-to-
  investigate when ML is used instead of purely rule-based alerting.
- A recurring pain point across reports is **alert fatigue** — security
  teams report high false-positive rates from rule-based systems, which is
  the core justification for an unsupervised, score-ranked approach like
  this prototype's rather than another keyword-matching rule engine.

**Competitive landscape:** established players (Splunk, Datadog, Sumo
Logic, LogRhythm, IBM QRadar, Graylog) offer far more complete platforms —
ingestion pipelines, long-term storage, compliance tooling, SOC workflows —
but are priced and scoped for enterprise deployments. SecureLog AI does not
compete with these; it demonstrates the **core ML technique** (unsupervised
anomaly scoring over log text and behavior) in a form small teams, students,
or a single engineer investigating an incident can run locally in minutes
with an entirely free, open-source stack.

*(Figures above are paraphrased from several 2026 industry market-research
reports; they are included to motivate the project's direction, not as
verified ground truth — see each report's methodology for details.)*

## 3. Project plan & timeline

Planned as a **6-week** part-time academic project for a two-person team.
Weeks can compress if working full-time.

| Phase | Weeks | Key deliverables | Owner focus |
|---|---|---|---|
| **1. Planning & research** | Week 1 | Problem definition, market research, architecture sketch, dataset selection, repo scaffolding | Both |
| **2. Parsing & data pipeline** | Week 2 | `parser.py`, sample dataset, log-format detection, unit tests | NLP/data member |
| **3. Feature engineering** | Week 3 | `features.py`, TF-IDF + temporal + volume features, notebook experiments | NLP/data member |
| **4. Model & scoring** | Week 3-4 | `model.py`, Isolation Forest tuning, score calibration, LOF comparison | NLP/data member |
| **5. Web interface** | Week 4-5 | `main.py`, templates, upload flow, results visualization | Interface/integration member |
| **6. Security hardening & testing** | Week 5 | Upload validation, `tests/`, edge-case handling | Both |
| **7. Docs, polish, reflection** | Week 6 | README, `docs/architecture.md`, demo prep, individual reflective journals | Both |

**Milestones / checkpoints:**
- End of Week 2: raw logs reliably parse into a clean DataFrame for 3+ log formats
- End of Week 4: end-to-end pipeline runs on synthetic data and flags injected anomalies
- End of Week 5: working web demo, deployable locally
- End of Week 6: documentation, tests, and reflective journals complete

## 4. Cost analysis & financial constraints

This project was scoped to run on **$0 recurring cost**, which shaped several
design decisions below.

| Category | Choice | Cost |
|---|---|---|
| Core ML/data stack | Python, pandas, NumPy, scikit-learn | Free, open-source |
| Web interface | FastAPI + Jinja2 (no paid frontend framework/build pipeline) | Free |
| Visualization | Matplotlib + Plotly (client-side rendering, no hosted charting service) | Free |
| Model | Isolation Forest (CPU-only, trains in-process) | Free — no GPU, no paid training compute |
| Hosting (prototype/demo) | Local run, or free tier of a PaaS (e.g. Render/Railway free tier) for a class demo | $0 for the scope of this project |
| Data | Synthetic + public dataset formats (no purchased log feeds) | Free |
| CI | GitHub Actions free tier (2,000 min/month on public repos) | Free |

**Why these choices were made under the constraint:**
- **No managed ML API calls** (e.g., no per-request cloud anomaly-detection
  API) — the model runs locally so there's no usage-based bill and no
  dependency on an external service staying available for grading/demo day.
- **No GPU dependency** — Isolation Forest and TF-IDF are CPU-efficient at
  the row counts this prototype targets (tens of thousands of log lines),
  which avoids needing paid compute.
- **No managed database** — CSV files under `data/` and `outputs/` are
  sufficient for a prototype's scale and keep the project deployable on a
  free hosting tier with no database bill.
- **If this were scaled toward production**, the main new costs would be:
  log ingestion/storage at volume (this is the single biggest line item in
  every commercial log-management product, per the market research above),
  a persistent database instead of CSV files, and possibly a hosted
  vector/text index if the text-similarity features were scaled up.

## 5. System design

### 5.1 Data flow

```
                 ┌──────────────┐
   raw log file  │   parser.py  │  → structured DataFrame
   (.log/.csv) ─►│              │    (timestamp, level, source, message)
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │ features.py  │  → numeric feature matrix
                 │ - temporal   │    (scaled, ready for the model)
                 │ - severity   │
                 │ - volume     │
                 │ - TF-IDF/SVD │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │  model.py    │  → anomaly_score (0-100) + is_anomaly
                 │ Isolation    │
                 │ Forest       │
                 └──────┬───────┘
                        │
                 ┌──────▼───────┐
                 │  main.py     │  → results.html (ranked table + plot)
                 │  (FastAPI)   │    outputs/anomaly_results.csv
                 └──────────────┘    outputs/anomaly_plot.png
```

Full component and sequence diagrams, plus the reasoning behind the model
choice, are in [`docs/architecture.md`](docs/architecture.md).

### 5.2 Why Isolation Forest

Isolation Forest was chosen over alternatives for this prototype because it:
- requires **no labeled anomalies** (log anomalies are essentially never
  pre-labeled in practice);
- produces a **continuous anomaly score**, not just a binary flag, so
  results can be ranked and thresholds adjusted without retraining;
- scales close to linearly with row count, so it stays fast enough for
  interactive use on a laptop;
- generalizes to new events without needing to be refit for every scoring
  request (unlike density-based methods such as Local Outlier Factor,
  which was evaluated for comparison — see the notebook).

## 6. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Data processing | pandas, NumPy |
| ML | scikit-learn (Isolation Forest, TF-IDF, TruncatedSVD, StandardScaler) |
| Web interface | FastAPI, Jinja2 templates, Uvicorn |
| Visualization | Matplotlib (static PNG), Plotly (interactive HTML) |
| Testing | pytest |
| Model persistence | joblib |

Kept deliberately simple, per the brief: no deep learning framework, no
external database, no paid APIs.

## 7. Development approach

- **Modular pipeline** — parsing, feature engineering, modeling, and
  presentation are separate modules (`parser.py`, `features.py`,
  `model.py`, `utils.py`) so each can be unit-tested and iterated on
  independently, and so the notebook in `notebooks/` can reuse the exact
  same functions as the production app instead of drifting into a separate
  copy.
- **Notebook-first experimentation** — feature and model choices (e.g. the
  Isolation Forest `contamination` default, the Isolation Forest vs. Local
  Outlier Factor comparison) were explored in
  `notebooks/anomaly_experiments.ipynb` before being locked into the app.
- **Fail open, not silent** — the parser keeps unparseable lines rather
  than dropping them, and the UI surfaces the detected format and parse
  rate, so a bad upload is visible rather than silently mishandled.
- **Version control** — feature branches per module, PR review between the
  two team members before merging to `main`, and commit history used as
  input to `docs/team-contributions.md`.

## 8. Security

Security was treated as a first-class requirement, not an afterthought,
given the target use case (analyzing security/operational logs):

- **Upload validation** (`app/utils.py::validate_upload`) — file extension
  allow-list (`.log`, `.txt`, `.csv` only), a 10 MB size cap, and an empty-
  file check, to reduce the risk of a malicious or oversized upload
  crashing or stalling the service.
- **Row-count cap** (`MAX_ROWS`) — protects the TF-IDF/Isolation Forest step
  from a memory-exhaustion attack via an artificially huge file.
- **No code execution on log content** — log messages are only ever
  tokenized (TF-IDF) or displayed as escaped text via Jinja2's autoescaping;
  they are never `eval`'d, templated, or passed to a shell.
- **Message truncation in the UI** (`redact_message`) — defends the results
  table from being broken by pathologically long or crafted messages.
- **No outbound network calls** — the entire pipeline runs locally; log
  content is never sent to a third-party API, which matters because logs
  can contain sensitive operational or personal data.
- **Local-first data handling** — this prototype does not persist uploaded
  logs anywhere beyond the current session's `outputs/` files, and does not
  include user authentication, since it's designed for local/single-user
  use, not multi-tenant deployment.

**What a production deployment would still need** (explicitly out of scope
for this academic prototype, called out here rather than glossed over):
authentication and per-tenant data isolation, PII/secret redaction *before*
storage (not just UI truncation), encryption at rest for `outputs/`,
structured audit logging of who ran what analysis, and rate limiting on the
`/analyze` endpoint.

## 9. Getting started

```bash
# 1. Clone and enter the repo
git clone <this-repo-url>
cd securelog-ai

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) regenerate the sample dataset
python data/generate_sample_logs.py

# 5. Run the app
uvicorn app.main:app --reload

# 6. Open the UI
# http://127.0.0.1:8000
```

Upload `data/sample_logs.csv` through the UI for a working demo — it
contains a synthetic brute-force burst and several scattered anomalies for
the model to find.

## 10. Testing

```bash
pytest tests/ -v
```

Covers log parsing (multiple formats, malformed-line handling, CSV
dispatch) and the anomaly model (fit/score shapes, score bounds, save/load
round-trip, unfitted-model guard rail).

## 11. Repository layout

```
securelog-ai/
├── README.md                     # this file
├── requirements.txt
├── .gitignore
├── app/
│   ├── main.py                   # FastAPI app & routes
│   ├── parser.py                 # raw log/CSV -> structured DataFrame
│   ├── features.py               # feature engineering
│   ├── model.py                  # Isolation Forest wrapper
│   ├── utils.py                  # validation, logging, plotting
│   └── templates/
│       ├── index.html            # upload page
│       └── results.html          # results / anomaly table + plot
├── data/
│   ├── sample_logs.csv           # synthetic demo dataset
│   ├── processed_logs.csv        # sample dataset after the pipeline
│   └── generate_sample_logs.py   # regenerates both files above
├── notebooks/
│   └── anomaly_experiments.ipynb # feature/model experimentation
├── outputs/
│   ├── anomaly_results.csv       # latest run's ranked results
│   └── anomaly_plot.png          # latest run's static plot
├── tests/
│   ├── test_parser.py
│   └── test_model.py
└── docs/
    ├── architecture.md           # detailed design + diagrams
    └── team-contributions.md     # contribution log per team member
```

## 12. Limitations & scope

- **Contamination is a guess.** Isolation Forest needs a prior on what
  fraction of events are anomalous; the default (5%) is a reasonable
  starting point, not a calibrated value for any specific real log source.
- **No ground-truth evaluation.** Without labeled anomalies, results are
  validated qualitatively (does it catch the injected synthetic anomalies?)
  rather than with precision/recall metrics.
- **Single-file, single-session.** There's no persistent storage, streaming
  ingestion, or multi-user support — each upload is analyzed independently.
- **Log format coverage is intentionally limited** to a generic timestamped
  format, syslog, Apache/Nginx combined log format, and CSV exports; a
  production system would need a much broader/pluggable parser library.
- **English-oriented text features** — the TF-IDF step assumes
  English-language log messages.

## 13. Team

See [`docs/team-contributions.md`](docs/team-contributions.md) for the
per-member contribution breakdown used for individual reflective journals.

---

*This is an academic prototype built for a hackathon/coursework
submission. It is not audited, load-tested, or intended for production
security monitoring.*
