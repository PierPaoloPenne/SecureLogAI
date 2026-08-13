"""
main.py
-------
FastAPI application tying the pipeline together:

upload -> parser.parse_log_file -> features.build_feature_matrix
-> model.run_detection -> utils plots -> results.html

Run locally with:
uvicorn app.main:app --reload
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import features, model, parser, utils

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

logger = utils.setup_logging()

app = FastAPI(
    title="SecureLog AI",
    description="ML-powered log anomaly detector prototype",
    version="0.1.0",
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"request": request},
    )


@app.get("/health")
async def health():
    """Simple liveness endpoint for deployment health checks."""
    return {"status": "ok"}


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, log_file: UploadFile = File(...)):
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]

    file_bytes = await log_file.read()

    try:
        utils.validate_upload(log_file.filename, file_bytes)
    except utils.UploadValidationError as exc:
        logger.warning("[%s] Rejected upload: %s", request_id, exc)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request, "error": str(exc)},
            status_code=400,
        )

    parsed = parser.parse_log_file(log_file.filename, file_bytes)
    df = utils.enforce_row_limit(parsed.df)

    if df.empty:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "error": "No parsable log lines were found in the uploaded file.",
            },
            status_code=422,
        )

    feature_df, _scaler, _vectorizer, _svd = features.build_feature_matrix(df)
    scores = model.run_detection(feature_df)

    results = pd.concat([df.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    results["message"] = results["message"].apply(utils.redact_message)
    results = results.sort_values("anomaly_score", ascending=False)

    results_csv_path = OUTPUTS_DIR / "anomaly_results.csv"
    results.to_csv(results_csv_path, index=False)

    plot_path = OUTPUTS_DIR / "anomaly_plot.png"
    utils.generate_static_plot(results, plot_path)
    interactive_plot_html = utils.generate_interactive_plot_html(results)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "[%s] Analyzed %s rows (%s anomalies) in %sms — format=%s parse_rate=%.1f%%",
        request_id,
        len(results),
        int(results["is_anomaly"].sum()),
        elapsed_ms,
        parsed.fmt_detected,
        parsed.parse_rate * 100,
    )

    summary = {
        "total_events": len(results),
        "anomaly_count": int(results["is_anomaly"].sum()),
        "anomaly_rate": round(results["is_anomaly"].mean() * 100, 2),
        "detected_format": parsed.fmt_detected,
        "parse_rate": round(parsed.parse_rate * 100, 1),
        "elapsed_ms": elapsed_ms,
    }

    top_anomalies = results[results["is_anomaly"]].head(25).to_dict(orient="records")

    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "request": request,
            "summary": summary,
            "anomalies": top_anomalies,
            "plot_html": interactive_plot_html,
            "csv_url": "/outputs/anomaly_results.csv",
        },
    )