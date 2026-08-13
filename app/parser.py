"""
parser.py
---------
Turns raw log input (plain-text log files or CSV exports) into a clean,
structured pandas DataFrame that the rest of the pipeline can work with.

Supported inputs:
    1. CSV files that already have columns like timestamp/level/message.
    2. Plain-text logs in a common "syslog-ish" shape:
           2026-06-01 08:14:02 INFO  auth-service: user 4471 logged in
       or Apache/Nginx combined-log-format lines.

The parser is intentionally forgiving: if a line doesn't match a known
pattern it is still kept, with the raw text stored in `message` and the
other fields left as NaN, rather than being silently dropped. Dropping
data before a security review is worse than keeping a partially-parsed row.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------------------
# Regex patterns for the log shapes we know how to parse
# --------------------------------------------------------------------------

# 2026-06-01 08:14:02 INFO  auth-service: user 4471 logged in
GENERIC_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>[A-Za-z]+)\s+"
    r"(?P<source>[\w\-\.]+)?:?\s*"
    r"(?P<message>.*)$"
)

# Jun  1 08:14:02 host process[1234]: message text
SYSLOG_PATTERN = re.compile(
    r"^(?P<timestamp>[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<source>\S+)\s+"
    r"(?P<process>\S+?):\s*"
    r"(?P<message>.*)$"
)

# 127.0.0.1 - - [01/Jun/2026:08:14:02 +0000] "GET /login HTTP/1.1" 401 512
APACHE_COMBINED_PATTERN = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<request>[^"]*)"\s+(?P<status>\d{3})\s+(?P<size>\S+)'
)

KNOWN_LEVELS = {"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL", "FATAL"}


@dataclass
class ParseResult:
    df: pd.DataFrame
    total_lines: int
    parsed_lines: int
    fmt_detected: str

    @property
    def parse_rate(self) -> float:
        return 0.0 if self.total_lines == 0 else self.parsed_lines / self.total_lines


def _parse_generic_line(line: str) -> dict | None:
    m = GENERIC_PATTERN.match(line)
    if not m:
        return None
    d = m.groupdict()
    level = (d.get("level") or "INFO").upper()
    if level not in KNOWN_LEVELS:
        level = "INFO"
    return {
        "timestamp": d["timestamp"],
        "level": level,
        "source": d.get("source") or "unknown",
        "message": d.get("message") or "",
    }


def _parse_syslog_line(line: str) -> dict | None:
    m = SYSLOG_PATTERN.match(line)
    if not m:
        return None
    d = m.groupdict()
    return {
        "timestamp": d["timestamp"],
        "level": "INFO",
        "source": d.get("source") or "unknown",
        "message": f'{d.get("process", "")}: {d.get("message", "")}'.strip(": "),
    }


def _parse_apache_line(line: str) -> dict | None:
    m = APACHE_COMBINED_PATTERN.match(line)
    if not m:
        return None
    d = m.groupdict()
    status = int(d["status"])
    level = "ERROR" if status >= 500 else ("WARN" if status >= 400 else "INFO")
    return {
        "timestamp": d["timestamp"],
        "level": level,
        "source": d.get("ip") or "unknown",
        "message": f'{d.get("request", "")} status={status} size={d.get("size", "")}',
    }


LINE_PARSERS = [
    ("generic", _parse_generic_line),
    ("syslog", _parse_syslog_line),
    ("apache_combined", _parse_apache_line),
]


def parse_text_log(raw_text: str) -> ParseResult:
    """Parse a raw multi-line log string into a DataFrame."""
    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return ParseResult(pd.DataFrame(columns=["timestamp", "level", "source", "message"]), 0, 0, "empty")

    # Detect the best-fitting parser using a small sample of lines.
    sample = lines[: min(25, len(lines))]
    scores = {}
    for name, fn in LINE_PARSERS:
        scores[name] = sum(1 for ln in sample if fn(ln) is not None)
    fmt_name, fn = max(zip(scores.keys(), (dict(LINE_PARSERS)[k] for k in scores)), key=lambda kv: scores[kv[0]])

    rows = []
    parsed_count = 0
    for ln in lines:
        rec = fn(ln)
        if rec is None:
            # Fall back to a raw row so nothing is silently discarded.
            rec = {"timestamp": None, "level": None, "source": None, "message": ln}
        else:
            parsed_count += 1
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # If parsing produced too many NaT timestamps, forward-fill isn't safe for
    # security data, so we simply keep NaT and let downstream feature code cope.
    return ParseResult(df, len(lines), parsed_count, fmt_name)


def parse_csv_log(file_bytes: bytes) -> ParseResult:
    """Parse a CSV log export. Expects at least a message-like column."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip().lower() for c in df.columns]

    col_map = {}
    for c in df.columns:
        if c in {"timestamp", "time", "datetime", "date"}:
            col_map[c] = "timestamp"
        elif c in {"level", "loglevel", "severity"}:
            col_map[c] = "level"
        elif c in {"source", "host", "service", "component"}:
            col_map[c] = "source"
        elif c in {"message", "msg", "log", "text"}:
            col_map[c] = "message"
    df = df.rename(columns=col_map)

    for required in ("timestamp", "level", "source", "message"):
        if required not in df.columns:
            df[required] = None

    df = df[["timestamp", "level", "source", "message"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["level"] = df["level"].fillna("INFO").astype(str).str.upper()
    df["source"] = df["source"].fillna("unknown")
    df["message"] = df["message"].fillna("")

    return ParseResult(df, len(df), int(df["message"].notna().sum()), "csv")


def parse_log_file(filename: str, file_bytes: bytes) -> ParseResult:
    """Entry point used by the API layer — dispatches on file extension."""
    if filename.lower().endswith(".csv"):
        return parse_csv_log(file_bytes)
    text = file_bytes.decode("utf-8", errors="replace")
    return parse_text_log(text)
