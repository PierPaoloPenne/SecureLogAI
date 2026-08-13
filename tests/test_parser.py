import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import parser


def test_parse_generic_text_log():
    raw = (
        "2026-06-01 08:14:02 INFO auth-service: user 4471 logged in\n"
        "2026-06-01 08:14:05 ERROR auth-service: failed login attempt for user 4471\n"
    )
    result = parser.parse_text_log(raw)
    assert result.total_lines == 2
    assert result.parsed_lines == 2
    assert list(result.df["level"]) == ["INFO", "ERROR"]
    assert result.df.loc[0, "source"] == "auth-service"
    assert "logged in" in result.df.loc[0, "message"]


def test_parse_syslog_line():
    raw = "Jun  1 08:14:02 web01 sshd[1234]: Accepted password for root\n"
    result = parser.parse_text_log(raw)
    assert result.parsed_lines == 1
    assert result.df.loc[0, "source"] == "web01"


def test_parse_apache_combined_log():
    raw = '127.0.0.1 - - [01/Jun/2026:08:14:02 +0000] "GET /login HTTP/1.1" 401 512\n'
    result = parser.parse_text_log(raw)
    assert result.parsed_lines == 1
    assert result.df.loc[0, "level"] == "WARN"
    assert result.df.loc[0, "source"] == "127.0.0.1"


def test_parse_csv_log():
    csv_bytes = b"timestamp,level,source,message\n2026-06-01 08:00:00,INFO,api,request ok\n"
    result = parser.parse_csv_log(csv_bytes)
    assert result.total_lines == 1
    assert result.df.loc[0, "message"] == "request ok"


def test_unparseable_line_is_kept_not_dropped():
    raw = "this is not a recognizable log line at all\n"
    result = parser.parse_text_log(raw)
    assert len(result.df) == 1
    assert result.df.loc[0, "message"] == raw.strip()


def test_empty_input_returns_empty_frame():
    result = parser.parse_text_log("")
    assert result.df.empty
    assert result.total_lines == 0


def test_parse_log_file_dispatches_on_extension():
    csv_bytes = b"timestamp,level,source,message\n2026-06-01 08:00:00,INFO,api,ok\n"
    result = parser.parse_log_file("sample.csv", csv_bytes)
    assert result.fmt_detected == "csv"

    text_bytes = b"2026-06-01 08:14:02 INFO api: ok\n"
    result2 = parser.parse_log_file("sample.log", text_bytes)
    assert result2.fmt_detected != "csv"
