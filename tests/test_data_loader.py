"""Stage 1 tests. Run from the project folder with:  python -m pytest -v"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.charts import price_chart  # noqa: E402
from src.data_loader import find_gaps, load_csv  # noqa: E402
from src.sample_data import make_synthetic_sample  # noqa: E402

HEADER = "Date,Symbol,Open,High,Low,Close,Volume\n"


def load(rows: str):
    return load_csv(io.StringIO(HEADER + rows))


def test_good_file_is_sorted_per_symbol():
    res = load(
        "2026-03-03,B,10,11,9,10.5,100\n"
        "2026-03-02,B,10,11,9,10.0,100\n"
        "2026-03-02,A,50,51,49,50.5,200\n"
    )
    assert res.ok
    assert list(res.data["Symbol"]) == ["A", "B", "B"]
    b = res.data[res.data["Symbol"] == "B"]
    assert b["Date"].is_monotonic_increasing


def test_missing_column_is_reported_by_name():
    res = load_csv(io.StringIO("Date,Symbol,Open,High,Low,Close\n2026-03-02,A,1,2,1,1.5\n"))
    assert not res.ok
    assert "Volume" in res.errors[0]


def test_duplicate_symbol_date_is_an_error_but_same_date_different_symbol_is_fine():
    ok = load("2026-03-02,A,1,2,1,1.5,10\n2026-03-02,B,1,2,1,1.5,10\n")
    assert ok.ok
    bad = load("2026-03-02,A,1,2,1,1.5,10\n2026-03-02,A,1,2,1,1.6,12\n")
    assert not bad.ok
    assert any("Duplicate" in e and "A on 2026-03-02" in e for e in bad.errors)


def test_invalid_prices_zero_negative_text_blank():
    res = load(
        "2026-03-02,A,0,2,1,1.5,10\n"
        "2026-03-03,A,1,2,1,-3,10\n"
        "2026-03-04,A,1,2,1,abc,10\n"
        "2026-03-05,A,1,2,1,,10\n"
    )
    assert not res.ok
    text = " ".join(res.errors)
    assert "Open must be greater than 0" in text
    assert "Close must be greater than 0" in text
    assert "Close must be a plain number" in text


def test_negative_volume_is_an_error_but_zero_volume_is_a_warning():
    assert not load("2026-03-02,A,1,2,1,1.5,-1\n").ok
    zero = load("2026-03-02,A,1,2,1,1.5,0\n")
    assert zero.ok and zero.warnings


def test_high_below_low_is_an_error():
    res = load("2026-03-02,A,1,1,2,1.5,10\n")
    assert not res.ok and any("High is lower than Low" in e for e in res.errors)


def test_ambiguous_date_format_is_rejected_with_example():
    res = load("04/03/2026,A,1,2,1,1.5,10\n")
    assert not res.ok and "YYYY-MM-DD" in res.errors[0]


def test_all_problems_are_reported_together_with_csv_row_numbers():
    bad = (Path(__file__).resolve().parent.parent / "data" / "bad_example.csv").read_text()
    res = load_csv(io.StringIO(bad))
    assert not res.ok
    assert len(res.errors) >= 5
    assert any("row(s) 8" in e for e in res.errors)  # the DD/MM/YYYY date is CSV row 8


def test_excel_bom_and_lowercase_headers_are_accepted():
    text = "﻿date,symbol,open,high,low,close,volume\n2026-03-02, abcd ,1,2,1,1.5,10\n"
    res = load_csv(io.BytesIO(text.encode("utf-8")))
    assert res.ok and res.data.loc[0, "Symbol"] == "ABCD"


def test_gaps_are_found_and_no_rows_are_invented():
    res = load(
        "2026-03-02,A,1,2,1,1.5,10\n"
        "2026-03-03,A,1,2,1,1.5,10\n"
        "2026-03-20,A,1,2,1,1.5,10\n"
    )
    gaps = find_gaps(res.data, min_gap_days=5)
    assert len(gaps) == 1 and gaps.loc[0, "Calendar days between"] == 17
    assert len(res.data) == 3  # still exactly the rows that were uploaded


def test_chart_breaks_line_at_gap_without_touching_stored_data():
    res = load(
        "2026-03-02,A,1,2,1,1.5,10\n"
        "2026-03-03,A,1,2,1,1.6,10\n"
        "2026-03-20,A,1,2,1,1.7,10\n"
    )
    before = res.data.copy()
    fig = price_chart(res.data, "A", min_gap_days=5)
    y = np.array(fig.data[0].y, dtype=float)
    assert np.isnan(y).sum() == 1          # one empty point in the plot only
    assert res.data.equals(before)         # stored data unchanged
    assert len(res.data) == 3


def test_synthetic_sample_is_valid_and_clearly_synthetic():
    res = load_csv(io.StringIO(make_synthetic_sample().to_csv(index=False)))
    assert res.ok
    assert all(s.startswith("SYNTH-") for s in res.data["Symbol"].unique())
    a = res.data[res.data["Symbol"] == "SYNTH-A"]
    assert (find_gaps(a, 10)["Calendar days between"] >= 14).any()  # the built-in 2-week gap
