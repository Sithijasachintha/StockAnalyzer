"""Stage 1: read, validate and tidy a CSV of daily stock prices.

Design rules (worth defending in your presentation):
  * Validation collects ALL problems at once, so the user can fix the file in one go.
  * Nothing is ever invented: no filling of missing sessions, no interpolation.
  * Dates must be YYYY-MM-DD. Formats like 03/04/2026 are ambiguous (3 April or
    4 March?) and a silent wrong guess would corrupt every later calculation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
DATE_FORMAT = "%Y-%m-%d"
MAX_EXAMPLES = 5  # how many bad rows to list per problem


@dataclass
class ValidationResult:
    """What the loader hands back to the app."""

    data: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)    # file cannot be used
    warnings: list[str] = field(default_factory=list)  # file is usable, but look

    @property
    def ok(self) -> bool:
        return not self.errors and self.data is not None


def _rows_text(mask: pd.Series) -> str:
    """'4, 9, 12 (and 3 more)'. Row numbers match the CSV file (header = row 1)."""
    rows = [int(i) + 2 for i in mask[mask].index]
    text = ", ".join(str(r) for r in rows[:MAX_EXAMPLES])
    extra = len(rows) - MAX_EXAMPLES
    return f"{text} (and {extra} more)" if extra > 0 else text


def load_csv(source) -> ValidationResult:
    """Read a CSV (path or file-like object) and validate it."""
    try:
        # dtype=str: we convert and check every column ourselves, so a bad value
        # becomes a readable message instead of a pandas crash.
        # utf-8-sig: Excel's "CSV UTF-8" adds an invisible marker at the start.
        raw = pd.read_csv(
            source,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
            skipinitialspace=True,
        )
    except pd.errors.EmptyDataError:
        return ValidationResult(errors=["The file is empty."])
    except UnicodeDecodeError:
        return ValidationResult(
            errors=["The file is not UTF-8 text. In Excel use Save As > 'CSV UTF-8'."]
        )
    except pd.errors.ParserError as exc:
        return ValidationResult(errors=[f"The file is not a well-formed CSV: {exc}"])
    return validate_prices(raw)


def validate_prices(raw: pd.DataFrame) -> ValidationResult:
    """Check a raw table and return clean data (sorted by Symbol, then Date)."""
    result = ValidationResult()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # --- 1. required columns (case-insensitive) ---------------------------------
    lookup = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED_COLUMNS if c.lower() not in lookup]
    if missing:
        result.errors.append(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Columns found: {', '.join(df.columns) or '(none)'}."
        )
        return result
    df = df.rename(columns={lookup[c.lower()]: c for c in REQUIRED_COLUMNS})
    df = df[REQUIRED_COLUMNS]
    if df.empty:
        result.errors.append("The file has a header but no data rows.")
        return result

    clean = pd.DataFrame(index=df.index)

    # --- 2. dates -----------------------------------------------------------------
    date_text = df["Date"].astype("string").fillna("").str.strip()
    clean["Date"] = pd.to_datetime(date_text, format=DATE_FORMAT, errors="coerce")
    bad_date = clean["Date"].isna()
    if bad_date.any():
        result.errors.append(
            "Dates must be written YYYY-MM-DD (for example 2026-03-27). "
            f"Invalid date on CSV row(s) {_rows_text(bad_date)}; "
            f"first bad value: {date_text[bad_date].iloc[0]!r}."
        )

    # --- 3. symbols ---------------------------------------------------------------
    clean["Symbol"] = (
        df["Symbol"].astype("string").fillna("").str.strip().str.upper().astype(str)
    )
    blank_symbol = clean["Symbol"] == ""
    if blank_symbol.any():
        result.errors.append(f"Empty Symbol on CSV row(s) {_rows_text(blank_symbol)}.")

    # --- 4. numbers: must be plain numbers, no commas or currency signs ------------
    for col in PRICE_COLUMNS + ["Volume"]:
        text = df[col].astype("string").fillna("").str.strip()
        values = pd.to_numeric(text, errors="coerce").astype("float64")
        clean[col] = values
        not_a_number = ~np.isfinite(values)
        if not_a_number.any():
            result.errors.append(
                f"{col} must be a plain number (no commas, no 'LKR'). Missing or "
                f"non-numeric value on CSV row(s) {_rows_text(not_a_number)}; "
                f"first bad value: {text[not_a_number].iloc[0]!r}."
            )

    # --- 5. price and volume rules --------------------------------------------------
    for col in PRICE_COLUMNS:
        non_positive = clean[col] <= 0  # NaN compares False, so no double-reporting
        if non_positive.any():
            result.errors.append(
                f"{col} must be greater than 0. Invalid price on CSV row(s) "
                f"{_rows_text(non_positive)}."
            )
    negative_volume = clean["Volume"] < 0
    if negative_volume.any():
        result.errors.append(
            f"Volume cannot be negative. Problem on CSV row(s) {_rows_text(negative_volume)}."
        )
    high_below_low = clean["High"] < clean["Low"]
    if high_below_low.any():
        result.errors.append(
            f"High is lower than Low on CSV row(s) {_rows_text(high_below_low)}."
        )

    # --- 6. duplicate Symbol + Date ----------------------------------------------------
    usable = ~bad_date & ~blank_symbol
    dup = clean.duplicated(["Symbol", "Date"], keep=False) & usable
    if dup.any():
        groups = clean[dup].groupby(["Symbol", "Date"])
        examples = []
        for (symbol, date), grp in groups:
            rows = ", ".join(str(i + 2) for i in grp.index)
            examples.append(f"{symbol} on {date:%Y-%m-%d} (CSV rows {rows})")
            if len(examples) == MAX_EXAMPLES:
                break
        more = groups.ngroups - len(examples)
        suffix = f" ... and {more} more" if more > 0 else ""
        result.errors.append(
            "Duplicate Symbol + Date rows (each symbol may appear once per date): "
            + "; ".join(examples)
            + suffix
            + "."
        )

    if result.errors:
        return result  # no data is returned for a file with errors

    # --- 7. warnings: usable file, but the user should know -------------------------------
    zero_volume = clean["Volume"] == 0
    if zero_volume.any():
        result.warnings.append(
            f"{int(zero_volume.sum())} row(s) have zero volume (no shares traded). "
            "Their prices may be carried over from an earlier session."
        )
    outside = (clean["Open"] > clean["High"]) | (clean["Open"] < clean["Low"]) | (
        clean["Close"] > clean["High"]
    ) | (clean["Close"] < clean["Low"])
    if outside.any():
        result.warnings.append(
            f"{int(outside.sum())} row(s) have an Open or Close outside the Low-High "
            f"range (CSV row(s) {_rows_text(outside)}). Check for typos or adjusted prices."
        )

    # --- 8. tidy: sort each symbol by date. Rows are NEVER added or filled in. -------
    result.data = clean[REQUIRED_COLUMNS].sort_values(
        ["Symbol", "Date"], kind="stable"
    ).reset_index(drop=True)
    return result


def find_gaps(symbol_df: pd.DataFrame, min_gap_days: int = 5) -> pd.DataFrame:
    """List stretches with no recorded session for ONE symbol.

    A gap is two consecutive recorded dates at least `min_gap_days` calendar days
    apart. A normal weekend is 3 days (Fri -> Mon), a single public holiday makes
    4, so the default of 5 flags only the longer silences. The reason for a gap
    (holiday, no trades, missing data) cannot be known from prices alone.
    """
    dates = symbol_df.sort_values("Date")["Date"].reset_index(drop=True)
    days_between = dates.diff().dt.days
    is_gap = days_between >= min_gap_days
    return pd.DataFrame(
        {
            "Last record before gap": dates.shift(1)[is_gap].dt.date.values,
            "First record after gap": dates[is_gap].dt.date.values,
            "Calendar days between": days_between[is_gap].astype(int).values,
        }
    )
