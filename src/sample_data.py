"""Synthetic demo data. NOT real prices, NOT from the CSE.

The prices are a random walk: yesterday's movement tells you nothing about
tomorrow's. That is deliberate. Later, a sensible model should score about 50%
on this data. If it scored 90%, you would know you had a leak in your code.
Never present anything computed from this data as evidence about the CSE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Made-up names that cannot be confused with real CSE tickers.
SYNTHETIC_START_PRICES_LKR = {"SYNTH-A": 120.0, "SYNTH-B": 45.0, "SYNTH-C": 310.0}


def make_synthetic_sample(seed: int = 42, periods: int = 300) -> pd.DataFrame:
    """Return a table in the same format as the CSV the app expects."""
    rng = np.random.default_rng(seed)
    weekdays = pd.bdate_range("2025-01-01", periods=periods)  # Monday-Friday only

    frames = []
    for symbol, start_price in SYNTHETIC_START_PRICES_LKR.items():
        # Each symbol is missing a different ~4% of sessions (like holidays or
        # days with no trading), so symbols do NOT share identical dates.
        dates = weekdays[rng.random(len(weekdays)) > 0.04]
        if symbol == "SYNTH-A":  # one deliberate two-week hole, to see a visible gap
            keep = (dates < pd.Timestamp("2025-06-02")) | (dates >= pd.Timestamp("2025-06-16"))
            dates = dates[keep]

        n = len(dates)
        close = start_price * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
        open_ = np.concatenate(([start_price], close[:-1])) * np.exp(rng.normal(0, 0.005, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
        volume = rng.lognormal(mean=9.5, sigma=0.6, size=n).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "Date": dates.strftime("%Y-%m-%d"),
                    "Symbol": symbol,
                    "Open": open_.round(2),
                    "High": high.round(2),
                    "Low": low.round(2),
                    "Close": close.round(2),
                    "Volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
