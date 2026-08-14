"""
analyze.py
----------
Step 1 (continued) of the Stock Analyzer AI project.

What this does:
  Loads the CSV saved by fetch_data.py, computes two simple moving
  averages (20-day and 50-day), prints some basic stats, and saves a
  chart showing price + moving averages.

Why moving averages first:
  A moving average is the simplest possible "feature" you can build from
  price data — it smooths out daily noise so you can see trend direction.
  The classic "golden cross" (20-day MA crossing above 50-day MA) is a
  beginner technical signal, and a good first thing to visualize before
  you build anything ML-related in Phase 3.

Usage:
  python src/analyze.py AAPL
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_data(ticker: str, data_dir: str = "data") -> pd.DataFrame:
    path = os.path.join(data_dir, f"{ticker.upper()}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file at {path}. Run fetch_data.py first: "
            f"python src/fetch_data.py {ticker}"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    return df


def print_summary(df: pd.DataFrame, ticker: str):
    latest = df.iloc[-1]
    change = df["Close"].iloc[-1] - df["Close"].iloc[-2]
    pct_change = (change / df["Close"].iloc[-2]) * 100

    print(f"\n--- {ticker.upper()} Summary ---")
    print(f"Latest close:   ${latest['Close']:.2f}")
    print(f"Daily change:   {change:+.2f} ({pct_change:+.2f}%)")
    print(f"20-day avg:     ${latest['MA20']:.2f}" if pd.notna(latest["MA20"]) else "20-day avg:     n/a (not enough data yet)")
    print(f"50-day avg:     ${latest['MA50']:.2f}" if pd.notna(latest["MA50"]) else "50-day avg:     n/a (not enough data yet)")

    if pd.notna(latest["MA20"]) and pd.notna(latest["MA50"]):
        if latest["MA20"] > latest["MA50"]:
            print("Signal:         MA20 above MA50 -> short-term trend is bullish")
        else:
            print("Signal:         MA20 below MA50 -> short-term trend is bearish")


def plot_chart(df: pd.DataFrame, ticker: str, out_dir: str = "data"):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Close Price", linewidth=1.2)
    plt.plot(df.index, df["MA20"], label="20-day MA", linewidth=1)
    plt.plot(df.index, df["MA50"], label="50-day MA", linewidth=1)
    plt.title(f"{ticker.upper()} Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker.upper()}_chart.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize stock data.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    args = parser.parse_args()

    try:
        df = load_data(args.ticker)
        df = add_moving_averages(df)
        print_summary(df, args.ticker)
        plot_chart(df, args.ticker)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()