"""Plotly charts. Charts only DRAW data; they never change what is stored."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def price_chart(symbol_df: pd.DataFrame, symbol: str, min_gap_days: int = 5) -> go.Figure:
    """Closing-price line for ONE symbol, with trading gaps kept visible.

    Plotly joins neighbouring points with a straight line, which would hide a
    two-week hole and make it look like a smooth move. So, only in this plotting
    copy, we add an empty (NaN) point inside each gap. Plotly then breaks the
    line there. No price is invented: the stored data is untouched.
    """
    df = symbol_df.sort_values("Date").reset_index(drop=True)
    days_between = df["Date"].diff().dt.days
    gap_rows = df.index[days_between >= min_gap_days]

    breaks = pd.DataFrame(
        {
            "Date": df.loc[gap_rows, "Date"] - pd.Timedelta(days=1),
            "Close": np.nan,
        }
    )
    plot_df = pd.concat([df[["Date", "Close"]], breaks]).sort_values("Date")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["Date"],
            y=plot_df["Close"],
            mode="lines+markers",
            connectgaps=False,
            marker={"size": 4},
            line={"width": 1.5},
            name="Close",
            hovertemplate="%{x|%d %b %Y}<br>Close: LKR %{y:,.2f}<extra></extra>",
        )
    )
    for i in gap_rows:  # shade each gap so it is obvious, not just a break
        fig.add_vrect(
            x0=df.loc[i - 1, "Date"].to_pydatetime(),
            x1=df.loc[i, "Date"].to_pydatetime(),
            fillcolor="grey",
            opacity=0.18,
            line_width=0,
            layer="below",
        )
    fig.update_layout(
        title=f"{symbol}: daily closing price",
        xaxis_title="Date",
        yaxis_title="Closing price (LKR)",
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        showlegend=False,
    )
    return fig
