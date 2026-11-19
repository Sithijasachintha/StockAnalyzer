"""CSE stock movement dashboard. Stage 1: import, validate, select, chart.

Run from the project folder with:  streamlit run app.py
"""
import io

import streamlit as st

from src.charts import price_chart
from src.data_loader import find_gaps, load_csv
from src.sample_data import make_synthetic_sample

st.set_page_config(page_title="CSE Stock Movement Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def parse_csv_bytes(content: bytes):
    """Streamlit re-runs this script on every click; caching avoids re-parsing."""
    return load_csv(io.BytesIO(content))


EXAMPLE_CSV = (
    "Date,Symbol,Open,High,Low,Close,Volume\n"
    "2026-03-02,ABCD.N0000,100.00,103.50,99.20,102.10,15400\n"
    "2026-03-03,ABCD.N0000,102.10,104.00,101.00,101.50,9800\n"
)
SAMPLE_CSV = make_synthetic_sample().to_csv(index=False).encode("utf-8")

# ---------------------------------------------------------------- sidebar ----
st.sidebar.header("Data")
SOURCE_UPLOAD = "Upload a CSV"
SOURCE_SAMPLE = "Use synthetic sample (demo only)"
source = st.sidebar.radio("Data source", [SOURCE_UPLOAD, SOURCE_SAMPLE])
uploaded = None
if source == SOURCE_UPLOAD:
    uploaded = st.sidebar.file_uploader("Daily prices (CSV)", type=["csv"])

# ------------------------------------------------------------------- main ----
st.title("CSE Stock Movement Dashboard")
st.caption("Stage 1: data import and price exploration. Prices are in LKR.")

with st.expander("Required CSV format", expanded=(source == SOURCE_UPLOAD and uploaded is None)):
    st.markdown(
        "- One row per symbol per trading day, with these 7 columns: "
        "`Date, Symbol, Open, High, Low, Close, Volume`\n"
        "- `Date` must be **YYYY-MM-DD**. Prices are plain numbers in LKR "
        "(no commas, no 'LKR').\n"
        "- Prices must be above 0, Volume cannot be negative, and a symbol cannot "
        "appear twice on the same date.\n"
        "- Missing trading days are fine. They are shown as gaps, never filled in."
    )
    st.code(EXAMPLE_CSV, language="text")
    st.download_button(
        "Download synthetic sample CSV (random data, not CSE)",
        data=SAMPLE_CSV,
        file_name="synthetic_sample.csv",
        mime="text/csv",
    )

if source == SOURCE_UPLOAD:
    if uploaded is None:
        st.info("Upload a CSV in the sidebar, or switch to the synthetic sample to try the interface.")
        st.stop()
    result = parse_csv_bytes(uploaded.getvalue())
    is_synthetic = False
else:
    result = parse_csv_bytes(SAMPLE_CSV)
    is_synthetic = True

if not result.ok:
    st.error(f"This file cannot be used yet. {len(result.errors)} problem(s) found:")
    for message in result.errors:
        st.markdown(f"- {message}")
    st.stop()

if is_synthetic:
    st.warning(
        "SYNTHETIC DEMO DATA. These prices were generated randomly by this program. "
        "They are not from the Colombo Stock Exchange, and nothing shown here says "
        "anything about real CSE performance."
    )
for message in result.warnings:
    st.warning(message)

data = result.data
symbols = sorted(data["Symbol"].unique())
symbol = st.sidebar.selectbox("Company (symbol)", symbols)
min_gap_days = st.sidebar.number_input(
    "Flag gaps of at least (calendar days)", min_value=2, max_value=60, value=5
)

# Everything below uses ONLY this one symbol's rows. Keeping each symbol separate
# is what stops one company's prices leaking into another company's analysis.
one = data[data["Symbol"] == symbol].copy()

st.subheader(f"What happened? {symbol}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recorded sessions", f"{len(one):,}")
c2.metric("First record", f"{one['Date'].min():%d %b %Y}")
c3.metric("Last record", f"{one['Date'].max():%d %b %Y}")
c4.metric("Latest close (LKR)", f"{one.iloc[-1]['Close']:,.2f}")

st.plotly_chart(price_chart(one, symbol, int(min_gap_days)), width="stretch")
st.caption(
    "Grey bands and broken lines mark stretches with no recorded session. They may be "
    "holidays, days the share did not trade, or missing data; the prices alone cannot "
    "tell which, so none are filled in."
)

gaps = find_gaps(one, int(min_gap_days))
if gaps.empty:
    st.caption(f"No gaps of {int(min_gap_days)} calendar days or more for {symbol}.")
else:
    st.markdown(f"**Gaps of {int(min_gap_days)}+ calendar days for {symbol}**")
    st.dataframe(gaps, hide_index=True, width="stretch")

with st.expander("Show recorded rows"):
    st.dataframe(one.sort_values("Date", ascending=False), hide_index=True, width="stretch")
