"""Interactive configuration and CCXT backtest dashboard."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd
import streamlit as st

from engine import Config, performance_summary, prepare_inputs, response, simulate
from scripts.ccxt_backtest import DEFAULT_SYMBOLS, build_panels, fetch_daily

PROJECT = Path(__file__).resolve().parent
SAVED_DAILY = PROJECT / "results" / "ccxt_daily.csv"
SAVED_SUMMARY = PROJECT / "results" / "ccxt_summary.json"

st.set_page_config(page_title="Crypto Trend Following", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp {background:radial-gradient(circle at 78% 4%,rgba(45,212,191,.10),transparent 26%),radial-gradient(circle at 20% 18%,rgba(99,102,241,.10),transparent 30%);}
.block-container {max-width: 1220px; padding-top: 1.5rem;}
.hero {padding:1.4rem 1.6rem;border-left:5px solid #2dd4bf;background:linear-gradient(110deg,rgba(45,212,191,.13),rgba(99,102,241,.08));border-radius:0 18px 18px 0;margin-bottom:1.4rem;}
.hero-kicker {color:#2dd4bf;letter-spacing:.14em;text-transform:uppercase;font-size:.78rem;font-weight:600;margin-bottom:.35rem;}
.hero h1 {padding:0;margin:0 0 .35rem;font-size:2.25rem;}
.hero p {margin:0;opacity:.72;max-width:760px;}
.pipeline {display:flex;align-items:flex-start;gap:.35rem;margin:.4rem 0 1.7rem;}
.pipe-step {flex:1;min-width:0;text-align:center;position:relative;padding:.25rem;}
.pipe-step:not(:last-child):after {content:"";position:absolute;top:18px;left:64%;width:72%;height:2px;background:linear-gradient(90deg,#2dd4bf,rgba(99,102,241,.35));z-index:0;}
.pipe-number {width:36px;height:36px;margin:0 auto .55rem;border-radius:50%;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#2dd4bf,#6366f1);color:#07111c;font-weight:700;position:relative;z-index:1;box-shadow:0 0 0 5px rgba(45,212,191,.08);}
.pipe-label {font-size:.84rem;line-height:1.25;opacity:.82;}
div[data-testid="stMetric"] {background:linear-gradient(145deg,rgba(45,212,191,.08),rgba(99,102,241,.06));border:1px solid rgba(148,163,184,.16);padding:1rem;border-radius:16px;}
div[data-testid="stMetricValue"] {color:#5eead4;}
section[data-testid="stSidebar"] {border-right:1px solid rgba(45,212,191,.15);}
div[data-testid="stButton"] button[kind="primary"] {background:linear-gradient(90deg,#0f766e,#4f46e5);border:0;}
@media(max-width:900px){.pipeline{display:grid;grid-template-columns:repeat(2,1fr)}.pipe-step:not(:last-child):after{display:none}.hero h1{font-size:1.75rem}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_market_data(exchange_id: str, symbols: tuple[str, ...], start: str, end: str):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "timeout": 30_000})
    exchange.load_markets()
    since = exchange.parse8601(f"{start}T00:00:00Z")
    until = exchange.parse8601(f"{end}T00:00:00Z")
    frames = {}
    try:
        for symbol in symbols:
            if symbol in exchange.markets:
                frame = fetch_daily(exchange, symbol, since, until)
                if not frame.empty:
                    key = symbol.replace("/", "").replace(":USDT", "")
                    frames[key] = frame
    finally:
        exchange.close()
    return frames


def run_backtest(frames, config: Config):
    panels = build_panels(frames)
    inputs = prepare_inputs(panels, config)
    result = simulate(
        opens=inputs.opens, closes=inputs.closes, signal=inputs.signal,
        annual_vol=inputs.annual_vol, tradeable=inputs.tradeable,
        funding=inputs.funding, log_returns=inputs.log_returns, config=config,
    )
    return result, performance_summary(result.daily, "dashboard")


st.markdown("""
<div class="hero">
  <div class="hero-kicker">Ask a question. Change one assumption. Measure the answer.</div>
  <h1>Trend Atlas Research Lab</h1>
  <p>Explore when diversified crypto trends persist, where portfolio construction helps, and which assumptions make the evidence disappear.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Design your question")
    exchange_id = st.selectbox("Exchange", ["binanceusdm"], help="Public USD-M perpetual market data through CCXT.")
    symbols = st.multiselect("Assets", DEFAULT_SYMBOLS, default=list(DEFAULT_SYMBOLS))
    custom = st.text_input("Additional CCXT symbols", placeholder="TRX/USDT:USDT, SUI/USDT:USDT")
    if custom.strip():
        symbols = list(dict.fromkeys(symbols + [s.strip() for s in custom.split(",") if s.strip()]))
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start", date(2020, 1, 1), min_value=date(2019, 1, 1))
    end_date = c2.date_input("End", date.today(), max_value=date.today())
    response_kind = st.selectbox("Response function", ["tanh", "overextension", "linear", "binary"])
    target_vol = st.slider("Portfolio volatility target", 5, 40, 20, 1) / 100
    buffer = st.slider("No-trade buffer", 0, 40, 10, 1) / 100
    cost_bps = st.slider("Trading cost", 0, 30, 10, 1)
    max_leverage = st.slider("Maximum gross leverage", 0.5, 5.0, 2.0, 0.25)
    clusters = st.slider("Risk clusters", 1, 12, 4, 1)
    min_history = st.slider("Minimum history (days)", 60, 365, 120, 15)
    run_clicked = st.button("Run this experiment", type="primary", use_container_width=True)
    st.caption("CCXT mode uses complete daily candles and excludes historical funding. No API key is required.")

config = replace(
    Config(), start=str(start_date), end=str(end_date), response=response_kind,
    vol_target_annual=target_vol, buffer=buffer, cost_bps=float(cost_bps),
    max_gross_leverage=float(max_leverage), n_clusters=min(clusters, max(len(symbols), 1)),
    min_history_days=min_history, corr_min_obs=min(60, min_history),
    min_median_quote_volume=0.0, apply_funding=False,
)

# Show the repository's reproducible CCXT run immediately. A custom run made
# from the sidebar replaces this snapshot in the same performance section.
if "last_result" not in st.session_state and SAVED_DAILY.exists() and SAVED_SUMMARY.exists():
    saved_daily = pd.read_csv(SAVED_DAILY, parse_dates=["date"]).set_index("date")
    st.session_state["last_result"] = saved_daily
    st.session_state["last_summary"] = json.loads(SAVED_SUMMARY.read_text(encoding="utf-8"))
    st.session_state["last_assets"] = [
        symbol.replace("/", "").replace(":USDT", "") for symbol in DEFAULT_SYMBOLS
    ]
    st.session_state["last_source"] = "Bundled CCXT baseline"

steps = [
    f"{len(symbols)} markets", "Trend ensemble",
    f"{response_kind} sizing", f"{config.n_clusters} risk clusters",
    f"{target_vol:.0%} vol target", f"{buffer:.0%} trade buffer",
    f"{cost_bps} bps execution",
]
st.markdown(
    '<div class="pipeline">' + ''.join(
        f'<div class="pipe-step"><div class="pipe-number">{i}</div><div class="pipe-label">{label}</div></div>'
        for i, label in enumerate(steps, 1)
    ) + '</div>',
    unsafe_allow_html=True,
)

st.subheader("Question 1 · How should signal strength become exposure?")
explore_left, explore_right = st.columns([1, 2])
with explore_left:
    signal_value = st.slider("Combined trend signal", -4.0, 4.0, 1.0, 0.05)
    asset_vol = st.slider("Illustrative asset volatility", 10, 100, 20, 1) / 100
    signal_response = float(response(np.array([signal_value]), response_kind)[0])
    raw_weight = signal_response / asset_vol
    allocation = raw_weight * target_vol
    m1, m2, m3 = st.columns(3)
    m1.metric("Response", f"{signal_response:.2f}", "Long" if signal_response > 0 else "Short" if signal_response < 0 else "Flat")
    m2.metric("Raw risk weight", f"{abs(raw_weight):.2f}x")
    m3.metric("Illustrative allocation", f"{allocation:.1%}")
with explore_right:
    z = np.linspace(-4, 4, 321)
    curve = pd.DataFrame({"signal": z, "position strength": response(z, response_kind)}).set_index("signal")
    st.line_chart(curve, height=300, y_label="position strength f(z)", x_label="normalized trend signal z")

if run_clicked:
    if len(symbols) < 2:
        st.error("Select at least two assets.")
    elif start_date >= end_date:
        st.error("The end date must be after the start date.")
    elif (end_date - start_date) < timedelta(days=min_history + 200):
        st.error("Choose a longer date range so signals and the risk model have enough warm-up history.")
    else:
        with st.spinner(f"Fetching {len(symbols)} markets through CCXT and running the portfolio..."):
            try:
                frames = fetch_market_data(exchange_id, tuple(symbols), str(start_date), str(end_date))
                if len(frames) < 2:
                    raise RuntimeError("Fewer than two selected markets returned usable candles.")
                actual_config = replace(config, n_clusters=min(config.n_clusters, len(frames)))
                result, summary = run_backtest(frames, actual_config)
                st.session_state["last_result"] = result.daily
                st.session_state["last_summary"] = summary
                st.session_state["last_assets"] = sorted(frames)
                st.session_state["last_source"] = "Custom CCXT run"
            except Exception as exc:
                st.exception(exc)

if "last_result" in st.session_state:
    daily = st.session_state["last_result"]
    summary = st.session_state["last_summary"]
    st.divider()
    st.subheader("Evidence · Backtest performance")
    st.caption(st.session_state.get("last_source", "CCXT backtest"))
    cols = st.columns(6)
    cols[0].metric("Total return", f"{summary['total_return']:.1%}")
    cols[1].metric("CAGR", f"{summary['cagr']:.1%}")
    cols[2].metric("Net Sharpe", f"{summary['sharpe_net']:.2f}")
    cols[3].metric("Annual volatility", f"{summary['ann_vol']:.1%}")
    cols[4].metric("Maximum drawdown", f"{summary['max_drawdown']:.1%}")
    cols[5].metric("Annual turnover", f"{summary['ann_turnover']:.1f}x")

    equity = daily[["equity"]].rename(columns={"equity": "Portfolio equity"})
    peak = daily["equity"].cummax()
    drawdown = (daily["equity"] / peak - 1).to_frame("Drawdown")
    st.line_chart(equity, height=360, y_label="account value ($)")
    c1, c2 = st.columns(2)
    c1.line_chart(drawdown, height=240, y_label="drawdown")
    c2.line_chart(daily[["gross_exposure", "net_exposure"]], height=240, y_label="notional ($)")
    st.caption(f"Assets: {', '.join(st.session_state['last_assets'])}. Trading costs included; funding excluded in CCXT mode.")
    st.download_button("Download daily results (CSV)", daily.to_csv().encode("utf-8"), "trend_backtest_daily.csv", "text/csv")
else:
    st.info("Choose a configuration and click **Fetch data and run**. The first fetch may take a few seconds per asset; repeated configurations reuse cached candles for one hour.")
