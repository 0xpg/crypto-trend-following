"""Interactive configuration and CCXT backtest dashboard."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import ccxt
import numpy as np
import pandas as pd
import streamlit as st

from engine import Config, performance_summary, prepare_inputs, response, simulate
from scripts.ccxt_backtest import DEFAULT_SYMBOLS, build_panels, fetch_daily

st.set_page_config(page_title="Crypto Trend Following", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1220px; padding-top: 1.8rem;}
.pipeline {display:grid;grid-template-columns:repeat(7,minmax(110px,1fr));gap:.55rem;margin:.5rem 0 1.4rem;}
.pipe-step {background:#252c34;border-radius:12px;padding:.75rem .45rem;text-align:center;min-height:68px;display:flex;align-items:center;justify-content:center;}
@media(max-width:900px){.pipeline{grid-template-columns:repeat(2,1fr)}}
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


st.title("Crypto Trend-Following Lab")
st.caption("Configure the seven-component CTA system, fetch public perpetual candles through CCXT, and inspect the resulting portfolio.")

with st.sidebar:
    st.header("Backtest configuration")
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
    run_clicked = st.button("Fetch data and run", type="primary", use_container_width=True)
    st.caption("CCXT mode uses complete daily candles and excludes historical funding. No API key is required.")

config = replace(
    Config(), start=str(start_date), end=str(end_date), response=response_kind,
    vol_target_annual=target_vol, buffer=buffer, cost_bps=float(cost_bps),
    max_gross_leverage=float(max_leverage), n_clusters=min(clusters, max(len(symbols), 1)),
    min_history_days=min_history, corr_min_obs=min(60, min_history),
    min_median_quote_volume=0.0, apply_funding=False,
)

steps = [
    f"1. {len(symbols)} eligible markets", "2. MA trend signal",
    f"3. {response_kind} + inverse vol", f"4. {config.n_clusters} equal-risk clusters",
    f"5. {target_vol:.0%} portfolio vol target", f"6. {buffer:.0%} no-trade buffer",
    f"7. {cost_bps} bps costs + funding",
]
st.markdown('<div class="pipeline">' + ''.join(f'<div class="pipe-step">{s}</div>' for s in steps) + '</div>', unsafe_allow_html=True)

st.subheader("Signal-to-position explorer")
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
            except Exception as exc:
                st.exception(exc)

if "last_result" in st.session_state:
    daily = st.session_state["last_result"]
    summary = st.session_state["last_summary"]
    st.divider()
    st.subheader("Backtest results")
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
    st.caption(f"Fetched assets: {', '.join(st.session_state['last_assets'])}. Trading costs included; funding excluded in CCXT mode.")
    st.download_button("Download daily results (CSV)", daily.to_csv().encode("utf-8"), "trend_backtest_daily.csv", "text/csv")
else:
    st.info("Choose a configuration and click **Fetch data and run**. The first fetch may take a few seconds per asset; repeated configurations reuse cached candles for one hour.")
