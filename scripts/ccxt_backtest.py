#!/usr/bin/env python3
"""Fetch public daily perpetual candles through CCXT and run the strategy.

This is a compact, reproducible live-data route for an exploratory backtest.
It intentionally does not fetch funding history, so funding is disabled and
the output is not directly comparable with the archive-backed baseline.

Usage:
    python scripts/ccxt_backtest.py
    python scripts/ccxt_backtest.py --start 2022-01-01 --symbols BTC/USDT:USDT ETH/USDT:USDT
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from engine import Config, performance_summary, prepare_inputs, simulate  # noqa: E402
from paths import RESULTS_DIR, ensure_directories  # noqa: E402

DEFAULT_SYMBOLS = (
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
    "SOL/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
    "ADA/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT",
    "DOT/USDT:USDT", "LTC/USDT:USDT", "BCH/USDT:USDT",
)


def fetch_daily(exchange, symbol: str, since: int, until: int) -> pd.DataFrame:
    """Fetch complete UTC daily candles with explicit time pagination."""
    rows: list[list[float]] = []
    cursor = since
    day_ms = 86_400_000
    while cursor < until:
        batch = exchange.fetch_ohlcv(symbol, "1d", since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(row for row in batch if since <= row[0] < until)
        next_cursor = int(batch[-1][0]) + day_ms
        if next_cursor <= cursor:
            break
        cursor = next_cursor
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=("timestamp", "open", "high", "low", "close", "volume"))
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.pop("timestamp"), unit="ms", utc=True)
    ).tz_localize(None)
    frame["quote_volume"] = frame["volume"] * frame["close"]
    frame["hours"] = 24.0
    return frame


def build_panels(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    fields = ("open", "close", "high", "low", "quote_volume", "hours")
    index = pd.DatetimeIndex(sorted({d for frame in frames.values() for d in frame.index}))
    panels = {field: pd.DataFrame(index=index, dtype=np.float64) for field in fields}
    for symbol, frame in frames.items():
        for field in fields:
            panels[field][symbol] = frame[field].reindex(index)
    panels["funding"] = pd.DataFrame(np.nan, index=index, columns=sorted(frames))
    return panels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="binanceusdm")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"))
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    args = parser.parse_args()
    ensure_directories()

    exchange_class = getattr(ccxt, args.exchange)
    exchange = exchange_class({"enableRateLimit": True, "timeout": 30_000})
    exchange.load_markets()
    since = exchange.parse8601(f"{args.start}T00:00:00Z")
    until = exchange.parse8601(f"{args.end}T00:00:00Z")
    frames = {}
    try:
        for symbol in args.symbols:
            if symbol not in exchange.markets:
                print(f"skip unavailable market: {symbol}", flush=True)
                continue
            print(f"fetch {symbol}", flush=True)
            frame = fetch_daily(exchange, symbol, since, until)
            if not frame.empty:
                frames[symbol.replace("/", "").replace(":USDT", "")] = frame
    finally:
        exchange.close()
    if len(frames) < 2:
        raise RuntimeError("fewer than two markets returned usable candles")

    panels = build_panels(frames)
    config = replace(
        Config(), start=args.start, end=args.end, apply_funding=False,
        n_clusters=min(4, len(frames)), corr_min_obs=60,
        min_median_quote_volume=0.0,
    )
    inputs = prepare_inputs(panels, config)
    result = simulate(
        opens=inputs.opens, closes=inputs.closes, signal=inputs.signal,
        annual_vol=inputs.annual_vol, tradeable=inputs.tradeable,
        funding=inputs.funding, log_returns=inputs.log_returns, config=config,
    )
    summary = performance_summary(result.daily, "ccxt_binanceusdm")
    summary.update({
        "exchange": args.exchange, "symbols_requested": len(args.symbols),
        "symbols_fetched": len(frames), "funding_included": False,
        "data_source": "CCXT public fetch_ohlcv",
    })
    result.daily.to_csv(RESULTS_DIR / "ccxt_daily.csv")
    (RESULTS_DIR / "ccxt_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
