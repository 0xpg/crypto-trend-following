#!/usr/bin/env python3
"""Build the daily panel and the daily funding panel from the archive mirror.

Reads the mirror's hourly kline parquet files, resamples each to daily UTC
bars, and writes wide date-by-symbol frames:

    open, close, high, low, quote_volume, hours (bar completeness), funding

``funding`` at row t is the total rate charged to a long between the open of
day t and the open of day t+1, summed over that day's published funding
timestamps. Symbol-days with no published archive are NaN and are reported
as a coverage fraction rather than silently zeroed.

Usage:
    python scripts/build_panel.py [--workers 8]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from paths import (  # noqa: E402
    FUNDING_RAW_DIR,
    KLINES_1H_DIR,
    PANEL_DIR,
    RESULTS_DIR,
    ensure_directories,
)

STABLE_BASES = {"USDC", "BUSD", "TUSD", "DAI", "FDUSD", "USDP", "USD1", "AEUR",
                "EUR", "EURI", "USDE", "XUSD", "BFUSD", "USTC", "UST"}

FIELDS = ("open", "close", "high", "low", "quote_volume", "hours")


def daily_from_hourly(path: Path) -> tuple[str, pd.DataFrame] | None:
    symbol = path.stem
    df = pd.read_parquet(path)
    if df.empty:
        return None
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.assign(day=ts.dt.floor("D")).sort_values("open_time")
    g = df.groupby("day", sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "close": g["close"].last(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "quote_volume": g["quote_volume"].sum(),
        "hours": g.size().astype(float),
    })
    return symbol, out


def funding_from_zips(symbol: str) -> pd.Series | None:
    folder = FUNDING_RAW_DIR / symbol
    if not folder.is_dir():
        return None
    frames = []
    for zpath in sorted(folder.glob("*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                name = zf.namelist()[0]
                raw = zf.read(name)
        except (zipfile.BadZipFile, IndexError):
            continue
        part = pd.read_csv(io.BytesIO(raw))
        if "calc_time" not in part.columns:
            part = pd.read_csv(io.BytesIO(raw), header=None,
                               names=["calc_time", "funding_interval_hours",
                                      "last_funding_rate"])
        frames.append(part[["calc_time", "last_funding_rate"]])
    if not frames:
        return None
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[pd.to_numeric(ev["calc_time"], errors="coerce").notna()]
    ev["calc_time"] = ev["calc_time"].astype(np.int64)
    ev["last_funding_rate"] = pd.to_numeric(ev["last_funding_rate"],
                                            errors="coerce")
    ts = pd.to_datetime(ev["calc_time"], unit="ms", utc=True).dt.tz_localize(None)
    daily = ev.groupby(ts.dt.floor("D"))["last_funding_rate"].sum()
    daily.name = symbol
    return daily


def _funding_task(symbol: str) -> tuple[str, pd.Series | None]:
    return symbol, funding_from_zips(symbol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ensure_directories()

    files = sorted(KLINES_1H_DIR.glob("*.parquet"))
    files = [f for f in files
             if f.stem.endswith("USDT") and f.stem[:-4] not in STABLE_BASES]
    print(f"{len(files)} non-stablecoin USDT perpetuals", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        results = [r for r in ex.map(daily_from_hourly, files, chunksize=8)
                   if r is not None]
    print(f"{len(results)} symbols with daily bars", flush=True)

    index = pd.DatetimeIndex(sorted({d for _, f in results for d in f.index}))
    symbols = sorted(s for s, _ in results)
    panels = {k: pd.DataFrame(index=index, columns=symbols, dtype=np.float64)
              for k in FIELDS}
    for symbol, frame in results:
        frame = frame.reindex(index)
        for k in FIELDS:
            panels[k][symbol] = frame[k]

    print(f"panel {len(index)} days x {len(symbols)} symbols "
          f"({index[0].date()} to {index[-1].date()})", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        funding_series = list(ex.map(_funding_task, symbols, chunksize=8))
    have = {s: v for s, v in funding_series if v is not None and len(v)}
    funding = pd.DataFrame(index=index, columns=symbols, dtype=np.float64)
    for symbol, series in have.items():
        funding[symbol] = series.reindex(index)
    panels["funding"] = funding

    for name, frame in panels.items():
        frame.to_parquet(PANEL_DIR / f"{name}.parquet")

    traded = panels["open"].notna()
    covered = funding.notna() & traded
    meta = {
        "symbols": len(symbols),
        "days": len(index),
        "start": str(index[0].date()),
        "end": str(index[-1].date()),
        "symbol_days_with_bars": int(traded.to_numpy().sum()),
        "symbols_with_funding": len(have),
        "funding_coverage_of_traded_symbol_days": float(
            covered.to_numpy().sum() / max(traded.to_numpy().sum(), 1)),
    }
    (RESULTS_DIR / "panel_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
