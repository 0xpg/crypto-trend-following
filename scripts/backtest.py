#!/usr/bin/env python3
"""Run the frozen baseline plus every component ablation and sweep.

Writes one row per run to results/runs.csv, the baseline's daily series to
results/baseline_daily.csv, its risk-model diagnostics to
results/diagnostics.csv, and the frozen configuration to results/config.json.

Usage:
    python scripts/backtest.py [--workers 6] [--only baseline]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from engine import (  # noqa: E402
    Config,
    Inputs,
    config_dict,
    performance_summary,
    prepare_inputs,
    simulate,
    variant,
)
from paths import PANEL_DIR, RESULTS_DIR, ensure_directories  # noqa: E402

FIELDS = ("open", "close", "high", "low", "quote_volume", "hours", "funding")
SPLIT = "2024-01-01"

# Fields consumed by prepare_inputs. A run that changes any of them needs its
# own universe/volatility/signal frames; reusing the baseline's would silently
# turn the sweep into a duplicate of the baseline.
PREP_FIELDS = (
    "min_history_days", "volume_lookback", "min_median_quote_volume",
    "min_hours_per_day", "vol_short_window", "vol_long_cap_days",
    "vol_blend_short", "vol_floor_annual", "vol_forecast", "vol_ewma_span",
    "speeds", "signal_scale_window", "signal_scale_min_periods",
    "signal_clip", "start", "end",
)

_PANELS: dict[str, pd.DataFrame] | None = None
_CACHE: dict[tuple, Inputs] = {}


def load_panels() -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(PANEL_DIR / f"{name}.parquet")
            for name in FIELDS}


def prep_key(config: Config) -> tuple:
    return tuple(getattr(config, f) for f in PREP_FIELDS)


def _init() -> None:
    global _PANELS
    _PANELS = load_panels()


def inputs_for(config: Config) -> Inputs:
    key = prep_key(config)
    if key not in _CACHE:
        _CACHE[key] = prepare_inputs(_PANELS, config, speeds=config.speeds)
    return _CACHE[key]


def run_one(job: tuple[str, str, Config]) -> dict:
    group, label, config = job
    inputs = inputs_for(config)
    result = simulate(
        opens=inputs.opens, closes=inputs.closes,
        signal=inputs.signal, annual_vol=inputs.annual_vol,
        tradeable=inputs.tradeable, funding=inputs.funding,
        log_returns=inputs.log_returns, config=config)
    daily = result.daily
    row = performance_summary(daily, label)
    row["group"] = group
    for period, sub in (("design", daily.loc[daily.index < SPLIT]),
                        ("holdout", daily.loc[daily.index >= SPLIT])):
        s = performance_summary(sub, label)
        row[f"{period}_sharpe_net"] = s.get("sharpe_net")
        row[f"{period}_total_return"] = s.get("total_return")
    return row, daily["net_return"].rename(label)


def average_ex_ante_risk(result, end: str = SPLIT) -> float:
    daily = result.daily.loc[result.daily.index < end]
    return float(daily["held_ex_ante_vol"].mean(skipna=True))


def calibrate_fixed_multiplier(inputs: Inputs, base: Config,
                               tol: float = 0.005,
                               max_iter: int = 15) -> tuple[float, float]:
    """Constant multiplier matching design-period average ex-ante risk.

    Calibration uses only the design period. The multiplier is then frozen for
    the held-out run. Iteration is required because buffering and the gross cap
    make held risk a nonlinear function of the multiplier.
    """
    def run(config: Config):
        return simulate(
            opens=inputs.opens, closes=inputs.closes, signal=inputs.signal,
            annual_vol=inputs.annual_vol, tradeable=inputs.tradeable,
            funding=inputs.funding, log_returns=inputs.log_returns,
            config=config)

    want = average_ex_ante_risk(run(base))
    mult = 1.0
    best: tuple[float, float] | None = None
    for _ in range(max_iter):
        have = average_ex_ante_risk(
            run(variant(base, use_risk_targeting=False, fixed_multiplier=mult)))
        if not np.isfinite(have) or have <= 0:
            break
        rel = have / want
        if best is None or abs(rel - 1.0) < abs(best[1] - 1.0):
            best = (mult, rel)
        if abs(rel - 1.0) <= tol:
            return float(mult), float(rel)
        mult /= rel
    if best is None:
        raise RuntimeError("gross-exposure calibration produced no usable run")
    return float(best[0]), float(best[1])


def build_jobs(base: Config, fixed: float) -> list[tuple[str, str, Config]]:
    jobs: list[tuple[str, str, Config]] = [("baseline", "baseline", base)]

    jobs += [
        ("ablation", "no_sector_weights", variant(base, use_sector_weights=False)),
        ("ablation", "no_risk_targeting",
         variant(base, use_risk_targeting=False, fixed_multiplier=fixed)),
        ("ablation", "no_buffer", variant(base, buffer=0.0)),
        ("ablation", "equal_notional", variant(base, sizing="equal_notional")),
        ("ablation", "no_costs", variant(base, cost_bps=0.0, apply_funding=False)),
        ("ablation", "no_funding", variant(base, apply_funding=False)),
    ]

    jobs += [("response", f"response_{k}", variant(base, response=k))
             for k in ("tanh", "overextension", "linear", "binary")]

    jobs += [("vol_forecast", f"vol_{k}", variant(base, vol_forecast=k))
             for k in ("blend", "short", "long", "ewma")]

    jobs += [("buffer", f"buffer_{b:g}", variant(base, buffer=b))
             for b in (0.0, 0.05, 0.10, 0.20, 0.40)]

    jobs += [("cost", f"cost_{c:g}bps", variant(base, cost_bps=c))
             for c in (0.0, 2.0, 5.0, 10.0, 20.0)]

    speed_sets = {
        "speed_16_48": ((16, 48),),
        "speed_32_96": ((32, 96),),
        "speed_64_192": ((64, 192),),
        "speed_fast_ensemble": ((4, 12), (8, 24), (16, 48)),
        "speed_slow_ensemble": ((32, 96), (64, 192), (128, 384)),
        "speed_wide_ensemble": ((8, 24), (16, 48), (32, 96), (64, 192), (128, 384)),
    }
    jobs += [("speed", name, variant(base, speeds=sp))
             for name, sp in speed_sets.items()]

    jobs += [("target", f"vol_target_{v:g}", variant(base, vol_target_annual=v))
             for v in (0.10, 0.20, 0.40)]
    jobs += [("clusters", f"clusters_{k}", variant(base, n_clusters=k))
             for k in (2, 4, 8, 16)]
    jobs += [("universe", f"volume_floor_{v/1e6:g}m",
              variant(base, min_median_quote_volume=v))
             for v in (1e6, 5e6, 25e6, 1e8)]

    # section 13 extension: the supplement's p/(2m) asymmetric short cap
    jobs += [("short_cap", f"short_cap_{f*100:g}pct",
              variant(base, max_short_frac_gross=f))
             for f in (0.01, 0.02, 0.05, 0.10)]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None, help="run a single labeled job")
    args = ap.parse_args()
    ensure_directories()

    base = Config()
    base.validate()
    panels = load_panels()
    inputs = prepare_inputs(panels, base)
    print(f"inputs: {inputs.opens.shape[0]} days x {inputs.opens.shape[1]} symbols, "
          f"tradeable symbol-days {int(inputs.tradeable.to_numpy().sum()):,}",
          flush=True)

    baseline = simulate(
        opens=inputs.opens, closes=inputs.closes,
        signal=inputs.signal, annual_vol=inputs.annual_vol,
        tradeable=inputs.tradeable, funding=inputs.funding,
        log_returns=inputs.log_returns, config=base)
    fixed, achieved = calibrate_fixed_multiplier(inputs, base)
    print(f"calibrated constant multiplier for no_risk_targeting: {fixed:.6f} "
          f"(design-period average ex-ante risk {achieved:.4f}x target run)",
          flush=True)
    no_target_config = variant(
        base, use_risk_targeting=False, fixed_multiplier=fixed)
    no_target = simulate(
        opens=inputs.opens, closes=inputs.closes,
        signal=inputs.signal, annual_vol=inputs.annual_vol,
        tradeable=inputs.tradeable, funding=inputs.funding,
        log_returns=inputs.log_returns, config=no_target_config)
    baseline.daily.to_csv(RESULTS_DIR / "baseline_daily.csv")
    baseline.diagnostics.to_csv(RESULTS_DIR / "diagnostics.csv")
    (baseline.positions.abs().sum(axis=1)
     .to_frame("gross").to_csv(RESULTS_DIR / "baseline_gross.csv"))
    frozen = config_dict(base)
    frozen["no_target_fixed_multiplier"] = fixed
    frozen["no_target_design_risk_ratio"] = achieved
    (RESULTS_DIR / "config.json").write_text(json.dumps(frozen, indent=2))
    no_target.daily.to_csv(RESULTS_DIR / "no_target_daily.csv")
    print("baseline:", json.dumps(
        {k: v for k, v in performance_summary(baseline.daily, "baseline").items()
         if isinstance(v, (int, float))}, indent=2, default=float), flush=True)

    jobs = build_jobs(base, fixed)
    if args.only:
        jobs = [j for j in jobs if j[1] == args.only]

    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        rows, series = [], []
        for i, (row, ret) in enumerate(ex.map(run_one, jobs), 1):
            rows.append(row)
            series.append(ret)
            print(f"[{i}/{len(jobs)}] {row['label']:<24} "
                  f"sharpe_net={row.get('sharpe_net', float('nan')):+.2f} "
                  f"total={row.get('total_return', float('nan')):+.1%}",
                  flush=True)

    out = pd.DataFrame(rows)
    cols = ["group", "label", "days", "start", "end", "total_return", "cagr",
            "ann_vol", "sharpe_net", "sharpe_gross", "max_drawdown",
            "design_sharpe_net", "holdout_sharpe_net", "design_total_return",
            "holdout_total_return", "ann_turnover", "avg_gross_exposure",
            "cost_drag_ann", "funding_drag_ann", "long_pnl", "short_pnl",
            "hit_rate", "avg_positions", "avg_live", "ruined", "ruin_date"]
    cols += ["avg_held_ex_ante_vol", "avg_target_tracking_error"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(RESULTS_DIR / "runs.csv", index=False)
    returns = pd.concat(series, axis=1)
    returns = returns.loc[:, ~returns.columns.duplicated()]
    returns.to_csv(RESULTS_DIR / "run_returns.csv")

    # A sweep cell that silently reuses the baseline's inputs is a driver bug
    # that looks exactly like "this parameter does not matter". List every
    # run whose return series is bit-identical to the baseline so the ones
    # that are legitimately restatements of it can be checked by eye.
    base_ret = returns["baseline"]
    same = [c for c in returns.columns
            if c != "baseline" and returns[c].equals(base_ret)]
    print("runs identical to baseline (expected: restatements of the frozen "
          f"config only): {same}")
    print(f"\n{len(out)} runs -> {RESULTS_DIR / 'runs.csv'}")


if __name__ == "__main__":
    main()
