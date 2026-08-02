#!/usr/bin/env python3
"""Measure a set of trend-following research questions on the Binance universe.

C1-C5 are properties of the signal, not of the full book, so they are
measured on single-market vol-targeted return streams that reuse exactly the
frozen signal, volatility forecast and universe filter. C6-C9, C11 and C12
come from the full-engine runs in results/runs.csv. C10 is measured on the
baseline's risk diagnostics.

Writes results/claims.json plus one CSV per measured claim.

Usage:
    python scripts/claims.py [--draws 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from engine import (  # noqa: E402
    ANNUALIZE,
    Config,
    cluster_labels,
    effective_breadth,
    forward_mark_returns,
    prepare_inputs,
    response,
    sharpe_ratio,
    shrunk_correlation,
    stationary_bootstrap_sharpe_diff_ci,
)
from paths import PANEL_DIR, RESULTS_DIR, ensure_directories  # noqa: E402

FIELDS = ("open", "close", "high", "low", "quote_volume", "hours", "funding")


def single_market_returns(panels: dict[str, pd.DataFrame], config: Config,
                          speeds=None, net: bool = True) -> pd.DataFrame:
    """Daily return of a one-market version of the frozen strategy.

    Each market is held at ``vol_target / sigma_hat`` times its response, so
    every stream targets the same volatility on its own. This is the object
    the single-market trend-strength research question is about.
    """
    inputs = prepare_inputs(panels, config, speeds=speeds)
    resp = np.asarray(response(inputs.signal.to_numpy(), config.response))
    vol = inputs.annual_vol.to_numpy()
    live = inputs.tradeable.to_numpy()

    desired = np.where(
        live & np.isfinite(resp) & np.isfinite(vol),
        resp * config.vol_target_annual / np.maximum(vol, 1e-9), 0.0)
    desired = np.nan_to_num(desired)

    opens = inputs.opens.to_numpy()
    closes = inputs.closes.to_numpy()
    fwd, terminal = forward_mark_returns(opens, closes)

    valid_return = np.isfinite(fwd)
    actual = np.zeros_like(desired)
    turnover = np.zeros_like(desired)
    drift = np.zeros(desired.shape[1])
    for t in range(len(desired)):
        can_execute = np.isfinite(opens[t])
        actual[t] = np.where(can_execute, desired[t], drift)
        turnover[t] = np.where(can_execute, np.abs(actual[t] - drift), 0.0)
        turnover[t] += np.where(
            terminal[t], np.abs(actual[t] * (1.0 + fwd[t])), 0.0)
        carry = valid_return[t] & ~terminal[t]
        drift = np.where(
            carry, actual[t] * (1.0 + np.nan_to_num(fwd[t])), 0.0)

    pnl = actual * np.nan_to_num(fwd)
    if net:
        pnl = pnl - turnover * config.cost_bps / 10_000.0
        pnl = pnl - np.where(
            valid_return, actual * np.nan_to_num(inputs.funding.to_numpy()), 0.0)

    out = pd.DataFrame(pnl, index=inputs.opens.index, columns=inputs.opens.columns)
    observable = (inputs.tradeable.to_numpy() & np.isfinite(resp)
                  & np.isfinite(vol) & np.isfinite(opens))
    started = np.maximum.accumulate(observable, axis=0)
    remaining = np.maximum.accumulate(
        np.isfinite(opens)[::-1], axis=0)[::-1]
    sample = pd.DataFrame(started & remaining, index=out.index,
                          columns=out.columns)
    return out.where(sample)


def claim_c1(streams: pd.DataFrame, gross: pd.DataFrame,
             min_days: int = 730) -> tuple[dict, pd.DataFrame]:
    rows = []
    for sym in streams.columns:
        s = streams[sym].dropna()
        if len(s) < min_days:
            continue
        rows.append({
            "symbol": sym,
            "days": len(s),
            "sharpe_net": sharpe_ratio(s.to_numpy()),
            "sharpe_gross": sharpe_ratio(gross[sym].dropna().to_numpy()),
        })
    table = pd.DataFrame(rows).sort_values("sharpe_net", ascending=False)
    q = table["sharpe_net"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    summary = {
        "markets": len(table),
        "min_days": min_days,
        "mean_sharpe_net": float(table["sharpe_net"].mean()),
        "median_sharpe_net": float(table["sharpe_net"].median()),
        "mean_sharpe_gross": float(table["sharpe_gross"].mean()),
        "median_sharpe_gross": float(table["sharpe_gross"].median()),
        "quantiles_net": {str(k): float(v) for k, v in q.items()},
        "share_positive_net": float((table["sharpe_net"] > 0).mean()),
        "claimed": 0.2,
    }
    return summary, table


def claim_c2(gross_streams: pd.DataFrame, availability: pd.DataFrame,
             draws: int, seed: int = 0
             ) -> tuple[dict, pd.DataFrame]:
    """Gross point-in-time random-priority portfolios capped at N markets.

    Each draw assigns every symbol a fixed random priority. On each day the
    portfolio holds the first N currently available symbols at equal weight.
    New listings enter only when they become observable; unavailable slots
    remain absent rather than being selected with future information. This is
    explicitly a gross-return breadth diagnostic: dynamic selection turnover
    is not modeled or represented as net performance.
    """
    mat = gross_streams
    values = mat.fillna(0.0).to_numpy()
    available = availability.reindex_like(mat).fillna(False).to_numpy(dtype=bool)
    max_live = int(available.sum(axis=1).max())
    rng = np.random.default_rng(seed)
    sizes = [n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 512)
             if n <= max_live]
    if max_live not in sizes:
        sizes.append(max_live)

    sharpes = {n: [] for n in sizes}
    average_live = {n: [] for n in sizes}
    for _ in range(draws):
        order = rng.permutation(mat.shape[1])
        a = available[:, order]
        cumulative_count = np.cumsum(a, axis=1)
        cumulative_return = np.cumsum(np.where(a, values[:, order], 0.0), axis=1)
        total = cumulative_count[:, -1]
        for n in sizes:
            used = np.minimum(total, n)
            idx = np.argmax(cumulative_count >= n, axis=1)
            idx = np.where(total >= n, idx, mat.shape[1] - 1)
            summed = np.take_along_axis(
                cumulative_return, idx[:, None], axis=1)[:, 0]
            port = np.divide(summed, used, out=np.zeros_like(summed),
                             where=used > 0)
            sharpes[n].append(sharpe_ratio(port))
            average_live[n].append(float(used.mean()))

    rows = []
    for n in sizes:
        vals = np.asarray(sharpes[n], dtype=float)
        rows.append({
            "return_basis": "gross",
            "n_markets": n,
            "draws": draws,
            "average_live_markets": float(np.mean(average_live[n])),
            "sharpe_mean": float(np.nanmean(vals)),
            "sharpe_median": float(np.nanmedian(vals)),
            "sharpe_p10": float(np.nanpercentile(vals, 10)),
            "sharpe_p90": float(np.nanpercentile(vals, 90)),
        })
    table = pd.DataFrame(rows)

    one = float(table.loc[table["n_markets"] == 1, "sharpe_mean"].iloc[0])
    largest = float(table["sharpe_mean"].iloc[-1])
    summary = {
        "return_basis": "gross",
        "sharpe_at_1": one,
        "sharpe_at_max_n": largest,
        "max_n": int(table["n_markets"].iloc[-1]),
        "observed_multiple_1_to_max": largest / one if one else float("nan"),
    }
    for lo, hi in ((70, 320),):
        a = table.iloc[(table["n_markets"] - lo).abs().argmin()]
        b = table.iloc[(table["n_markets"] - hi).abs().argmin()]
        summary["c3_expansion"] = {
            "from_n": int(a["n_markets"]), "to_n": int(b["n_markets"]),
            "from_sharpe": float(a["sharpe_mean"]),
            "to_sharpe": float(b["sharpe_mean"]),
            "observed_multiple": float(b["sharpe_mean"] / a["sharpe_mean"]),
            "claimed_multiple": "1.5-2.0",
        }
    return summary, table


def _cluster_draw_metrics(corr: np.ndarray, labels: np.ndarray, size: int,
                          draws: int, rng: np.random.Generator
                          ) -> tuple[list[tuple[float, float]],
                                     list[tuple[float, float]]]:
    """Mean correlation and effective breadth within/across clusters."""
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    groups = [g for g in groups if len(g)]
    wide = [g for g in groups if len(g) >= size]
    conc, div = [], []

    def metric(pick) -> tuple[float, float]:
        sub = corr[np.ix_(pick, pick)]
        off = sub[~np.eye(len(pick), dtype=bool)]
        return float(off.mean()), effective_breadth(sub)

    for _ in range(draws):
        if wide:
            g = wide[rng.integers(len(wide))]
            pick = rng.choice(g, size=size, replace=False)
            conc.append(metric(pick))
        chosen = rng.permutation(len(groups))[:size]
        pick = [groups[i][rng.integers(len(groups[i]))] for i in chosen]
        while len(pick) < size:
            g = groups[rng.integers(len(groups))]
            pick.append(g[rng.integers(len(g))])
        div.append(metric(pick))
    return conc, div


def claim_c4(streams: pd.DataFrame, draws: int, min_days: int = 730, seed: int = 1,
             split: str = "2023-01-01") -> tuple[dict, pd.DataFrame]:
    """Correlation and effective breadth within and across clusters.

    Two measurements. The in-sample one clusters on the whole record and
    scores on the whole record: it decomposes the realized correlation
    structure and is not an implementable selection rule. The out-of-sample
    one clusters on data before ``split`` and scores correlations only after it, which is
    what an allocator could actually have done.
    """
    usable = [c for c in streams.columns if streams[c].notna().sum() >= min_days]
    mat = streams[usable]
    corr = shrunk_correlation(mat.to_numpy(), min_obs=250, shrinkage=0.0)
    breadth = effective_breadth(corr)

    n_groups = 8
    labels = cluster_labels(corr, n_groups)

    train = streams.loc[streams.index < split]
    test = streams.loc[streams.index >= split]
    pit_names = [c for c in streams.columns
                 if train[c].notna().sum() >= 250
                 and pd.notna(train[c].iloc[-1])]
    corr_pit = shrunk_correlation(train[pit_names].to_numpy(), min_obs=250,
                                  shrinkage=0.0)
    labels_pit = cluster_labels(corr_pit, n_groups)
    corr_test = shrunk_correlation(test[pit_names].to_numpy(), min_obs=100,
                                   shrinkage=0.0)

    rows = []
    for size in (2, 4, 8, 16):
        rng = np.random.default_rng(seed + size)
        conc, div = _cluster_draw_metrics(corr, labels, size, draws, rng)
        rng_pit = np.random.default_rng(seed + 1000 + size)
        conc_p, div_p = _cluster_draw_metrics(
            corr_test, labels_pit, size, draws, rng_pit)
        conc_arr = np.asarray(conc)
        div_arr = np.asarray(div)
        conc_p_arr = np.asarray(conc_p)
        div_p_arr = np.asarray(div_p)
        rows.append({
            "n_markets": size,
            "mean_corr_same_cluster": (float(np.nanmean(conc_arr[:, 0]))
                                       if conc else np.nan),
            "mean_corr_across_clusters": float(np.nanmean(div_arr[:, 0])),
            "breadth_same_cluster": (float(np.nanmean(conc_arr[:, 1]))
                                     if conc else np.nan),
            "breadth_across_clusters": float(np.nanmean(div_arr[:, 1])),
            "oos_mean_corr_same_cluster": (
                float(np.nanmean(conc_p_arr[:, 0])) if conc_p else np.nan),
            "oos_mean_corr_across_clusters": float(
                np.nanmean(div_p_arr[:, 0])),
            "oos_breadth_same_cluster": (
                float(np.nanmean(conc_p_arr[:, 1])) if conc_p else np.nan),
            "oos_breadth_across_clusters": float(
                np.nanmean(div_p_arr[:, 1])),
        })
    table = pd.DataFrame(rows)
    table["breadth_gain"] = (table["breadth_across_clusters"]
                             - table["breadth_same_cluster"])
    table["oos_breadth_gain"] = (table["oos_breadth_across_clusters"]
                                 - table["oos_breadth_same_cluster"])
    summary = {
        "markets": len(usable),
        "effective_breadth_of_strategy_returns": float(breadth),
        "effective_breadth_shrinkage_0p30": float(effective_breadth(
            shrunk_correlation(mat.to_numpy(), min_obs=250, shrinkage=0.30))),
        "mean_pairwise_correlation": float(
            corr[~np.eye(len(usable), dtype=bool)].mean()),
        "nominal_breadth": len(usable),
        "clusters": int(n_groups),
        "oos_split": split,
        "oos_markets": len(pit_names),
        "oos_effective_breadth_train": float(effective_breadth(corr_pit)),
    }
    return summary, table


def claim_c5(panels: dict, config: Config, min_days: int = 730
             ) -> tuple[dict, pd.DataFrame]:
    grid = [(4, 12), (8, 24), (16, 48), (32, 96), (64, 192), (128, 384),
            (256, 768)]
    rows = []
    for fast, slow in grid:
        streams = single_market_returns(panels, config, speeds=((fast, slow),))
        usable = [c for c in streams.columns
                  if streams[c].notna().sum() >= min_days]
        port = streams[usable].fillna(0.0).mean(axis=1)
        per = [sharpe_ratio(streams[c].dropna().to_numpy()) for c in usable]
        rows.append({
            "fast": fast, "slow": slow,
            "slow_months": round(slow / 30.4, 1),
            "portfolio_sharpe": sharpe_ratio(port.to_numpy()),
            "median_single_market_sharpe": float(np.nanmedian(per)),
            "markets": len(usable),
        })
    table = pd.DataFrame(rows)
    best = table.loc[table["portfolio_sharpe"].idxmax()]
    return {
        "best_slow_span_days": int(best["slow"]),
        "best_slow_months": float(best["slow_months"]),
        "best_portfolio_sharpe": float(best["portfolio_sharpe"]),
        "source_typical_months": "3-6",
    }, table


def claim_c10(no_target_daily: pd.DataFrame, target: float) -> dict:
    """Actual post-buffer ex-ante risk of the fixed-multiplier control."""
    d = no_target_daily.dropna(subset=["held_ex_ante_vol"])
    ratio = d["held_ex_ante_vol"] / target
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    return {
        "days": int(len(d)),
        "claimed_range": [0.3, 3.0],
        "risk_to_target_quantiles": {
            str(q): float(ratio.quantile(q)) for q in qs},
        "risk_to_target_min": float(ratio.min()),
        "risk_to_target_max": float(ratio.max()),
        "share_outside_claimed_range": float(
            ((ratio < 0.3) | (ratio > 3.0)).mean()),
    }


def ablation_table(runs: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = returns["baseline"].to_numpy()
    for _, r in runs.iterrows():
        label = r["label"]
        if label == "baseline" or label not in returns.columns:
            continue
        arr = returns[label].to_numpy()
        if np.allclose(arr - base, 0.0):
            lo = hi = 0.0
        else:
            lo, hi = stationary_bootstrap_sharpe_diff_ci(arr, base)
        rows.append({
            "group": r["group"], "label": label,
            "sharpe_net": r["sharpe_net"],
            "sharpe_vs_baseline": r["sharpe_net"] - float(
                runs.loc[runs["label"] == "baseline", "sharpe_net"].iloc[0]),
            "sharpe_diff_ci_lo": lo, "sharpe_diff_ci_hi": hi,
            "ann_turnover": r["ann_turnover"],
            "cost_drag_ann": r["cost_drag_ann"],
            "funding_drag_ann": r["funding_drag_ann"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()
    ensure_directories()

    config = Config()
    panels = {n: pd.read_parquet(PANEL_DIR / f"{n}.parquet") for n in FIELDS}

    print("building single-market streams", flush=True)
    net = single_market_returns(panels, config, net=True)
    gross = single_market_returns(panels, config, net=False)
    net.to_parquet(RESULTS_DIR / "single_market_returns.parquet")

    out: dict = {}

    print("C1 single-market Sharpe", flush=True)
    out["C1"], t1 = claim_c1(net, gross)
    t1.to_csv(RESULTS_DIR / "claim_c1_single_market.csv", index=False)

    print("C2/C3 breadth scaling", flush=True)
    prepared = prepare_inputs(panels, config)
    availability = (prepared.tradeable & prepared.signal.notna()
                    & prepared.annual_vol.notna() & prepared.opens.notna())
    out["C2_C3"], t2 = claim_c2(gross, availability, args.draws)
    t2.to_csv(RESULTS_DIR / "claim_c2_breadth.csv", index=False)

    print("C4 diversification", flush=True)
    out["C4"], t4 = claim_c4(net, args.draws)
    t4.to_csv(RESULTS_DIR / "claim_c4_clusters.csv", index=False)

    print("C5 horizons", flush=True)
    out["C5"], t5 = claim_c5(panels, config)
    t5.to_csv(RESULTS_DIR / "claim_c5_horizons.csv", index=False)

    no_target_daily = pd.read_csv(
        RESULTS_DIR / "no_target_daily.csv", index_col=0, parse_dates=True)
    print("C10 risk dispersion", flush=True)
    out["C10"] = claim_c10(no_target_daily, config.vol_target_annual)

    runs = pd.read_csv(RESULTS_DIR / "runs.csv")
    returns = pd.read_csv(RESULTS_DIR / "run_returns.csv", index_col=0,
                          parse_dates=True)
    print("ablation bootstrap", flush=True)
    ab = ablation_table(runs, returns)
    ab.to_csv(RESULTS_DIR / "ablation_significance.csv", index=False)

    diag = pd.read_csv(RESULTS_DIR / "diagnostics.csv", index_col=0,
                       parse_dates=True)
    out["risk_model"] = {
        "mean_pairwise_correlation_of_prices": float(diag["mean_corr"].mean()),
        "mean_effective_breadth_of_prices": float(
            diag["effective_breadth"].mean()),
        "last_effective_breadth": float(diag["effective_breadth"].iloc[-1]),
        "mean_markets_in_risk_model": float(diag["n_corr"].mean()),
        "max_markets_in_risk_model": float(diag["n_corr"].max()),
    }

    tradeable = prepared.tradeable
    funding_known = panels["funding"].loc[tradeable.index].notna() & tradeable
    out["funding_coverage_of_tradeable_symbol_days"] = float(
        funding_known.to_numpy().sum() / max(tradeable.to_numpy().sum(), 1))

    (RESULTS_DIR / "claims.json").write_text(json.dumps(out, indent=2,
                                                        default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
