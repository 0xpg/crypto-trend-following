#!/usr/bin/env python3
"""Figures for the trend-following report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from _style import C1, C2, C3, C4, C8, INK2, MUTED, plt  # noqa: E402
from paths import FIGURES_DIR, RESULTS_DIR, ensure_directories  # noqa: E402


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=160)
    plt.close(fig)
    print(f"  {name}")


def fig_equity(daily: pd.DataFrame) -> None:
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1]})
    eq = daily["equity"] / daily["equity"].iloc[0]
    ax.plot(eq.index, eq, color=C1, lw=1.3)
    ax.set_yscale("log")
    ax.set_ylabel("equity (log, start = 1)")
    ax.set_title("Baseline: seven-component trend book on Binance perpetuals",
                 loc="left", fontsize=10)
    dd = eq / eq.cummax() - 1.0
    ax2.fill_between(dd.index, dd, 0, color=C8, alpha=0.35, lw=0)
    ax2.set_ylabel("drawdown")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    save(fig, "fig1_equity.png")


def fig_breadth(t2_all: pd.DataFrame, claims: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    n = t2_all["average_live_markets"]
    ax.fill_between(n, t2_all["sharpe_p10"], t2_all["sharpe_p90"], color=C1,
                    alpha=0.15, lw=0, label="10-90% of random draws")
    ax.plot(n, t2_all["sharpe_mean"], color=C1, lw=1.6, marker="o", ms=3.5,
            label="observed mean Sharpe")
    eb = claims["C4"]["effective_breadth_of_strategy_returns"]
    ax.axvline(eb, color=C3, lw=1.1, ls=":",
               label=f"equal-weight effective breadth = {eb:.1f}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("average live markets in point-in-time portfolio")
    ax.set_ylabel("gross annualized Sharpe")
    ax.set_title("C2/C3: gross breadth and correlated saturation",
                 loc="left", fontsize=10)
    ax.legend(loc="upper left")
    save(fig, "fig2_breadth.png")


def fig_single_market(t1: pd.DataFrame, claims: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(t1["sharpe_net"], bins=45, color=C1, alpha=0.75, lw=0)
    med = t1["sharpe_net"].median()
    ax.axvline(med, color=C2, lw=1.4, label=f"median {med:+.2f}")
    ax.axvline(0.2, color=C3, lw=1.4, ls="--", label="0.2 reference")
    ax.axvline(0.0, color=MUTED, lw=0.8)
    ax.set_xlabel("annualized Sharpe, one market, net of costs and funding")
    ax.set_ylabel("markets")
    ax.set_title(f"C1: single-market trend Sharpe across "
                 f"{len(t1)} perpetuals", loc="left", fontsize=10)
    ax.legend()
    save(fig, "fig3_single_market.png")


def fig_horizons(t5: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t5["slow"], t5["portfolio_sharpe"], color=C1, lw=1.6, marker="o",
            ms=4, label="equal-weight portfolio")
    ax.plot(t5["slow"], t5["median_single_market_sharpe"], color=C2, lw=1.3,
            marker="s", ms=3.5, ls="--", label="median single market")
    ax.axvspan(91, 183, color=C3, alpha=0.12, lw=0)
    ax.text(130, ax.get_ylim()[1] * 0.05, "medium-term reference",
            ha="center", fontsize=8, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("slow EWMA span (days)")
    ax.set_ylabel("annualized Sharpe")
    ax.set_title("C5: Sharpe by trend horizon on crypto perpetuals",
                 loc="left", fontsize=10)
    ax.legend()
    save(fig, "fig4_horizons.png")


def fig_risk(daily: pd.DataFrame, no_target: pd.DataFrame, claims: dict) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    d = no_target.dropna(subset=["held_ex_ante_vol"])
    ratio = d["held_ex_ante_vol"] / 0.20
    ax.hist(ratio, bins=60, color=C1, alpha=0.8, lw=0)
    for x, c in ((0.3, C8), (3.0, C8)):
        ax.axvline(x, color=c, lw=1.3, ls="--")
    ax.text(0.31, ax.get_ylim()[1] * 0.9, "0.3x-3x reference range", fontsize=8,
            color=C8)
    ax.set_xlabel("untargeted risk / its own mean")
    ax.set_ylabel("days")
    ax.set_title("C10: dispersion of untargeted risk", loc="left",
                 fontsize=10)

    roll = d["net_return"].rolling(90).std() * np.sqrt(365)
    ax2.plot(roll.index, roll, color=C1, lw=1.1, label="realized (90d)")
    ax2.axhline(0.20, color=C2, lw=1.2, ls="--", label="20% target")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.set_ylabel("annualized volatility")
    ax2.set_title("Realized volatility against the 20% target", loc="left", fontsize=10)
    ax2.legend()
    save(fig, "fig5_risk_targeting.png")


def fig_buffer(runs: pd.DataFrame) -> None:
    b = runs[runs["group"] == "buffer"].copy()
    b["b"] = b["label"].str.replace("buffer_", "").astype(float)
    b = b.sort_values("b")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b["ann_turnover"], b["sharpe_net"], color=C1, lw=1.5, marker="o",
            ms=4)
    for _, r in b.iterrows():
        ax.annotate(f"b={r['b']:g}", (r["ann_turnover"], r["sharpe_net"]),
                    textcoords="offset points", xytext=(6, -3), fontsize=8,
                    color=INK2)
    ax.set_xlabel("annual turnover (multiples of capital)")
    ax.set_ylabel("net annualized Sharpe")
    ax.set_title("C12: buffering trades turnover against tracking error",
                 loc="left", fontsize=10)
    save(fig, "fig6_buffer.png")


def fig_components(runs: pd.DataFrame) -> None:
    order = ["no_costs", "baseline", "no_sector_weights", "no_buffer",
             "equal_notional", "no_risk_targeting"]
    nice = {"no_costs": "costs and funding removed", "baseline": "full system",
            "no_sector_weights": "without sector weights (C9)",
            "no_buffer": "without buffering (C12)",
            "equal_notional": "without inverse-vol sizing (C7)",
            "no_risk_targeting": "without risk targeting (C11)"}
    sub = runs.set_index("label").reindex(order).dropna(subset=["sharpe_net"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = [C3 if i == "no_costs" else C1 if i == "baseline" else C4
              for i in sub.index]
    ax.barh(range(len(sub)), sub["sharpe_net"], color=colors, height=0.6)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([nice[i] for i in sub.index])
    ax.invert_yaxis()
    for i, v in enumerate(sub["sharpe_net"]):
        ax.text(v + 0.02, i, f"{v:+.2f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("net annualized Sharpe, 2020-2026")
    ax.set_title("Which components carry the result", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_components.png")


def fig_universe(daily: pd.DataFrame, diag: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(daily.index, daily["n_live"], color=C1, lw=1.2,
            label="tradeable markets")
    ax.plot(diag.index, diag["effective_breadth"], color=C2, lw=1.4,
            label="effective breadth (price correlations)")
    ax.set_ylabel("markets")
    ax.set_title("Nominal universe growth does not buy independence",
                 loc="left", fontsize=10)
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(diag.index, diag["mean_corr"], color=C3, lw=1.0, alpha=0.8)
    ax2.set_ylabel("mean pairwise correlation", color=C3)
    ax2.grid(False)
    ax2.tick_params(axis="y", colors=C3)
    save(fig, "fig8_universe.png")


def main() -> None:
    ensure_directories()
    daily = pd.read_csv(RESULTS_DIR / "baseline_daily.csv", index_col=0,
                        parse_dates=True)
    no_target = pd.read_csv(RESULTS_DIR / "no_target_daily.csv", index_col=0,
                            parse_dates=True)
    diag = pd.read_csv(RESULTS_DIR / "diagnostics.csv", index_col=0,
                       parse_dates=True)
    runs = pd.read_csv(RESULTS_DIR / "runs.csv")
    claims = json.loads((RESULTS_DIR / "claims.json").read_text())
    t1 = pd.read_csv(RESULTS_DIR / "claim_c1_single_market.csv")
    t2 = pd.read_csv(RESULTS_DIR / "claim_c2_breadth.csv")
    t5 = pd.read_csv(RESULTS_DIR / "claim_c5_horizons.csv")

    print("figures:")
    fig_equity(daily)
    fig_breadth(t2, claims)
    fig_single_market(t1, claims)
    fig_horizons(t5)
    fig_risk(daily, no_target, claims)
    fig_buffer(runs)
    fig_components(runs)
    fig_universe(daily, diag)


if __name__ == "__main__":
    main()
