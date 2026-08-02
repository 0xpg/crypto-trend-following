# Trend Atlas

Trend Atlas is an open research workspace for studying systematic trend
following in crypto perpetual futures. It combines a causal portfolio engine,
a CCXT data route, synthetic correctness tests, and an interactive Streamlit
lab.

The project is organized around questions rather than a promised strategy:

- Does a diversified trend ensemble survive realistic trading costs?
- How sensitive is performance to the asset universe and test window?
- Do volatility targeting and risk clustering improve the path of returns?
- How much turnover can a no-trade buffer remove before it weakens tracking?
- Which assumptions would invalidate the result in live trading?

This is research software, not financial advice or an execution system.

## Start with a question

Run the lab and change one assumption at a time:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The sidebar lets you select:

- Binance USD-M perpetual assets, including custom CCXT symbols;
- backtest start and end dates;
- signal response: `tanh`, `overextension`, `linear`, or `binary`;
- annual portfolio volatility target;
- no-trade buffer;
- estimated trading cost in basis points;
- gross leverage ceiling;
- number of correlation-based risk clusters;
- minimum market-history requirement.

The dashboard initially displays the bundled CCXT research run. Selecting
**Fetch data and run** replaces it with the chosen experiment. Daily results
can be downloaded as CSV.

## Research design

The engine follows a seven-stage portfolio process.

| Stage | Question | Implementation |
|---|---|---|
| Universe | Which contracts were eligible at the decision time? | History, bar-completeness, and rolling-liquidity gates |
| Trend | Is price moving persistently across several horizons? | Log-price EWMA crossovers `(16,48)`, `(32,96)`, `(64,192)` |
| Sizing | How strongly should the portfolio express the signal? | Saturating response divided by forecast volatility |
| Diversification | Are nominally different assets carrying the same risk? | Shrunk correlations and hierarchical clusters |
| Portfolio risk | How large should the entire book be? | Covariance-aware annual volatility target |
| Trading | Is the target change large enough to justify a trade? | Symmetric no-trade buffer |
| Accounting | Does performance survive implementation drag? | Open-to-open P&L, costs, funding where available, and causal gap marks |

Signals use information available before the modeled fill. The engine never
turns a missing price into a favorable fill and does not infer a liquidation
from the later discovery that a contract was delisted.

## Default experiment

The bundled CCXT run is an accessible example, not a definitive estimate.

| Setting | Value |
|---|---|
| Venue | Binance USD-M perpetuals through CCXT |
| Assets | 12 fixed liquid contracts |
| Period | 2020-01-01 through 2026-08-01 |
| Starting equity | $100,000 |
| Response | `tanh` |
| Portfolio volatility target | 20% annualized |
| Trading cost | 10 bps of traded notional |
| Funding | Excluded from this CCXT example |

Measured output:

| Metric | Result |
|---|---:|
| Total return | +215.9% |
| CAGR | 19.1% |
| Net Sharpe | 1.03 |
| Annualized volatility | 18.7% |
| Maximum drawdown | -32.5% |
| Annual turnover | 7.5x |

These numbers are conditional on a small fixed universe and omit historical
funding. They should be used to formulate the next test, not to extrapolate a
future return.

## Two data workflows

### CCXT research loop

The quickest path fetches public daily OHLCV candles and needs no API key:

```powershell
python scripts/ccxt_backtest.py
```

Optional example:

```powershell
python scripts/ccxt_backtest.py --start 2022-01-01 --symbols BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT
```

Outputs:

- `results/ccxt_daily.csv`
- `results/ccxt_summary.json`

### Archive research loop

The larger workflow builds daily panels from a Binance archive mirror and can
include the published funding history:

```powershell
python scripts/fetch_funding.py
python scripts/build_panel.py
python scripts/backtest.py
python scripts/claims.py
```

Set `TREND_MIRROR_DIR`, `TREND_DATA_DIR`, or `TREND_RESULTS_DIR` to relocate
raw archives, derived panels, or outputs.

## Configuration model

All portfolio assumptions live in the immutable `Config` dataclass in
`engine.py`. Important defaults include:

```python
Config(
    min_history_days=120,
    vol_short_window=60,
    vol_floor_annual=0.20,
    speeds=((16, 48), (32, 96), (64, 192)),
    response="tanh",
    corr_window=180,
    n_clusters=8,
    vol_target_annual=0.20,
    max_gross_leverage=2.0,
    buffer=0.10,
    cost_bps=10.0,
)
```

Create variations with `dataclasses.replace` or `engine.variant`; avoid
mutating assumptions after observing a result.

## Verification

Run the synthetic suite without downloading market data:

```powershell
python -m unittest discover -s tests -v
```

The tests cover signal timing, missing observations, delisting behavior,
funding signs, trading costs, volatility targeting, leverage limits, risk
clustering, response functions, buffering, and portfolio ruin. A broad input
tampering test confirms that future observations do not alter earlier reported
columns.

## Repository map

| Path | Purpose |
|---|---|
| `app.py` | Interactive research dashboard |
| `engine.py` | Pure portfolio and accounting functions |
| `scripts/ccxt_backtest.py` | Public-data CCXT experiment |
| `scripts/build_panel.py` | Archive-to-daily panel builder |
| `scripts/backtest.py` | Baseline, ablations, and parameter sweeps |
| `scripts/claims.py` | Cross-sectional and portfolio diagnostics |
| `tests/test_engine.py` | Synthetic causality and accounting tests |

## Known limits

- A fixed asset list can introduce selection bias.
- Daily bars cannot model intraday liquidation paths.
- Flat basis-point costs do not model market impact or order-book depth.
- Crypto contracts provide less independent breadth than their listing count
  suggests.
- The CCXT example excludes funding history.
- The model has no exchange default, margin waterfall, or operational-risk
  simulation.

Treat every attractive result as an invitation to ask a harder question.

## License

MIT. See `LICENSE`.
