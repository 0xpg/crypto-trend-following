# Source and interpretation boundary

- Post: [1587591552691765251](https://x.com/i/web/status/1587591552691765251)
- Published: 2022-11-01 23:44 UTC, continued 2022-11-02
- Retrieved: 2026-07-26, post-by-post evidence map in
  [docs/source-thread.md](docs/source-thread.md)

The thread is a construction guide, not a strategy disclosure. It names
seven components and states how each is normally built:

1. **Universe selection**: trade as many *diversifying* markets as possible.
2. **Trend detection**: moving-average crossovers on log price, normalized
   by volatility.
3. **Mapping trend strength to desired position**: a saturating response
   function.
4. **Sector / asset-class weights**: equalize long-run risk across groups.
5. **Portfolio risk targeting**: control total strategy volatility.
6. **Trading rules**: buffering to suppress turnover.
7. **Execution**: algos for liquid markets.

## Source statements used in the evaluation

The thread mixes empirical claims with descriptions of common practitioner
construction. Performance claims are measured directly; construction
statements are checked for faithful implementation and reported with
sensitivity analysis rather than treated as claims about optimal Sharpe.

| # | Claim | Source tweet |
|---|---|---|
| C1 | Single-market trend following earns "around 0.2 Sharpe" | 6 |
| C2 | The strategy "scales like sqrt(breadth), where breadth is the number of independent markets" | 6 |
| C3 | Expanding the universe can raise Sharpe by "1.5-2x" | 9 |
| C4 | Adding a correlated market gives "almost no benefit"; an uncorrelated one "helps a lot" | 7 |
| C5 | Practitioners use horizons from 1 month to 1 year, averaging 3-6 months; >1y focus is rare | 12 |
| C6 | A saturating response (sigmoid, or `x*exp(-x^2)`) is a common mapping | 13, 14 |
| C7 | Inverse-volatility position sizing is the most common construction | 19 |
| C8 | A simple blend of long-run and 30-60 day realized volatility captures most of the benefit | 20 |
| C9 | Asset-class groups can be sized to roughly equal long-run risk | 21-24 |
| C10 | Without risk targeting, realized risk ranges "0.3x to 3x" the target | 25 |
| C11 | Sizing up when few markets trend and down when many do "improves sharpe ratios" | 27 |
| C12 | Buffering keeps positions near target while cutting turnover and cost | 31-32 |

## Supplementary posts

Five related posts bear on this project and are summarized in
[docs/source-supplements.md](docs/source-supplements.md). None
is part of the construction thread and none alters the frozen rule set:

| # | Post | Date | Bears on |
|---|---|---|---|
| S1 | overfitting primer (random-noise backtest) | 2021-03-12 | the freeze-before-measuring protocol |
| S2 | what practitioners ignore (incl. vol forecasting) | 2022-05-26 | the C8 sensitivity analysis |
| S3 | backtest errors day 1: universe look-ahead | 2022-12-02 | the point-in-time filter of README 3.2 |
| S4 | 5-minute rebalances after a 20x short squeeze | 2026-07-26 | the daily-rebalance limitation |
| S5 | cap shorts at p/(2m) of gross | 2026-07-26 | tested as a post-freeze extension, README section 13 |

## Unpublished details

- moving-average lengths, the number of speeds, or how speeds are combined;
- the exact normalization of the signal;
- the sigmoid parameters, or the constant in `x*exp(-x^2)`;
- the blend weights of the volatility forecast;
- the volatility target, leverage cap, or rebalance frequency;
- the buffer width;
- sector definitions or the risk-equalization algorithm;
- any backtest, chart, return series, or code.

Every number in this project's rule set is **a choice made here**,
not a source parameter. Section 3 of [README.md](README.md) labels each one.
The strategy is frozen in that document before any result is computed.

## Venue substitution

The thread describes cross-asset futures: equity indices, fixed income, FX,
commodities, credit. This project has one venue, Binance USDT-margined
perpetual futures, so the reimplementation runs inside a **single asset
class**. That is a material deviation and it is the point of the exercise
rather than a flaw to hide: claims C2, C3, C4 and C9 are all statements
about diversification, and a universe of 788 crypto perpetuals that move
together is the sharpest available test of what happens when nominal
breadth is large and effective breadth is not.

Crypto perpetuals also add a cost the thread's futures framing does not
have: funding. It is measured separately rather than folded into a
commission assumption.

## Reproduction verdict

The seven components produce an operational Binance-perpetual trend book:
net Sharpe 1.02, 20.0% realized volatility against a 20% target, and -33%
worst drawdown. Design-period and held-out Sharpes differ materially, so the
full-sample Sharpe is not treated as a stationary expectation.

The breadth evidence is consistent with the source's conditional logic.
Among 268 established strategy-return streams, equal-weight
variance-equivalent breadth is 3.81 despite the much larger nominal listing
count. Point-in-time portfolios saturate quickly as correlated contracts
are added. C2 defines breadth as the number of independent markets. Expanding
one crypto venue is not comparable to expanding a cross-asset futures
universe.

Inverse-volatility sizing contributes detectably in this implementation.
Risk targeting lowers the point estimate of risk dispersion and raises the
Sharpe estimate, but its paired confidence interval includes zero. Buffering
delivers the stated turnover/tracking-error trade-off. Industry-practice
statements about horizons, response functions and volatility forecasts are
implemented faithfully and reported as sensitivities. Full measurements are
in [README.md](README.md) sections 9-11 and
[results/report.md](results/report.md).
