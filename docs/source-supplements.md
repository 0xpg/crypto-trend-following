# Supplementary source notes

Retrieved: 2026-07-28

These neutral summaries cover five related posts outside the November 2022
construction thread. Post identifiers preserve the evidence trail. None of
the notes changes the baseline in README section 3. S4 and S5 motivate the
separate short-cap extension in README section 13.

## S1. Overfitting primer

Root post: [1370349076525514752](https://x.com/i/web/status/1370349076525514752)

Published: 2021-03-12

The six-part example starts with a price series that appears mean reverting,
fits moving-average speeds, selects the strongest in-sample parameter and
shows an attractive backtest. The final post reveals that the price series is
pure random noise. The example supports freezing parameters before reading
results and treating parameter sweeps as diagnostics rather than selection.

## S2. Practitioner priorities

Post: [1529771509639786496](https://x.com/i/web/status/1529771509639786496)

Published: 2022-05-26

The post lists several topics that receive substantial retail attention but
less attention in systematic practice, including volatility forecasting,
stationarity tests, cointegration, exits and stop losses. It supports the C8
comparison among simple volatility forecasts without implying an oracle.

## S3. Point-in-time universe selection

Root post: [1598823745681903616](https://x.com/i/web/status/1598823745681903616)

Published: 2022-12-02

The first entry in a backtest-error series explains why selecting assets from
current market capitalization, current liquidity or any later eligibility
introduces future information. It prescribes a rolling universe, repeated
eligibility checks and explicit handling of delistings. README section 3.2
implements that rule daily.

## S4. Rebalance frequency and squeeze risk

Post: [2081477331877519775](https://x.com/i/web/status/2081477331877519775)

Published: 2026-07-26

The post reports a move from 30-minute to 5-minute rebalancing after a short
position encountered a 20x intraday pump and approached liquidation. The
event motivates the daily-bar limitation in README sections 7 and 12.

## S5. Asymmetric short cap

Post: [2081478809111368154](https://x.com/i/web/status/2081478809111368154)

Published: 2026-07-26

The post distinguishes bounded long losses from unbounded short losses. With
margin fraction `p` and a required survival multiple `m`, it proposes a short
limit near `p/(2m)` of gross exposure. The numerical example `p=0.2`, `m=10`
produces a 2% cap. README section 13 reports that extension separately from
the frozen baseline.
