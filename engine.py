"""Pure functions for the seven-component trend-following system.

Nothing here reads or writes files. Every function takes and returns arrays
or frames indexed by date with symbols as columns, so the simulation can be
unit-tested without the archive.

The rule set implemented here is frozen in README.md sections 3.1-3.10.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ANNUALIZE = np.sqrt(365.0)

Response = Literal["tanh", "overextension", "linear", "binary"]
VolForecast = Literal["blend", "short", "long", "ewma"]
Sizing = Literal["inverse_vol", "equal_notional"]

# peak of z * exp(-z^2 / 4), attained at z = sqrt(2)
_OVEREXT_PEAK = np.sqrt(2.0) * np.exp(-0.5)


@dataclass(frozen=True)
class Config:
    """Every field is a README section 3 parameter."""

    # 3.2 universe
    min_history_days: int = 120
    volume_lookback: int = 30
    min_median_quote_volume: float = 5_000_000.0
    min_hours_per_day: int = 20

    # 3.3 volatility forecast
    vol_short_window: int = 60
    vol_long_cap_days: int = 3650
    vol_blend_short: float = 0.70
    vol_floor_annual: float = 0.20
    vol_forecast: VolForecast = "blend"
    vol_ewma_span: int = 60

    # 3.4 signal
    speeds: tuple[tuple[int, int], ...] = ((16, 48), (32, 96), (64, 192))
    signal_scale_window: int = 365
    signal_scale_min_periods: int = 60
    signal_clip: float = 6.0

    # 3.5 / 3.6 response and sizing
    response: Response = "tanh"
    sizing: Sizing = "inverse_vol"

    # 3.6 sector weights
    use_sector_weights: bool = True
    corr_rebuild_days: int = 7
    corr_window: int = 180
    corr_min_obs: int = 100
    corr_shrinkage: float = 0.30
    n_clusters: int = 8

    # 3.7 risk targeting
    vol_target_annual: float = 0.20
    use_risk_targeting: bool = True
    fixed_multiplier: float | None = None
    max_gross_leverage: float = 2.0

    # section 13 extension: asymmetric short cap, off in the frozen baseline.
    # Each short is capped at this fraction of the book's pre-cap gross
    # notional (the supplement's p/(2m) rule for leveraged crypto books).
    max_short_frac_gross: float | None = None

    # 3.8 trading rule
    buffer: float = 0.10

    # 3.9 costs
    cost_bps: float = 10.0
    apply_funding: bool = True

    # 3.10 accounting
    initial_equity: float = 100_000.0
    start: str = "2020-01-01"
    end: str = "2026-07-22"

    def validate(self) -> None:
        if self.vol_short_window <= 1:
            raise ValueError("vol_short_window must exceed 1")
        if not 0.0 <= self.vol_blend_short <= 1.0:
            raise ValueError("vol_blend_short must lie in [0, 1]")
        if self.vol_floor_annual <= 0:
            raise ValueError("vol_floor_annual must be positive")
        if not self.speeds:
            raise ValueError("at least one crossover speed is required")
        if any(s >= l for s, l in self.speeds):
            raise ValueError("each speed needs fast span < slow span")
        if not 0.0 <= self.corr_shrinkage <= 1.0:
            raise ValueError("corr_shrinkage must lie in [0, 1]")
        if self.n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if self.buffer < 0:
            raise ValueError("buffer must be non-negative")
        if self.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")
        if self.max_short_frac_gross is not None and \
                not 0.0 < self.max_short_frac_gross <= 1.0:
            raise ValueError("max_short_frac_gross must lie in (0, 1]")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")


def config_dict(config: Config) -> dict:
    out = asdict(config)
    out["speeds"] = [list(pair) for pair in config.speeds]
    return out


# --------------------------------------------------------------------------
# 3.3 volatility forecast
# --------------------------------------------------------------------------

def volatility_forecast(log_returns: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Annualized per-symbol volatility forecast, row t using returns to t.

    ``log_returns`` is close-to-close and indexed by day. Callers lag this by
    one day before using it to size a position filled at the open of t; see
    :func:`prepare_inputs`.
    """
    short = log_returns.rolling(config.vol_short_window, min_periods=20).std()
    long = log_returns.rolling(
        config.vol_long_cap_days, min_periods=60).std()
    ewma = log_returns.ewm(span=config.vol_ewma_span, min_periods=20).std()

    if config.vol_forecast == "short":
        raw = short
    elif config.vol_forecast == "long":
        raw = long.where(long.notna(), short)
    elif config.vol_forecast == "ewma":
        raw = ewma
    else:
        w = config.vol_blend_short
        raw = w * short + (1.0 - w) * long
        raw = raw.where(long.notna(), short)

    annual = raw * ANNUALIZE
    return annual.clip(lower=config.vol_floor_annual)


@dataclass
class Inputs:
    """Everything :func:`simulate` needs, with the lag convention applied."""

    opens: pd.DataFrame
    closes: pd.DataFrame
    signal: pd.DataFrame
    annual_vol: pd.DataFrame
    tradeable: pd.DataFrame
    funding: pd.DataFrame
    log_returns: pd.DataFrame


def prepare_inputs(panels: dict[str, pd.DataFrame], config: Config,
                   speeds: tuple[tuple[int, int], ...] | None = None) -> Inputs:
    """Derive eligibility, volatility and signal frames from the raw panel.

    Row t of every returned frame except ``opens`` and ``funding`` is
    computable at 00:00 UTC on day t, which is when it is used.
    """
    close = panels["close"]
    opens = panels["open"]
    hours = panels["hours"]
    qvol = panels["quote_volume"]

    complete = (hours >= config.min_hours_per_day) & close.notna() & opens.notna()

    # A gap must not erase the price move across it. Carry the last complete
    # close as the risk mark, then record the whole move when a complete bar
    # resumes. Missing/incomplete rows remain NaN rather than adding stale
    # zero returns to the volatility sample.
    risk_mark = close.where(complete).ffill()
    log_returns = np.log(risk_mark).diff().where(complete)

    vol_now = volatility_forecast(log_returns, config)
    annual_vol = vol_now.shift(1)
    signal = trend_signal(close, vol_now / ANNUALIZE, config, speeds=speeds)

    history = complete.cumsum().shift(1)
    med_volume = (qvol.where(complete)
                  .rolling(config.volume_lookback,
                           min_periods=config.volume_lookback // 2)
                  .median()
                  .shift(1))

    tradeable = (
        opens.notna()
        & complete.shift(1).fillna(False)
        & (history >= config.min_history_days)
        & (med_volume >= config.min_median_quote_volume)
    )

    funding = panels.get("funding")
    if funding is None:
        funding = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)

    mask = (opens.index >= pd.Timestamp(config.start)) & \
           (opens.index <= pd.Timestamp(config.end))
    sl = opens.index[mask]
    return Inputs(
        opens=opens.loc[sl],
        closes=close.loc[sl],
        signal=signal.loc[sl],
        annual_vol=annual_vol.loc[sl],
        tradeable=tradeable.loc[sl],
        funding=funding.loc[sl],
        log_returns=log_returns.loc[sl],
    )


# --------------------------------------------------------------------------
# 3.4 trend signal
# --------------------------------------------------------------------------

def trend_signal(
    close: pd.DataFrame,
    daily_vol_now: pd.DataFrame,
    config: Config,
    speeds: tuple[tuple[int, int], ...] | None = None,
) -> pd.DataFrame:
    """Volatility-normalized multi-speed EWMA crossover, usable at t.

    ``daily_vol_now`` is the *daily* (not annualized) volatility estimate
    computed through and including row t. The whole result is lagged one day
    at the end, so the returned row t depends only on closes through t-1.

    Speeds whose slow span is not yet filled contribute nothing; a young
    symbol therefore trades on its fast speeds alone until the slow ones
    become available.
    """
    speeds = speeds or config.speeds
    logp = np.log(close)
    parts = []
    for fast, slow in speeds:
        raw = (logp.ewm(span=fast, min_periods=fast).mean()
               - logp.ewm(span=slow, min_periods=slow).mean())
        y = raw / (daily_vol_now * np.sqrt(slow))
        scale = y.rolling(
            config.signal_scale_window,
            min_periods=config.signal_scale_min_periods,
        ).std()
        parts.append(y / scale.replace(0.0, np.nan))
    stacked = pd.concat(parts, axis=0, keys=range(len(parts)))
    combined = stacked.groupby(level=1).mean()
    combined = combined.reindex(close.index)
    return combined.clip(-config.signal_clip, config.signal_clip).shift(1)


# --------------------------------------------------------------------------
# 3.5 response function
# --------------------------------------------------------------------------

def response(z: pd.DataFrame | np.ndarray, kind: Response) -> pd.DataFrame | np.ndarray:
    if kind == "tanh":
        return np.tanh(z)
    if kind == "overextension":
        return z * np.exp(-np.square(z) / 4.0) / _OVEREXT_PEAK
    if kind == "linear":
        return np.clip(z, -1.0, 1.0)
    if kind == "binary":
        return np.sign(z)
    raise ValueError(f"unknown response {kind!r}")


# --------------------------------------------------------------------------
# 3.6 correlation, clusters, sector weights
# --------------------------------------------------------------------------

def shrunk_correlation(
    window_returns: np.ndarray, min_obs: int, shrinkage: float
) -> np.ndarray:
    """Pairwise-complete correlation of a (T, n) return block, then shrunk.

    Every pair is centered and standardized on its shared observations.
    Pairs with fewer than ``min_obs`` shared observations fall back to the
    mean correlation. The shrunk result is projected to the positive-
    semidefinite correlation cone before it is used as a risk model.
    """
    mask = np.isfinite(window_returns)
    x = np.where(mask, window_returns, 0.0)
    pair_counts = mask.astype(np.float64).T @ mask.astype(np.float64)
    pair_counts_safe = np.maximum(pair_counts, 1.0)
    sums = x.T @ mask.astype(np.float64)
    cross = x.T @ x
    squares = np.square(x).T @ mask.astype(np.float64)
    cov_num = cross - sums * sums.T / pair_counts_safe
    var_i = squares - np.square(sums) / pair_counts_safe
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov_num / np.sqrt(np.maximum(var_i * var_i.T, 0.0))

    n = corr.shape[0]
    off = ~np.eye(n, dtype=bool)
    valid = off & (pair_counts >= min_obs) & np.isfinite(corr)
    mean_corr = float(corr[valid].mean()) if valid.any() else 0.0
    corr = np.where(valid | np.eye(n, dtype=bool), corr, mean_corr)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    target = np.full((n, n), mean_corr)
    np.fill_diagonal(target, 1.0)
    out = (1.0 - shrinkage) * corr + shrinkage * target
    np.fill_diagonal(out, 1.0)
    out = 0.5 * (out + out.T)

    eigval, eigvec = np.linalg.eigh(out)
    eigval = np.maximum(eigval, 1e-10)
    out = (eigvec * eigval) @ eigvec.T
    scale = np.sqrt(np.maximum(np.diag(out), 1e-12))
    out = out / np.outer(scale, scale)
    np.fill_diagonal(out, 1.0)
    return 0.5 * (out + out.T)


def cluster_labels(corr: np.ndarray, n_clusters: int) -> np.ndarray:
    """Average-linkage clusters on the correlation distance sqrt(2(1-rho))."""
    n = corr.shape[0]
    if n <= n_clusters:
        return np.arange(n)
    dist = np.sqrt(np.maximum(2.0 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(z, t=n_clusters, criterion="maxclust")


def sector_weights(
    corr: np.ndarray,
    vols: np.ndarray,
    labels: np.ndarray,
    active: np.ndarray | None = None,
    sizing: Sizing = "inverse_vol",
) -> np.ndarray:
    """Equalize full-signal standalone risk across active clusters.

    Returns a per-symbol multiplier with mean 1 across active symbols, so
    switching sector weights on or off does not by itself change scale.
    """
    active = (np.ones(len(labels), dtype=bool)
              if active is None else np.asarray(active, dtype=bool))
    cov = corr * np.outer(vols, vols)
    base = np.where(active, 1.0, 0.0)
    if sizing == "inverse_vol":
        base = base / np.maximum(vols, 1e-9)
    out = np.ones(len(labels), dtype=np.float64)
    for lab in np.unique(labels):
        idx = np.flatnonzero((labels == lab) & active)
        if not len(idx):
            continue
        w = np.zeros(len(labels), dtype=np.float64)
        w[idx] = base[idx]
        risk = np.sqrt(max(float(w @ cov @ w), 1e-12))
        out[idx] = 1.0 / risk
    mean = out[active].mean() if active.any() else 1.0
    return out / mean


def cap_short_positions(pos: np.ndarray, frac: float | None,
                        gross: float | None = None) -> np.ndarray:
    """Cap each short at ``frac`` of a reference gross notional.

    ``gross`` is the reference scale; when omitted it is the NaN-safe gross
    of ``pos`` itself. One pass, not a fixed-point iteration: an all-short
    book (an ordinary state for a trend follower in a bear market) keeps its
    shape instead of collapsing toward zero, and a trimmed short can still
    exceed ``frac`` of the final, smaller gross. Longs are never touched;
    the asymmetry is the point, since a short can lose more than its
    notional while a long cannot.
    """
    if frac is None:
        return pos
    if gross is None:
        gross = float(np.nansum(np.abs(pos)))
    if not np.isfinite(gross) or gross <= 0:
        return pos
    return np.maximum(pos, -frac * gross)


def constrain_book(pos: np.ndarray, gross_cap: float, short_frac: float | None,
                   short_ref_gross: float | None) -> np.ndarray:
    """The hard book constraints, in one place and one order.

    Scale the whole book pro rata down to the gross-leverage cap, then cap
    each short against the day's reference gross. simulate() calls this for
    both the target book and the post-buffer held book, so a constraint
    added here binds both identically instead of relying on two hand-pasted
    blocks staying in sync.
    """
    gross = np.abs(pos).sum()
    if gross > gross_cap and gross > 0:
        pos = pos * (gross_cap / gross)
    return cap_short_positions(pos, short_frac, short_ref_gross)


def effective_breadth(corr: np.ndarray) -> float:
    """Equal-weight diversification breadth implied by a correlation matrix.

    This is ``1 / (w.T @ corr @ w)`` for equal weights summing to one. It is
    the number of independent equal-volatility return streams that would
    produce the same portfolio variance, hence the breadth relevant to the
    square-root Sharpe scaling rule.
    """
    n = corr.shape[0]
    if n == 0:
        return float("nan")
    variance = float(corr.sum()) / n**2
    if variance <= 0:
        return float("nan")
    return float(1.0 / variance)


# --------------------------------------------------------------------------
# 3.7-3.10 simulation
# --------------------------------------------------------------------------

def forward_mark_returns(opens: np.ndarray, closes: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Causal forward returns through gaps and at the backtest endpoint.

    A missing next open is not an executable price. In that case the current
    day's close becomes the observable mark, which is then carried until an
    open resumes. This avoids scanning future rows to decide that a symbol's
    current observation is its last one. Only positions executable on the
    predeclared final row are liquidated at that row's close.
    """
    if opens.shape != closes.shape:
        raise ValueError("opens and closes must have the same shape")
    if opens.ndim != 2:
        raise ValueError("opens and closes must be two-dimensional")

    fwd = np.full_like(opens, np.nan, dtype=np.float64)
    terminal = np.zeros_like(opens, dtype=bool)
    mark = np.full(opens.shape[1], np.nan, dtype=np.float64)
    for t in range(len(opens)):
        current_open = np.isfinite(opens[t])
        mark = np.where(current_open, opens[t], mark)

        if t + 1 < len(opens):
            next_open = np.isfinite(opens[t + 1])
            end_mark = np.where(next_open, opens[t + 1], mark)
            current_close = np.isfinite(closes[t])
            end_mark = np.where(~next_open & current_close, closes[t], end_mark)
        else:
            current_close = current_open & np.isfinite(closes[t])
            end_mark = np.where(current_close, closes[t], mark)
            terminal[t] = current_close

        with np.errstate(divide="ignore", invalid="ignore"):
            fwd[t] = end_mark / mark - 1.0
        mark = end_mark
    return fwd, terminal


@dataclass
class SimulationResult:
    daily: pd.DataFrame
    positions: pd.DataFrame
    diagnostics: pd.DataFrame

    @property
    def returns(self) -> pd.Series:
        return self.daily["net_return"]


def simulate(
    *,
    opens: pd.DataFrame,
    closes: pd.DataFrame | None = None,
    signal: pd.DataFrame,
    annual_vol: pd.DataFrame,
    tradeable: pd.DataFrame,
    funding: pd.DataFrame | None,
    log_returns: pd.DataFrame | None = None,
    config: Config,
) -> SimulationResult:
    """Run the daily book.

    All inputs are date-indexed frames sharing one column set. ``signal`` and
    ``annual_vol`` are already lagged so row t is knowable at 00:00 UTC on t;
    ``opens`` row t is the fill price on t. Missing opens freeze the existing
    position at the latest observable mark until an open resumes. ``closes``
    supplies causal marks when the next open is missing and liquidation prices
    at the global endpoint. ``log_returns`` is the point-in-time return matrix
    used by the correlation model. ``funding`` row t is the total funding rate
    charged to a long between the open of t and the open of t+1.
    """
    config.validate()
    dates = opens.index
    symbols = opens.columns
    n = len(symbols)

    open_arr = opens.to_numpy(dtype=np.float64)
    close_arr = (closes.to_numpy(dtype=np.float64)
                 if closes is not None else open_arr.copy())
    sig_arr = signal.to_numpy(dtype=np.float64)
    vol_arr = annual_vol.to_numpy(dtype=np.float64)
    trade_arr = tradeable.to_numpy(dtype=bool)
    fund_arr = (funding.to_numpy(dtype=np.float64)
                if funding is not None and config.apply_funding
                else np.zeros_like(open_arr))
    fund_arr = np.nan_to_num(fund_arr)

    fwd, terminal = forward_mark_returns(open_arr, close_arr)

    if log_returns is None:
        log_returns = np.log(closes.ffill()).diff()
    log_ret = log_returns.to_numpy(dtype=np.float64)

    resp = np.asarray(response(sig_arr, config.response), dtype=np.float64)

    equity = config.initial_equity
    pos = np.zeros(n)                     # signed notional held from this open
    rows: list[dict] = []
    pos_history = np.zeros((len(dates), n))
    diag: list[dict] = []

    corr = None
    labels = None
    sector = np.ones(n)
    corr_cols: np.ndarray = np.zeros(0, dtype=int)
    last_rebuild = -10**9
    ruined = False

    in_corr = np.zeros(n, dtype=bool)

    for t, date in enumerate(dates):
        if ruined:
            rows.append({
                "date": date, "ruined": True, "equity": 0.0,
                "start_equity": 0.0, "gross_pnl": 0.0,
                "cost": 0.0, "funding": 0.0, "net_pnl": 0.0,
                "gross_return": 0.0, "net_return": 0.0, "traded_notional": 0.0,
                "gross_exposure": 0.0, "net_exposure": 0.0, "long_pnl": 0.0,
                "short_pnl": 0.0, "n_positions": 0, "n_live": 0,
                "multiplier": 0.0, "ex_ante_vol": np.nan,
                "ex_ante_vol_full_signal": np.nan,
            })
            continue

        # positions drift with the market between opens
        if t > 0:
            pos = pos * (1.0 + np.nan_to_num(fwd[t - 1]))

        live = trade_arr[t] & np.isfinite(resp[t]) & np.isfinite(vol_arr[t]) \
            & np.isfinite(open_arr[t])

        # ---- weekly correlation / cluster rebuild (uses data through t-1)
        if t - last_rebuild >= config.corr_rebuild_days and live.sum() >= 5:
            lo = max(0, t - config.corr_window)
            block = log_ret[lo:t]
            cand = np.flatnonzero(live & (np.isfinite(block).sum(axis=0)
                                          >= config.corr_min_obs))
            if len(cand) >= 5:
                corr = shrunk_correlation(
                    block[:, cand], config.corr_min_obs, config.corr_shrinkage)
                labels = cluster_labels(corr, config.n_clusters)
                corr_cols = cand
                in_corr = np.zeros(n, dtype=bool)
                in_corr[cand] = True
                last_rebuild = t
                diag.append({
                    "date": date,
                    "n_corr": len(cand),
                    "mean_corr": float(
                        corr[~np.eye(len(cand), dtype=bool)].mean()),
                    "effective_breadth": effective_breadth(corr),
                    "n_clusters": int(len(np.unique(labels))),
                })

        # only symbols carried by the risk model may take exposure, so the
        # ex-ante volatility below covers every position in the book
        live = live & in_corr

        # ---- unscaled target exposure
        if config.sizing == "equal_notional":
            u = np.where(live, resp[t], 0.0)
        else:
            u = np.where(live, resp[t] / np.maximum(vol_arr[t], 1e-9), 0.0)
        u = np.nan_to_num(u)

        sector = np.ones(n)
        if config.use_sector_weights and corr is not None and labels is not None:
            vols_c = np.nan_to_num(vol_arr[t][corr_cols], nan=1.0)
            sector[corr_cols] = sector_weights(
                corr, vols_c, labels, active=live[corr_cols],
                sizing=config.sizing)
            u = u * sector

        # ---- portfolio risk targeting
        ex_ante = np.nan
        ex_ante_full = np.nan
        if corr is not None and len(corr_cols):
            uc = u[corr_cols]
            vc = np.nan_to_num(vol_arr[t][corr_cols], nan=0.0)
            cov = corr * np.outer(vc, vc)
            ex_ante = float(np.sqrt(max(uc @ cov @ uc, 0.0)))
            # the same book with every live market at full signal strength:
            # dividing by this removes universe-size drift from the risk
            # series, leaving only how much the trend configuration expresses
            full = np.where(live[corr_cols],
                            sector[corr_cols] / np.maximum(vc, 1e-9), 0.0)
            ex_ante_full = float(np.sqrt(max(full @ cov @ full, 0.0)))

        # The multiplier is deliberately unbounded: u is a notional-per-dollar
        # weight whose scale depends on the number of live markets and their
        # volatilities, so any fixed bound on m is dimensionally meaningless.
        # The gross-leverage cap below is the binding constraint.
        if config.use_risk_targeting:
            mult = (config.vol_target_annual / ex_ante
                    if np.isfinite(ex_ante) and ex_ante > 0 else 0.0)
        else:
            mult = config.fixed_multiplier if config.fixed_multiplier else 0.0

        target = mult * u * equity

        cap = config.max_gross_leverage * equity
        # one reference scale per day for the short cap: the leverage-capped
        # target gross. Both constraint applications below use it, so a
        # short that drifted larger overnight is trimmed against the
        # intended book size rather than its own inflated one.
        ref_gross = min(float(np.abs(target).sum()), cap)
        target = constrain_book(target, cap, config.max_short_frac_gross,
                                ref_gross)

        # ---- buffering: trade only outside the no-trade band
        if config.buffer > 0:
            tol = config.buffer * mult * equity / np.maximum(vol_arr[t], 1e-9)
            tol = np.nan_to_num(tol)
            if config.sizing == "equal_notional":
                tol = np.full(n, config.buffer * mult * equity)
            gap = target - pos
            move = np.where(
                np.abs(gap) > tol, gap - np.sign(gap) * tol, 0.0)
            new_pos = pos + move
        else:
            new_pos = target.copy()

        # A symbol that is no longer eligible is flattened when an execution
        # price exists. With no current open, its position must remain frozen.
        can_execute = np.isfinite(open_arr[t])
        new_pos = np.where(live | (pos == 0.0), new_pos, 0.0)
        new_pos = np.where(can_execute, new_pos, pos)

        # both constraints are margin/survival rules, so they override the
        # no-trade band: a short pinned at the cap is trimmed daily even
        # when the buffer would not have traded it, by design
        new_pos = constrain_book(new_pos, cap, config.max_short_frac_gross,
                                 ref_gross)
        new_pos = np.where(can_execute, new_pos, pos)

        traded = np.abs(new_pos - pos).sum()
        close_notional = np.where(
            terminal[t], np.abs(new_pos * (1.0 + fwd[t])), 0.0)
        traded += close_notional.sum()
        cost = traded * config.cost_bps / 10_000.0

        r = np.nan_to_num(fwd[t])
        held = np.where(np.isfinite(fwd[t]), new_pos, 0.0)
        tracking_error = float(np.abs(new_pos - target).sum() / equity)
        held_ex_ante = np.nan
        if corr is not None and len(corr_cols):
            held_w = new_pos[corr_cols] / equity
            held_ex_ante = float(np.sqrt(max(held_w @ cov @ held_w, 0.0)))
        pnl = float((held * r).sum())
        funding_cost = float((held * fund_arr[t]).sum())

        net_pnl = pnl - cost - funding_cost
        start_equity = equity
        equity = equity + net_pnl
        if equity <= 0:
            # An account that reaches zero is ruined, not floored: it stops
            # trading and every later day is reported flat at zero equity.
            ruined = True
            equity = 0.0
            new_pos = np.zeros(n)

        recorded_pos = new_pos.copy()
        carry_pos = np.where(terminal[t], 0.0, new_pos)
        pos_history[t] = recorded_pos
        rows.append({
            "date": date,
            "ruined": ruined,
            "equity": equity,
            "start_equity": start_equity,
            "gross_pnl": pnl,
            "cost": cost,
            "funding": funding_cost,
            "net_pnl": net_pnl,
            "gross_return": pnl / start_equity,
            "net_return": net_pnl / start_equity,
            "traded_notional": traded,
            "gross_exposure": float(np.abs(new_pos).sum()),
            "net_exposure": float(new_pos.sum()),
            "long_pnl": float((np.where(held > 0, held, 0.0) * r).sum()),
            "short_pnl": float((np.where(held < 0, held, 0.0) * r).sum()),
            "n_positions": int((np.abs(new_pos) > 0).sum()),
            "n_live": int(live.sum()),
            "multiplier": float(mult),
            "ex_ante_vol": ex_ante,
            "ex_ante_vol_full_signal": ex_ante_full,
            "held_ex_ante_vol": held_ex_ante,
            "target_tracking_error": tracking_error,
        })
        pos = carry_pos

    daily = pd.DataFrame(rows).set_index("date")
    positions = pd.DataFrame(pos_history, index=dates, columns=symbols)
    diagnostics = (pd.DataFrame(diag).set_index("date")
                   if diag else pd.DataFrame())
    return SimulationResult(daily, positions, diagnostics)


# --------------------------------------------------------------------------
# performance statistics
# --------------------------------------------------------------------------

def performance_summary(daily: pd.DataFrame, label: str = "") -> dict:
    net = daily["net_return"].to_numpy(dtype=np.float64)
    gross = daily["gross_return"].to_numpy(dtype=np.float64)
    equity = daily["equity"].to_numpy(dtype=np.float64)
    n = len(net)
    if n == 0:
        return {"label": label, "days": 0}

    first_start = float(daily["start_equity"].iloc[0])
    peak = np.maximum.accumulate(np.r_[first_start, equity])[1:]
    dd = np.where(peak > 0, equity / np.maximum(peak, 1e-12) - 1.0, 0.0)
    years = n / 365.0
    initial = equity[0] / (1.0 + net[0]) if net[0] > -1 else equity[0]
    total = equity[-1] / initial - 1.0 if initial > 0 else float("nan")
    ruined = bool(daily["ruined"].iloc[-1]) if "ruined" in daily else False
    ruin_date = None
    if ruined:
        first = daily.index[daily["ruined"].to_numpy().argmax()]
        ruin_date = str(pd.Timestamp(first).date())

    def sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=1)
        return float(x.mean() / sd * ANNUALIZE) if sd > 0 else float("nan")

    live_equity = equity[equity > 0]
    avg_equity = float(live_equity.mean()) if len(live_equity) else 1.0
    return {
        "label": label,
        "days": n,
        "start": str(daily.index[0].date()),
        "end": str(daily.index[-1].date()),
        "ruined": ruined,
        "ruin_date": ruin_date,
        "total_return": float(total),
        "cagr": float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 else -1.0,
        "ann_vol": float(net.std(ddof=1) * ANNUALIZE),
        "sharpe_net": sharpe(net),
        "sharpe_gross": sharpe(gross),
        "max_drawdown": float(dd.min()),
        "hit_rate": float((net > 0).mean()),
        "ann_turnover": float(daily["traded_notional"].sum() / avg_equity / years),
        "avg_gross_exposure": float(
            (daily["gross_exposure"]
             / daily["start_equity"].replace(0.0, np.nan)).mean(skipna=True)),
        "cost_drag_ann": float(daily["cost"].sum() / avg_equity / years),
        "funding_drag_ann": float(daily["funding"].sum() / avg_equity / years),
        "long_pnl": float(daily["long_pnl"].sum()),
        "short_pnl": float(daily["short_pnl"].sum()),
        "avg_positions": float(daily["n_positions"].mean()),
        "avg_live": float(daily["n_live"].mean()),
        "avg_held_ex_ante_vol": float(
            daily.get("held_ex_ante_vol", pd.Series(dtype=float)).mean()),
        "avg_target_tracking_error": float(
            daily.get("target_tracking_error", pd.Series(dtype=float)).mean()),
    }


def sharpe_ratio(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2:
        return float("nan")
    sd = returns.std(ddof=1)
    return float(returns.mean() / sd * ANNUALIZE) if sd > 0 else float("nan")


def _stationary_indices(n: int, mean_block: int, n_resamples: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary-bootstrap indices, built without a loop.

    A new block starts with probability 1/mean_block, and every day inside a
    block is one step past the day that started it.
    """
    p = 1.0 / mean_block
    positions = np.arange(n)
    restart = rng.random((n_resamples, n)) < p
    restart[:, 0] = True
    block_start = np.maximum.accumulate(np.where(restart, positions, -1), axis=1)
    offset = positions - block_start
    origins = rng.integers(n, size=(n_resamples, n))
    return (np.take_along_axis(origins, block_start, axis=1) + offset) % n


def _sharpe_rows(sample: np.ndarray) -> np.ndarray:
    mean = sample.mean(axis=1)
    sd = sample.std(axis=1, ddof=1)
    return np.where(sd > 0, mean / np.where(sd > 0, sd, 1.0) * ANNUALIZE, np.nan)


def stationary_bootstrap_ci(
    diff: np.ndarray,
    mean_block: int = 21,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Confidence interval for the annualized Sharpe of one return series."""
    diff = np.asarray(diff, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = _stationary_indices(n, mean_block, n_resamples, rng)
    stats = _sharpe_rows(diff[idx])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def stationary_bootstrap_sharpe_diff_ci(
    variant_returns: np.ndarray,
    base_returns: np.ndarray,
    mean_block: int = 21,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Confidence interval for Sharpe(variant) - Sharpe(base).

    The two series are resampled with the *same* block indices, because the
    variants share market paths and only their differences are informative.
    """
    a = np.asarray(variant_returns, dtype=np.float64)
    b = np.asarray(base_returns, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = len(a)
    if n < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = _stationary_indices(n, mean_block, n_resamples, rng)
    stats = _sharpe_rows(a[idx]) - _sharpe_rows(b[idx])
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def variant(config: Config, **kwargs) -> Config:
    return replace(config, **kwargs)
