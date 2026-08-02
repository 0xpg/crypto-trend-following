"""Unit tests for the trend-following engine.

These run entirely on synthetic panels, so `make test` needs no archive.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from engine import (  # noqa: E402
    ANNUALIZE,
    Config,
    cap_short_positions,
    cluster_labels,
    effective_breadth,
    forward_mark_returns,
    performance_summary,
    prepare_inputs,
    response,
    sector_weights,
    shrunk_correlation,
    simulate,
    trend_signal,
    variant,
    volatility_forecast,
)
from scripts.claims import claim_c2, single_market_returns  # noqa: E402


def synthetic_panels(n_days=900, n_symbols=6, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    cols = [f"S{i}USDT" for i in range(n_symbols)]
    shocks = rng.normal(drift, 0.03, size=(n_days, n_symbols))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    close = pd.DataFrame(prices, index=dates, columns=cols)
    opens = close.shift(1).fillna(100.0)
    return {
        "open": opens,
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "quote_volume": pd.DataFrame(1e9, index=dates, columns=cols),
        "hours": pd.DataFrame(24.0, index=dates, columns=cols),
        "funding": pd.DataFrame(0.0, index=dates, columns=cols),
    }


def run(panels, config):
    inputs = prepare_inputs(panels, config)
    return simulate(
        opens=inputs.opens, closes=inputs.closes,
        signal=inputs.signal, annual_vol=inputs.annual_vol,
        tradeable=inputs.tradeable, funding=inputs.funding,
        log_returns=inputs.log_returns, config=config)


class TestResponse(unittest.TestCase):
    def test_saturation(self):
        z = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        t = response(z, "tanh")
        self.assertAlmostEqual(t[0], -1.0, places=6)
        self.assertAlmostEqual(t[4], 1.0, places=6)
        self.assertAlmostEqual(t[2], 0.0)

    def test_overextension_peaks_at_one(self):
        self.assertAlmostEqual(
            float(response(np.array([np.sqrt(2.0)]), "overextension")[0]),
            1.0, places=12)
        z = np.linspace(-6, 6, 200001)
        f = response(z, "overextension")
        self.assertLessEqual(float(f.max()), 1.0 + 1e-12)
        self.assertAlmostEqual(float(z[f.argmax()]), np.sqrt(2.0), places=3)
        # the defining property: large signals get smaller positions
        self.assertLess(response(np.array([5.0]), "overextension")[0],
                        response(np.array([np.sqrt(2)]), "overextension")[0])

    def test_linear_and_binary(self):
        z = np.array([-3.0, -0.5, 0.5, 3.0])
        np.testing.assert_allclose(response(z, "linear"), [-1, -0.5, 0.5, 1])
        np.testing.assert_allclose(response(z, "binary"), [-1, -1, 1, 1])

    def test_unknown_response_rejected(self):
        with self.assertRaises(ValueError):
            response(np.zeros(3), "quadratic")


class TestVolatility(unittest.TestCase):
    def test_floor_and_annualization(self):
        dates = pd.date_range("2020-01-01", periods=400, freq="D")
        rets = pd.DataFrame(0.0, index=dates, columns=["A"])
        rets.iloc[:, 0] = 0.01 * np.sin(np.arange(400))
        cfg = Config(vol_floor_annual=0.20)
        vol = volatility_forecast(rets, cfg)
        self.assertTrue((vol.dropna() >= 0.20 - 1e-12).all().all())

    def test_blend_weights(self):
        dates = pd.date_range("2020-01-01", periods=800, freq="D")
        rng = np.random.default_rng(1)
        rets = pd.DataFrame(rng.normal(0, 0.05, (800, 1)), index=dates,
                            columns=["A"])
        cfg = Config(vol_floor_annual=1e-6)
        blend = volatility_forecast(rets, cfg).iloc[-1, 0]
        short = volatility_forecast(variant(cfg, vol_forecast="short"),
                                    ) if False else None
        s = volatility_forecast(rets, variant(cfg, vol_forecast="short")).iloc[-1, 0]
        lg = volatility_forecast(rets, variant(cfg, vol_forecast="long")).iloc[-1, 0]
        self.assertAlmostEqual(blend, 0.70 * s + 0.30 * lg, places=10)

    def test_gap_return_is_included_when_complete_bars_resume(self):
        panels = synthetic_panels(n_days=700, seed=2)
        sym = panels["close"].columns[0]
        gap = 500
        prior = float(panels["close"].iloc[gap - 1, 0])
        panels["close"].iloc[gap + 1, 0] = prior * 0.25
        for key in ("open", "close", "high", "low", "quote_volume", "hours"):
            panels[key].iloc[gap, 0] = np.nan

        inputs = prepare_inputs(panels, Config())
        expected = np.log(float(panels["close"].iloc[gap + 1, 0]) / prior)
        resumed = inputs.log_returns.index[gap + 1]
        self.assertAlmostEqual(
            float(inputs.log_returns.at[resumed, sym]), expected, places=12)


class TestSignalTiming(unittest.TestCase):
    def test_no_lookahead_in_signal(self):
        panels = synthetic_panels(n_days=700, seed=3)
        cfg = Config()
        base = prepare_inputs(panels, cfg).signal
        cut = 500
        tampered = {k: v.copy() for k, v in panels.items()}
        tampered["close"].iloc[cut:] *= 3.0
        after = prepare_inputs(tampered, cfg).signal
        pd.testing.assert_frame_equal(
            base.iloc[:cut + 1], after.iloc[:cut + 1], check_exact=False,
            atol=1e-12)

    def test_signal_follows_trend_direction(self):
        dates = pd.date_range("2020-01-01", periods=800, freq="D")
        up = pd.DataFrame(
            100 * np.exp(np.linspace(0, 1.5, 800))[:, None] +
            np.zeros((800, 1)), index=dates, columns=["A"])
        vol = pd.DataFrame(0.03, index=dates, columns=["A"])
        z = trend_signal(up, vol, Config())
        self.assertGreater(z.iloc[-1, 0], 0.0)
        z_down = trend_signal(1.0 / up, vol, Config())
        self.assertLess(z_down.iloc[-1, 0], 0.0)


class TestCorrelation(unittest.TestCase):
    def test_recovers_known_correlation(self):
        rng = np.random.default_rng(5)
        f = rng.normal(size=(4000, 1))
        x = np.hstack([f + rng.normal(size=(4000, 1)),
                       f + rng.normal(size=(4000, 1))])
        corr = shrunk_correlation(x, min_obs=100, shrinkage=0.0)
        self.assertAlmostEqual(corr[0, 1], 0.5, delta=0.05)
        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_shrinkage_pulls_toward_mean(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(500, 8))
        raw = shrunk_correlation(x, 100, 0.0)
        shrunk = shrunk_correlation(x, 100, 1.0)
        off = ~np.eye(8, dtype=bool)
        self.assertLess(shrunk[off].std(), raw[off].std())

    def test_missing_data_tolerated(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(400, 5))
        x[:200, 4] = np.nan
        corr = shrunk_correlation(x, min_obs=100, shrinkage=0.1)
        self.assertTrue(np.isfinite(corr).all())
        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_pairwise_complete_matches_shared_observations(self):
        x = np.array([
            [0.0, 0.0], [1.0, np.nan], [2.0, 2.0], [np.nan, 3.0],
        ])
        corr = shrunk_correlation(x, min_obs=2, shrinkage=0.0)
        expected = pd.DataFrame(x).corr(min_periods=2).iloc[0, 1]
        self.assertAlmostEqual(corr[0, 1], expected, places=9)

    def test_output_is_positive_semidefinite(self):
        rng = np.random.default_rng(70)
        x = rng.normal(size=(180, 40))
        x[rng.random(x.shape) < 0.25] = np.nan
        corr = shrunk_correlation(x, min_obs=80, shrinkage=0.3)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(corr).min()), -1e-9)

    def test_effective_breadth_bounds(self):
        n = 10
        self.assertAlmostEqual(effective_breadth(np.eye(n)), n, places=6)
        allone = np.full((n, n), 0.999999)
        np.fill_diagonal(allone, 1.0)
        self.assertLess(effective_breadth(allone), 1.05)

    def test_clusters_separate_blocks(self):
        block = np.full((4, 4), 0.9)
        np.fill_diagonal(block, 1.0)
        corr = np.full((8, 8), 0.05)
        corr[:4, :4] = block
        corr[4:, 4:] = block
        np.fill_diagonal(corr, 1.0)
        labels = cluster_labels(corr, 2)
        self.assertEqual(len(set(labels[:4])), 1)
        self.assertEqual(len(set(labels[4:])), 1)
        self.assertNotEqual(labels[0], labels[4])


class TestSectorWeights(unittest.TestCase):
    def test_equalizes_presector_cluster_risk(self):
        corr = np.eye(6)
        vols = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
        labels = np.array([1, 1, 1, 2, 2, 2])
        g = sector_weights(corr, vols, labels)
        self.assertAlmostEqual(float(g.mean()), 1.0, places=10)
        cov = corr * np.outer(vols, vols)
        base = g / vols
        risks = []
        for lab in np.unique(labels):
            w = np.where(labels == lab, base, 0.0)
            risks.append(np.sqrt(w @ cov @ w))
        np.testing.assert_allclose(risks, risks[0], rtol=1e-10)

    def test_cluster_size_does_not_change_allocated_risk(self):
        corr = np.eye(5)
        vols = np.full(5, 0.2)
        labels = np.array([1, 2, 2, 2, 2])
        g = sector_weights(corr, vols, labels)
        cov = corr * np.outer(vols, vols)
        base = g / vols
        one = np.where(labels == 1, base, 0.0)
        four = np.where(labels == 2, base, 0.0)
        self.assertAlmostEqual(
            float(np.sqrt(one @ cov @ one)),
            float(np.sqrt(four @ cov @ four)), places=10)


class TestSimulation(unittest.TestCase):
    def test_drawdown_includes_initial_equity(self):
        idx = pd.date_range("2020-01-01", periods=2, freq="D")
        daily = pd.DataFrame({
            "net_return": [-0.10, 0.0], "gross_return": [-0.10, 0.0],
            "equity": [90.0, 90.0], "start_equity": [100.0, 90.0],
            "ruined": [False, False], "traded_notional": [0.0, 0.0],
            "gross_exposure": [0.0, 0.0], "cost": [0.0, 0.0],
            "funding": [0.0, 0.0], "long_pnl": [0.0, 0.0],
            "short_pnl": [0.0, 0.0], "n_positions": [0, 0],
            "n_live": [0, 0],
        }, index=idx)
        summary = performance_summary(daily)
        self.assertAlmostEqual(summary["max_drawdown"], -0.10, places=12)

    def test_no_lookahead_in_pnl(self):
        panels = synthetic_panels(n_days=800, seed=11)
        cfg = Config()
        base = run(panels, cfg).daily
        cut = 600
        tampered = {k: v.copy() for k, v in panels.items()}
        for key in ("open", "close", "high", "low"):
            tampered[key].iloc[cut:] *= 5.0
        after = run(tampered, cfg).daily
        np.testing.assert_allclose(
            base["net_pnl"].to_numpy()[:cut - 1],
            after["net_pnl"].to_numpy()[:cut - 1], atol=1e-8)

    def test_no_lookahead_in_any_reported_column(self):
        """Tamper with every input after a cutoff; nothing before it may move.

        Stronger than the P&L check: it covers positions, exposure, the risk
        multiplier, the ex-ante volatility, the universe filter and funding,
        so a leak through any one of them fails here.
        """
        panels = synthetic_panels(n_days=900, n_symbols=8, seed=31)
        cfg = Config()
        base = run(panels, cfg)
        cut = 700
        tampered = {k: v.copy() for k, v in panels.items()}
        for key in ("open", "close", "high", "low"):
            tampered[key].iloc[cut:] *= 7.0
        tampered["quote_volume"].iloc[cut:] = 1.0
        tampered["hours"].iloc[cut:] = 3.0
        tampered["funding"].iloc[cut:] = 0.05
        after = run(tampered, cfg)

        # row t is decided at the open of t and settled with the open of t+1,
        # so the last untouched decision is t = cut - 2
        end = cut - 2
        pd.testing.assert_frame_equal(
            base.daily.iloc[:end], after.daily.iloc[:end],
            check_exact=False, atol=1e-9)
        pd.testing.assert_frame_equal(
            base.positions.iloc[:end], after.positions.iloc[:end],
            check_exact=False, atol=1e-9)

    def test_zero_buffer_hits_target_exactly(self):
        panels = synthetic_panels(n_days=700, seed=12)
        res = run(panels, Config(buffer=0.0, max_gross_leverage=1e9))
        pos = res.positions
        last = pos.index[-1]
        gross = float(pos.loc[last].abs().sum())
        self.assertGreater(gross, 0.0)
        active = res.daily["n_positions"] > 0
        self.assertAlmostEqual(
            float(res.daily.loc[active, "target_tracking_error"].max()),
            0.0, places=12)

    def test_buffer_monotonically_reduces_turnover(self):
        panels = synthetic_panels(n_days=800, seed=13)
        turnovers = []
        for b in (0.0, 0.1, 0.4, 1.0):
            daily = run(panels, Config(buffer=b)).daily
            turnovers.append(daily["traded_notional"].sum())
        self.assertTrue(all(a >= b for a, b in zip(turnovers, turnovers[1:])),
                        f"turnover not monotone in buffer: {turnovers}")

    def test_costs_only_reduce_pnl(self):
        panels = synthetic_panels(n_days=700, seed=14)
        free = run(panels, Config(cost_bps=0.0)).daily["net_pnl"].sum()
        paid = run(panels, Config(cost_bps=25.0)).daily["net_pnl"].sum()
        self.assertLess(paid, free)

    def test_funding_sign_charges_longs(self):
        panels = synthetic_panels(n_days=700, seed=15, drift=0.004)
        cfg = Config()
        without = run(panels, variant(cfg, apply_funding=False)).daily
        charged = {k: v.copy() for k, v in panels.items()}
        charged["funding"] = pd.DataFrame(
            0.001, index=panels["open"].index, columns=panels["open"].columns)
        with_funding = run(charged, cfg).daily
        # a trend book on rising prices is net long, so a positive funding
        # rate must cost it money
        self.assertGreater(without["net_exposure"].mean(), 0.0)
        self.assertGreater(with_funding["funding"].sum(), 0.0)
        self.assertLess(with_funding["net_pnl"].sum(), without["net_pnl"].sum())

    def test_risk_targeting_tracks_the_target(self):
        panels = synthetic_panels(n_days=1200, n_symbols=12, seed=16)
        cfg = Config(vol_target_annual=0.20, buffer=0.0,
                     max_gross_leverage=1e9)
        daily = run(panels, cfg).daily
        active = daily[daily["n_positions"] > 0]
        realized = active["net_return"].std(ddof=1) * ANNUALIZE
        self.assertGreater(realized, 0.08)
        self.assertLess(realized, 0.45)
        ex_ante = active["held_ex_ante_vol"].dropna()
        np.testing.assert_allclose(ex_ante, 0.20, rtol=1e-9, atol=1e-9)

    def test_correlation_model_uses_prepared_gap_aware_returns(self):
        panels = synthetic_panels(n_days=700, n_symbols=6, seed=159)
        cfg = Config(corr_rebuild_days=1000, corr_shrinkage=0.0)
        inputs = prepare_inputs(panels, cfg)
        common = np.sin(np.arange(len(inputs.opens), dtype=float))
        risk_returns = pd.DataFrame(
            np.repeat(common[:, None], inputs.opens.shape[1], axis=1),
            index=inputs.opens.index, columns=inputs.opens.columns)
        result = simulate(
            opens=inputs.opens, closes=inputs.closes,
            signal=inputs.signal, annual_vol=inputs.annual_vol,
            tradeable=inputs.tradeable, funding=inputs.funding,
            log_returns=risk_returns, config=cfg)
        self.assertGreater(float(result.diagnostics["mean_corr"].iloc[0]), 0.999)

    def test_archive_end_does_not_trigger_hindsight_liquidation(self):
        panels = synthetic_panels(n_days=800, n_symbols=6, seed=160)
        cut = 700
        last_open = panels["open"].iloc[cut - 1].copy()
        last_close = panels["close"].iloc[cut - 1].copy()
        for key in ("open", "close", "high", "low", "quote_volume", "hours"):
            panels[key].iloc[cut:] = np.nan
        panels["funding"].iloc[cut:] = np.nan
        result = run(panels, Config(buffer=0.0))
        terminal = result.daily.index[cut - 1]
        after = result.daily.index[cut]
        self.assertGreater(float(result.positions.loc[terminal].abs().sum()), 0.0)
        self.assertEqual(float(result.daily.at[after, "traded_notional"]), 0.0)
        expected = result.positions.loc[terminal] * (last_close / last_open)
        np.testing.assert_allclose(result.positions.loc[after], expected)
        self.assertGreater(float(result.daily.at[after, "gross_exposure"]), 0.0)

    def test_internal_open_gap_freezes_position_until_mark_resumes(self):
        panels = synthetic_panels(n_days=800, n_symbols=6, seed=161)
        gap = 700
        prior_opens = panels["open"].iloc[gap - 1].copy()
        resumed_opens = panels["open"].iloc[gap + 1].copy()
        for key in ("open", "close", "high", "low", "quote_volume", "hours"):
            panels[key].iloc[gap] = np.nan
        panels["funding"].iloc[gap] = np.nan

        result = run(panels, Config(buffer=0.0))
        gap_date = result.daily.index[gap]
        prior_date = result.daily.index[gap - 1]
        resumed_date = result.daily.index[gap + 1]

        prior_position = result.positions.loc[prior_date]
        prior_closes = panels["close"].iloc[gap - 1]
        marked_position = prior_position * prior_closes / prior_opens
        np.testing.assert_allclose(result.positions.loc[gap_date], marked_position)
        self.assertEqual(float(result.daily.at[gap_date, "traded_notional"]), 0.0)
        expected = float((prior_position
                          * (resumed_opens / prior_opens - 1.0)).sum())
        self.assertAlmostEqual(
            float(result.daily.loc[[prior_date, gap_date], "gross_pnl"].sum()),
            expected, places=8)
        self.assertGreater(float(result.daily.at[resumed_date, "traded_notional"]),
                           0.0)

    def test_forward_marks_are_causal_through_gaps(self):
        opens = np.array([[100.0], [np.nan], [110.0], [120.0], [np.nan]])
        closes = np.array([[101.0], [np.nan], [111.0], [126.0], [np.nan]])
        fwd, terminal = forward_mark_returns(opens, closes)
        np.testing.assert_allclose(
            fwd[:, 0], [0.01, 110.0 / 101.0 - 1.0,
                        120.0 / 110.0 - 1.0, 0.05, 0.0])
        self.assertFalse(bool(terminal.any()))

        # A later resumption must not retroactively turn the last observed
        # close into an earlier liquidation.
        dormant = np.array([[100.0], [np.nan], [np.nan], [np.nan]])
        resumed = dormant.copy()
        resumed[3, 0] = 120.0
        gap_closes = np.array([[105.0], [np.nan], [np.nan], [np.nan]])
        base_fwd, base_terminal = forward_mark_returns(dormant, gap_closes)
        extended_fwd, extended_terminal = forward_mark_returns(resumed, gap_closes)
        np.testing.assert_allclose(base_fwd[:2], extended_fwd[:2])
        self.assertFalse(bool(base_terminal[0, 0]))
        self.assertFalse(bool(extended_terminal[0, 0]))

    def test_only_global_endpoint_is_liquidated(self):
        opens = np.array([[100.0], [110.0]])
        closes = np.array([[105.0], [115.0]])
        fwd, terminal = forward_mark_returns(opens, closes)
        np.testing.assert_allclose(fwd[:, 0], [0.10, 115.0 / 110.0 - 1.0])
        np.testing.assert_array_equal(terminal[:, 0], [False, True])

    def test_untargeted_risk_is_more_dispersed(self):
        panels = synthetic_panels(n_days=1200, n_symbols=12, seed=17)
        targeted = run(panels, Config(buffer=0.0)).daily
        flat = run(panels, Config(buffer=0.0, use_risk_targeting=False,
                                  fixed_multiplier=0.01)).daily
        a = targeted["gross_exposure"] / targeted["start_equity"]
        b = flat["gross_exposure"] / flat["start_equity"]
        self.assertLess(a[a > 0].std() / a[a > 0].mean(),
                        b[b > 0].std() / b[b > 0].mean())

    def test_gross_leverage_cap_binds(self):
        panels = synthetic_panels(n_days=900, n_symbols=10, seed=18)
        cfg = Config(max_gross_leverage=0.25, buffer=0.0)
        daily = run(panels, cfg).daily
        ratio = (daily["gross_exposure"] / daily["start_equity"]).max()
        self.assertLessEqual(ratio, 0.25 + 1e-6)

    def test_short_cap_semantics(self):
        pos = np.array([100.0, -50.0, -10.0, 30.0])
        capped = cap_short_positions(pos, 0.10)  # gross 190, cap 19
        np.testing.assert_allclose(capped, [100.0, -19.0, -10.0, 30.0])
        # longs and small shorts untouched, None is a no-op
        np.testing.assert_allclose(cap_short_positions(pos, None), pos)
        # an explicit reference gross overrides the book's own
        np.testing.assert_allclose(
            cap_short_positions(pos, 0.10, gross=100.0),
            [100.0, -10.0, -10.0, 30.0])
        # an all-short book keeps its shape (one pass against pre-cap gross)
        allshort = np.array([-10.0, -10.0])
        np.testing.assert_allclose(
            cap_short_positions(allshort, 0.05), [-1.0, -1.0])
        # a NaN element must not poison the healthy positions
        with_nan = np.array([100.0, -50.0, np.nan])
        out = cap_short_positions(with_nan, 0.10)  # NaN-safe gross 150
        np.testing.assert_allclose(out[:2], [100.0, -15.0])
        self.assertTrue(np.isnan(out[2]))

    def test_short_cap_binds_in_simulation(self):
        # three rising and three falling markets so the book is long/short
        frac = 0.02
        panels = synthetic_panels(
            n_days=900, n_symbols=6, seed=7,
            drift=np.array([0.002, 0.002, 0.002, -0.002, -0.002, -0.002]))
        base = run(panels, Config())
        capped = run(panels, variant(Config(), max_short_frac_gross=frac))
        worst_base = base.positions.to_numpy().min()
        worst_capped = capped.positions.to_numpy().min()
        self.assertLess(worst_base, 0.0)          # shorts exist to cap
        self.assertGreater(worst_capped, worst_base)  # cap binds
        # longs survive the cap
        self.assertGreater(capped.positions.to_numpy().max(), 0.0)
        # and the cap level is right, not just present: no short may exceed
        # frac of the day's reference gross. The reported held gross can sit
        # a little below that reference (buffer slack, one-pass trim), so
        # the bound carries a 25% allowance; a regression that made 2%
        # behave like 20% still fails it by an order of magnitude.
        daily_gross = capped.daily["gross_exposure"].to_numpy()
        worst_by_day = capped.positions.to_numpy().min(axis=1)
        ok = worst_by_day >= -frac * 1.25 * daily_gross - 1e-6
        self.assertTrue(ok.all())

    def test_untradeable_symbols_hold_no_position(self):
        panels = synthetic_panels(n_days=800, seed=19)
        panels["quote_volume"].iloc[:, 0] = 1.0  # far below the floor
        res = run(panels, Config())
        self.assertAlmostEqual(float(res.positions.iloc[:, 0].abs().sum()), 0.0)

    def test_ruin_is_recorded_not_floored(self):
        panels = synthetic_panels(n_days=700, seed=20)
        # a crushing per-unit cost drains the account
        daily = run(panels, Config(cost_bps=500_000.0)).daily
        self.assertTrue(bool(daily["ruined"].iloc[-1]))
        self.assertEqual(float(daily["equity"].iloc[-1]), 0.0)
        summary = performance_summary(daily, "ruined")
        self.assertTrue(summary["ruined"])
        self.assertIsNotNone(summary["ruin_date"])


class TestUniverseFilter(unittest.TestCase):
    def test_history_and_volume_gates(self):
        panels = synthetic_panels(n_days=600, seed=21)
        cfg = Config(min_history_days=200, min_median_quote_volume=1e6)
        inputs = prepare_inputs(panels, cfg)
        first = inputs.tradeable.any(axis=1).idxmax()
        self.assertGreaterEqual(
            (first - inputs.tradeable.index[0]).days, 200)

    def test_incomplete_days_are_untradeable(self):
        panels = synthetic_panels(n_days=600, seed=22)
        panels["hours"].iloc[400:, 0] = 5.0
        inputs = prepare_inputs(panels, Config())
        self.assertFalse(bool(inputs.tradeable.iloc[420:, 0].any()))


class TestClaimStreams(unittest.TestCase):
    def test_breadth_diagnostic_is_explicitly_gross(self):
        index = pd.date_range("2020-01-01", periods=30, freq="D")
        wave = np.sin(np.arange(len(index))) * 0.001
        streams = pd.DataFrame({"A": wave + 0.0002, "B": -wave}, index=index)
        availability = pd.DataFrame(True, index=index, columns=streams.columns)
        summary, table = claim_c2(streams, availability, draws=2)
        self.assertEqual(summary["return_basis"], "gross")
        self.assertEqual(set(table["return_basis"]), {"gross"})

    def test_cash_days_are_retained_inside_account_lifetime(self):
        panels = synthetic_panels(n_days=900, seed=23)
        panels["quote_volume"].iloc[700:800, 0] = 1.0
        stream = single_market_returns(panels, Config(), net=True).iloc[:, 0]
        # Once the standalone account exists, a temporary liquidity failure
        # is a zero-return cash interval, not a set of observations removed
        # before annualizing Sharpe.
        cash = stream.iloc[760:790]
        self.assertTrue(cash.notna().all())
        self.assertAlmostEqual(float(cash.abs().sum()), 0.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
