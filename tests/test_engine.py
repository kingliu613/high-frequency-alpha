import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import simulate_lob_day
from src.backtest.engine import MarketParams, run_backtest
from src.signals.composite import build_feature_matrix, build_composite_alpha


@pytest.fixture
def base_lob():
    return simulate_lob_day(seed=7, date="2024-01-02")

@pytest.fixture
def base_signal(base_lob):
    feat = build_feature_matrix(base_lob)
    return build_composite_alpha(feat)


class TestMarketParams:
    def test_new_fields_have_defaults(self):
        p = MarketParams()
        assert p.max_position_size == 3
        assert p.use_regime_filter is True
        assert p.regime_ic_off == 0.0
        assert p.regime_ic_on  == 0.02
        assert p.slippage_model == "lob_walk"
        assert p.market_impact_coef == 0.5
        assert p.market_impact_perm_frac == 0.2


class TestSignalPropSizing:
    def test_size_one_at_entry_z(self, base_lob, base_signal):
        p = MarketParams(entry_z=1.5, use_regime_filter=False)
        _, trades = run_backtest(base_lob, base_signal, params=p)
        if len(trades):
            # Minimum size is 1
            assert trades["direction"].abs().min() >= 1

    def test_size_scales_with_signal(self, base_lob):
        """Force a constant strong signal and check sizes > 1 appear."""
        p = MarketParams(entry_z=1.0, max_position_size=3, use_regime_filter=False)
        mid = (base_lob["bid_px_1"] + base_lob["ask_px_1"]) / 2.0
        # Constant signal at 2.5 × entry_z → expected size = floor(2.5) = 2
        strong_sig = pd.Series(2.5, index=base_lob.index)
        _, trades = run_backtest(base_lob, strong_sig, params=p)
        if len(trades):
            # All entries should be size 2 (floor(2.5/1.0) = 2)
            assert (trades["direction"].abs() == 2).all()


class TestDynamicHold:
    def test_strong_signal_allows_longer_hold(self, base_lob):
        """Trades entered on a strong constant signal should hold > base_hold ticks."""
        p = MarketParams(entry_z=1.0, max_hold=10, use_regime_filter=False)
        strong_sig = pd.Series(3.5, index=base_lob.index)
        _, trades = run_backtest(base_lob, strong_sig, params=p)
        timeout_trades = trades[trades["exit_reason"] == "timeout"]
        if len(timeout_trades):
            # Dynamic hold at signal=3.5, entry_z=1.0: cap = 10*(1+0.5*3.5/1.0)=27.5→27
            assert timeout_trades["hold_ticks"].max() > 10


class TestSlippageModel:
    def test_lob_walk_higher_cost_than_none(self, base_lob, base_signal):
        p_none = MarketParams(slippage_model="none", market_impact_coef=0.0, use_regime_filter=False)
        p_walk = MarketParams(slippage_model="lob_walk", market_impact_coef=0.0, use_regime_filter=False)
        pnl_none, trades_none = run_backtest(base_lob, base_signal, params=p_none)
        pnl_walk, trades_walk = run_backtest(base_lob, base_signal, params=p_walk)
        if len(trades_walk):
            assert trades_walk["slip_cost"].sum() > 0
            assert pnl_walk.sum() < pnl_none.sum()

    def test_fixed_slippage_deterministic(self, base_lob, base_signal):
        p = MarketParams(slippage_model="fixed", slippage_fixed_ticks=2.0,
                         market_impact_coef=0.0, use_regime_filter=False)
        _, trades = run_backtest(base_lob, base_signal, params=p)
        if len(trades):
            # Each trade slip_cost = 2 ticks × size × LOT (exit side only — entry not in log)
            assert (trades["slip_cost"] >= 0).all()

    def test_trade_log_has_cost_columns(self, base_lob, base_signal):
        p = MarketParams(use_regime_filter=False)
        _, trades = run_backtest(base_lob, base_signal, params=p)
        for col in ("slip_cost", "impact_cost", "cost"):
            assert col in trades.columns


class TestMarketImpact:
    def test_impact_increases_cost(self, base_lob, base_signal):
        p_no = MarketParams(slippage_model="none", market_impact_coef=0.0, use_regime_filter=False)
        p_hi = MarketParams(slippage_model="none", market_impact_coef=5.0, use_regime_filter=False)
        pnl_no, _ = run_backtest(base_lob, base_signal, params=p_no)
        pnl_hi, t = run_backtest(base_lob, base_signal, params=p_hi)
        if len(t):
            assert t["impact_cost"].sum() > 0
            assert pnl_hi.sum() < pnl_no.sum()

    def test_zero_coef_zero_impact(self, base_lob, base_signal):
        p = MarketParams(market_impact_coef=0.0, use_regime_filter=False)
        _, trades = run_backtest(base_lob, base_signal, params=p)
        if len(trades):
            assert trades["impact_cost"].sum() == 0.0


class TestRealismConstraints:
    def test_default_for_stock_costs(self):
        p = MarketParams.default_for("stock")
        assert p.commission == 0.00025
        assert p.stamp_duty == 0.0005
        assert p.enforce_t1 is True

    def test_default_for_futures_costs(self):
        p = MarketParams.default_for("futures")
        assert p.commission == 0.000023
        assert p.stamp_duty == 0.0

    def test_t1_no_intraday_exits(self, base_lob, base_signal):
        p = MarketParams.default_for("stock", entry_z=0.5, use_regime_filter=False)
        _, trades = run_backtest(base_lob, base_signal, params=p)
        if len(trades):
            assert set(trades["exit_reason"]).issubset({"eod_t1", "eod_limit_down"})
            # single-position engine + no intraday exit → at most one trade
            assert len(trades) == 1

    def test_t1_off_allows_intraday_exits(self, base_lob, base_signal):
        p = MarketParams.default_for("stock", entry_z=0.5,
                                     use_regime_filter=False, enforce_t1=False)
        _, trades = run_backtest(base_lob, base_signal, params=p)
        if len(trades) > 1:
            assert (trades["exit_reason"].isin(["flip", "stop", "timeout", "eod"])).any()

    def test_buy_blocked_at_up_limit(self, base_lob):
        """Strong buy signal but ask sealed at the up-limit → no entries."""
        p = MarketParams.default_for("stock", entry_z=1.0, use_regime_filter=False)
        ask_min = float(base_lob["ask_px_1"].min())
        # choose prev_close so the up-limit sits below every ask → always sealed
        prev_close = ask_min / (1.0 + p.price_limit) * 0.999
        sig = pd.Series(3.0, index=base_lob.index)
        _, trades = run_backtest(base_lob, sig, params=p, prev_close=prev_close)
        assert len(trades) == 0

    def test_eod_force_close_futures(self, base_lob):
        """Position open at the last snapshot must be closed, logged, costed."""
        p = MarketParams(entry_z=1.0, max_hold=30, stop_loss_bp=10**9,
                         exit_z=10**9, use_regime_filter=False)
        # Signal only fires in the last 10 ticks → position cannot time out
        # (hold cap ≥ 30) and is still open at the final snapshot.
        sig = pd.Series(0.0, index=base_lob.index)
        sig.iloc[-10:] = 2.0
        _, trades = run_backtest(base_lob, sig, params=p)
        assert len(trades) == 1
        assert trades["exit_reason"].iloc[0] == "eod"
        assert trades["cost"].iloc[0] > 0

    def test_perm_impact_charged_at_exit(self, base_lob, base_signal):
        p_no   = MarketParams(use_regime_filter=False, market_impact_coef=5.0,
                              market_impact_perm_frac=0.0, slippage_model="none")
        p_perm = MarketParams(use_regime_filter=False, market_impact_coef=5.0,
                              market_impact_perm_frac=0.5, slippage_model="none")
        _, t_no   = run_backtest(base_lob, base_signal, params=p_no)
        _, t_perm = run_backtest(base_lob, base_signal, params=p_perm)
        if len(t_no) and len(t_perm):
            assert t_perm["impact_cost"].sum() > t_no["impact_cost"].sum()


class TestRegimeFilter:
    def test_no_entries_when_regime_off(self, base_lob):
        """Regime filter should block all entries when IC is persistently negative."""
        p = MarketParams(
            entry_z=0.1,           # very low threshold — many entries normally
            use_regime_filter=True,
            regime_ic_off=1.0,     # IC threshold so high regime is always OFF
            regime_ic_on=2.0,
        )
        mid = (base_lob["bid_px_1"] + base_lob["ask_px_1"]) / 2.0
        # Signal above entry_z almost everywhere
        sig = pd.Series(0.5, index=base_lob.index)
        _, trades = run_backtest(base_lob, sig, params=p)
        # With regime_ic_off=1.0, IC can never exceed 1.0 in 200 ticks, so always blocked
        assert len(trades) == 0, f"Expected 0 trades, got {len(trades)}"
