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
