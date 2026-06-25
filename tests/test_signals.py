import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import simulate_lob_day
from src.signals.features import price_limit_signal


class TestPriceLimitSignal:

    def test_zero_far_from_limit(self):
        """Signal is 0 when mid is more than 3% from either limit."""
        df = simulate_lob_day(seed=0, prev_close=100.0)
        sig = price_limit_signal(df, prev_close=100.0)
        # mid will be near 100; 3% activation zone starts at 107.3 (up) or 92.7 (down)
        # synthetic data won't go that far with signal_strength=0.05
        assert (sig.abs() < 1e-6).mean() > 0.90, "Expected mostly zeros far from limit"

    def test_positive_at_up_limit(self):
        """Up-limit hit state produces positive signal."""
        mid_px = 110.0
        spread = 0.02
        row = {"bid_px_1": mid_px - spread/2, "ask_px_1": mid_px + spread/2}
        for lv in range(2, 11):
            row[f"bid_px_{lv}"] = row["bid_px_1"] - lv * 0.01
            row[f"ask_px_{lv}"] = row["ask_px_1"] + lv * 0.01
            row[f"bid_vol_{lv}"] = 100
            row[f"ask_vol_{lv}"] = 100
        row["bid_vol_1"] = 100
        row["ask_vol_1"] = 100
        row["cum_buy_vol"] = 0
        row["cum_sell_vol"] = 0
        df = pd.DataFrame([row], index=pd.DatetimeIndex(["2024-01-02 10:00:00"]))
        sig = price_limit_signal(df, prev_close=100.0, limit_pct=0.10)
        assert float(sig.iloc[0]) == 1.0

    def test_exact_limit_price_uses_tick_tolerance(self):
        """A quote exactly at the computed limit should be treated as a hit."""
        row = {
            "bid_px_1": 109.98,
            "ask_px_1": 110.00,
            "bid_vol_1": 100,
            "ask_vol_1": 100,
        }
        df = pd.DataFrame([row], index=pd.DatetimeIndex(["2024-01-02 10:00:00"]))
        sig = price_limit_signal(df, prev_close=100.0, limit_pct=0.10)
        assert float(sig.iloc[0]) == 1.0

    def test_negative_at_down_limit(self):
        """Down-limit hit state produces negative signal."""
        mid_px = 90.0
        spread = 0.02
        row = {"bid_px_1": mid_px - spread/2, "ask_px_1": mid_px + spread/2}
        for lv in range(2, 11):
            row[f"bid_px_{lv}"] = row["bid_px_1"] - lv * 0.01
            row[f"ask_px_{lv}"] = row["ask_px_1"] + lv * 0.01
            row[f"bid_vol_{lv}"] = 100
            row[f"ask_vol_{lv}"] = 100
        row["bid_vol_1"] = 100
        row["ask_vol_1"] = 100
        row["cum_buy_vol"] = 0
        row["cum_sell_vol"] = 0
        df = pd.DataFrame([row], index=pd.DatetimeIndex(["2024-01-02 10:00:00"]))
        sig = price_limit_signal(df, prev_close=100.0, limit_pct=0.10)
        assert float(sig.iloc[0]) == -1.0

    def test_returns_series_same_index(self):
        df = simulate_lob_day(seed=0)
        sig = price_limit_signal(df, prev_close=4000.0)
        assert isinstance(sig, pd.Series)
        assert sig.index.equals(df.index)
        assert sig.name == "price_limit"
