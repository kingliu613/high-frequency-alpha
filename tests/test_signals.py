import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import simulate_lob_day, simulate_etf_series
from src.signals.features import price_limit_signal, etf_basis_signal


class TestPriceLimitSignal:

    def test_zero_far_from_limit(self):
        """Signal is 0 when mid is more than 3% from either limit."""
        df = simulate_lob_day(seed=0, prev_close=100.0)
        sig = price_limit_signal(df, prev_close=100.0)
        # mid will be near 100; 3% activation zone starts at 107.3 (up) or 92.7 (down)
        # synthetic data won't go that far with signal_strength=0.05
        assert (sig.abs() < 1e-6).mean() > 0.90, "Expected mostly zeros far from limit"

    def test_positive_near_up_limit(self):
        """Approaching up-limit produces positive signal."""
        # Construct a LOB snapshot where mid ≈ 109 (within 3% of 110 up-limit)
        mid_px = 109.5
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
        assert float(sig.iloc[0]) > 0.5, f"Expected > 0.5 near up-limit, got {float(sig.iloc[0]):.4f}"

    def test_negative_near_down_limit(self):
        """Approaching down-limit produces negative signal."""
        mid_px = 90.5
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
        assert float(sig.iloc[0]) < -0.5, f"Expected < -0.5 near down-limit, got {float(sig.iloc[0]):.4f}"

    def test_returns_series_same_index(self):
        df = simulate_lob_day(seed=0)
        sig = price_limit_signal(df, prev_close=4000.0)
        assert isinstance(sig, pd.Series)
        assert sig.index.equals(df.index)
        assert sig.name == "price_limit"


class TestEtfBasisSignal:

    def test_returns_series_same_index(self):
        df  = simulate_lob_day(seed=0)
        etf = simulate_etf_series(df, seed=0)
        sig = etf_basis_signal(df, etf)
        assert isinstance(sig, pd.Series)
        assert sig.index.equals(df.index)
        assert sig.name == "etf_basis"

    def test_sign_is_mean_reverting(self):
        """Positive ETF premium → negative signal (sell expensive ETF)."""
        df  = simulate_lob_day(seed=0)
        mid = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
        # Force ETF price 2% above mid (expensive ETF)
        etf_expensive = mid * 1.02
        sig = etf_basis_signal(df, etf_expensive)
        # After burn-in (200 ticks), signal should be negative
        assert float(sig.iloc[250:].mean()) < 0, "Expensive ETF should give negative signal"

    def test_zero_signal_at_par(self):
        """ETF at exact NAV → basis = 0 → signal = 0."""
        df  = simulate_lob_day(seed=0)
        mid = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
        sig = etf_basis_signal(df, mid)   # ETF = mid exactly
        # rolling std of a zero series is 0, so fillna(0) → all zeros
        assert (sig.abs() < 1e-9).all()
