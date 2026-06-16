import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import simulate_lob_day, simulate_etf_series


def test_signal_strength_default_is_001():
    import inspect
    sig = inspect.signature(simulate_lob_day)
    assert sig.parameters["signal_strength"].default == 0.01


def test_lob_vol_has_persistence():
    """Consecutive bid_vol_1 values should be correlated (OU process)."""
    df = simulate_lob_day(seed=42)
    v = df["bid_vol_1"].astype(float)
    lag1_corr = v.corr(v.shift(1))
    assert lag1_corr > 0.3, f"Expected lag-1 autocorr > 0.3, got {lag1_corr:.3f}"


def test_simulate_etf_series_returns_series():
    df = simulate_lob_day(seed=0)
    etf = simulate_etf_series(df, seed=0)
    assert isinstance(etf, pd.Series)
    assert len(etf) == len(df)
    assert etf.index.equals(df.index)


def test_simulate_etf_series_premium_is_small():
    """ETF should trade within ±3% of mid (AR(1) σ=10bps)."""
    df = simulate_lob_day(seed=0)
    mid = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
    etf = simulate_etf_series(df, seed=0)
    premium = (etf / mid - 1.0).abs()
    assert premium.max() < 0.03, f"Max premium {premium.max():.4f} too large"
