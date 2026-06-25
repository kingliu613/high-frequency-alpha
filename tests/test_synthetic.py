import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import simulate_lob_day


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


def test_lob_has_exact_factor_inputs():
    df = simulate_lob_day(seed=42)
    required = {
        "cum_buy_count",
        "cum_sell_count",
        "buy_count",
        "sell_count",
        "limit_buy_vol",
        "limit_sell_vol",
        "cancel_buy_vol",
        "cancel_sell_vol",
        "market_buy_vol",
        "market_sell_vol",
        "bid_depth",
        "ask_depth",
    }
    assert required.issubset(df.columns)
