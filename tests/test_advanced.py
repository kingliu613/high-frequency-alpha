import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import simulate_lob_day
from src.signals.advanced import (
    herding_intensity,
    aggressive_passive_imbalance,
    order_execution_imbalance,
    realized_vol,
    vpin,
    kyle_lambda,
    kyle_lambda_state,
    cancel_spike_imbalance,
    exposure_gate,
)
from src.signals.composite import (
    build_feature_matrix,
    build_composite_alpha,
    DEFAULT_WEIGHTS,
    FACTOR_REGISTRY,
    FACTOR_GROUPS,
    expand_factor_selection,
    missing_required_columns,
)


# ---------------------------------------------------------------------------
# Helpers — hand-built minimal 5-level LOB frames for directional assertions
# ---------------------------------------------------------------------------

def _index(n):
    return pd.date_range("2024-01-02 09:30:00", periods=n, freq="3s")


def _book(n, *, bid_px1=10.0, ask_px1=10.02, bid_vol=10000, ask_vol=10000,
          cum_buy=None, cum_sell=None, n_levels=5, tick=0.01):
    idx = _index(n)
    data = {"mid_price": np.full(n, (bid_px1 + ask_px1) / 2.0)}
    for lv in range(1, n_levels + 1):
        data[f"bid_px_{lv}"]  = np.full(n, bid_px1 - (lv - 1) * tick)
        data[f"ask_px_{lv}"]  = np.full(n, ask_px1 + (lv - 1) * tick)
        data[f"bid_vol_{lv}"] = np.full(n, float(bid_vol))
        data[f"ask_vol_{lv}"] = np.full(n, float(ask_vol))
    data["cum_buy_vol"]  = np.arange(n, dtype=float) * 1000.0 if cum_buy is None else cum_buy
    data["cum_sell_vol"] = np.arange(n, dtype=float) * 1000.0 if cum_sell is None else cum_sell
    dvb = pd.Series(data["cum_buy_vol"]).diff().clip(lower=0).fillna(0.0).to_numpy()
    dvs = pd.Series(data["cum_sell_vol"]).diff().clip(lower=0).fillna(0.0).to_numpy()
    total_trade = dvb + dvs
    data["last_price"] = np.where(dvb >= dvs, ask_px1, bid_px1)
    data["last_volume"] = total_trade
    data["buy_count"] = (dvb > 0).astype(float)
    data["sell_count"] = (dvs > 0).astype(float)
    data["cum_buy_count"] = np.cumsum(data["buy_count"])
    data["cum_sell_count"] = np.cumsum(data["sell_count"])
    data["market_buy_vol"] = dvb
    data["market_sell_vol"] = dvs
    data["limit_buy_vol"] = np.full(n, float(bid_vol) * 0.1)
    data["limit_sell_vol"] = np.full(n, float(ask_vol) * 0.1)
    data["cancel_buy_vol"] = np.zeros(n)
    data["cancel_sell_vol"] = np.zeros(n)
    data["bid_depth"] = np.full(n, float(bid_vol) * n_levels)
    data["ask_depth"] = np.full(n, float(ask_vol) * n_levels)
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Shape / alignment — every factor returns aligned, finite, NaN-free Series
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    lambda d: herding_intensity(d),
    lambda d: aggressive_passive_imbalance(d),
    lambda d: order_execution_imbalance(d),
    lambda d: realized_vol(d),
])
def test_factor_shape_and_finite(fn):
    df = simulate_lob_day(seed=7)
    s = fn(df)
    assert isinstance(s, pd.Series)
    assert len(s) == len(df)
    assert s.index.equals(df.index)
    assert np.isfinite(s.to_numpy()).all()


def test_api_is_finite_and_has_variance():
    df = simulate_lob_day(seed=3)
    s = aggressive_passive_imbalance(df)
    assert np.isfinite(s.to_numpy()).all()
    assert s.std() > 0


# ---------------------------------------------------------------------------
# Directional sign checks on constructed flow
# ---------------------------------------------------------------------------

def test_herding_positive_when_all_buys():
    n = 60
    df = _book(n, cum_buy=np.arange(n) * 5000.0, cum_sell=np.zeros(n))
    s = herding_intensity(df, window=20)
    assert s.iloc[30:].mean() > 0


def test_api_positive_on_aggressive_buys():
    n = 60
    df = _book(n, cum_buy=np.arange(n) * 5000.0, cum_sell=np.zeros(n))
    s = aggressive_passive_imbalance(df, window=10)
    assert s.iloc[20:].mean() > 0


def test_oei_positive_when_ask_consumed():
    n = 60
    # buys only → ask side being executed → bullish OEI
    df = _book(n, cum_buy=np.arange(n) * 5000.0, cum_sell=np.zeros(n))
    s = order_execution_imbalance(df, window=10)
    assert s.iloc[20:].mean() > 0


def test_exact_factors_require_their_paper_columns():
    n = 50
    df = _book(n)
    with pytest.raises(ValueError, match="buy/sell count columns"):
        herding_intensity(df.drop(columns=["cum_buy_count", "cum_sell_count", "buy_count", "sell_count"]))
    with pytest.raises(ValueError, match="order-event columns"):
        aggressive_passive_imbalance(df.drop(columns=["limit_buy_vol"]))


# ---------------------------------------------------------------------------
# Composite integration
# ---------------------------------------------------------------------------

def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)


def test_factor_registry_declares_roles_and_inputs():
    assert FACTOR_REGISTRY["trade_imbalance"].formula_id == "polarity=(NOB-NOS)/(NOB+NOS)"
    assert FACTOR_REGISTRY["api"].output_role == "gate"
    assert FACTOR_REGISTRY["vpin"].output_role == "gate"
    assert FACTOR_REGISTRY["price_limit"].output_role == "label"
    assert FACTOR_REGISTRY["herding"].strict_supported is False


def test_feature_matrix_default_is_strict_alpha_only():
    df = simulate_lob_day(seed=11)
    feat = build_feature_matrix(df, ofi_levels=5)
    for col in ["mlofi", "agg_ofi", "trade_imbalance"]:
        assert col in feat.columns
    for col in ["api", "oei", "queue_imbalance", "herding", "cancel_spike", "auction_signal", "price_limit"]:
        assert col not in feat.columns


def test_stock_mode_does_not_add_price_limit_by_default():
    df = simulate_lob_day(seed=12, is_futures=False, prev_close=100.0)
    feat = build_feature_matrix(df, instrument="stock", prev_close=100.0)
    assert "price_limit" not in feat.columns


def test_composite_runs_with_all_factors():
    df = simulate_lob_day(seed=13)
    feat = build_feature_matrix(df, auction_value=0.1, ofi_levels=5)
    alpha = build_composite_alpha(feat)
    assert len(alpha) == len(df)
    assert np.isfinite(alpha.to_numpy()).all()


# ---------------------------------------------------------------------------
# Tier 3 — VPIN / Kyle λ / cancel-spike / exposure gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    lambda d: vpin(d),
    lambda d: kyle_lambda(d),
    lambda d: cancel_spike_imbalance(d),
    lambda d: exposure_gate(d),
])
def test_tier3_shape_and_finite(fn):
    df = simulate_lob_day(seed=21)
    s = fn(df)
    assert isinstance(s, pd.Series)
    assert len(s) == len(df)
    assert s.index.equals(df.index)
    assert np.isfinite(s.to_numpy()).all()


def test_vpin_bounded_zero_one():
    df = simulate_lob_day(seed=22)
    v = vpin(df)
    assert v.min() >= 0.0
    assert v.max() <= 1.0


def test_vpin_high_for_one_sided_flow():
    n = 400
    one_sided = _book(n, cum_buy=np.arange(n) * 5000.0, cum_sell=np.zeros(n))
    one_sided["last_price"] = 10.0 + np.arange(n) * 0.01
    rng = np.random.default_rng(0)
    buys  = rng.integers(0, 5000, n).astype(float)
    sells = rng.integers(0, 5000, n).astype(float)
    balanced = _book(n, cum_buy=np.cumsum(buys), cum_sell=np.cumsum(sells))
    balanced["last_price"] = 10.0 + np.cumsum(rng.normal(0.0, 0.01, n))
    v_one = vpin(one_sided).iloc[-1]
    v_bal = vpin(balanced).iloc[-1]
    assert v_one > v_bal
    assert v_one > 0.8


def test_kyle_lambda_zero_without_impact():
    n = 300
    # constant price, random flow → no impact → λ z-score ≈ 0
    rng = np.random.default_rng(1)
    buys  = rng.integers(0, 5000, n).astype(float)
    sells = rng.integers(0, 5000, n).astype(float)
    df = _book(n, cum_buy=np.cumsum(buys), cum_sell=np.cumsum(sells))
    lam = kyle_lambda(df, window=60)
    assert lam.abs().max() < 1e-6


def test_kyle_lambda_raw_and_state_are_separate():
    n = 300
    signed = np.arange(n, dtype=float)
    df = _book(n, cum_buy=np.cumsum(signed + 1.0), cum_sell=np.zeros(n))
    df["bid_px_1"] = 10.0 + signed * 0.001
    df["ask_px_1"] = 10.02 + signed * 0.001
    raw = kyle_lambda(df, window=60)
    state = kyle_lambda_state(df, window=60)
    assert raw.name == "kyle_lambda"
    assert state.name == "kyle_lambda_state"
    assert np.isfinite(raw.to_numpy()).all()
    assert np.isfinite(state.to_numpy()).all()


def test_cancel_spike_positive_on_ask_withdrawal():
    n = 400
    df = _book(n)
    df["cancel_sell_vol"] = 0.0
    df.iloc[350:, df.columns.get_loc("cancel_sell_vol")] = 5000.0
    s = cancel_spike_imbalance(df, window=10)
    assert s.iloc[355:].max() > 0          # bullish: ask cancels dominate
    assert (s.iloc[:340] == 0.0).all()     # silent without spike


def test_cancel_spike_small_lookback_does_not_error():
    df = _book(20)
    df["cancel_sell_vol"] = np.r_[np.zeros(10), np.ones(10) * 1000.0]
    s = cancel_spike_imbalance(df, window=2, spike_lookback=3)
    assert np.isfinite(s.to_numpy()).all()


def test_exposure_gate_in_range():
    df = simulate_lob_day(seed=23)
    g = exposure_gate(df, floor=0.2)
    assert g.min() >= 0.2 - 1e-9
    assert g.max() <= 1.0 + 1e-9


def test_backtest_accepts_exposure_scale():
    from src.backtest.engine import run_backtest, MarketParams
    df = simulate_lob_day(seed=24)
    feat = build_feature_matrix(df, ofi_levels=5)
    alpha = build_composite_alpha(feat)
    gate = exposure_gate(df)
    pnl_g, trades_g = run_backtest(df, alpha, MarketParams(), exposure_scale=gate)
    pnl_n, trades_n = run_backtest(df, alpha, MarketParams())
    assert len(pnl_g) == len(df)
    assert np.isfinite(pnl_g.to_numpy()).all()
    # gate can only reduce or keep entry count (blocks some entries)
    assert len(trades_g) <= len(trades_n) + 1   # +1 slack for hold-window shifts


# ---------------------------------------------------------------------------
# Modular factor selection
# ---------------------------------------------------------------------------

def test_expand_selection_groups_and_names():
    sel = expand_factor_selection(["flow", "auction_signal"])
    assert "mlofi" in sel and "agg_ofi" in sel and "auction_signal" in sel
    assert "api" not in sel
    assert "queue_imbalance" not in sel


def test_missing_columns_reported_for_explicit_factor():
    df = simulate_lob_day(seed=41).drop(columns=["limit_buy_vol"])
    assert missing_required_columns(df, "api") == ["limit_buy_vol"]
    with pytest.raises(ValueError, match="Factor 'api' requires columns"):
        build_feature_matrix(df, factors=["api"])


def test_expand_selection_unknown_raises():
    with pytest.raises(ValueError, match="Unknown factor"):
        expand_factor_selection(["flow", "nonsense_factor"])


def test_feature_matrix_subset_only_selected():
    df = simulate_lob_day(seed=42)
    feat = build_feature_matrix(df, factors=["book"])
    assert set(feat.columns) == set(FACTOR_GROUPS["book"])


def test_subset_composite_runs():
    df = simulate_lob_day(seed=43)
    feat = build_feature_matrix(df, auction_value=0.2,
                                factors=["flow", "auction"])
    alpha = build_composite_alpha(feat)
    assert np.isfinite(alpha.to_numpy()).all()
    assert "auction_signal" in feat.columns


def test_empty_selection_raises():
    df = simulate_lob_day(seed=44)
    # auction group selected but no auction scalars provided → empty matrix
    with pytest.raises(ValueError, match="empty feature set"):
        build_feature_matrix(df, factors=["auction"])


def test_stock_mode_selection_includes_price_limit_only():
    df = simulate_lob_day(seed=45, is_futures=False, prev_close=100.0)
    feat = build_feature_matrix(df, instrument="stock", prev_close=100.0,
                                factors=["limit"])
    assert set(feat.columns) == {"price_limit"}


def test_vpin_is_causal():
    """VPIN at tick t must not change when future data changes (no look-ahead)."""
    df = simulate_lob_day(seed=30)
    k = len(df) // 2
    v_full = vpin(df)
    v_half = vpin(df.iloc[:k])
    pd.testing.assert_series_equal(v_full.iloc[:k], v_half, check_names=False)


def test_exposure_gate_is_causal():
    df = simulate_lob_day(seed=32)
    k = len(df) // 2
    g_full = exposure_gate(df)
    g_half = exposure_gate(df.iloc[:k])
    pd.testing.assert_series_equal(g_full.iloc[:k], g_half, check_names=False)


def test_backtest_zero_scale_blocks_all_entries():
    from src.backtest.engine import run_backtest, MarketParams
    df = simulate_lob_day(seed=25)
    feat = build_feature_matrix(df, ofi_levels=5)
    alpha = build_composite_alpha(feat)
    zero_gate = pd.Series(0.0, index=df.index)
    pnl, trades = run_backtest(df, alpha, MarketParams(), exposure_scale=zero_gate)
    assert len(trades) == 0
    assert pnl.abs().sum() == 0.0
