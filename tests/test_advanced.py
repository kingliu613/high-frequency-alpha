import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import simulate_lob_day, simulate_close_auction_data
from src.signals.auction import close_auction_imbalance, close_auction_signal_series
from src.signals.advanced import (
    big_order_flow,
    herding_intensity,
    aggressive_passive_imbalance,
    order_execution_imbalance,
    order_book_slope,
    book_resiliency,
    signed_jump_reversal,
    realized_vol,
    sealing_strength,
    vpin,
    kyle_lambda,
    cancel_spike_imbalance,
    exposure_gate,
    spoof_filtered_qi,
    institutional_seal,
)
from src.signals.composite import (
    build_feature_matrix,
    build_composite_alpha,
    DEFAULT_WEIGHTS,
    FACTOR_GROUPS,
    expand_factor_selection,
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
    data["last_price"]  = np.full(n, (bid_px1 + ask_px1) / 2.0)
    data["last_volume"] = np.zeros(n)
    data["cum_buy_vol"]  = np.arange(n, dtype=float) * 1000.0 if cum_buy is None else cum_buy
    data["cum_sell_vol"] = np.arange(n, dtype=float) * 1000.0 if cum_sell is None else cum_sell
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Shape / alignment — every factor returns aligned, finite, NaN-free Series
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    lambda d: big_order_flow(d),
    lambda d: herding_intensity(d),
    lambda d: aggressive_passive_imbalance(d),
    lambda d: order_execution_imbalance(d),
    lambda d: order_book_slope(d),
    lambda d: book_resiliency(d),
    lambda d: signed_jump_reversal(d),
    lambda d: realized_vol(d),
])
def test_factor_shape_and_finite(fn):
    df = simulate_lob_day(seed=7)
    s = fn(df)
    assert isinstance(s, pd.Series)
    assert len(s) == len(df)
    assert s.index.equals(df.index)
    assert np.isfinite(s.to_numpy()).all()


def test_bounded_factors_in_range():
    df = simulate_lob_day(seed=3)
    for s in (aggressive_passive_imbalance(df), order_book_slope(df), book_resiliency(df)):
        assert s.abs().max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Directional sign checks on constructed flow
# ---------------------------------------------------------------------------

def test_big_order_flow_positive_on_net_buys():
    n = 120
    df = _book(n, cum_buy=np.arange(n) * 5000.0, cum_sell=np.zeros(n))
    s = big_order_flow(df, window=20)
    assert s.iloc[40:].mean() > 0


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


def test_book_slope_positive_when_bid_steeper():
    n = 40
    df = _book(n, bid_vol=30000, ask_vol=8000)  # thicker bid over same price span
    s = order_book_slope(df)
    assert s.iloc[-1] > 0


def test_resiliency_positive_when_bid_refills():
    n = 60
    df = _book(n)
    # bid depth grows (refills), ask depth shrinks (depletes)
    df["bid_vol_1"] = 5000.0 + np.arange(n) * 100.0
    df["ask_vol_1"] = 12000.0 - np.arange(n) * 100.0
    s = book_resiliency(df, window=20)
    assert s.iloc[30:].mean() > 0


def test_signed_jump_is_reversal_after_up_move():
    n = 140
    df = _book(n)
    mid = np.full(n, 10.0)
    mid[80:] = 10.20           # single up-jump at t=80 (past z-score warm-up)
    df["bid_px_1"] = mid - 0.01
    df["ask_px_1"] = mid + 0.01
    s = signed_jump_reversal(df, window=20)
    # window straddling the up-jump → negative (expect pullback)
    assert s.iloc[81:100].mean() < 0


def test_sealing_strength_positive_at_up_limit():
    n = 40
    prev_close = 10.0
    df = _book(n, bid_px1=11.0, ask_px1=11.02, bid_vol=50000,
               cum_buy=np.arange(n) * 1000.0, cum_sell=np.zeros(n))
    s = sealing_strength(df, prev_close=prev_close, limit_pct=0.10)
    assert s.iloc[-1] > 0


def test_factors_robust_to_missing_trade_columns():
    """big_flow/herding/api degrade to finite (≈0) when no trade data present."""
    n = 50
    df = _book(n)
    df = df.drop(columns=["cum_buy_vol", "cum_sell_vol", "last_price", "last_volume"])
    for fn in (big_order_flow, herding_intensity, aggressive_passive_imbalance):
        s = fn(df)
        assert np.isfinite(s.to_numpy()).all()


# ---------------------------------------------------------------------------
# Closing call auction
# ---------------------------------------------------------------------------

def test_close_auction_imbalance_in_range():
    close_df, close_px = simulate_close_auction_data(day_close=4000.0, seed=1)
    imb = close_auction_imbalance(close_df)
    assert -1.0 <= imb <= 1.0
    assert close_px > 0


def test_close_auction_signal_decays():
    df = simulate_lob_day(seed=5)
    s = close_auction_signal_series(df, close_auction_value=0.5, half_life_min=30.0)
    assert s.iloc[0] == pytest.approx(0.5, abs=1e-6)
    assert abs(s.iloc[-1]) < abs(s.iloc[0])   # decays toward zero


# ---------------------------------------------------------------------------
# Composite integration
# ---------------------------------------------------------------------------

def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)


def test_feature_matrix_includes_new_factors():
    df = simulate_lob_day(seed=11)
    feat = build_feature_matrix(df, close_auction_value=0.2, ofi_levels=5)
    for col in ["api", "oei", "big_flow", "herding", "book_slope",
                "resiliency", "signed_jump", "close_auction"]:
        assert col in feat.columns


def test_stock_mode_adds_sealing():
    df = simulate_lob_day(seed=12, is_futures=False, prev_close=100.0)
    feat = build_feature_matrix(df, instrument="stock", prev_close=100.0)
    assert "sealing" in feat.columns
    assert "price_limit" in feat.columns


def test_composite_runs_with_all_factors():
    df = simulate_lob_day(seed=13)
    feat = build_feature_matrix(df, auction_value=0.1, close_auction_value=0.1, ofi_levels=5)
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
    rng = np.random.default_rng(0)
    buys  = rng.integers(0, 5000, n).astype(float)
    sells = rng.integers(0, 5000, n).astype(float)
    balanced = _book(n, cum_buy=np.cumsum(buys), cum_sell=np.cumsum(sells))
    v_one = vpin(one_sided).iloc[-1]
    v_bal = vpin(balanced).iloc[-1]
    assert v_one > v_bal
    assert v_one == pytest.approx(1.0, abs=1e-6)


def test_kyle_lambda_zero_without_impact():
    n = 300
    # constant price, random flow → no impact → λ z-score ≈ 0
    rng = np.random.default_rng(1)
    buys  = rng.integers(0, 5000, n).astype(float)
    sells = rng.integers(0, 5000, n).astype(float)
    df = _book(n, cum_buy=np.cumsum(buys), cum_sell=np.cumsum(sells))
    lam = kyle_lambda(df, window=60)
    assert lam.abs().max() < 1e-6


def test_cancel_spike_positive_on_ask_withdrawal():
    n = 400
    df = _book(n)
    # no trades at all → every depth drop is a cancel
    df["cum_buy_vol"] = 0.0
    df["cum_sell_vol"] = 0.0
    # steady book until t=350, then ask side evaporates (sellers withdrawing)
    av = np.full(n, 20000.0)
    av[350:] = 20000.0 - np.arange(n - 350) * 2000.0
    df["ask_vol_1"] = np.clip(av, 100.0, None)
    s = cancel_spike_imbalance(df, window=10)
    assert s.iloc[355:].max() > 0          # bullish: ask cancels dominate
    assert (s.iloc[:340] == 0.0).all()     # silent without spike


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
# Interaction factors — spoof-filtered QI, institutional seal
# ---------------------------------------------------------------------------

def test_spoof_filter_mutes_cancelled_wall():
    """Bid wall being cancelled → filtered signal much weaker than raw QI."""
    n = 400
    df = _book(n, bid_vol=50000, ask_vol=5000)          # fat bid wall → QI > 0
    df["cum_buy_vol"] = 0.0                              # no trades at all:
    df["cum_sell_vol"] = 0.0                             # depth drops = cancels
    # wall quietly evaporates over the last 50 ticks (spoof being pulled)
    bv = np.full(n, 50000.0)
    bv[350:] = 50000.0 - np.arange(n - 350) * 4000.0
    df["bid_vol_1"] = np.clip(bv, 100.0, None)

    from src.signals.features import queue_imbalance
    raw  = queue_imbalance(df)
    filt = spoof_filtered_qi(df, window=10)

    spoof_zone = slice(355, 398)
    assert raw.iloc[spoof_zone].mean() > 0              # raw still bullish
    assert filt.iloc[spoof_zone].mean() < raw.iloc[spoof_zone].mean() * 0.8
    # quiet period: no contradiction → filtered ≈ raw
    calm = slice(250, 340)
    assert np.allclose(filt.iloc[calm], raw.iloc[calm], atol=1e-6)


def test_spoof_filter_never_flips_sign():
    df = simulate_lob_day(seed=40)
    from src.signals.features import queue_imbalance
    raw  = queue_imbalance(df)
    filt = spoof_filtered_qi(df)
    assert ((filt * raw) >= -1e-12).all()               # same sign or zero


def test_institutional_seal_big_beats_retail():
    """Equal turnover, different size distribution: institutional prints
    (volume concentrated in few big trades) must outscore retail dribble."""
    n = 300
    prev_close = 10.0

    def sealed(big: bool):
        rng = np.random.default_rng(3)
        if big:
            # 80% tiny prints + 20% institutional blocks; mean ≈ 996/tick
            base = np.full(n, 100.0)
            base[::5] = 4580.0
        else:
            # uniform retail lots, same mean ≈ 1000/tick
            base = rng.integers(1, 20, n).astype(float) * 100.0
        cum = np.cumsum(base)
        return _book(n, bid_px1=11.0, ask_px1=11.02, bid_vol=80000,
                     cum_buy=cum, cum_sell=np.zeros(n))

    s_big    = institutional_seal(sealed(True),  prev_close)
    s_retail = institutional_seal(sealed(False), prev_close)
    assert s_big.iloc[-1] > s_retail.iloc[-1]
    assert s_retail.iloc[-1] >= 0.0


def test_institutional_seal_zero_away_from_limit():
    df = simulate_lob_day(seed=41, is_futures=False, prev_close=100.0)
    # synthetic day rarely hits ±10%; signal should be ~all zeros
    s = institutional_seal(df, prev_close=100.0)
    assert (s == 0.0).mean() > 0.95


# ---------------------------------------------------------------------------
# Modular factor selection
# ---------------------------------------------------------------------------

def test_expand_selection_groups_and_names():
    sel = expand_factor_selection(["flow", "close_auction"])
    assert "mlofi" in sel and "api" in sel and "close_auction" in sel
    assert "queue_imbalance" not in sel


def test_expand_selection_unknown_raises():
    with pytest.raises(ValueError, match="Unknown factor"):
        expand_factor_selection(["flow", "nonsense_factor"])


def test_feature_matrix_subset_only_selected():
    df = simulate_lob_day(seed=42)
    feat = build_feature_matrix(df, factors=["book"])
    assert set(feat.columns) == set(FACTOR_GROUPS["book"])


def test_subset_composite_runs():
    df = simulate_lob_day(seed=43)
    feat = build_feature_matrix(df, close_auction_value=0.2,
                                factors=["flow", "auction"])
    alpha = build_composite_alpha(feat)
    assert np.isfinite(alpha.to_numpy()).all()
    assert "close_auction" in feat.columns
    assert "auction_signal" not in feat.columns   # no auction_value provided


def test_empty_selection_raises():
    df = simulate_lob_day(seed=44)
    # auction group selected but no auction scalars provided → empty matrix
    with pytest.raises(ValueError, match="empty feature set"):
        build_feature_matrix(df, factors=["auction"])


def test_stock_mode_selection_includes_seal_inst():
    df = simulate_lob_day(seed=45, is_futures=False, prev_close=100.0)
    feat = build_feature_matrix(df, instrument="stock", prev_close=100.0,
                                factors=["limit"])
    assert set(feat.columns) == {"price_limit", "sealing", "seal_inst"}


def test_vpin_is_causal():
    """VPIN at tick t must not change when future data changes (no look-ahead)."""
    df = simulate_lob_day(seed=30)
    k = len(df) // 2
    v_full = vpin(df)
    v_half = vpin(df.iloc[:k])
    pd.testing.assert_series_equal(v_full.iloc[:k], v_half, check_names=False)


def test_big_order_flow_is_causal():
    df = simulate_lob_day(seed=31)
    k = len(df) // 2
    f_full = big_order_flow(df)
    f_half = big_order_flow(df.iloc[:k])
    pd.testing.assert_series_equal(f_full.iloc[:k], f_half, check_names=False)


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
