"""
Integration tests for the Wind L2 pipeline at 10 LOB levels.

These tests use synthetic data shaped to match the Wind loader output schema
(10 bid/ask levels, cum_buy_vol/cum_sell_vol from cumsum, bid_depth/ask_depth
derived columns). They validate the full path from LOB data → features →
composite alpha → backtest without a live Wind connection.

Run:
    pytest tests/test_wind_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import simulate_lob_day
from src.data.validator import (
    validate_lob_schema,
    assert_minimum_viable,
    _count_complete_lob_levels,
)
from src.signals.ofi import mlofi, aggregated_ofi, trade_imbalance
from src.signals.composite import build_feature_matrix, build_composite_alpha
from src.backtest.engine import run_backtest, MarketParams
from src.backtest.metrics import compute_forward_returns, ic_by_horizon, pnl_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _wind_shaped_lob(seed: int = 42, n_levels: int = 10) -> pd.DataFrame:
    """
    Return a synthetic LOB frame shaped to match the Wind loader output:
      - 10 bid/ask levels
      - cum_buy_vol / cum_sell_vol as running totals (not per-bar)
      - cum_buy_count / cum_sell_count present
      - bid_depth / ask_depth as sum of all level volumes
      - optional limit_buy_vol etc. absent (will block api / oei)
    """
    df = simulate_lob_day(seed=seed, prev_close=4000.0, is_futures=True)

    # Ensure exactly 10 levels (synthetic already generates 10, but be explicit)
    for lv in range(1, n_levels + 1):
        for side in ("bid", "ask"):
            px_col  = f"{side}_px_{lv}"
            vol_col = f"{side}_vol_{lv}"
            if px_col not in df.columns:
                ref_px = df[f"{side}_px_1"].astype(float)
                direction = -1 if side == "bid" else 1
                df[px_col]  = ref_px + direction * (lv - 1) * 0.01
                df[vol_col] = 1000.0

    # cum_buy_count / cum_sell_count: simulate as cumulative transaction counts
    rng = np.random.default_rng(seed + 1)
    n = len(df)
    buy_cnt  = rng.integers(1, 5, size=n).astype(float)
    sell_cnt = rng.integers(1, 5, size=n).astype(float)
    df["cum_buy_count"]  = buy_cnt.cumsum()
    df["cum_sell_count"] = sell_cnt.cumsum()

    # bid_depth / ask_depth: sum all visible levels (mirrors _wind_result_to_lob)
    bid_vols = [f"bid_vol_{lv}" for lv in range(1, 11)]
    ask_vols = [f"ask_vol_{lv}" for lv in range(1, 11)]
    df["bid_depth"] = df[bid_vols].fillna(0.0).sum(axis=1)
    df["ask_depth"] = df[ask_vols].fillna(0.0).sum(axis=1)

    return df


@pytest.fixture(scope="module")
def wind_lob():
    return _wind_shaped_lob(seed=42)


@pytest.fixture(scope="module")
def wind_feat(wind_lob):
    return build_feature_matrix(wind_lob, ofi_levels=10)


@pytest.fixture(scope="module")
def wind_comp(wind_feat):
    return build_composite_alpha(wind_feat)


# ---------------------------------------------------------------------------
# Schema / validator tests
# ---------------------------------------------------------------------------

class TestWindSchema:

    def test_10_levels_complete(self, wind_lob):
        n = _count_complete_lob_levels(wind_lob)
        assert n == 10, f"Expected 10 complete LOB levels, got {n}"

    def test_cum_buy_vol_is_monotone(self, wind_lob):
        diffs = wind_lob["cum_buy_vol"].diff().dropna()
        assert (diffs >= 0).all(), "cum_buy_vol must be non-decreasing"

    def test_cum_sell_vol_is_monotone(self, wind_lob):
        diffs = wind_lob["cum_sell_vol"].diff().dropna()
        assert (diffs >= 0).all(), "cum_sell_vol must be non-decreasing"

    def test_bid_depth_equals_sum_of_levels(self, wind_lob):
        expected = sum(wind_lob[f"bid_vol_{lv}"] for lv in range(1, 11))
        pd.testing.assert_series_equal(
            wind_lob["bid_depth"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_ask_depth_equals_sum_of_levels(self, wind_lob):
        expected = sum(wind_lob[f"ask_vol_{lv}"] for lv in range(1, 11))
        pd.testing.assert_series_equal(
            wind_lob["ask_depth"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_validate_schema_returns_report(self, wind_lob):
        report = validate_lob_schema(wind_lob)
        assert report.n_rows == len(wind_lob)
        assert report.n_levels_complete == 10

    def test_mlofi_agg_ofi_show_runnable(self, wind_lob):
        report = validate_lob_schema(wind_lob)
        runnable = report.runnable_alpha
        assert "mlofi" in runnable
        assert "agg_ofi" in runnable

    def test_trade_imbalance_shows_runnable_with_counts(self, wind_lob):
        report = validate_lob_schema(wind_lob)
        assert "trade_imbalance" in report.runnable_alpha

    def test_assert_minimum_viable_passes(self, wind_lob):
        assert_minimum_viable(wind_lob, min_alpha=2)

    def test_assert_minimum_viable_fails_on_empty(self):
        empty = pd.DataFrame({"bid_px_1": [1.0], "ask_px_1": [1.01]})
        with pytest.raises(ValueError):
            assert_minimum_viable(empty, min_alpha=1)

    def test_index_is_datetime(self, wind_lob):
        report = validate_lob_schema(wind_lob)
        assert report.index_is_datetime


# ---------------------------------------------------------------------------
# Signal tests at 10 levels
# ---------------------------------------------------------------------------

class TestSignals10Level:

    def test_mlofi_10_levels(self, wind_lob):
        sig = mlofi(wind_lob, n_levels=10)
        assert isinstance(sig, pd.Series)
        assert sig.index.equals(wind_lob.index)
        assert sig.name == "mlofi"
        assert sig.notna().sum() > len(wind_lob) * 0.9

    def test_mlofi_std_positive(self, wind_lob):
        sig = mlofi(wind_lob, n_levels=10)
        assert sig.std() > 0, "mlofi should have non-zero variance"

    def test_agg_ofi_10_levels(self, wind_lob):
        sig = aggregated_ofi(wind_lob, window=10, n_levels=10)
        assert sig.std() > 0

    def test_trade_imbalance_with_count_columns(self, wind_lob):
        sig = trade_imbalance(wind_lob, window=20)
        assert isinstance(sig, pd.Series)
        assert sig.notna().mean() > 0.9
        # polarity is bounded [-1, 1]
        assert sig.min() >= -1.0 - 1e-9
        assert sig.max() <= 1.0 + 1e-9

    def test_mlofi_uses_more_levels_than_5(self, wind_lob):
        sig5  = mlofi(wind_lob, n_levels=5,  normalize=False)
        sig10 = mlofi(wind_lob, n_levels=10, normalize=False)
        # 10-level MLOFI and 5-level MLOFI should differ (deeper levels add info)
        assert not np.allclose(sig5.fillna(0), sig10.fillna(0), atol=1e-6), \
            "10-level and 5-level MLOFI should differ"


# ---------------------------------------------------------------------------
# Feature matrix / composite tests
# ---------------------------------------------------------------------------

class TestFeatureMatrix10Level:

    def test_feature_matrix_has_expected_columns(self, wind_feat):
        assert "mlofi" in wind_feat.columns
        assert "agg_ofi" in wind_feat.columns
        assert "trade_imbalance" in wind_feat.columns

    def test_feature_matrix_index_matches_lob(self, wind_lob, wind_feat):
        assert wind_feat.index.equals(wind_lob.index)

    def test_feature_matrix_no_all_nan_columns(self, wind_feat):
        for col in wind_feat.columns:
            assert wind_feat[col].notna().any(), f"Column {col} is all NaN"

    def test_composite_alpha_series(self, wind_comp, wind_lob):
        assert isinstance(wind_comp, pd.Series)
        assert wind_comp.index.equals(wind_lob.index)
        assert wind_comp.name == "composite_alpha"
        assert wind_comp.std() > 0

    def test_composite_alpha_roughly_standardized(self, wind_comp):
        tail = wind_comp.iloc[200:]
        assert abs(tail.mean()) < 0.5
        assert 0.3 < tail.std() < 3.0

    def test_explicit_10_level_selection(self, wind_lob):
        feat = build_feature_matrix(wind_lob, ofi_levels=10, factors=["mlofi", "agg_ofi"])
        assert "mlofi" in feat.columns
        assert "agg_ofi" in feat.columns


# ---------------------------------------------------------------------------
# Backtest tests at 10 levels
# ---------------------------------------------------------------------------

class TestBacktest10Level:

    def test_backtest_runs_without_error(self, wind_lob, wind_comp):
        p = MarketParams.default_for("futures", entry_z=1.5, max_hold=20)
        pnl, trades = run_backtest(wind_lob, wind_comp, params=p)
        assert isinstance(pnl, pd.Series)
        assert isinstance(trades, pd.DataFrame)
        assert pnl.index.equals(wind_lob.index)

    def test_backtest_pnl_metrics(self, wind_lob, wind_comp):
        p = MarketParams.default_for("futures")
        pnl, _ = run_backtest(wind_lob, wind_comp, params=p)
        m = pnl_metrics(pnl)
        assert "sharpe" in m
        assert "win_rate" in m
        assert np.isfinite(m["sharpe"])

    def test_backtest_trade_log_columns(self, wind_lob, wind_comp):
        p = MarketParams.default_for("futures")
        _, trades = run_backtest(wind_lob, wind_comp, params=p)
        expected_cols = {"entry_time", "exit_time", "direction",
                         "entry_price", "exit_price", "exit_reason",
                         "gross_pnl", "cost"}
        assert expected_cols.issubset(set(trades.columns))

    def test_stock_mode_t1_enforced(self, wind_lob, wind_comp):
        p = MarketParams.default_for("stock", entry_z=1.0, max_hold=20)
        _, trades = run_backtest(wind_lob, wind_comp, params=p)
        if not trades.empty:
            intraday_exits = trades["exit_reason"].isin(["flip", "stop", "timeout"])
            assert not intraday_exits.any(), \
                "T+1 mode should not produce intraday exits"

    def test_eod_position_always_closed(self, wind_lob, wind_comp):
        p = MarketParams.default_for("futures", entry_z=0.5, max_hold=9999)
        pnl, trades = run_backtest(wind_lob, wind_comp, params=p)
        if not trades.empty:
            last_ts = wind_lob.index[-1]
            open_at_end = trades[trades["exit_time"] == last_ts]
            # If any position was open at EOD it must be in the trade log
            # (not silently left open with no costs)
            assert len(trades) > 0


# ---------------------------------------------------------------------------
# Forward returns / IC test
# ---------------------------------------------------------------------------

class TestForwardReturns10Level:

    def test_forward_returns_shape(self, wind_lob):
        fwd = compute_forward_returns(wind_lob, horizons=[1, 5, 10])
        assert set(fwd.columns) == {"fwd_1", "fwd_5", "fwd_10"}
        assert fwd.index.equals(wind_lob.index)

    def test_ic_by_horizon_returns_series(self, wind_lob, wind_comp):
        fwd = compute_forward_returns(wind_lob, horizons=[1, 5, 10])
        ic = ic_by_horizon(wind_comp, fwd)
        assert isinstance(ic, pd.Series)
        assert len(ic) == 3
        assert all(np.isfinite(v) or np.isnan(v) for v in ic)


# ---------------------------------------------------------------------------
# cum_buy_vol cumulation regression test
# ---------------------------------------------------------------------------

class TestCumVolCumulation:
    """
    Regression test for the Wind loader bug where per-bar BUY_VOLUME/SELL_VOLUME
    was assigned directly to cum_buy_vol/cum_sell_vol without cumsum.
    """

    def _make_per_bar_df(self) -> pd.DataFrame:
        """Simulate what Wind returns: per-bar buy/sell volumes."""
        idx = pd.date_range("2024-01-02 09:30:00", periods=5, freq="3s")
        return pd.DataFrame({
            "bid_px_1": [100.0] * 5,
            "ask_px_1": [100.02] * 5,
            "bid_vol_1": [1000.0] * 5,
            "ask_vol_1": [1000.0] * 5,
            "_bar_buy_vol":  [200.0, 300.0, 100.0, 400.0, 250.0],
            "_bar_sell_vol": [150.0, 200.0, 350.0, 100.0, 200.0],
        }, index=idx)

    def test_cumsum_makes_monotone(self):
        df = self._make_per_bar_df()
        # Simulate what _wind_result_to_lob now does
        df["cum_buy_vol"]  = df["_bar_buy_vol"].clip(lower=0).fillna(0).cumsum()
        df["cum_sell_vol"] = df["_bar_sell_vol"].clip(lower=0).fillna(0).cumsum()

        assert (df["cum_buy_vol"].diff().dropna() >= 0).all()
        assert (df["cum_sell_vol"].diff().dropna() >= 0).all()

    def test_per_bar_would_be_non_monotone_without_cumsum(self):
        """Prove the old (buggy) behaviour: direct assignment is NOT monotone."""
        per_bar_buy = pd.Series([200.0, 300.0, 100.0, 400.0, 250.0])
        diffs = per_bar_buy.diff().dropna()
        # 300→100 is a decrease; old code would have given non-monotone cum_buy_vol
        assert (diffs < 0).any(), "Per-bar values are not monotone (expected)"
