"""
Integration tests for the L2+L3 real data loader.

These tests read actual parquet files from data/20250102/.
Skip automatically if the data directory is absent (CI without data).

Run:
    pytest tests/test_l2_loader.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_DATE = "20250102"
TEST_CODE = 1   # 000001.SZ (平安银行)

# Skip entire module if data not available
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATA_DIR, TEST_DATE)),
    reason="Real data not available (data/20250102/ missing)",
)


@pytest.fixture(scope="module")
def lob():
    from src.data.l2_loader import load_l2_day
    return load_l2_day(DATA_DIR, TEST_DATE, TEST_CODE, session="continuous", include_trans=True)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestSchema:

    def test_returns_dataframe(self, lob):
        assert isinstance(lob, pd.DataFrame)
        assert len(lob) > 100

    def test_datetime_index(self, lob):
        assert isinstance(lob.index, pd.DatetimeIndex)
        assert str(lob.index[0].date()) == "2025-01-02"

    def test_lob_columns_present(self, lob):
        for lv in range(1, 11):
            assert f"bid_px_{lv}" in lob.columns,  f"missing bid_px_{lv}"
            assert f"bid_vol_{lv}" in lob.columns, f"missing bid_vol_{lv}"
            assert f"ask_px_{lv}" in lob.columns,  f"missing ask_px_{lv}"
            assert f"ask_vol_{lv}" in lob.columns, f"missing ask_vol_{lv}"

    def test_event_columns_present(self, lob):
        for col in ["limit_buy_vol", "limit_sell_vol",
                    "cancel_buy_vol", "cancel_sell_vol",
                    "market_buy_vol", "market_sell_vol"]:
            assert col in lob.columns, f"missing {col}"

    def test_cumulative_columns_present(self, lob):
        for col in ["cum_buy_vol", "cum_sell_vol", "cum_buy_count", "cum_sell_count"]:
            assert col in lob.columns, f"missing {col}"

    def test_depth_columns_present(self, lob):
        assert "bid_depth" in lob.columns
        assert "ask_depth" in lob.columns

    def test_no_internal_columns(self, lob):
        internal = [c for c in lob.columns if c.startswith("_")]
        assert internal == [], f"internal columns leaked: {internal}"

    def test_no_raw_id_columns(self, lob):
        for col in ["code", "date", "time"]:
            assert col not in lob.columns


# ---------------------------------------------------------------------------
# Data quality tests
# ---------------------------------------------------------------------------

class TestDataQuality:

    def test_session_in_continuous_hours(self, lob):
        hours = lob.index.hour + lob.index.minute / 60
        assert hours.min() >= 9.5 - 0.01
        assert hours.max() <= 15.0 + 0.01

    def test_bid_px1_positive(self, lob):
        valid = lob["bid_px_1"].replace(0.0, np.nan).dropna()
        assert (valid > 0).all()

    def test_spread_nonnegative(self, lob):
        spread = lob["ask_px_1"] - lob["bid_px_1"]
        # Allow zero spread (crossed book at auction edges) but not negative
        assert (spread.replace(0.0, np.nan).dropna() >= 0).all()

    def test_event_columns_nonnegative(self, lob):
        for col in ["limit_buy_vol", "limit_sell_vol",
                    "cancel_buy_vol", "cancel_sell_vol",
                    "market_buy_vol", "market_sell_vol"]:
            assert (lob[col] >= 0).all(), f"{col} has negative values"

    def test_event_volumes_nonzero_total(self, lob):
        total_limit = lob["limit_buy_vol"].sum() + lob["limit_sell_vol"].sum()
        total_cancel = lob["cancel_buy_vol"].sum() + lob["cancel_sell_vol"].sum()
        assert total_limit > 0, "No limit order volume recorded"
        assert total_cancel > 0, "No cancel volume recorded"

    def test_cum_buy_vol_monotone(self, lob):
        diffs = lob["cum_buy_vol"].diff().dropna()
        assert (diffs >= 0).all(), "cum_buy_vol is not non-decreasing"

    def test_cum_sell_vol_monotone(self, lob):
        diffs = lob["cum_sell_vol"].diff().dropna()
        assert (diffs >= 0).all(), "cum_sell_vol is not non-decreasing"

    def test_depth_positive(self, lob):
        assert (lob["bid_depth"] >= 0).all()
        assert (lob["ask_depth"] >= 0).all()

    def test_no_all_nan_columns(self, lob):
        for col in lob.columns:
            assert lob[col].notna().any(), f"{col} is all NaN"


# ---------------------------------------------------------------------------
# Pipeline integration test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:

    def test_validator_reports_alpha_signals(self, lob):
        from src.data.validator import validate_lob_schema
        report = validate_lob_schema(lob)
        assert len(report.runnable_alpha) >= 2, (
            f"Expected ≥2 runnable alpha signals, got: {report.runnable_alpha}"
        )
        assert "mlofi" in report.runnable_alpha
        assert "agg_ofi" in report.runnable_alpha

    def test_build_feature_matrix(self, lob):
        from src.signals.composite import build_feature_matrix
        feat = build_feature_matrix(lob, ofi_levels=10)
        assert isinstance(feat, pd.DataFrame)
        assert len(feat) == len(lob)
        assert "mlofi" in feat.columns

    def test_composite_alpha_finite(self, lob):
        from src.signals.composite import build_feature_matrix, build_composite_alpha
        feat = build_feature_matrix(lob, ofi_levels=10)
        alpha = build_composite_alpha(feat)
        assert np.isfinite(alpha.to_numpy()).all()
        assert alpha.std() > 0

    def test_forward_returns_shape(self, lob):
        from src.backtest.metrics import compute_forward_returns
        fwd = compute_forward_returns(lob, horizons=[1, 5, 10])
        assert set(fwd.columns) == {"fwd_1", "fwd_5", "fwd_10"}
        assert len(fwd) == len(lob)

    def test_ic_finite(self, lob):
        from src.signals.composite import build_feature_matrix, build_composite_alpha
        from src.backtest.metrics import compute_forward_returns, ic_by_horizon
        feat  = build_feature_matrix(lob, ofi_levels=10)
        alpha = build_composite_alpha(feat)
        fwd   = compute_forward_returns(lob, horizons=[1, 5])
        ic    = ic_by_horizon(alpha, fwd)
        assert len(ic) == 2
        # IC may be NaN at short horizons if variance is zero; just check no crash
        assert all(np.isnan(v) or np.isfinite(v) for v in ic.values)


# ---------------------------------------------------------------------------
# Session filter test
# ---------------------------------------------------------------------------

def test_auction_session_loads():
    from src.data.l2_loader import load_l2_day
    if not os.path.isdir(os.path.join(DATA_DIR, TEST_DATE)):
        pytest.skip("data not available")
    df = load_l2_day(DATA_DIR, TEST_DATE, TEST_CODE, session="auction")
    assert len(df) > 0
    hours = df.index.hour + df.index.minute / 60
    assert hours.max() < 9.5, "auction session contains post-9:30 ticks"


def test_list_available_dates():
    from src.data.l2_loader import list_available_dates
    dates = list_available_dates(DATA_DIR)
    assert len(dates) > 0
    assert all(d.isdigit() and len(d) == 8 for d in dates)
    assert dates == sorted(dates)
