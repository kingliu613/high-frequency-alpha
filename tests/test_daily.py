import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import simulate_lob_day
from src.signals.composite import build_feature_matrix
from src.signals.daily import (
    aggregate_daily_factors,
    build_panel,
    cross_sectional_rank_ic,
    quantile_longshort,
)


@pytest.fixture(scope="module")
def one_day():
    lob = simulate_lob_day(seed=42, is_futures=False, prev_close=100.0)
    feat = build_feature_matrix(lob, auction_value=0.1, close_auction_value=0.2,
                                prev_close=100.0, instrument="stock")
    return lob, feat


def test_aggregate_produces_mean_and_tail(one_day):
    lob, feat = one_day
    row = aggregate_daily_factors(feat, lob, auction_value=0.1,
                                  close_auction_value=0.2)
    assert "api" in row and "api_tail" in row
    assert "big_flow" in row and "big_flow_tail" in row
    assert row["auction_imb"] == 0.1
    assert row["close_auction_imb"] == 0.2
    assert row["day_rv"] >= 0.0
    assert "sealing_max" in row
    assert row["open"] > 0 and row["close"] > 0
    # decay-shape columns excluded
    assert "auction_signal" not in row and "close_auction" not in row


def test_aggregate_all_finite(one_day):
    lob, feat = one_day
    row = aggregate_daily_factors(feat, lob)
    for k, v in row.items():
        assert np.isfinite(v), f"{k} not finite"


def _toy_panel(n_dates=12, n_stocks=10, perfect=True, seed=0):
    """Panel where factor 'f' perfectly (or not at all) ranks next-day ret_cc."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = f"2024-02-{d+1:02d}"
        for s in range(n_stocks):
            f_val = float(s)
            ret = float(s) if perfect else float(rng.normal())
            rows.append({"date": date, "ticker": f"s{s}",
                         "open": 10.0, "close": 10.0,
                         "f": f_val, "_ret": ret})
    panel = pd.DataFrame(rows).set_index(["date", "ticker"])
    panel["ret_cc"] = panel["_ret"]
    panel["ret_oo"] = panel["_ret"]
    return panel


def test_rank_ic_perfect_factor_is_one():
    panel = _toy_panel(perfect=True)
    ic = cross_sectional_rank_ic(panel, ["f"], ret_col="ret_cc")
    assert ic.loc["f", "mean_ic"] == pytest.approx(1.0)
    assert ic.loc["f", "n_days"] == 12


def test_rank_ic_random_factor_near_zero():
    panel = _toy_panel(perfect=False)
    ic = cross_sectional_rank_ic(panel, ["f"], ret_col="ret_cc")
    assert abs(ic.loc["f", "mean_ic"]) < 0.5   # noise band for 10 names


def test_quantile_longshort_positive_for_perfect_factor():
    panel = _toy_panel(perfect=True)
    ls = quantile_longshort(panel, "f", ret_col="ret_oo", n_quantiles=5)
    assert (ls > 0).all()
    lo = quantile_longshort(panel, "f", ret_col="ret_oo", n_quantiles=5,
                            long_only=True)
    assert (lo > 0).all()
    assert lo.mean() < ls.mean()   # excess-vs-mean < top-minus-bottom


def test_build_panel_return_alignment():
    """ret_cc(t) must equal close(t+1)/close(t)-1; ret_oo(t) = open(t+2)/open(t+1)-1."""
    rows = []
    closes = [10.0, 11.0, 12.1, 13.31]
    opens  = [9.0, 10.0, 11.0, 12.0]
    for i, (o, c) in enumerate(zip(opens, closes)):
        rows.append({"date": f"2024-03-{i+1:02d}", "ticker": "sX",
                     "open": o, "close": c, "f": 0.0})
    panel = build_panel(rows)

    r = panel.xs("sX", level="ticker")
    assert r["ret_cc"].iloc[0] == pytest.approx(11.0 / 10.0 - 1.0)
    assert r["ret_oo"].iloc[0] == pytest.approx(11.0 / 10.0 - 1.0)  # open(t+2)/open(t+1)
    assert np.isnan(r["ret_cc"].iloc[-1])
    assert np.isnan(r["ret_oo"].iloc[-2])
