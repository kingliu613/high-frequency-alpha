#!/usr/bin/env python3
"""
高频因子日频化 · 截面选股评估 — Cross-Sectional Daily Factor Pipeline

Usage:
    python scripts/run_cross_sectional.py                     # 10 stocks × 8 days demo
    python scripts/run_cross_sectional.py --stocks 30 --days 20

Per (stock, day): simulate L2 → tick factors → daily aggregates.
Then: cross-sectional RankIC vs next-day return, ICIR, t-stat,
top-quintile long-only excess return (T+1-executable open-to-open).

HONESTY NOTE: the synthetic generator plants only INTRADAY flow→price
coupling; it embeds no cross-sectional next-day alpha. Expected RankIC here
is ≈ 0 — that is correct behaviour, and it makes this script a true null
pipeline test. Real cross-sectional IC requires real data (20+ days ×
30–50 names minimum; see research/hft_factors_china.md).

Swap simulate-loop for a real loader: for each (date, ticker) provide
lob_df + auction values, everything downstream is unchanged.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from src.data.synthetic import (
    simulate_lob_day,
    simulate_auction_data,
    simulate_close_auction_data,
)
from src.signals.auction import auction_composite, close_auction_imbalance
from src.signals.composite import build_feature_matrix
from src.signals.daily import (
    aggregate_daily_factors,
    build_panel,
    cross_sectional_rank_ic,
    quantile_longshort,
)

SEP = "=" * 64


def simulate_universe(n_stocks: int, n_days: int, signal_strength: float = 0.01) -> pd.DataFrame:
    """
    Simulate a stock universe: per-stock price chains across days
    (today's close = tomorrow's prev_close), daily factor aggregation.
    """
    dates = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    rows = []

    prev_closes = {s: 10.0 + 5.0 * (s % 7) for s in range(n_stocks)}

    total = n_stocks * n_days
    done = 0
    for s in range(n_stocks):
        ticker = f"{600000 + s}.SH"
        for d_i, date in enumerate(dates):
            seed = s * 1009 + d_i          # distinct, deterministic
            pc   = prev_closes[s]

            auction_df, open_px = simulate_auction_data(
                ticker=ticker, date=date, prev_close=pc, seed=seed)
            lob_df = simulate_lob_day(
                ticker=ticker, date=date, prev_close=open_px,
                is_futures=False, seed=seed, signal_strength=signal_strength)
            auc_val = auction_composite(auction_df, open_px)

            close_df, _ = simulate_close_auction_data(
                ticker=ticker, date=date,
                day_close=float((lob_df["bid_px_1"].iloc[-1] + lob_df["ask_px_1"].iloc[-1]) / 2),
                seed=seed)
            close_val = close_auction_imbalance(close_df)

            feat_df = build_feature_matrix(
                lob_df,
                auction_value=auc_val,
                close_auction_value=close_val,
                ofi_levels=5,
                prev_close=pc,
                instrument="stock",
            )

            row = aggregate_daily_factors(
                feat_df, lob_df,
                auction_value=auc_val,
                close_auction_value=close_val,
            )
            row["date"]   = date
            row["ticker"] = ticker
            rows.append(row)

            prev_closes[s] = row["close"]
            done += 1
            if done % 20 == 0:
                print(f"  simulated {done}/{total} stock-days …")

    return build_panel(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", type=int, default=10)
    ap.add_argument("--days",   type=int, default=8)
    ap.add_argument("--signal-strength", type=float, default=0.01)
    ap.add_argument("--quantiles", type=int, default=5)
    args = ap.parse_args()

    print(f"\n{SEP}\n  高频因子日频化 · 截面评估  "
          f"({args.stocks} stocks × {args.days} days)\n{SEP}")

    panel = simulate_universe(args.stocks, args.days, args.signal_strength)

    factor_cols = [c for c in panel.columns
                   if c not in ("open", "close", "ret_cc", "ret_oo")]

    print(f"\nPanel: {len(panel)} stock-days, {len(factor_cols)} daily factors")

    # ── RankIC (IC convention: close→close) ─────────────────────────────
    ic = cross_sectional_rank_ic(panel, factor_cols, ret_col="ret_cc")
    print(f"\n{SEP}\n  Cross-sectional RankIC vs next-day return (close→close)\n{SEP}")
    print("  (|mean_ic|>0.03 usable, >0.05 strong; |t|≥2 minimum credibility)\n")
    with pd.option_context("display.float_format", "{:+.4f}".format):
        print(ic.head(15).to_string())

    # ── Top-quintile long-only excess (executable open→open, T+1) ───────
    print(f"\n{SEP}\n  Top-quintile long-only excess return "
          f"(executable: open(t+1)→open(t+2))\n{SEP}")
    best = ic.dropna().head(5).index.tolist()
    for f in best:
        lo = quantile_longshort(panel, f, ret_col="ret_oo",
                                n_quantiles=args.quantiles, long_only=True)
        if len(lo) < 2 or lo.std() == 0:
            print(f"  {f:<28} insufficient data")
            continue
        ann_sh = np.sqrt(252) * lo.mean() / lo.std()
        print(f"  {f:<28} mean={lo.mean()*1e4:+7.1f} bp/d   "
              f"ann_sharpe={ann_sh:+5.2f}   n={len(lo)}")

    print(f"\n{SEP}\n  Notes\n{SEP}")
    print("  - Synthetic data embeds NO cross-sectional next-day alpha;")
    print("    RankIC ≈ 0 here is the CORRECT null result (pipeline test).")
    print("  - Costs not subtracted (~10 bp/side daily turnover).")
    print("  - Real validation: 20+ days × 30–50 names of real L2 / tick data.")


if __name__ == "__main__":
    main()
