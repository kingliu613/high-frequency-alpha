#!/usr/bin/env python3
"""
Real-data research using free sources (AKShare / BaoStock).

AKShare path  — needs trading hours, gives 5-level LOB:
    python scripts/run_real_data.py --source akshare --ticker 000001

BaoStock path — full history, 5-min bars only (no LOB):
    python scripts/run_real_data.py --source baostock --ticker sh.600519

Install deps first:
    pip install akshare baostock
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings; warnings.filterwarnings("ignore")
import pandas as pd

from src.signals.composite import build_feature_matrix, build_composite_alpha
from src.backtest.metrics  import (
    compute_forward_returns, ic_by_horizon, signal_decay_table, pnl_metrics
)
from src.backtest.engine   import run_backtest, MarketParams


def run_akshare(ticker: str) -> None:
    from src.data.loader import load_tick_day_akshare

    print(f"\nLoading AKShare tick data for {ticker} …")
    lob_df = load_tick_day_akshare(ticker)
    print(f"  {len(lob_df):,} snapshots loaded")

    # AKShare tick data has trade-side counts but no real order-event stream.
    # Strict mode therefore uses only the paper polarity factor here.
    feat = build_feature_matrix(
        lob_df,
        ofi_levels=1,
        ofi_window=10,
        factors=["trade_imbalance"],
    )
    comp = build_composite_alpha(feat)

    fwd  = compute_forward_returns(lob_df, horizons=[1, 5, 10, 20, 40])
    ic   = ic_by_horizon(comp, fwd)

    print("\nComposite IC by horizon:")
    for h, v in ic.items():
        print(f"  {h:>4} ticks ({h*3:>4}s): {v:+.4f}")

    decay = signal_decay_table(comp, lob_df, max_ticks=80, step=4)
    if not decay.empty:
        peak = decay.loc[decay["IC"].abs().idxmax()]
        print(f"\n  Peak IC @ {peak['seconds']:.0f}s: {peak['IC']:.4f}")

    p   = MarketParams(instrument="stock", entry_z=1.5, max_hold=20)
    pnl, trades = run_backtest(lob_df, comp, params=p)
    m = pnl_metrics(pnl)
    print(f"\n  Sharpe: {m['sharpe']:.2f}   Win rate: {m['win_rate']:.2%}   "
          f"Trades: {m['n_nonzero']}")


def run_baostock(ticker: str, start: str, end: str) -> None:
    from src.data.loader import load_minute_baostock

    print(f"\nLoading BaoStock 5-min bars: {ticker}  {start}→{end} …")
    df = load_minute_baostock(ticker, start_date=start, end_date=end, freq="5")
    if df.empty:
        print("  No data returned — check ticker format (sh.XXXXXX / sz.XXXXXX)")
        return
    print(f"  {len(df):,} bars loaded")

    print("  No strict intraday paper factors are available on 5-min OHLCV bars.")
    print("  Use tick/order-event data for OFI, trade polarity, API, OEI, VPIN, and Kyle lambda.")
    return



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  choices=["akshare", "baostock"], default="akshare")
    ap.add_argument("--ticker",  default="000001")
    ap.add_argument("--start",   default="2024-01-01")
    ap.add_argument("--end",     default="2024-03-31")
    args = ap.parse_args()

    if args.source == "akshare":
        run_akshare(args.ticker)
    else:
        run_baostock(args.ticker, args.start, args.end)


if __name__ == "__main__":
    main()
