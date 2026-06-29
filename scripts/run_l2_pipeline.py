"""
Real L2+L3 pipeline runner.

Usage:
    python3 scripts/run_l2_pipeline.py --date 20250102 --code 1
    python3 scripts/run_l2_pipeline.py --all-dates --code 1
    python3 scripts/run_l2_pipeline.py --date 20250102 --code 600000

Output:
    - Schema validation report
    - Feature matrix columns available
    - IC by horizon (1, 5, 10, 20 ticks)
    - Composite alpha backtest summary
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import pandas as pd

from src.data.l2_loader import load_l2_day, list_available_dates
from src.data.validator import validate_lob_schema, print_validation_report
from src.signals.composite import build_feature_matrix, build_composite_alpha
from src.backtest.metrics import compute_forward_returns, ic_by_horizon, pnl_metrics
from src.backtest.engine import run_backtest, MarketParams


DATA_DIR = "data"


def run_one_day(date: str, code: int, verbose: bool = True) -> dict:
    print(f"\n{'='*60}")
    print(f"  Date: {date}  Code: {code:06d}")
    print(f"{'='*60}")

    # Load
    print("Loading L2+L3 data...")
    df = load_l2_day(DATA_DIR, date, code, session="continuous", include_trans=True)
    print(f"  Loaded {len(df)} ticks  |  {df.index[0]} → {df.index[-1]}")

    # Validate
    report = validate_lob_schema(df)
    if verbose:
        print_validation_report(report)
    else:
        print(f"  Runnable alpha: {report.runnable_alpha}")

    if not report.runnable_alpha:
        print("  No alpha signals available — skipping.")
        return {}

    # Feature matrix
    feat = build_feature_matrix(df, ofi_levels=10)
    print(f"\nFeature matrix columns: {list(feat.columns)}")

    # Forward returns
    fwd = compute_forward_returns(df, horizons=[1, 5, 10, 20])

    # Per-signal IC
    print("\nPer-signal IC by horizon:")
    print(f"  {'Signal':<22} {'IC@1':>8} {'IC@5':>8} {'IC@10':>8} {'IC@20':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for col in feat.columns:
        ics = ic_by_horizon(feat[col], fwd)
        vals = [f"{v:8.4f}" if np.isfinite(v) else "     nan" for v in ics.values]
        print(f"  {col:<22} {'  '.join(vals)}")

    # Composite IC
    alpha = build_composite_alpha(feat)
    composite_ic = ic_by_horizon(alpha, fwd)
    print(f"\nComposite IC:  " + "  ".join(f"@{h}={v:.4f}" for h, v in zip([1,5,10,20], composite_ic.values)))

    # Backtest
    params = MarketParams.default_for("stock", entry_z=1.5, max_hold=20)
    pnl, trades = run_backtest(df, alpha, params=params)
    m = pnl_metrics(pnl)
    print(f"\nBacktest:  Sharpe={m['sharpe']:.3f}  WinRate={m['win_rate']:.1%}  Trades={len(trades)}")

    return {
        "date": date,
        "code": code,
        "n_ticks": len(df),
        "signals": list(feat.columns),
        "composite_ic": composite_ic.to_dict(),
        "sharpe": m["sharpe"],
        "n_trades": len(trades),
    }


def main():
    parser = argparse.ArgumentParser(description="Run L2+L3 real data pipeline")
    parser.add_argument("--date",      type=str, help="Trading date YYYYMMDD")
    parser.add_argument("--all-dates", action="store_true", help="Run all available dates")
    parser.add_argument("--code",      type=int, default=1, help="Stock code (default: 1 = 000001.SZ)")
    parser.add_argument("--quiet",     action="store_true", help="Suppress schema validation report")
    args = parser.parse_args()

    if args.all_dates:
        dates = list_available_dates(DATA_DIR)
        print(f"Found {len(dates)} dates: {dates[0]} → {dates[-1]}")
    elif args.date:
        dates = [args.date]
    else:
        parser.error("Provide --date YYYYMMDD or --all-dates")

    results = []
    for date in dates:
        try:
            r = run_one_day(date, args.code, verbose=not args.quiet)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ERROR {date}: {e}")

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  Multi-day summary ({len(results)} days, code={args.code:06d})")
        print(f"{'='*60}")
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        for h in [1, 5, 10, 20]:
            key = f"fwd_{h}"
            avg_ic = np.nanmean([r["composite_ic"].get(key, np.nan) for r in results])
            print(f"  Composite IC@{h:2d}: {avg_ic:.4f}")
        print(f"  Avg Sharpe:     {avg_sharpe:.3f}")


if __name__ == "__main__":
    main()
