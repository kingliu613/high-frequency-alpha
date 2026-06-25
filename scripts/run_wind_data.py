#!/usr/bin/env python3
"""
Wind L2 Research Pipeline — end-to-end from Wind terminal to IC / backtest report.

Usage examples:
    # Single ticker, single day (futures)
    python scripts/run_wind_data.py --ticker IF2501.CFFEX --date 2024-01-02

    # A-share stock (T+1, 10% price limit)
    python scripts/run_wind_data.py --ticker 000001.SZ --date 2024-01-02 --instrument stock

    # Multiple dates — IC stability across days
    python scripts/run_wind_data.py --ticker IF2501.CFFEX --dates 2024-01-02,2024-01-03,2024-01-04

    # Force-refresh cache (re-fetch from Wind even if cached)
    python scripts/run_wind_data.py --ticker IF2501.CFFEX --date 2024-01-02 --refresh

    # Use 10 LOB levels for mlofi/agg_ofi (default: 5 to match tested range)
    python scripts/run_wind_data.py --ticker IF2501.CFFEX --date 2024-01-02 --levels 10

    # Night session for commodity futures (e.g. CU)
    python scripts/run_wind_data.py --ticker CU2501.SHFE --date 2024-01-02 --night

Pre-requisites:
    pip install WindPy pyarrow
    Wind terminal running + L2 data subscription active.

Cache:
    Fetched data is saved to ./data/wind_cache/ as Parquet files.
    Re-running the same ticker+date loads from disk, not Wind.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.data.cache         import load_or_fetch_wind, list_cached
from src.data.validator     import validate_lob_schema, print_validation_report
from src.signals.composite  import build_feature_matrix, build_composite_alpha
from src.signals.advanced   import vpin, exposure_gate
from src.backtest.metrics   import (
    compute_forward_returns, ic_by_horizon,
    rolling_ic, icir, signal_decay_table, pnl_metrics,
)
from src.backtest.engine    import run_backtest, MarketParams


SEP = "=" * 64


def hdr(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ---------------------------------------------------------------------------
# Single day pipeline
# ---------------------------------------------------------------------------

def run_one_day(
    ticker: str,
    date: str,
    instrument: str = "futures",
    ofi_levels: int = 5,
    include_night: bool = False,
    force_refresh: bool = False,
    prev_close: float | None = None,
) -> dict | None:
    """
    Full pipeline for one (ticker, date).
    Returns dict of summary metrics, or None if data unavailable.
    """
    hdr(f"{ticker}  {date}  [{instrument}]")

    # --- 1. Load ---
    print("Loading from cache / Wind ...")
    df = load_or_fetch_wind(
        ticker, date,
        include_night=include_night,
        force_refresh=force_refresh,
    )
    if df.empty:
        print("  No data returned — skipping.")
        return None
    print(f"  {len(df):,} snapshots loaded")

    # --- 2. Validate ---
    report = validate_lob_schema(df)
    print_validation_report(report, verbose=True)

    if not report.runnable_alpha:
        print("  No alpha signals available on this data — stopping.")
        return None

    # Clamp ofi_levels to what the data actually has
    actual_levels = min(ofi_levels, report.n_levels_complete)
    if actual_levels < ofi_levels:
        print(f"  ofi_levels clamped: requested {ofi_levels} but data only has "
              f"{report.n_levels_complete} complete levels → using {actual_levels}")

    # --- 3. Features ---
    hdr("Signal computation")
    feat = build_feature_matrix(
        df,
        ofi_levels=actual_levels,
        prev_close=prev_close,
        instrument=instrument,
    )
    print(f"  Features computed: {list(feat.columns)}")
    print(f"  NaN fraction per feature:")
    for col in feat.columns:
        frac = feat[col].isna().mean()
        if frac > 0:
            print(f"    {col}: {frac:.1%}")

    comp = build_composite_alpha(feat)
    print(f"\n  Composite alpha — mean={comp.mean():.4f}  std={comp.std():.4f}  "
          f"min={comp.min():.3f}  max={comp.max():.3f}")

    # --- 4. Exposure gate (VPIN + Kyle λ) ---
    gate_scale = None
    if "last_price" in df.columns and "last_volume" in df.columns:
        try:
            gate_feat = build_feature_matrix(df, factors=["vpin", "kyle_lambda"])
            gate_scale = exposure_gate(gate_feat)
            print(f"  Exposure gate — mean={gate_scale.mean():.3f}")
        except Exception as e:
            print(f"  Exposure gate skipped: {e}")

    # --- 5. IC decay ---
    hdr("IC decay curve")
    fwd = compute_forward_returns(df, horizons=[1, 3, 5, 10, 20, 40, 80, 200])
    ic_series = ic_by_horizon(comp, fwd)
    print(f"  {'Horizon':>8}  {'Seconds':>8}  {'IC':>10}")
    print(f"  {'-'*32}")
    for h, ic_val in ic_series.items():
        print(f"  {h:>8}  {h*3:>8}  {ic_val:>+10.4f}")

    peak_idx = ic_series.abs().idxmax()
    print(f"\n  Peak |IC| @ {peak_idx} ticks ({peak_idx*3}s): "
          f"{ic_series[peak_idx]:+.4f}")

    # Rolling IC stability
    ric = rolling_ic(comp, fwd, horizon=10, window=500)
    if len(ric) > 1:
        print(f"  Rolling IC (h=10, w=500) — mean={ric.mean():+.4f}  "
              f"std={ric.std():.4f}  ICIR={icir(ric):+.3f}")

    # --- 6. Backtest ---
    hdr("Backtest")
    p = MarketParams.default_for(instrument, entry_z=1.5, max_hold=20)
    pnl, trades = run_backtest(
        df, comp, params=p,
        exposure_scale=gate_scale,
        prev_close=prev_close,
    )
    m = pnl_metrics(pnl)

    print(f"  Sharpe      : {m['sharpe']:+.2f}")
    print(f"  Max drawdown: {m['max_drawdown']:+.1f} RMB")
    print(f"  Win rate    : {m['win_rate']:.1%}")
    print(f"  Trades      : {m['n_nonzero']}")
    print(f"  Total PnL   : {m['total_pnl']:+.2f} RMB  "
          f"(ann. rate {m['ann_return_rmb']:+.0f} RMB/yr)")

    if not trades.empty:
        print(f"\n  Exit reason breakdown:")
        for reason, cnt in trades["exit_reason"].value_counts().items():
            print(f"    {reason:<18}: {cnt}")

    return {
        "ticker":        ticker,
        "date":          date,
        "n_snapshots":   len(df),
        "n_alpha":       len(report.runnable_alpha),
        "ofi_levels":    actual_levels,
        "peak_ic_ticks": int(peak_idx),
        "peak_ic":       float(ic_series[peak_idx]),
        "sharpe":        m["sharpe"],
        "win_rate":      m["win_rate"],
        "total_pnl":     m["total_pnl"],
        "n_trades":      m["n_nonzero"],
    }


# ---------------------------------------------------------------------------
# Multi-day roll-up
# ---------------------------------------------------------------------------

def run_multi_day(
    ticker: str,
    dates: list[str],
    instrument: str = "futures",
    ofi_levels: int = 5,
    include_night: bool = False,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Loop over dates, collect per-day metrics, print summary table.
    """
    rows = []
    for date in dates:
        try:
            r = run_one_day(
                ticker, date, instrument=instrument,
                ofi_levels=ofi_levels, include_night=include_night,
                force_refresh=force_refresh,
            )
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  ERROR on {date}: {e}")

    if not rows:
        print("\nNo results to summarize.")
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    hdr("Multi-day summary")
    print(summary[["date", "n_snapshots", "ofi_levels",
                    "peak_ic", "sharpe", "win_rate", "n_trades"]].to_string(index=False))

    print(f"\n  Mean Sharpe  : {summary['sharpe'].mean():+.2f}")
    print(f"  Mean peak IC : {summary['peak_ic'].mean():+.4f}")
    print(f"  Days positive: {(summary['total_pnl'] > 0).sum()} / {len(summary)}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Wind L2 research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ticker",     required=True,
                    help="Wind ticker, e.g. IF2501.CFFEX or 000001.SZ")
    ap.add_argument("--date",       default=None,
                    help="Single trading day YYYY-MM-DD")
    ap.add_argument("--dates",      default=None,
                    help="Comma-separated list of dates for multi-day run")
    ap.add_argument("--instrument", choices=["futures", "stock"], default="futures")
    ap.add_argument("--levels",     type=int, default=5,
                    help="LOB depth levels for mlofi/agg_ofi (max 10)")
    ap.add_argument("--night",      action="store_true",
                    help="Include prior night session (commodity futures only)")
    ap.add_argument("--refresh",    action="store_true",
                    help="Ignore cache and re-fetch from Wind")
    ap.add_argument("--prev-close", type=float, default=None,
                    help="Previous close for price-limit gating (stocks)")
    ap.add_argument("--list-cache", action="store_true",
                    help="Print cached files and exit")

    args = ap.parse_args()

    if args.list_cache:
        entries = list_cached()
        if not entries:
            print("Cache is empty.")
        else:
            print(f"\n{'Ticker':<20} {'Date':<12} {'Size MB':>8}  Path")
            print("-" * 70)
            for e in entries:
                print(f"  {e['ticker']:<18} {e['date']:<12} {e['size_mb']:>7.2f}  {e['path']}")
        return

    if args.dates:
        dates = [d.strip() for d in args.dates.split(",")]
        run_multi_day(
            args.ticker, dates,
            instrument=args.instrument,
            ofi_levels=args.levels,
            include_night=args.night,
            force_refresh=args.refresh,
        )
    elif args.date:
        run_one_day(
            args.ticker, args.date,
            instrument=args.instrument,
            ofi_levels=args.levels,
            include_night=args.night,
            force_refresh=args.refresh,
            prev_close=args.prev_close,
        )
    else:
        ap.error("Provide --date or --dates.")


if __name__ == "__main__":
    main()
