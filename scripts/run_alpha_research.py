#!/usr/bin/env python3
"""
HFT Alpha Research Pipeline — Chinese Market

Usage:
    python scripts/run_alpha_research.py               # single day
    python scripts/run_alpha_research.py --multiday    # 20-day IC stability

Generates synthetic Chinese L2 data, computes all signals, runs
IC-decay analysis, and prints a full backtest summary.

Replace simulate_lob_day() / simulate_auction_data() with real
Tushare-Pro / Wind L2 data loaders when moving to production.
"""

import sys
import os
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from src.data.synthetic  import simulate_lob_day, simulate_auction_data, simulate_etf_series
from src.signals.auction import auction_composite
from src.signals.composite import build_feature_matrix, build_composite_alpha
from src.backtest.metrics import (
    compute_forward_returns,
    ic_by_horizon,
    rolling_ic,
    icir,
    signal_decay_table,
    pnl_metrics,
)
from src.backtest.engine import run_backtest, MarketParams


SEP = "=" * 64


def hdr(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ---------------------------------------------------------------------------
# Single-day pipeline
# ---------------------------------------------------------------------------

def run_day(
    date: str,
    ticker: str = "IF2401.CFFEX",
    is_futures: bool = True,
    seed: int = 42,
    verbose: bool = True,
    params: Optional[MarketParams] = None,
    use_etf: bool = False,
) -> dict:

    # --- Data ---
    auction_df, open_price = simulate_auction_data(
        ticker=ticker, date=date, seed=seed
    )
    lob_df = simulate_lob_day(
        ticker=ticker, date=date,
        prev_close=open_price,
        is_futures=is_futures,
        seed=seed,
    )
    auc_val = auction_composite(auction_df, open_price)

    if params is None:
        params = MarketParams(
            instrument = "futures" if is_futures else "stock",
            entry_z    = 1.5,
            exit_z     = 0.3,
            max_hold   = 20,
        )

    etf_series = simulate_etf_series(lob_df, seed=seed) if use_etf else None

    # --- Features ---
    feat_df = build_feature_matrix(
        lob_df,
        auction_value         = auc_val,
        ofi_levels            = 5,
        ofi_window            = 10,
        prev_close            = float(open_price) if not is_futures else None,
        instrument            = params.instrument,
        etf_series            = etf_series,
    )
    composite = build_composite_alpha(feat_df)

    # --- IC analysis ---
    fwd = compute_forward_returns(lob_df, horizons=[1, 5, 10, 20, 40, 100, 200])
    ic_all = ic_by_horizon(composite, fwd)
    decay  = signal_decay_table(composite, lob_df, max_ticks=200, step=5)

    # IC per feature
    feat_ics: dict[str, float] = {}
    for col in feat_df.columns:
        sig_c = feat_df[col]
        if sig_c.std() < 1e-10:
            continue
        ic_s = ic_by_horizon(sig_c, fwd)
        feat_ics[col] = float(ic_s.get(10, np.nan))

    # --- Backtest ---
    pnl, trades = run_backtest(lob_df, composite, params=params)
    m = pnl_metrics(pnl)

    if verbose:
        print(f"\nDate: {date}   Ticker: {ticker}   "
              f"{'Futures' if is_futures else 'Stock'}")
        print(f"Snapshots: {len(lob_df):,}   Auction signal: {auc_val:+.4f}")

        print("\nIndividual signal IC @ 10 ticks (30s):")
        for name, ic_val in sorted(feat_ics.items(), key=lambda x: -abs(x[1])):
            bar = "#" * int(abs(ic_val) * 200)
            print(f"  {name:<22} {ic_val:+.4f}  {bar}")

        print("\nComposite alpha IC by horizon:")
        for h in [1, 5, 10, 20, 40, 100, 200]:
            ic_v = ic_all.get(h, np.nan)
            bar  = "#" * int(abs(ic_v) * 200) if not np.isnan(ic_v) else ""
            print(f"  {h:>4} ticks  ({h*3:>5}s): {ic_v:+.4f}  {bar}")

        if not decay.empty:
            peak_idx = decay["IC"].abs().idxmax()
            peak_sec = decay.loc[peak_idx, "seconds"]
            peak_ic  = decay.loc[peak_idx, "IC"]
            print(f"\n  Peak IC @ {peak_sec:.0f}s: {peak_ic:.4f}")

        print("\nBacktest:")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k:<20} {v:>10.4f}")
            else:
                print(f"  {k:<20} {v:>10}")

        if len(trades):
            avg_hold = trades["hold_ticks"].mean() * 3
            print(f"\n  Avg hold: {avg_hold:.1f}s   "
                  f"Trades: {len(trades)}   "
                  f"Exit reasons: {trades['exit_reason'].value_counts().to_dict()}")

    return {
        "lob_df":    lob_df,
        "feat_df":   feat_df,
        "composite": composite,
        "fwd":       fwd,
        "ic_all":    ic_all,
        "decay":     decay,
        "feat_ics":  feat_ics,
        "pnl":       pnl,
        "trades":    trades,
        "metrics":   m,
    }


# ---------------------------------------------------------------------------
# Multi-day stability
# ---------------------------------------------------------------------------

def run_multiday(
    n_days: int = 20,
    ticker: str = "IF2401.CFFEX",
    is_futures: bool = True,
) -> pd.DataFrame:

    dates = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    rows  = []

    print(f"\nRunning {n_days} days …")
    for i, d in enumerate(dates):
        r   = run_day(d, ticker=ticker, is_futures=is_futures, seed=i, verbose=False)
        ic  = r["ic_all"]
        row = {"date": d, "daily_pnl": r["pnl"].sum()}
        for h in [1, 5, 10, 20, 40]:
            row[f"ic_{h}"] = float(ic.get(h, np.nan))
        rows.append(row)
        print(f"  {d}  IC@10={row['ic_10']:+.4f}  PnL={row['daily_pnl']:+.1f}")

    df = pd.DataFrame(rows)

    hdr("Multi-Day IC Stability")
    print(df.round(4).to_string(index=False))

    print("\nSummary (IC @ 10 ticks = 30s):")
    for col in [c for c in df.columns if c.startswith("ic_")]:
        mean = df[col].mean()
        std  = df[col].std()
        ir   = mean / std if std > 0 else np.nan
        h    = col.split("_")[1]
        print(f"  h={h:>3} ticks: mean={mean:+.4f}  std={std:.4f}  ICIR={ir:.3f}")

    pnl_all = df["daily_pnl"]
    print(f"\n  Daily PnL: mean={pnl_all.mean():+.2f}  "
          f"std={pnl_all.std():.2f}  "
          f"Sharpe={np.sqrt(252)*pnl_all.mean()/pnl_all.std():.2f}")

    return df


# ---------------------------------------------------------------------------
# Walk-forward optimization
# ---------------------------------------------------------------------------

def run_walkforward(
    n_days:     int  = 20,
    ticker:     str  = "IF2401.CFFEX",
    is_futures: bool = True,
    is_window:  int  = 10,
    oos_window: int  = 2,
) -> pd.DataFrame:
    """
    Walk-forward parameter optimization.

    Grid-searches entry_z × max_hold on IS window, applies best params
    to next OOS window, rolls forward. Reports OOS-only metrics.
    """
    entry_z_grid  = [0.8, 1.0, 1.2, 1.5, 2.0]
    max_hold_grid = [10, 15, 20, 30, 40]

    dates    = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    date_idx = {d: i for i, d in enumerate(dates)}
    oos_rows = []

    print(f"\n  Grid: entry_z×{entry_z_grid}  max_hold×{max_hold_grid}  "
          f"({len(entry_z_grid)*len(max_hold_grid)} combos per IS window)")

    for oos_start in range(is_window, n_days, oos_window):
        is_dates  = dates[oos_start - is_window : oos_start]
        oos_dates = dates[oos_start : oos_start + oos_window]
        if not oos_dates:
            break

        # ── IS grid search ──────────────────────────────────────────────
        best_sharpe = -np.inf
        best_ez, best_mh = 1.5, 20

        for ez in entry_z_grid:
            for mh in max_hold_grid:
                p = MarketParams(
                    instrument        = "futures" if is_futures else "stock",
                    entry_z           = ez,
                    exit_z            = 0.3,
                    max_hold          = mh,
                    use_regime_filter = False,   # skip for speed during grid search
                )
                daily_pnls = []
                for d in is_dates:
                    r = run_day(d, ticker=ticker, is_futures=is_futures,
                                seed=date_idx[d], verbose=False, params=p)
                    daily_pnls.append(float(r["pnl"].sum()))
                s = pd.Series(daily_pnls)
                sh = float(np.sqrt(252) * s.mean() / s.std()) if s.std() > 0 else 0.0
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_ez, best_mh = ez, mh

        # ── OOS evaluation ───────────────────────────────────────────────
        p_oos = MarketParams(
            instrument = "futures" if is_futures else "stock",
            entry_z    = best_ez,
            exit_z     = 0.3,
            max_hold   = best_mh,
        )
        oos_pnls, oos_ics = [], []
        for d in oos_dates:
            r = run_day(d, ticker=ticker, is_futures=is_futures,
                        seed=date_idx[d], verbose=False, params=p_oos)
            oos_pnls.append(float(r["pnl"].sum()))
            oos_ics.append(float(r["ic_all"].get(10, np.nan)))

        row = {
            "oos_start":   oos_dates[0],
            "oos_end":     oos_dates[-1],
            "entry_z":     best_ez,
            "max_hold":    best_mh,
            "is_sharpe":   round(best_sharpe, 2),
            "oos_pnl":     round(sum(oos_pnls), 1),
            "oos_ic_mean": round(float(np.nanmean(oos_ics)), 4),
        }
        oos_rows.append(row)
        print(f"  OOS {row['oos_start']}–{row['oos_end']}  "
              f"params=(ez={best_ez}, mh={best_mh})  "
              f"IS_sh={best_sharpe:+.2f}  OOS_pnl={row['oos_pnl']:+.0f}  "
              f"IC@10={row['oos_ic_mean']:+.4f}")

    df = pd.DataFrame(oos_rows)

    hdr("Walk-Forward Summary  (IS=10d  OOS=2d)")
    print(df.to_string(index=False))

    all_pnl    = pd.Series([r["oos_pnl"] for r in oos_rows])
    oos_sharpe = float(np.sqrt(252) * all_pnl.mean() / all_pnl.std()) if all_pnl.std() > 0 else 0.0
    n_prof     = int((all_pnl > 0).sum())
    mean_ic    = float(np.nanmean([r["oos_ic_mean"] for r in oos_rows]))

    print(f"\n  OOS-concatenated Sharpe : {oos_sharpe:+.2f}")
    print(f"  Profitable OOS windows  : {n_prof}/{len(oos_rows)}")
    print(f"  Mean OOS IC@10          : {mean_ic:+.4f}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiday", action="store_true")
    ap.add_argument("--stock",    action="store_true", help="use stock mode (long-only T+1)")
    ap.add_argument("--date",     default="2024-01-02")
    ap.add_argument("--ticker",   default="IF2401.CFFEX")
    ap.add_argument("--days",        type=int, default=20)
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--etf",         action="store_true", help="include ETF basis signal")
    args = ap.parse_args()

    is_futures = not args.stock
    ticker     = args.ticker if is_futures else "600519.SH"  # Moutai example

    hdr("HFT Alpha Research — Chinese Market")
    print("  Instrument : " + ("CSI 300 Futures (IF) on CFFEX" if is_futures
                                else "A-share (long-only, T+1)"))
    print("  Signals    : Multi-Level OFI · Auction Imbalance · Micro-Price")
    print("               Queue Imbalance · Depth Tilt · Momentum")
    print("  Data       : Synthetic Chinese L2 (3-second, 10-level LOB)")

    hdr("Single Day Deep-Dive")
    run_day(date=args.date, ticker=ticker, is_futures=is_futures,
            verbose=True, use_etf=args.etf)

    if args.multiday:
        run_multiday(n_days=args.days, ticker=ticker, is_futures=is_futures)

    if args.walkforward:
        hdr("Walk-Forward Optimization")
        run_walkforward(n_days=args.days, ticker=ticker, is_futures=is_futures)

    hdr("Done")
    print("  Next steps:")
    print("  1. Replace simulate_lob_day() with real Wind/Tushare L2 feed")
    print("  2. Grid-search ofi_levels, ofi_window, entry_z, max_hold")
    print("  3. Validate IC on 6+ months out-of-sample before trading live")
    print("  4. For stocks: add 融券 (margin short) or ETF-arb pairs")
    print("  5. For futures: add basis spread (IF vs spot ETF 510300)")


if __name__ == "__main__":
    main()
