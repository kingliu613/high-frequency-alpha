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

from src.data.synthetic  import (
    simulate_lob_day,
    simulate_auction_data,
)
from src.signals.auction import auction_imbalance
from src.signals.advanced import exposure_gate, vpin
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
    signal_strength: float = 0.01,
    use_gate: bool = False,
    factors=None,
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
        signal_strength=signal_strength,
    )
    auc_val = auction_imbalance(auction_df)

    if params is None:
        params = MarketParams.default_for(
            "futures" if is_futures else "stock",
            entry_z  = 1.5,
            exit_z   = 0.3,
            max_hold = 20,
        )

    # --- Features ---
    feat_df = build_feature_matrix(
        lob_df,
        auction_value         = auc_val,
        ofi_levels            = 5,
        ofi_window            = 10,
        prev_close            = float(open_price) if not is_futures else None,
        instrument            = params.instrument,
        factors               = factors,
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
    gate = exposure_gate(lob_df) if use_gate else None
    pnl, trades = run_backtest(
        lob_df, composite, params=params, exposure_scale=gate,
        prev_close=float(open_price) if not is_futures else None,
    )
    m = pnl_metrics(pnl)

    if verbose:
        print(f"\nDate: {date}   Ticker: {ticker}   "
              f"{'Futures' if is_futures else 'Stock'}")
        print(f"Snapshots: {len(lob_df):,}   Auction signal: {auc_val:+.4f}   "
              f"Features: {len(feat_df.columns)}")
        if use_gate and gate is not None:
            v = vpin(lob_df)
            print(f"Gate: mean scale={gate.mean():.2f}   "
                  f"blocked(<0.3)={float((gate < 0.3).mean()):.1%}   "
                  f"VPIN mean={v.mean():.3f} max={v.max():.3f}")

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
    signal_strength: float = 0.01,
) -> pd.DataFrame:
    """
    Run n_days and report IC stability and PnL.

    Also runs a null baseline (signal_strength=0.0) on the same seeds so
    you can see how much of the IC/Sharpe is guaranteed by the synthetic
    data generator vs actual signal content.
    """
    dates = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    rows, null_rows = [], []

    print(f"\nRunning {n_days} days  (signal_strength={signal_strength}) …")
    for i, d in enumerate(dates):
        r      = run_day(d, ticker=ticker, is_futures=is_futures, seed=i,
                         verbose=False, signal_strength=signal_strength)
        r_null = run_day(d, ticker=ticker, is_futures=is_futures, seed=i,
                         verbose=False, signal_strength=0.0)
        ic     = r["ic_all"]
        ic_n   = r_null["ic_all"]
        row = {"date": d, "daily_pnl": r["pnl"].sum(),
               "null_pnl": r_null["pnl"].sum()}
        for h in [1, 5, 10, 20, 40]:
            row[f"ic_{h}"]      = float(ic.get(h, np.nan))
            row[f"null_ic_{h}"] = float(ic_n.get(h, np.nan))
        rows.append(row)
        print(f"  {d}  IC@10={row['ic_10']:+.4f}  "
              f"(null={row['null_ic_10']:+.4f})  "
              f"PnL={row['daily_pnl']:+.1f}  (null={row['null_pnl']:+.1f})")

    df = pd.DataFrame(rows)

    hdr("Multi-Day IC Stability")
    print(df.round(4).to_string(index=False))

    print(f"\nSummary (IC @ 10 ticks = 30s)  signal_strength={signal_strength} vs null=0.0:")
    for h in [1, 5, 10, 20, 40]:
        mean  = df[f"ic_{h}"].mean();      std  = df[f"ic_{h}"].std()
        nmean = df[f"null_ic_{h}"].mean(); nstd = df[f"null_ic_{h}"].std()
        ir    = mean  / std  if std  > 0 else np.nan
        nir   = nmean / nstd if nstd > 0 else np.nan
        print(f"  h={h:>3}t: IC={mean:+.4f} (ICIR={ir:.2f})  "
              f"null={nmean:+.4f} (ICIR={nir:.2f})  "
              f"edge={mean-nmean:+.4f}")

    pnl_all  = df["daily_pnl"]
    null_all = df["null_pnl"]
    ann_sh   = np.sqrt(252) * pnl_all.mean()  / pnl_all.std()  if pnl_all.std()  > 0 else 0.0
    null_sh  = np.sqrt(252) * null_all.mean() / null_all.std() if null_all.std() > 0 else 0.0
    print(f"\n  PnL  mean={pnl_all.mean():+.0f}  Sharpe={ann_sh:.2f}")
    print(f"  Null mean={null_all.mean():+.0f}  Sharpe={null_sh:.2f}  "
          f"(baseline from circular data alone)")

    if signal_strength > 0.01:
        print(f"\n  WARNING: signal_strength={signal_strength} > 0.01. "
              f"IC/Sharpe are inflated by circular data coupling.")

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
                p = MarketParams.default_for(
                    "futures" if is_futures else "stock",
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
        p_oos = MarketParams.default_for(
            "futures" if is_futures else "stock",
            entry_z  = best_ez,
            exit_z   = 0.3,
            max_hold = best_mh,
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
    ap.add_argument("--days",        type=int,   default=20)
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--gate",        action="store_true",
                    help="apply VPIN + Kyle-λ exposure gate to position sizing")
    ap.add_argument("--factors",     default=None,
                    help="comma-separated factor groups and/or factor names "
                         "(e.g. 'flow,auction' or 'mlofi,api,auction_signal'). "
                         "Groups: flow, book, behavior, auction, limit, "
                         "interaction. Default: all.")
    ap.add_argument(
        "--signal-strength", type=float, default=0.01,
        help="Synthetic data: coupling between latent flow and price/LOB. "
             "0.0 = null baseline (no embedded alpha). Default 0.01.",
    )
    args = ap.parse_args()

    is_futures = not args.stock
    ticker     = args.ticker if is_futures else "600519.SH"  # Moutai example

    hdr("HFT Alpha Research — Chinese Market")
    print("  Instrument : " + ("CSI 300 Futures (IF) on CFFEX" if is_futures
                                else "A-share (long-only, T+1)"))
    print("  Signals    : Strict alpha = OFI · Trade Polarity · API/OEI")
    print("               Gates/diagnostics are explicit opt-ins")
    print("  Data       : Synthetic Chinese L2 (3-second, 10-level LOB)")

    ss = args.signal_strength
    if ss > 0.01:
        print(f"\n  WARNING: --signal-strength={ss} > 0.01. "
              f"Results will be inflated by circular synthetic-data coupling.")

    hdr("Single Day Deep-Dive")
    factors = [f.strip() for f in args.factors.split(",")] if args.factors else None
    if factors:
        from src.signals.composite import expand_factor_selection, FACTOR_GROUPS
        try:
            expand_factor_selection(factors)
        except ValueError as e:
            print(f"\n  ERROR: {e}\n")
            print("  Available groups:")
            for g, members in FACTOR_GROUPS.items():
                print(f"    {g:<12} {', '.join(members)}")
            sys.exit(1)
        print(f"\n  Factor selection: {factors}")

    run_day(date=args.date, ticker=ticker, is_futures=is_futures,
            verbose=True, signal_strength=ss,
            use_gate=args.gate, factors=factors)

    if args.multiday:
        run_multiday(n_days=args.days, ticker=ticker, is_futures=is_futures,
                     signal_strength=ss)

    if args.walkforward:
        hdr("Walk-Forward Optimization")
        run_walkforward(n_days=args.days, ticker=ticker, is_futures=is_futures)

    hdr("Done")
    print("  Next steps:")
    print("  1. Replace simulate_lob_day() with real L2 feed including counts and order events")
    print("  2. Grid-search ofi_levels, ofi_window, entry_z, max_hold")
    print("  3. Validate IC on 6+ months out-of-sample before trading live")
    print("  4. Audit event-column mappings against the exchange/vendor schema")
    print("  5. Re-run paper formula checks before adding any new factor")


if __name__ == "__main__":
    main()
