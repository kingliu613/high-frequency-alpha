"""
Performance metrics for HFT alpha evaluation.

Standard HFT alpha evaluation workflow:
  1. compute_forward_returns() – build return targets at multiple horizons
  2. ic_by_horizon()           – IC curve (signal decay analysis)
  3. rolling_ic()              – time-series stability check
  4. pnl_metrics()             – Sharpe, drawdown, win-rate from PnL series

IC thresholds (empirical HFT benchmarks):
  IC > 0.02   worthwhile at scale
  IC > 0.05   strong for intraday
  IC > 0.10   exceptional (rare in mature markets)
  ICIR > 0.5  acceptable stability
  ICIR > 1.0  strong
"""

import numpy as np
import pandas as pd
from typing import Union


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation without scipy dependency."""
    ra = a.rank().values.astype(float)
    rb = b.rank().values.astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def compute_forward_returns(
    lob_df: pd.DataFrame,
    horizons: list[int] = (1, 5, 10, 20, 40, 100, 200),
) -> pd.DataFrame:
    """
    Mid-price returns at each tick horizon.

        r_h(t) = mid(t+h) / mid(t) - 1

    For 3-second LOB snapshots:
        h=1   →  3s
        h=5   → 15s
        h=10  → 30s
        h=20  →  1 min
        h=40  →  2 min
        h=100 →  5 min
        h=200 → 10 min
    """
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]).astype(float) / 2.0
    out = {}
    for h in horizons:
        out[f"fwd_{h}"] = mid.shift(-h) / mid - 1.0
    return pd.DataFrame(out, index=lob_df.index)


# ---------------------------------------------------------------------------
# IC / ICIR
# ---------------------------------------------------------------------------

def ic_by_horizon(
    signal: pd.Series,
    fwd_df: pd.DataFrame,
) -> pd.Series:
    """
    Spearman rank IC at every horizon in fwd_df.

    Returns Series indexed by tick horizon (e.g. 1, 5, 10, ...).
    """
    result = {}
    for col in fwd_df.columns:
        h    = int(col.split("_")[1])
        mask = signal.notna() & fwd_df[col].notna()
        if mask.sum() < 50:
            result[h] = np.nan
            continue
        s = signal[mask]
        r = fwd_df[col][mask]
        result[h] = _spearman(s, r)
    return pd.Series(result, name="IC").sort_index()


def rolling_ic(
    signal: pd.Series,
    fwd_df: pd.DataFrame,
    horizon: int = 10,
    window: int = 500,
) -> pd.Series:
    """
    Rolling Spearman IC over `window` snapshots at one horizon.

    Use to check:
      - Is IC consistent, or driven by a few days?
      - Is alpha decaying over time (crowding, regime change)?
    """
    col  = f"fwd_{horizon}"
    mask = signal.notna() & fwd_df[col].notna()
    s    = signal[mask]
    r    = fwd_df[col][mask]

    ics, idx = [], []
    for i in range(window, len(s)):
        sw  = s.iloc[i - window : i]
        rw  = r.iloc[i - window : i]
        ics.append(_spearman(sw, rw))
        idx.append(s.index[i])

    return pd.Series(ics, index=idx, name=f"rolling_ic_h{horizon}")


def icir(ic_ts: pd.Series) -> float:
    """IC Information Ratio: mean(IC) / std(IC)."""
    std = ic_ts.std()
    return float(ic_ts.mean() / std) if std > 0 else np.nan


def signal_decay_table(
    signal: pd.Series,
    lob_df: pd.DataFrame,
    max_ticks: int = 200,
    step: int    = 5,
) -> pd.DataFrame:
    """
    Build IC-decay curve: IC vs tick horizon from 1 to max_ticks.

    Returns DataFrame with columns:
        ticks, seconds, IC
    """
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]).astype(float) / 2.0
    rows = []

    for h in range(1, max_ticks + 1, step):
        fwd  = mid.shift(-h) / mid - 1.0
        mask = signal.notna() & fwd.notna()
        if mask.sum() < 100:
            continue
        s = signal[mask]
        r = fwd[mask]
        ic_val = _spearman(s, r)
        rows.append({"ticks": h, "seconds": h * 3, "IC": ic_val})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PnL metrics
# ---------------------------------------------------------------------------

def pnl_metrics(
    pnl: pd.Series,
    snapshots_per_day: int = 4800,
    capital_rmb: float = None,
) -> dict:
    """
    Compute standard performance metrics from a per-snapshot PnL series.

    snapshots_per_day ≈ 4h × 60min × 20 snapshots/min  = 4800
    (for 3-second data: 4 × 3600 / 3 = 4800 exactly)

    capital_rmb : margin/capital deployed in RMB.  When provided, returns
                  ann_return_pct as a percentage.  Omit when running on
                  synthetic data — the number is not meaningful without a
                  real capital base.
    """
    pnl   = pnl.dropna()
    cum   = pnl.cumsum()
    total = float(cum.iloc[-1]) if len(cum) else 0.0

    # Annualised Sharpe (assume 252 trading days)
    if isinstance(pnl.index, pd.DatetimeIndex):
        daily = pnl.resample("D").sum()
    else:
        daily = pnl.groupby(np.arange(len(pnl)) // snapshots_per_day).sum()

    sharpe = (
        float(np.sqrt(252) * daily.mean() / daily.std())
        if daily.std() > 0 else 0.0
    )

    # Max drawdown
    running_max = cum.cummax()
    dd          = cum - running_max
    max_dd      = float(dd.min())

    # ann_return_rmb: scale observed mean-daily PnL to a full year.
    # On single-day synthetic data this is days_observed=1, so multiply
    # by 252 gives a "per-year at this rate" figure — NOT a realised return.
    ann_return_rmb = total / len(pnl) * snapshots_per_day * 252
    calmar         = ann_return_rmb / abs(max_dd) if max_dd != 0 else 0.0

    wins        = int((pnl > 0).sum())
    losses      = int((pnl < 0).sum())
    win_rate    = wins / (wins + losses) if (wins + losses) > 0 else 0.0
    gross_win   = float(pnl[pnl > 0].sum())
    gross_loss  = float(abs(pnl[pnl < 0].sum()))
    profit_fac  = gross_win / gross_loss if gross_loss > 0 else np.inf

    out = {
        "total_pnl":      total,
        "ann_return_rmb": ann_return_rmb,
        "sharpe":         sharpe,
        "max_drawdown":   max_dd,
        "calmar":         calmar,
        "win_rate":       win_rate,
        "profit_factor":  profit_fac,
        "n_nonzero":      int((pnl != 0).sum()),
    }
    if capital_rmb is not None and capital_rmb > 0:
        out["ann_return_pct"] = ann_return_rmb / capital_rmb * 100.0
    return out
