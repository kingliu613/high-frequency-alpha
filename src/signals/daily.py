"""
高频因子日频化 — daily aggregation of tick-level factors for
cross-sectional stock selection (截面选股).

Industry-standard workflow (海通/国盛金工 style):
  1. Per stock per day: compute tick-level factor series (composite.py)
  2. Aggregate each factor into daily scalar(s)        ← this module
  3. Cross-sectional rank across the universe per day
  4. RankIC vs NEXT-day return; ICIR; quantile long-short

Aggregation scheme
------------------
For every tick-level factor column two daily values are produced:
    <factor>        full-day mean        (sustained pressure)
    <factor>_tail   last-30-min mean     (尾盘 — closest to the next-day
                                          horizon, empirically strongest
                                          for flow factors in A-shares)
Plus scalar factors that are already daily by construction:
    auction_imb, day_rv

Return conventions (T+1 reality)
--------------------------------
    ret_cc    close(t) → close(t+1)   standard IC convention, NOT executable
    ret_oo    open(t+1) → open(t+2)   executable: signal known at close(t),
                                      buy next open, T+1 allows selling the
                                      open after that
Both are provided; report IC on ret_cc for comparability with research,
report strategy PnL on ret_oo for honesty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


TAIL_MINUTES = 30.0


# ---------------------------------------------------------------------------
# Per (stock, day) aggregation
# ---------------------------------------------------------------------------

def aggregate_daily_factors(
    feat_df: pd.DataFrame,
    lob_df: pd.DataFrame,
    auction_value: Optional[float] = None,
    tail_minutes: float = TAIL_MINUTES,
) -> dict:
    """
    Collapse one day of tick-level factors into daily scalars.

    Parameters
    ----------
    feat_df  : output of build_feature_matrix() for one stock-day
    lob_df   : the matching LOB frame (for prices / tail window)
    auction_value : daily scalar passed through
    tail_minutes : size of the end-of-day window for the _tail aggregates

    Returns dict of daily factor values + open/close prices for return
    construction. The intraday `auction_signal` projection is excluded because
    the daily scalar is passed through separately as `auction_imb`.
    """
    mid = (lob_df["bid_px_1"].astype(float) + lob_df["ask_px_1"].astype(float)) / 2.0

    end_ts     = lob_df.index[-1]
    tail_start = end_ts - pd.Timedelta(minutes=tail_minutes)
    tail_mask  = lob_df.index >= tail_start

    row: dict = {
        "open":  float(mid.iloc[0]),
        "close": float(mid.iloc[-1]),
    }

    skip = {"auction_signal"}   # scalar daily info is carried as auction_imb
    for col in feat_df.columns:
        if col in skip:
            continue
        s = feat_df[col].astype(float)
        row[col]           = float(s.mean())
        row[f"{col}_tail"] = float(s[tail_mask].mean())

    # Scalar daily factors
    if auction_value is not None:
        row["auction_imb"] = float(auction_value)

    # Day realized variance from mid returns (annualisation-free, cross-
    # sectionally comparable)
    r = np.log(mid.replace(0.0, np.nan)).diff()
    row["day_rv"] = float((r ** 2).sum())

    return row


def build_panel(rows: list[dict]) -> pd.DataFrame:
    """
    Assemble per-(date, ticker) dicts into a MultiIndex panel.

    Each dict must carry "date" and "ticker" keys (added by the caller).
    Adds ret_cc (close→close, IC convention) and ret_oo (open(t+1)→open(t+2),
    executable under T+1) per ticker.
    """
    panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_index()
        close = g["close"]
        opn   = g["open"]
        g["ret_cc"] = close.shift(-1) / close - 1.0
        g["ret_oo"] = opn.shift(-2) / opn.shift(-1) - 1.0
        return g

    return panel.groupby(level="ticker", group_keys=False).apply(_per_ticker)


# ---------------------------------------------------------------------------
# Cross-sectional evaluation
# ---------------------------------------------------------------------------

def cross_sectional_rank_ic(
    panel: pd.DataFrame,
    factor_cols: list[str],
    ret_col: str = "ret_cc",
    min_names: int = 5,
) -> pd.DataFrame:
    """
    Per-factor cross-sectional Spearman RankIC vs next-period return.

    For each date: rank factor values across the universe, rank returns,
    correlate. Returns summary DataFrame indexed by factor:

        mean_ic   mean daily RankIC
        icir      mean / std of the daily IC series
        t_stat    icir × √n_days
        n_days    dates with enough names

    Thresholds (daily-horizon industry norms): |mean_ic| > 0.03 usable,
    > 0.05 strong; |t| ≥ 2 minimum credibility.
    """
    out = []
    dates = panel.index.get_level_values("date").unique()

    for col in factor_cols:
        ics = []
        for d in dates:
            sub = panel.loc[d]
            mask = sub[col].notna() & sub[ret_col].notna()
            if mask.sum() < min_names:
                continue
            f = sub.loc[mask, col].rank()
            r = sub.loc[mask, ret_col].rank()
            if f.std() == 0 or r.std() == 0:
                continue
            ics.append(float(np.corrcoef(f, r)[0, 1]))

        if not ics:
            out.append({"factor": col, "mean_ic": np.nan, "icir": np.nan,
                        "t_stat": np.nan, "n_days": 0})
            continue

        ics_a = np.array(ics)
        icir  = ics_a.mean() / ics_a.std() if ics_a.std() > 0 else np.nan
        out.append({
            "factor":  col,
            "mean_ic": float(ics_a.mean()),
            "icir":    float(icir) if np.isfinite(icir) else np.nan,
            "t_stat":  float(icir * np.sqrt(len(ics_a))) if np.isfinite(icir) else np.nan,
            "n_days":  len(ics_a),
        })

    return (pd.DataFrame(out).set_index("factor")
            .sort_values("mean_ic", key=lambda s: s.abs(), ascending=False))


def quantile_longshort(
    panel: pd.DataFrame,
    factor: str,
    ret_col: str = "ret_oo",
    n_quantiles: int = 5,
    long_only: bool = False,
) -> pd.Series:
    """
    Daily-rebalanced quantile portfolio returns for one factor.

    Each date: sort universe by factor, equal-weight top quantile (long);
    long-short subtracts the bottom quantile. long_only=True returns the top
    quantile MINUS the universe mean (excess return) — the honest A-share
    benchmark, since shorting single names is impractical (融券 thin; hedge
    via IF/IC futures ≈ universe-mean subtraction).

    Default ret_col="ret_oo": executable under T+1 (signal at close, buy next
    open, sell the open after). Costs NOT included — subtract ~10 bp/side for
    a realistic daily-turnover figure.
    """
    rets = []
    idx  = []
    for d in panel.index.get_level_values("date").unique():
        sub  = panel.loc[d]
        mask = sub[factor].notna() & sub[ret_col].notna()
        sub  = sub.loc[mask]
        if len(sub) < n_quantiles:
            continue
        q = pd.qcut(sub[factor].rank(method="first"), n_quantiles, labels=False)
        top = sub.loc[q == n_quantiles - 1, ret_col].mean()
        if long_only:
            rets.append(float(top - sub[ret_col].mean()))
        else:
            bot = sub.loc[q == 0, ret_col].mean()
            rets.append(float(top - bot))
        idx.append(d)

    return pd.Series(rets, index=pd.Index(idx, name="date"),
                     name=f"{factor}_{'lo' if long_only else 'ls'}")
