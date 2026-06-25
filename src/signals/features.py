"""
LOB-derived microstructure features for Chinese L2 data.

Signals:
  - Queue imbalance                       (Cao, Chen, Wang 2009)
  - Spread features                       (spread, relative spread, depth ratio)
  - Price-limit hit state                 (Chinese daily limit regime)

All signals are stationary and suitable for cross-sectional
normalization or direct z-score ranking.
"""

import numpy as np
import pandas as pd

N_LEVELS = 10


# ---------------------------------------------------------------------------
# Queue imbalance
# ---------------------------------------------------------------------------

def queue_imbalance(
    lob_df: pd.DataFrame,
    n_levels: int = 1,
    decay_lambda: float = 0.3,
) -> pd.Series:
    """
    Queue imbalance from displayed bid/ask depth.

    The paper-defined best-queue form is

        QI = (V_bid_1 - V_ask_1) / (V_bid_1 + V_ask_1)

    `n_levels=1` is therefore the default and the exact paper-aligned mode.
    Passing n_levels > 1 computes the same depth-imbalance extension across
    levels with exponential weights for exploratory use.

    Positive → deeper bid side → price support → bullish.
    Negative → deeper ask side → selling pressure → bearish.

    Note: can be gamed by spoofing; cross-validate with trade_imbalance.
    """
    weights = np.exp(-decay_lambda * np.arange(n_levels))
    weights /= weights.sum()

    wb = pd.Series(0.0, index=lob_df.index)
    wa = pd.Series(0.0, index=lob_df.index)

    for lv in range(1, n_levels + 1):
        w = weights[lv - 1]
        wb += w * lob_df[f"bid_vol_{lv}"].astype(float)
        wa += w * lob_df[f"ask_vol_{lv}"].astype(float)

    total = (wb + wa).replace(0.0, np.nan)
    return ((wb - wa) / total).fillna(0.0).rename("queue_imbalance")

# ---------------------------------------------------------------------------
# Spread & LOB shape
# ---------------------------------------------------------------------------

def spread_features(lob_df: pd.DataFrame) -> pd.DataFrame:
    """
    Quoted spread, relative spread, and 5-level depth ratio.

    depth_ratio > 1  → more bid depth → bullish
    depth_ratio < 1  → more ask depth → bearish
    high rel_spread  → low liquidity / wide market → avoid trading
    """
    pb = lob_df["bid_px_1"].astype(float)
    pa = lob_df["ask_px_1"].astype(float)
    mid = (pb + pa) / 2.0

    spread     = pa - pb
    rel_spread = spread / mid.replace(0.0, np.nan)

    bid5 = sum(lob_df[f"bid_vol_{lv}"].astype(float) for lv in range(1, 6))
    ask5 = sum(lob_df[f"ask_vol_{lv}"].astype(float) for lv in range(1, 6))
    depth_ratio = bid5 / ask5.replace(0.0, np.nan)

    return pd.DataFrame({
        "spread":      spread,
        "rel_spread":  rel_spread.fillna(0.0),
        "depth_ratio": depth_ratio.fillna(1.0),
    }, index=lob_df.index)


# ---------------------------------------------------------------------------
# Price-limit hit-state signal  (stock mode only)
# ---------------------------------------------------------------------------

def price_limit_state(
    lob_df: pd.DataFrame,
    prev_close: float,
    limit_pct: float = 0.10,
    activation_pct: float | None = None,
    tick_size: float = 0.01,
) -> pd.Series:
    """
    Price-limit hit state under the Chinese daily limit rule.

    The price-limit papers study hit states and post-hit dynamics, not a
    hand-tuned exponential "approach" score. This function therefore emits:

        +1 when the best ask/mid is at or above the up-limit
        -1 when the best bid/mid is at or below the down-limit
         0 otherwise

    `activation_pct` is retained only for backward-compatible call sites and
    is intentionally ignored. `tick_size` supplies the tolerance used when a
    computed limit price cannot be represented exactly in binary floating point.
    """
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]).astype(float) / 2.0
    bid = lob_df["bid_px_1"].astype(float)
    ask = lob_df["ask_px_1"].astype(float)

    up_limit   = prev_close * (1.0 + limit_pct)
    down_limit = prev_close * (1.0 - limit_pct)

    tol = max(float(tick_size) / 2.0, 1e-9)

    sig = pd.Series(0.0, index=lob_df.index)
    sig = sig.mask((ask >= up_limit - tol) | (mid >= up_limit - tol), 1.0)
    sig = sig.mask((bid <= down_limit + tol) | (mid <= down_limit + tol), -1.0)
    return sig.rename("price_limit")


def price_limit_signal(
    lob_df: pd.DataFrame,
    prev_close: float,
    limit_pct: float = 0.10,
    activation_pct: float | None = None,
) -> pd.Series:
    """Backward-compatible alias for the strict price-limit state label."""
    return price_limit_state(
        lob_df,
        prev_close=prev_close,
        limit_pct=limit_pct,
        activation_pct=activation_pct,
    )
