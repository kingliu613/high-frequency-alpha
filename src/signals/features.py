"""
LOB-derived microstructure features for Chinese L2 data.

Signals:
  - Micro-price & micro-price deviation  (Stoikov 2018)
  - Queue imbalance                       (Cao, Chen, Wang 2009)
  - Depth tilt                            (price-weighted asymmetry)
  - Short-term momentum                   (mid-price returns)
  - Spread features                       (spread, relative spread, depth ratio)

All signals are stationary and suitable for cross-sectional
normalization or direct z-score ranking.
"""

import numpy as np
import pandas as pd

N_LEVELS = 10


# ---------------------------------------------------------------------------
# Micro-price
# ---------------------------------------------------------------------------

def micro_price(lob_df: pd.DataFrame) -> pd.Series:
    """
    Volume-weighted mid-price using best bid/ask.

        μ = (V_ask · P_bid + V_bid · P_ask) / (V_bid + V_ask)

    Equals mid-price when V_bid = V_ask.
    Moves toward the ask when bid is thicker (buyers committed).
    Moves toward the bid when ask is thicker (sellers committed).

    Reference: Stoikov (2018) "The micro-price", Quantitative Finance.
    """
    vb = lob_df["bid_vol_1"].astype(float)
    va = lob_df["ask_vol_1"].astype(float)
    pb = lob_df["bid_px_1"].astype(float)
    pa = lob_df["ask_px_1"].astype(float)

    total = (vb + va).replace(0.0, np.nan)
    return ((va * pb + vb * pa) / total).rename("micro_price")


def micro_price_dev(lob_df: pd.DataFrame) -> pd.Series:
    """
    Signed deviation of micro-price from arithmetic mid, scaled by half-spread.

        dev = (μ - mid) / (0.5 · spread)   ∈ [-1, +1]

    +1  → all volume on bid side (strong buy pressure)
    -1  → all volume on ask side (strong sell pressure)

    This is equivalent to the signed queue imbalance at best level
    but expressed in price units.
    """
    pb = lob_df["bid_px_1"].astype(float)
    pa = lob_df["ask_px_1"].astype(float)
    mid        = (pb + pa) / 2.0
    half_spread = (pa - pb) / 2.0

    mp  = micro_price(lob_df)
    dev = (mp - mid) / half_spread.replace(0.0, np.nan)

    return dev.fillna(0.0).rename("micro_price_dev")


# ---------------------------------------------------------------------------
# Queue imbalance
# ---------------------------------------------------------------------------

def queue_imbalance(
    lob_df: pd.DataFrame,
    n_levels: int = 5,
    decay_lambda: float = 0.3,
) -> pd.Series:
    """
    Multi-level queue imbalance.

        QI = (Σ w_l · V_bid_l - Σ w_l · V_ask_l)
           / (Σ w_l · V_bid_l + Σ w_l · V_ask_l)

    w_l = exp(-λ·(l-1)); best-level weighted most heavily.

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
# Depth tilt
# ---------------------------------------------------------------------------

def depth_tilt(lob_df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """
    Price-proximity-weighted depth asymmetry.

    Upweights volume that sits close to the spread (more likely to trade),
    downweights volume far from mid (often noise/layering in A-shares).

        tilt = (Σ w_bid_l · V_bid_l - Σ w_ask_l · V_ask_l)
             / (Σ w_bid_l · V_bid_l + Σ w_ask_l · V_ask_l)

    where w_bid_l ∝ 1/(1 + distance_from_best_bid_in_ticks)

    Captures LOB shape information not captured by simple queue imbalance.
    """
    pb1 = lob_df["bid_px_1"].astype(float)
    pa1 = lob_df["ask_px_1"].astype(float)

    wb_sum = pd.Series(0.0, index=lob_df.index)
    wa_sum = pd.Series(0.0, index=lob_df.index)

    for lv in range(1, n_levels + 1):
        bid_dist = ((pb1 - lob_df[f"bid_px_{lv}"]) / 0.01).clip(lower=0)
        ask_dist = ((lob_df[f"ask_px_{lv}"] - pa1) / 0.01).clip(lower=0)

        w_b = 1.0 / (1.0 + bid_dist)
        w_a = 1.0 / (1.0 + ask_dist)

        wb_sum += w_b * lob_df[f"bid_vol_{lv}"].astype(float)
        wa_sum += w_a * lob_df[f"ask_vol_{lv}"].astype(float)

    total = (wb_sum + wa_sum).replace(0.0, np.nan)
    return ((wb_sum - wa_sum) / total).fillna(0.0).rename("depth_tilt")


# ---------------------------------------------------------------------------
# Short-term momentum
# ---------------------------------------------------------------------------

def short_term_momentum(
    lob_df: pd.DataFrame,
    windows: list[int] = [5, 20],
) -> pd.DataFrame:
    """
    Z-scored mid-price returns at multiple tick horizons.

    5  ticks = 15s  (very short, mean-reversion dominant)
    20 ticks = 60s  (momentum slightly more persistent)

    In Chinese A-shares:
    - 5-tick momentum tends to mean-revert (high retail noise)
    - 20-tick momentum shows mild continuation in first 30 min
    """
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]).astype(float) / 2.0
    result = {}

    for w in windows:
        ret   = mid.pct_change(w)
        rstd  = ret.rolling(200, min_periods=50).std()
        z     = (ret / rstd.replace(0.0, np.nan)).fillna(0.0)
        result[f"mom_{w}"] = z

    return pd.DataFrame(result, index=lob_df.index)


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
# Price-limit approach signal  (stock mode only)
# ---------------------------------------------------------------------------

def price_limit_signal(
    lob_df: pd.DataFrame,
    prev_close: float,
    limit_pct: float = 0.10,
    activation_pct: float = 0.03,
) -> pd.Series:
    """
    Exponential signal as mid-price approaches daily ±10% price limit.

    Positive → approaching up-limit (涨停 momentum).
    Negative → approaching down-limit (跌停 momentum).

    Only active within `activation_pct` of the limit; zero otherwise.
    Stock-mode only — futures have no fixed daily price limits.

    Reference: PMC4395215 (2015) price continuation probability 0.68 after
    up-limit hit.
    """
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]).astype(float) / 2.0

    up_limit   = prev_close * (1.0 + limit_pct)
    down_limit = prev_close * (1.0 - limit_pct)

    pct_to_up   = (up_limit   - mid) / up_limit
    pct_to_down = (mid - down_limit) / down_limit

    sig_up   = np.exp(-10.0 * pct_to_up.clip(lower=0.0))
    sig_down = -np.exp(-10.0 * pct_to_down.clip(lower=0.0))

    mask_up   = pct_to_up   < activation_pct
    mask_down = pct_to_down < activation_pct

    sig = sig_up.where(mask_up, 0.0) + sig_down.where(mask_down, 0.0)
    return sig.rename("price_limit")


# Stub — full implementation in Task 4
def etf_basis_signal(lob_df: pd.DataFrame, etf_series) -> "pd.Series":
    raise NotImplementedError("etf_basis_signal not yet implemented")
