"""
Order Flow Imbalance (OFI) signals for Chinese L2 data.

Primary reference: Cont, Kukanov, Stoikov (2014)
    "The Price Impact of Order Book Events"
    Journal of Financial Econometrics

Extended to multi-level: Kolm, Turku, Westray (2023)
    "Multi-Level Order-Flow Imbalance in a Limit Order Book"

Chinese market application: Chen Hu, Kouxiao Zhang (2025)
    "Stochastic Price Dynamics in Response to Order Flow Imbalance:
     Evidence from CSI 300 Index Futures"
    arXiv:2505.17388

All functions expect a LOB snapshot DataFrame produced by
src/data/synthetic.simulate_lob_day() or equivalent real-data loader.
"""

import numpy as np
import pandas as pd


def ofi_level(lob_df: pd.DataFrame, level: int = 1) -> pd.Series:
    """
    CKS Order Flow Imbalance for a single LOB level.

    For each snapshot transition t-1 → t:

      e_bid(t) = V_bid(t) · 1[P_bid(t) ≥ P_bid(t-1)]
               - V_bid(t-1) · 1[P_bid(t) ≤ P_bid(t-1)]

      e_ask(t) = V_ask(t) · 1[P_ask(t) ≤ P_ask(t-1)]
               - V_ask(t-1) · 1[P_ask(t) ≥ P_ask(t-1)]

      OFI_l(t) = e_bid(t) - e_ask(t)

    Positive OFI → buy pressure > sell pressure → expect price rise.

    The three cases for e_bid collapse correctly:
      P_bid up   → +V_bid(t)         (new aggressive buyer)
      P_bid same → V_bid(t)-V_bid(t-1) (net order accumulation)
      P_bid down → -V_bid(t-1)       (buyer withdrew)
    """
    bp  = lob_df[f"bid_px_{level}"].astype(float)
    bv  = lob_df[f"bid_vol_{level}"].astype(float)
    ap  = lob_df[f"ask_px_{level}"].astype(float)
    av  = lob_df[f"ask_vol_{level}"].astype(float)

    bp_p = bp.shift(1)
    bv_p = bv.shift(1)
    ap_p = ap.shift(1)
    av_p = av.shift(1)

    e_bid = bv * (bp >= bp_p).astype(float) - bv_p * (bp <= bp_p).astype(float)
    e_ask = av * (ap <= ap_p).astype(float) - av_p * (ap >= ap_p).astype(float)

    return (e_bid - e_ask).rename(f"ofi_l{level}")


def mlofi(
    lob_df: pd.DataFrame,
    n_levels: int = 5,
    decay_lambda: float = 0.5,
    normalize: bool = True,
    norm_window: int = 120,
) -> pd.Series:
    """
    Multi-Level OFI: exponentially-weighted sum across LOB levels.

    w_l = exp(-λ·(l-1)) / Σ exp(-λ·(k-1))

    Rationale: deeper levels have staleness and lower information content,
    but do provide incremental signal about sustained directional pressure.

    λ = 0.5 (default): level-2 weight is ~60% of level-1.
    λ = 1.0: level-2 weight is ~37% of level-1 (more focused on best bid/ask).

    normalize=True returns z-score via rolling std (window = norm_window ticks).
    Use raw MLOFI (normalize=False) when aggregating across windows.
    """
    weights = np.exp(-decay_lambda * np.arange(n_levels))
    weights /= weights.sum()

    signal = pd.Series(0.0, index=lob_df.index)
    for lv in range(1, n_levels + 1):
        signal += weights[lv - 1] * ofi_level(lob_df, lv)

    if normalize:
        roll_std = signal.rolling(norm_window, min_periods=20).std()
        signal   = (signal / roll_std.replace(0.0, np.nan)).fillna(0.0)

    return signal.rename("mlofi")


def aggregated_ofi(
    lob_df: pd.DataFrame,
    window: int = 10,
    n_levels: int = 5,
    decay_lambda: float = 0.5,
) -> pd.Series:
    """
    Rolling cumulative MLOFI over `window` ticks (default = 30s at 3s cadence).

    Captures sustained directional order flow pressure rather than
    single-snapshot noise. Empirically peaks IC at 5–20 tick horizon
    for Chinese index futures (Chen & Zhang, 2025).
    """
    raw = mlofi(lob_df, n_levels=n_levels, decay_lambda=decay_lambda, normalize=False)
    agg = raw.rolling(window, min_periods=max(1, window // 2)).sum()

    roll_std = agg.rolling(240, min_periods=60).std()
    agg = (agg / roll_std.replace(0.0, np.nan)).fillna(0.0)

    return agg.rename("agg_ofi")


def trade_imbalance(lob_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Paper-exact transaction polarity from buy/sell participant counts.

    Following "Trading Imbalance in Chinese Stock Market--A High-Frequency
    View", polarity over an interval is

        (NOB - NOS) / (NOB + NOS)

    where NOB and NOS are the numbers of buyer-initiated and seller-initiated
    participants/orders in the interval. `window` is the interval length in
    snapshots; use a 20-snapshot window for 1-minute data at 3s cadence.

    Required columns:
      - cumulative counts: `cum_buy_count`, `cum_sell_count`, or
      - per-snapshot counts: `buy_count`, `sell_count`.
    """
    if "cum_buy_count" in lob_df.columns and "cum_sell_count" in lob_df.columns:
        db = lob_df["cum_buy_count"].astype(float).diff().clip(lower=0).fillna(0.0)
        ds = lob_df["cum_sell_count"].astype(float).diff().clip(lower=0).fillna(0.0)
    elif "buy_count" in lob_df.columns and "sell_count" in lob_df.columns:
        db = lob_df["buy_count"].astype(float).clip(lower=0).fillna(0.0)
        ds = lob_df["sell_count"].astype(float).clip(lower=0).fillna(0.0)
    elif "cum_buy_vol" in lob_df.columns and "cum_sell_vol" in lob_df.columns:
        # Volume fallback: Wind L2 3-second bars provide buy/sell volume but
        # not participant counts. Volume-weighted polarity is NOT the paper
        # NOB/NOS formula but is the best available approximation when count
        # data is absent. Downstream composite normalisation reduces the bias.
        db = lob_df["cum_buy_vol"].astype(float).diff().clip(lower=0).fillna(0.0)
        ds = lob_df["cum_sell_vol"].astype(float).diff().clip(lower=0).fillna(0.0)
    else:
        raise ValueError(
            "trade_imbalance requires buy/sell count columns for the paper "
            "polarity formula: cum_buy_count/cum_sell_count or buy_count/sell_count. "
            "Volume fallback (cum_buy_vol/cum_sell_vol) is also accepted but is "
            "not the paper-exact formula."
        )

    rb = db.rolling(window, min_periods=max(1, window // 4)).sum()
    rs = ds.rolling(window, min_periods=max(1, window // 4)).sum()

    total = (rb + rs).replace(0.0, np.nan)
    return ((rb - rs) / total).fillna(0.0).rename("trade_imbalance")
