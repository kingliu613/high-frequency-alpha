"""
Opening call auction signals for Chinese A-shares / index futures.

Chinese auction schedule:
  09:15–09:20  Order submission (cancellations allowed)
  09:20–09:25  Order submission (NO cancellations)
  09:25        Auction clears at volume-maximising price; opening price set
  09:25–09:30  Cooling period (no trading, positions can be viewed)
  09:30        Continuous double-auction begins

Key insight:
  The non-cancellable window (09:20–09:25) traps informed traders
  who submitted orders before 09:20. The resulting order imbalance
  therefore carries genuine information about overnight news and
  institutional positioning, unlike cancellable-order LOB signals
  which can be spoofed cheaply.

  Empirically (MDPI Entropy 2020, "Trading Imbalance in Chinese Stock
  Market—A High-Frequency View"):
    - Auction buy/sell imbalance predicts next-day returns
    - Opening gap continuation lasts ~20–30 min then mean-reverts
    - Small-cap effect: imbalance signal stronger for mid/small-caps

  For futures (IF/IC): T+0 allows capitalising on directional signal
  immediately after 09:30 open within the same trading day.
"""

import numpy as np
import pandas as pd


def auction_imbalance(auction_df: pd.DataFrame) -> float:
    """
    Buy/sell volume imbalance at end of call auction.

    Returns float in [-1, +1]:
        +1 → pure buy pressure
        -1 → pure sell pressure
         0 → balanced

    Uses the final snapshot (09:24:57) which captures committed
    non-cancellable orders submitted after 09:20.
    """
    final = auction_df.iloc[-1]
    buy  = float(final["cum_buy_vol"])
    sell = float(final["cum_sell_vol"])
    total = buy + sell
    if total == 0.0:
        return 0.0
    return (buy - sell) / total


def auction_gap(auction_df: pd.DataFrame, open_price: float) -> float:
    """
    Opening gap as fraction of previous close.

        gap = (open_price - prev_close) / prev_close

    Positive → gapped up (overnight demand/news).
    Negative → gapped down.

    Clip interpretation: gaps > 5% on A-shares often indicate
    price-limit approach dynamics next session, not pure momentum.
    """
    prev_close = float(auction_df["prev_close"].iloc[0])
    return (open_price - prev_close) / prev_close


def auction_composite(
    auction_df: pd.DataFrame,
    open_price: float,
    imbalance_weight: float = 0.65,
    gap_weight: float = 0.35,
) -> float:
    """
    Composite auction signal: imbalance + gap direction.

    Default weights: imbalance dominates (more manipulation-resistant).
    Gap clipped at ±5% before normalisation to avoid price-limit outliers.

    Returns scalar in [-1, +1].
    """
    imb    = auction_imbalance(auction_df)
    gap    = auction_gap(auction_df, open_price)
    gap_n  = float(np.clip(gap / 0.05, -1.0, 1.0))   # normalize to ±1

    composite = imbalance_weight * imb + gap_weight * gap_n
    return float(np.clip(composite, -1.0, 1.0))


def close_auction_imbalance(close_df: pd.DataFrame) -> float:
    """
    Buy/sell volume imbalance at the end of the CLOSING call auction (14:57–15:00).

    China added a closing call auction in 2018 (SSE/SZSE). Index-rebalance flow,
    ETF creation/redemption hedges, and end-of-day informed positioning concentrate
    here, so the closing imbalance predicts the OVERNIGHT return and the next-day
    open gap — the mirror of the opening-auction edge on the other information window.

    Returns float in [-1, +1] from the final closing snapshot.
    Strongest for index constituents on rebalance days.
    """
    final = close_df.iloc[-1]
    buy  = float(final["cum_buy_vol"])
    sell = float(final["cum_sell_vol"])
    total = buy + sell
    if total == 0.0:
        return 0.0
    return (buy - sell) / total


def close_auction_signal_series(
    lob_df: pd.DataFrame,
    close_auction_value: float,
    half_life_min: float = 30.0,
) -> pd.Series:
    """
    Carry the prior session's closing-auction imbalance into today's session as a
    decaying prior, anchored at the 09:30 open.

        s(t) = close_auction_value · exp(-ln2 · t / t_{1/2})

    where t is minutes since 09:30. The closing imbalance predicts the overnight
    move; by the open much of it is realised, so the residual prior decays over the
    first ~30 min. Positive → prior-close buy pressure → bullish bias into the open.

    Returns Series aligned to lob_df.index.
    """
    open_ts   = lob_df.index[0]
    elapsed_m = (lob_df.index - open_ts).total_seconds() / 60.0
    decay     = np.exp(-np.log(2.0) * elapsed_m / half_life_min)

    return pd.Series(
        close_auction_value * decay,
        index=lob_df.index,
        name="close_auction",
    )


def auction_signal_series(
    lob_df: pd.DataFrame,
    auction_value: float,
    half_life_min: float = 20.0,
) -> pd.Series:
    """
    Project scalar auction signal onto continuous LOB timestamps
    with exponential decay.

        s(t) = auction_value · exp(-ln2 · t / t_{1/2})

    where t is minutes elapsed since 09:30.

    Half-life of ~20 min calibrated to Chinese A-share data
    (imbalance information fully absorbed by ~09:50–10:00).
    Use shorter half-life (10 min) for index futures where price
    discovery is faster.

    Returns Series aligned to lob_df.index.
    """
    open_ts   = lob_df.index[0]
    elapsed_m = (lob_df.index - open_ts).total_seconds() / 60.0
    decay     = np.exp(-np.log(2.0) * elapsed_m / half_life_min)

    return pd.Series(
        auction_value * decay,
        index=lob_df.index,
        name="auction_signal",
    )
