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
  Market—A High-Frequency View"), auction buy/sell imbalance predicts
  next-day returns and is stronger for mid/small-caps.

  For futures (IF/IC): T+0 allows capitalising on directional signal
  immediately after 09:30 open within the same trading day.
"""

import pandas as pd


def auction_imbalance(auction_df: pd.DataFrame) -> float:
    """
    Buy/sell imbalance at end of call auction.

    When available, use the paper's count polarity:

        (NOB - NOS) / (NOB + NOS)

    with NOB/NOS represented by `cum_buy_count` / `cum_sell_count`.
    Volume columns are accepted only for older auction fixtures that do not
    yet carry participant/order counts.

    Returns float in [-1, +1]:
        +1 → pure buy pressure
        -1 → pure sell pressure
         0 → balanced

    Uses the final snapshot (09:24:57) which captures committed
    non-cancellable orders submitted after 09:20.
    """
    final = auction_df.iloc[-1]
    if "cum_buy_count" in auction_df.columns and "cum_sell_count" in auction_df.columns:
        buy = float(final["cum_buy_count"])
        sell = float(final["cum_sell_count"])
    else:
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
    price-limit state risk next session, not pure momentum.
    """
    prev_close = float(auction_df["prev_close"].iloc[0])
    return (open_price - prev_close) / prev_close


def auction_signal_series(
    lob_df: pd.DataFrame,
    auction_value: float,
    half_life_min: float = 20.0,
) -> pd.Series:
    """
    Project the paper-defined auction imbalance onto continuous timestamps.

    The cited papers define the auction imbalance scalar; they do not define
    a fitted intraday decay curve. `half_life_min` is retained only for
    backward-compatible call sites and is intentionally ignored.

    Returns Series aligned to lob_df.index.
    """
    return pd.Series(
        float(auction_value),
        index=lob_df.index,
        name="auction_signal",
    )
