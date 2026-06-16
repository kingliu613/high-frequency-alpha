"""
Real data loaders for Chinese market data (free sources).

Source hierarchy:
  1. AKShare   — tick trades + 5-level LOB snapshots (real-time, ~5-day history)
  2. BaoStock  — 5-min OHLCV, full history (2006+), no L2
  3. Tushare   — daily bars only on free tier; paid for L2

AKShare covers most of what we need for signal research:
  - stock_intraday_em()  → tick-by-tick trades (price, volume, side)
  - stock_bid_ask_em()   → current 5-level bid/ask snapshot

Limitations vs real Wind L2:
  - Only 5 LOB levels (not 10)  → adapt N_LEVELS=5 everywhere
  - No 3-second snapshot cadence → derive snapshots from tick stream
  - Real-time only: no historical LOB book (use ticks to reconstruct)

Install:
    pip install akshare baostock

Usage:
    from src.data.loader import load_tick_day_akshare, load_minute_baostock
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

TICK = 0.01
LOT  = 100


# ---------------------------------------------------------------------------
# AKShare — tick + 5-level LOB
# ---------------------------------------------------------------------------

def load_tick_day_akshare(
    ticker: str,
    date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load intraday tick data for one stock via AKShare.

    ticker examples:
        "000001"   → Ping An Bank (SZSE)
        "600519"   → Kweichow Moutai (SSE)
        "510300"   → CSI 300 ETF (SSE)

    Returns DataFrame with columns matching simulate_lob_day() schema
    but only 5 LOB levels, derived from AKShare tick stream.

    Note: AKShare only exposes ~5 days of intraday history.
    For longer history use load_minute_baostock() instead.
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("Run: pip install akshare")

    # Tick-by-tick trade data
    raw = ak.stock_intraday_em(symbol=ticker)
    # Columns: 时间, 成交价, 手数, 买卖盘性质  (time, price, volume, side)
    raw.columns = ["time", "price", "volume_lot", "side_code", "change", "change_pct"]

    raw["price"]      = pd.to_numeric(raw["price"],      errors="coerce")
    raw["volume_lot"] = pd.to_numeric(raw["volume_lot"], errors="coerce")
    raw["volume"]     = raw["volume_lot"] * LOT

    if date is None:
        date = pd.Timestamp.today().strftime("%Y-%m-%d")

    raw["timestamp"] = pd.to_datetime(date + " " + raw["time"])

    # side: '买盘' = buy-initiated, '卖盘' = sell-initiated
    raw["is_buy"] = raw["side_code"].str.contains("买", na=False)

    # Build cumulative volumes (needed for trade_imbalance signal)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    raw["cum_buy_vol"]  = (raw["volume"] *  raw["is_buy"]).cumsum()
    raw["cum_sell_vol"] = (raw["volume"] * ~raw["is_buy"]).cumsum()

    # Snap to 3-second grid (to match synthetic data cadence)
    raw = raw.set_index("timestamp")
    raw = raw[~raw.index.duplicated(keep="last")]
    grid = pd.date_range(
        raw.index[0].floor("30min"),
        raw.index[-1].ceil("30min"),
        freq="3s",
    )
    snapped = raw.reindex(grid, method="ffill").dropna(subset=["price"])

    # We only have mid from last trade price — use as best estimate
    # 5-level LOB comes from a separate snapshot call if needed
    mid = snapped["price"]

    out = pd.DataFrame(index=snapped.index)
    out["mid_price"]    = mid
    out["last_price"]   = mid
    out["last_volume"]  = snapped["volume"].fillna(0)
    out["cum_buy_vol"]  = snapped["cum_buy_vol"].ffill()
    out["cum_sell_vol"] = snapped["cum_sell_vol"].ffill()
    out["ticker"]       = ticker

    # Fake 1-tick spread for levels we cannot observe from tick stream alone
    for lv in range(1, 6):
        out[f"bid_px_{lv}"]  = mid - lv * TICK
        out[f"bid_vol_{lv}"] = LOT * 10
        out[f"ask_px_{lv}"]  = mid + lv * TICK
        out[f"ask_vol_{lv}"] = LOT * 10

    return out


def load_lob_snapshot_akshare(ticker: str) -> pd.DataFrame:
    """
    Fetch CURRENT 5-level bid/ask snapshot for one ticker via AKShare.

    Returns single-row DataFrame in the same column schema as simulate_lob_day().
    Useful for building a live signal monitor that polls every few seconds.

    Call this in a loop to build your own LOB snapshot history:
        while True:
            snap = load_lob_snapshot_akshare("000001")
            history.append(snap)
            time.sleep(3)
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("Run: pip install akshare")

    raw = ak.stock_bid_ask_em(symbol=ticker)
    # raw typically has rows: 卖5..卖1, 买1..买5 with price and volume

    now = pd.Timestamp.now().floor("s")
    row: dict = {"timestamp": now, "ticker": ticker}

    # Parse AKShare bid/ask format
    ask_rows = raw[raw["item"].str.contains("卖", na=False)].sort_values("item", ascending=False)
    bid_rows = raw[raw["item"].str.contains("买", na=False)].sort_values("item")

    for i, (_, r) in enumerate(bid_rows.head(5).iterrows(), start=1):
        row[f"bid_px_{i}"]  = float(r["price"])
        row[f"bid_vol_{i}"] = int(r["volume"]) * LOT

    for i, (_, r) in enumerate(ask_rows.head(5).iterrows(), start=1):
        row[f"ask_px_{i}"]  = float(r["price"])
        row[f"ask_vol_{i}"] = int(r["volume"]) * LOT

    # Fill missing levels if fewer than 5 returned
    if "bid_px_1" in row and "ask_px_1" in row:
        mid = (row["bid_px_1"] + row["ask_px_1"]) / 2.0
    else:
        mid = 0.0

    for lv in range(1, 6):
        row.setdefault(f"bid_px_{lv}",  mid - lv * TICK)
        row.setdefault(f"bid_vol_{lv}", LOT * 5)
        row.setdefault(f"ask_px_{lv}",  mid + lv * TICK)
        row.setdefault(f"ask_vol_{lv}", LOT * 5)

    row["mid_price"]    = mid
    row["last_price"]   = mid
    row["last_volume"]  = 0
    row["cum_buy_vol"]  = 0
    row["cum_sell_vol"] = 0

    return pd.DataFrame([row]).set_index("timestamp")


# ---------------------------------------------------------------------------
# BaoStock — 5-min OHLCV (full history, free, no LOB)
# ---------------------------------------------------------------------------

def load_minute_baostock(
    ticker: str,
    start_date: str,
    end_date: str,
    freq: str = "5",
) -> pd.DataFrame:
    """
    Load minute-bar OHLCV from BaoStock (free, full history since 2006).

    ticker format: "sh.600519" (SSE) or "sz.000001" (SZSE)

    freq: "1" | "5" | "15" | "30" | "60"  (minutes)

    Returns DataFrame with OHLCV + basic derived signals.
    No LOB data — use for lower-frequency momentum / auction gap signals only.

    Note: BaoStock requires login (free, no registration needed):
        import baostock as bs
        bs.login()   # called automatically here
    """
    try:
        import baostock as bs
    except ImportError:
        raise ImportError("Run: pip install baostock")

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")

    rs = bs.query_history_k_data_plus(
        ticker,
        fields="date,time,code,open,high,low,close,volume,amount,adjustflag",
        start_date=start_date,
        end_date=end_date,
        frequency=freq,
        adjustflag="3",    # no adjustment
    )

    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    bs.logout()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df.set_index("timestamp").sort_index()

    # Derive pseudo mid-price and basic LOB columns so it can feed
    # into momentum and auction gap signals (not OFI — no book data)
    df["mid_price"]    = (df["open"] + df["close"]) / 2.0
    df["bid_px_1"]     = df["close"] - TICK
    df["ask_px_1"]     = df["close"] + TICK
    df["bid_vol_1"]    = df["volume"] // 2
    df["ask_vol_1"]    = df["volume"] // 2
    df["last_price"]   = df["close"]
    df["last_volume"]  = df["volume"]
    df["cum_buy_vol"]  = (df["volume"] * 0.5).cumsum().astype(int)
    df["cum_sell_vol"] = (df["volume"] * 0.5).cumsum().astype(int)
    df["ticker"]       = ticker

    return df


# ---------------------------------------------------------------------------
# Live polling loop (AKShare)
# ---------------------------------------------------------------------------

def poll_lob_history(
    ticker: str,
    n_snapshots: int = 200,
    interval_sec: float = 3.0,
) -> pd.DataFrame:
    """
    Build a LOB snapshot history by polling AKShare every 3 seconds.

    Collects n_snapshots rows then returns the DataFrame.
    Run during trading hours (09:30–15:00 Beijing time).

    Returns same schema as simulate_lob_day() — directly compatible
    with build_feature_matrix() and run_backtest().

    Example:
        lob_df = poll_lob_history("000001", n_snapshots=100)
        feat   = build_feature_matrix(lob_df, ofi_levels=5)
        signal = build_composite_alpha(feat)
    """
    import time

    snaps = []
    print(f"Polling {ticker} every {interval_sec}s × {n_snapshots} …")

    for i in range(n_snapshots):
        snap = load_lob_snapshot_akshare(ticker)
        snaps.append(snap)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_snapshots}")
        time.sleep(interval_sec)

    df = pd.concat(snaps)
    df["cum_buy_vol"]  = 0   # not available from snapshot alone
    df["cum_sell_vol"] = 0
    return df
