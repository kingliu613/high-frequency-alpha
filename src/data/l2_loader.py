"""
L2+L3 real data loader for raw A-share parquet files.

Converts per-day parquet files:
  data/{YYYYMMDD}/tick.parquet    — 10-level LOB snapshots (3s cadence, all stocks)
  data/{YYYYMMDD}/order.parquet   — per-order events (all stocks)
  data/{YYYYMMDD}/trans.parquet   — per-trade records (all stocks)

→ single-stock LOB DataFrame in the standard pipeline schema,
  identical format to simulate_lob_day() so all existing signals
  and build_feature_matrix() work without modification.

order_kind encoding (ASCII):
  '0' (48) = call-auction limit order
  'A' (65) = continuous-session limit order submission  → limit_{buy,sell}_vol
  'D' (68) = cancellation                              → cancel_{buy,sell}_vol
  '1' (49) = market order                              → market_{buy,sell}_vol
  'U' (85) = order update (rare, ignored)

func_code encoding:
  'B' (66) = buy side
  'S' (83) = sell side

trans.bs encoding:
  'B' (66) = buyer-initiated trade
  'S' (83) = seller-initiated trade
  0        = unknown (opening-auction matched, ignored for direction)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SESSION_BOUNDS: dict[str, tuple[int, int]] = {
    "continuous": (93000000, 150000000),   # 9:30:00–15:00:00
    "auction":    (91500000, 92500000),    # 9:15:00–9:25:00
    "all":        (0, 999999999),
}

_LIMIT_KINDS  = frozenset([48, 65])   # '0' (SZSE limit), 'A' (SSE limit)
_CANCEL_KINDS = frozenset([68])       # 'D' (SSE cancel — SZSE uses trans.parquet instead)
_MARKET_KINDS = frozenset([49])       # '1'
_BUY  = 66   # 'B'
_SELL = 83   # 'S'

_TRANS_CANCEL = 67   # 'C' in trans.parquet — SZSE cancellation records
_SZSE_MAX_CODE = 599999   # codes below this are SZSE (Shenzhen); above are SSE (Shanghai)

_TICK_RENAME: dict[str, str] = {
    **{f"bid_price_{i}": f"bid_px_{i}"  for i in range(1, 11)},
    **{f"bid_volume_{i}": f"bid_vol_{i}" for i in range(1, 11)},
    **{f"ask_price_{i}": f"ask_px_{i}"  for i in range(1, 11)},
    **{f"ask_volume_{i}": f"ask_vol_{i}" for i in range(1, 11)},
    "bid_vol_all":   "bid_depth",
    "ask_vol_all":   "ask_depth",
    "pre_close":     "prev_close",
    "price":         "last_price",
    "volume":        "last_volume",
    "acc_volume":    "_acc_volume",
    "acc_turnover":  "_acc_turnover",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_to_timedelta(t: pd.Series) -> pd.Series:
    """Convert HHMMSSMMM integer → pd.TimedeltaIndex."""
    ms   = t % 1000
    rest = t // 1000
    ss   = rest % 100
    rest = rest // 100
    mm   = rest % 100
    hh   = rest // 100
    return (
        pd.to_timedelta(hh * 3600 + mm * 60 + ss, unit="s")
        + pd.to_timedelta(ms, unit="ms")
    )


def _read_stock(path: Path, code: int, columns: list[str] | None = None) -> pd.DataFrame:
    """Read parquet with code-level predicate pushdown (avoids loading all stocks)."""
    kwargs: dict = {"filters": [("code", "==", code)]}
    if columns is not None:
        kwargs["columns"] = ["code"] + [c for c in columns if c != "code"]
    return pq.read_table(str(path), **kwargs).to_pandas()


def _assign_bins(event_times: np.ndarray, tick_times: np.ndarray) -> np.ndarray:
    """
    Map each event time to the preceding tick snapshot index.
    Events in [tick_times[i], tick_times[i+1]) → bin i.
    Events before tick_times[0] → bin -1 (caller must filter out).
    """
    return np.searchsorted(tick_times, event_times, side="right") - 1


def _agg_order_col(
    order: pd.DataFrame,
    kind_set: frozenset,
    side: int,
    n_ticks: int,
) -> np.ndarray:
    mask = order["order_kind"].isin(kind_set) & (order["func_code"] == side)
    vol  = order.loc[mask].groupby("_bin")["trade_vol"].sum()
    return vol.reindex(range(n_ticks), fill_value=0).values.astype(float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_l2_day(
    data_dir: str | Path,
    date: str,
    code: int,
    session: str = "continuous",
    include_trans: bool = True,
) -> pd.DataFrame:
    """
    Load one stock's L2+L3 data for one trading day from raw parquet files.

    Parameters
    ----------
    data_dir      : root data directory (contains {YYYYMMDD}/ subdirs)
    date          : trading date "YYYYMMDD"
    code          : integer stock code  (1 = 000001.SZ, 600000 = 600000.SH)
    session       : "continuous" (9:30–15:00), "auction" (9:15–9:25), "all"
    include_trans : use trans.parquet for exact cum_buy/sell_vol (recommended)
                    when False, estimates direction from market order volumes

    Returns
    -------
    pd.DataFrame with DatetimeIndex, compatible with simulate_lob_day() schema:
      bid_px_1..10, bid_vol_1..10, ask_px_1..10, ask_vol_1..10
      bid_depth, ask_depth, prev_close
      last_price, last_volume
      limit_buy_vol, limit_sell_vol
      cancel_buy_vol, cancel_sell_vol
      market_buy_vol, market_sell_vol
      cum_buy_vol, cum_sell_vol, cum_buy_count, cum_sell_count
      mid_price
    """
    day_dir = Path(data_dir) / date
    t_lo, t_hi = _SESSION_BOUNDS[session]

    # ------------------------------------------------------------------
    # Step 1: LOB snapshots from tick.parquet
    # ------------------------------------------------------------------
    tick_raw = _read_stock(day_dir / "tick.parquet", code)
    tick_raw = (
        tick_raw[(tick_raw["time"] >= t_lo) & (tick_raw["time"] < t_hi)]
        .sort_values("time")
        .reset_index(drop=True)
    )
    if tick_raw.empty:
        raise ValueError(f"No tick data for code={code} date={date} session={session}")

    trade_date = pd.Timestamp(date)
    idx = trade_date + _int_to_timedelta(tick_raw["time"])

    tick = tick_raw.rename(columns=_TICK_RENAME).copy()
    tick.index = idx

    # Forward-fill last_price (=0 when no trade occurred at this snapshot)
    tick["last_price"] = tick["last_price"].replace(0.0, np.nan).ffill().fillna(
        tick.get("bid_px_1", pd.Series(np.nan, index=tick.index))
    )
    tick["mid_price"] = (tick["bid_px_1"] + tick["ask_px_1"]) / 2.0

    # ------------------------------------------------------------------
    # Step 2: order events → 6 event columns
    # ------------------------------------------------------------------
    order_raw = _read_stock(
        day_dir / "order.parquet", code,
        columns=["time", "order_kind", "func_code", "trade_vol"],
    )
    order_raw = order_raw[
        (order_raw["time"] >= t_lo) & (order_raw["time"] < t_hi)
    ].copy()

    tick_times = tick_raw["time"].values
    n = len(tick_times)

    order_raw["_bin"] = _assign_bins(order_raw["time"].values, tick_times)
    order_valid = order_raw[order_raw["_bin"] >= 0]

    tick["limit_buy_vol"]   = _agg_order_col(order_valid, _LIMIT_KINDS,  _BUY,  n)
    tick["limit_sell_vol"]  = _agg_order_col(order_valid, _LIMIT_KINDS,  _SELL, n)
    tick["cancel_buy_vol"]  = _agg_order_col(order_valid, _CANCEL_KINDS, _BUY,  n)
    tick["cancel_sell_vol"] = _agg_order_col(order_valid, _CANCEL_KINDS, _SELL, n)
    tick["market_buy_vol"]  = _agg_order_col(order_valid, _MARKET_KINDS, _BUY,  n)
    tick["market_sell_vol"] = _agg_order_col(order_valid, _MARKET_KINDS, _SELL, n)

    # ------------------------------------------------------------------
    # Step 3: trade records → cumulative buy/sell + SZSE cancel volumes
    #
    # SZSE (code ≤ 599999): trans.parquet func_code='C' (67) = cancellation
    #   Direction inferred: bid_order>0 & ask_order==0 → buy cancel
    #                       ask_order>0 & bid_order==0 → sell cancel
    # SSE  (code ≥ 600000): cancels already in order.parquet as order_kind='D'
    # ------------------------------------------------------------------
    is_szse = code <= _SZSE_MAX_CODE

    if include_trans:
        trans_cols = ["time", "bs", "trd_vol", "func_code"]
        if is_szse:
            trans_cols += ["ask_order", "bid_order"]
        trans_raw = _read_stock(day_dir / "trans.parquet", code, columns=trans_cols)
        trans_raw = trans_raw[
            (trans_raw["time"] >= t_lo) & (trans_raw["time"] < t_hi)
        ].copy()

        trans_raw["_bin"] = _assign_bins(trans_raw["time"].values, tick_times)
        trans_valid = trans_raw[trans_raw["_bin"] >= 0]

        # Execution records only (func_code='0' = 48, exclude cancel='C' = 67)
        trans_exec = trans_valid[trans_valid["func_code"] == 48]

        def _cum_pair(side: int) -> tuple[np.ndarray, np.ndarray]:
            sub = trans_exec[trans_exec["bs"] == side]
            vol = sub.groupby("_bin")["trd_vol"].sum().reindex(range(n), fill_value=0)
            cnt = sub.groupby("_bin").size().reindex(range(n), fill_value=0)
            return vol.values.astype(float).cumsum(), cnt.values.astype(float).cumsum()

        tick["cum_buy_vol"],  tick["cum_buy_count"]  = _cum_pair(_BUY)
        tick["cum_sell_vol"], tick["cum_sell_count"] = _cum_pair(_SELL)

        # SZSE: extract cancel volumes from trans (SSE already has them from order)
        if is_szse:
            tc = trans_valid[trans_valid["func_code"] == _TRANS_CANCEL].copy()
            buy_mask  = (tc["bid_order"] > 0) & (tc["ask_order"] == 0)
            sell_mask = (tc["ask_order"] > 0) & (tc["bid_order"] == 0)
            c_buy  = tc[buy_mask].groupby("_bin")["trd_vol"].sum()
            c_sell = tc[sell_mask].groupby("_bin")["trd_vol"].sum()
            tick["cancel_buy_vol"]  = c_buy.reindex(range(n),  fill_value=0).values.astype(float)
            tick["cancel_sell_vol"] = c_sell.reindex(range(n), fill_value=0).values.astype(float)

    else:
        # Estimate direction from market order volumes (fallback, less accurate)
        acc      = tick["_acc_volume"].astype(float).clip(lower=0)
        bar_vol  = acc.diff().clip(lower=0).fillna(0.0)
        mkt_buy  = tick["market_buy_vol"]
        mkt_sell = tick["market_sell_vol"]
        total    = (mkt_buy + mkt_sell).replace(0, np.nan)
        buy_frac = (mkt_buy / total).fillna(0.5)
        tick["cum_buy_vol"]   = (bar_vol * buy_frac).cumsum()
        tick["cum_sell_vol"]  = (bar_vol * (1 - buy_frac)).cumsum()
        tick["cum_buy_count"] = mkt_buy.gt(0).astype(float).cumsum()
        tick["cum_sell_count"] = mkt_sell.gt(0).astype(float).cumsum()

    # ------------------------------------------------------------------
    # Step 4: drop internal/raw columns not in pipeline schema
    # ------------------------------------------------------------------
    drop = [c for c in tick.columns if c.startswith("_") or c in {"code", "date", "time"}]
    tick = tick.drop(columns=[c for c in drop if c in tick.columns])

    return tick


def list_available_dates(data_dir: str | Path) -> list[str]:
    """Return sorted list of date strings with complete parquet data."""
    base = Path(data_dir)
    dates = sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        and (d / "tick.parquet").exists()
        and (d / "order.parquet").exists()
    )
    return dates
