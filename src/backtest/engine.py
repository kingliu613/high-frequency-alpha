"""
Threshold-based backtest engine with Chinese market constraints.

Key Chinese market rules modelled:
  Stocks (A-share)
    - T+1: shares bought today cannot be sold today
      → long-only mode; short via 融券 (margin lending) is possible but
        illiquid; default: stocks treated as long-only
    - Stamp duty: 0.1% on sell side only (2023 reduced to 0.05%)
    - Commission: ~0.025–0.03% per side
    - Price limits: ±10% (main board), ±20% (科创板/创业板)

  Index Futures (IF/IC/IH/IM on CFFEX)
    - T+0: can buy and sell same contract same day
    - No stamp duty
    - Commission: ~2–3 RMB per 10,000 RMB notional (≈ 0.000023)
    - No individual price limit (but circuit breakers at ±5%/10%)
    - Margin: ~12% of notional for IF

Strategy logic (threshold):
  Entry : |signal| > entry_z  →  long (sig > 0) or short (sig < 0)
  Exit  : signal reverses past -exit_z / +exit_z, OR max_hold exceeded,
          OR stop-loss triggered
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


TICK = 0.01
LOT  = 100


@dataclass
class MarketParams:
    """Configurable per-instrument parameters."""

    instrument: str   = "futures"
    price_limit: float = 0.10

    commission:   float = 0.000023
    stamp_duty:   float = 0.0005

    entry_z:      float = 1.5
    exit_z:       float = 0.3
    max_hold:     int   = 20
    stop_loss_bp: float = 15.0

    position_size:     int   = 1      # kept for backward compat; sizing now signal-driven
    max_position_size: int   = 3
    use_regime_filter: bool  = True
    regime_ic_off:     float = 0.0
    regime_ic_on:      float = 0.02


@dataclass
class _Pos:
    direction:    int   = 0
    entry_price:  float = 0.0
    entry_idx:    int   = 0
    size:         int   = 0
    entry_signal: float = 0.0


def _txn_cost(price: float, size: int, p: MarketParams, is_sell: bool) -> float:
    notional = price * size * LOT
    cost = notional * p.commission
    if p.instrument == "stock" and is_sell:
        cost += notional * p.stamp_duty
    return cost


def run_backtest(
    lob_df: pd.DataFrame,
    signal: pd.Series,
    params: Optional[MarketParams] = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Vectorised-style (but loop-based for correctness) threshold backtest.

    Returns
    -------
    pnl : pd.Series   per-snapshot PnL (zero when flat)
    trades : pd.DataFrame  trade log with entry/exit details
    """
    if params is None:
        params = MarketParams()

    bid = lob_df["bid_px_1"].astype(float)
    ask = lob_df["ask_px_1"].astype(float)
    mid = (bid + ask) / 2.0
    sig = signal.reindex(lob_df.index).fillna(0.0)

    pnl    = pd.Series(0.0, index=lob_df.index)
    trades = []
    pos    = _Pos()

    # Regime filter state
    from collections import deque
    regime_buf     = deque(maxlen=200)   # (signal_t-1, realized_return_t-1→t)
    regime_ic      = 0.0
    # Start in regime-off if filter is enabled and initial IC (0.0) is below ic_off
    in_regime_off  = params.use_regime_filter and (0.0 < params.regime_ic_off)

    for i in range(1, len(mid)):
        s           = float(sig.iloc[i])
        m           = float(mid.iloc[i])
        ts          = mid.index[i]
        prev        = float(mid.iloc[i - 1])
        half_spread = (float(ask.iloc[i]) - float(bid.iloc[i])) / 2.0

        # --- Regime filter update (no look-ahead: uses realized return at i-1→i) ---
        if i >= 2:
            realized = m / prev - 1.0 if prev != 0.0 else 0.0
            regime_buf.append((float(sig.iloc[i - 1]), realized))

        if params.use_regime_filter and len(regime_buf) >= 50 and i % 20 == 0:
            sigs_r = [x[0] for x in regime_buf]
            rets_r = [x[1] for x in regime_buf]
            ra = np.argsort(np.argsort(sigs_r)).astype(float)
            rb = np.argsort(np.argsort(rets_r)).astype(float)
            denom = ra.std() * rb.std()
            regime_ic = float(np.corrcoef(ra, rb)[0, 1]) if denom > 0 else 0.0
            if in_regime_off and regime_ic > params.regime_ic_on:
                in_regime_off = False
            elif not in_regime_off and regime_ic < params.regime_ic_off:
                in_regime_off = True

        # --- Mark-to-market (mid-to-mid) ---
        if pos.direction != 0:
            pnl.iloc[i] = pos.direction * pos.size * (m - prev) * LOT

            hold_ticks = i - pos.entry_idx
            unreal_ret = pos.direction * (m - pos.entry_price) / pos.entry_price if pos.entry_price != 0 else 0.0

            # Dynamic hold cap: base × (1 + 0.5 × |entry_signal| / entry_z), max 40
            hold_cap = int(params.max_hold * (1.0 + 0.5 * pos.entry_signal / params.entry_z))
            hold_cap = min(hold_cap, 40)

            exit_on_flip  = (pos.direction > 0 and s < -params.exit_z) or \
                            (pos.direction < 0 and s >  params.exit_z)
            exit_on_hold  = hold_ticks >= hold_cap
            exit_on_stop  = unreal_ret < -(params.stop_loss_bp / 10_000)

            if exit_on_flip or exit_on_hold or exit_on_stop:
                spread_cost = half_spread * pos.size * LOT
                cost        = _txn_cost(m, pos.size, params, is_sell=(pos.direction > 0))
                pnl.iloc[i] -= (spread_cost + cost)

                reason = (
                    "flip"    if exit_on_flip  else
                    "stop"    if exit_on_stop  else
                    "timeout"
                )
                trades.append({
                    "entry_time":  mid.index[pos.entry_idx],
                    "exit_time":   ts,
                    "direction":   pos.direction * pos.size,
                    "entry_price": pos.entry_price,
                    "exit_price":  m,
                    "hold_ticks":  hold_ticks,
                    "exit_reason": reason,
                    "gross_pnl":   pos.direction * (m - pos.entry_price) * pos.size * LOT,
                    "cost":        cost + spread_cost,
                })
                pos = _Pos()

        # --- Entry (skipped when in regime-off) ---
        if pos.direction == 0 and not in_regime_off:
            want_long  = s >  params.entry_z
            want_short = s < -params.entry_z and params.instrument == "futures"

            if want_long or want_short:
                direction   = 1 if want_long else -1
                # Signal-proportional size: floor(|s|/entry_z), clamped [1, max_position_size]
                size        = int(np.clip(np.floor(abs(s) / params.entry_z), 1, params.max_position_size))
                spread_cost = half_spread * size * LOT
                cost        = _txn_cost(m, size, params, is_sell=False)
                pnl.iloc[i] -= (spread_cost + cost)
                pos = _Pos(
                    direction    = direction,
                    entry_price  = m,
                    entry_idx    = i,
                    size         = size,
                    entry_signal = abs(s),
                )

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "direction", "entry_price",
                 "exit_price", "hold_ticks", "exit_reason", "gross_pnl", "cost"]
    )
    return pnl, trades_df
