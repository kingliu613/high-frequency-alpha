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

    # T+1 settlement (stocks only): shares bought today cannot be sold today.
    # When True and instrument == "stock", intraday exits are disabled and the
    # position is force-closed on the final snapshot (next-day-open proxy).
    enforce_t1:   bool  = True

    entry_z:      float = 1.5
    exit_z:       float = 0.3
    max_hold:     int   = 20
    stop_loss_bp: float = 15.0

    position_size:     int   = 1      # kept for backward compat; sizing now signal-driven
    max_position_size: int   = 3
    use_regime_filter: bool  = True
    regime_ic_off:     float = 0.0
    regime_ic_on:      float = 0.02

    @classmethod
    def default_for(cls, instrument: str, **overrides) -> "MarketParams":
        """
        Realistic per-instrument cost defaults.

        futures (IF/IC/IH on CFFEX): commission ≈ 0.23 bp, no stamp duty
        stock (A-share): commission ≈ 2.5 bp/side + 0.05% stamp duty on sells,
                         T+1 enforced, ±10% price limit
        """
        base: dict = {"instrument": instrument}
        if instrument == "stock":
            base.update(commission=0.00025, stamp_duty=0.0005,
                        enforce_t1=True, price_limit=0.10)
        else:
            base.update(commission=0.000023, stamp_duty=0.0)
        base.update(overrides)
        return cls(**base)

    # Toxicity gate: entries blocked when exposure_scale < gate_block_below
    gate_block_below: float = 0.3

    # Slippage & market impact
    slippage_model:          str   = "lob_walk"  # "none" | "fixed" | "lob_walk"
    slippage_fixed_ticks:    float = 1.0          # ticks per side (fixed model only)
    market_impact_coef:      float = 0.5          # bps × √(size_lots / top_depth_lots)
    market_impact_perm_frac: float = 0.2          # permanent fraction charged at both legs


@dataclass
class _Pos:
    direction:    int   = 0
    entry_price:  float = 0.0
    entry_idx:    int   = 0
    size:         int   = 0
    entry_signal: float = 0.0
    entry_impact: float = 0.0   # temporary impact paid at entry (for perm component)


def _txn_cost(price: float, size: int, p: MarketParams, is_sell: bool) -> float:
    notional = price * size * LOT
    cost = notional * p.commission
    if p.instrument == "stock" and is_sell:
        cost += notional * p.stamp_duty
    return cost


def _slippage_cost(
    lob_row: "pd.Series",
    direction: int,
    size: int,
    params: MarketParams,
) -> float:
    """
    Slippage cost in RMB for `size` lots traded in `direction`.

    "none"     : zero (signal-research mode)
    "fixed"    : params.slippage_fixed_ticks × TICK × size × LOT per side
    "lob_walk" : walk the visible 10-level book; penalise remainder beyond
                 level 10 with 5 extra ticks in the adverse direction.

    direction: +1 = buy (lift ask), -1 = sell (hit bid).
    Returns cost >= 0.
    """
    if params.slippage_model == "none":
        return 0.0

    mid = (float(lob_row["bid_px_1"]) + float(lob_row["ask_px_1"])) / 2.0

    if params.slippage_model == "fixed":
        return params.slippage_fixed_ticks * TICK * size * LOT

    # lob_walk
    side = "ask" if direction > 0 else "bid"
    remaining    = float(size)
    filled_value = 0.0

    for lv in range(1, 11):
        px         = float(lob_row[f"{side}_px_{lv}"])
        depth_lots = float(lob_row[f"{side}_vol_{lv}"]) / LOT
        take       = min(remaining, depth_lots)
        filled_value += take * px
        remaining    -= take
        if remaining <= 0.0:
            break

    if remaining > 0.0:
        # Beyond visible book: estimate 5-tick penalty per lot
        last_px = float(lob_row[f"{side}_px_10"])
        filled_value += remaining * (last_px + direction * 5.0 * TICK)

    avg_fill          = filled_value / size
    slip_per_share    = direction * (avg_fill - mid)   # positive for buys
    return max(0.0, slip_per_share) * size * LOT


def _market_impact_cost(
    lob_row: "pd.Series",
    size: int,
    params: MarketParams,
) -> float:
    """
    Square-root temporary market impact in RMB (Almgren-Chriss inspired).

        impact_bps = coef × √(size_lots / avg_top_depth_lots)

    perm_frac of this is also charged at the opposing leg to model permanent
    price shift.  The temporary component is charged at both entry and exit.
    """
    if params.market_impact_coef <= 0.0:
        return 0.0

    mid = (float(lob_row["bid_px_1"]) + float(lob_row["ask_px_1"])) / 2.0
    top_depth_lots = (
        float(lob_row["bid_vol_1"]) + float(lob_row["ask_vol_1"])
    ) / (2.0 * LOT)
    top_depth_lots = max(top_depth_lots, 0.01)

    impact_bps = params.market_impact_coef * np.sqrt(size / top_depth_lots)
    return mid * (impact_bps / 10_000.0) * size * LOT


def run_backtest(
    lob_df: pd.DataFrame,
    signal: pd.Series,
    params: Optional[MarketParams] = None,
    exposure_scale: Optional[pd.Series] = None,
    prev_close: Optional[float] = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Vectorised-style (but loop-based for correctness) threshold backtest.

    Parameters
    ----------
    exposure_scale : optional Series in [0, 1] aligned to lob_df (e.g. from
        signals.advanced.exposure_gate). Entry size is multiplied by the scale
        at entry time (rounded, min 1 lot); entries are blocked entirely when
        the scale drops below params.gate_block_below. Models VPIN/λ toxicity
        gating — high order-flow toxicity or thin liquidity cuts position size
        rather than flipping direction.
    prev_close : previous close price; enables price-limit non-tradability for
        stocks (buy entries blocked at a sealed up-limit — there is no ask side
        to lift; sells flagged at the down-limit).

    Chinese market realism enforced here:
      - T+1 (stock + enforce_t1): no intraday exits; position force-closed on
        the final snapshot as a next-day-open proxy (exit_reason "eod_t1").
      - Price limits (stock + prev_close): buy entries blocked when the ask is
        at/above the up-limit; an EOD close with the bid at/below the down-limit
        is tagged "eod_limit_down" — in reality that position could NOT be sold.
      - Any position still open at the last snapshot is closed with full costs
        (futures too — previously dangling positions vanished without exit
        costs or a trade-log row).

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

    if exposure_scale is not None:
        exp_scale = exposure_scale.reindex(lob_df.index).ffill().fillna(1.0).clip(0.0, 1.0)
    else:
        exp_scale = None

    is_stock = params.instrument == "stock"
    t1_mode  = is_stock and params.enforce_t1
    up_limit   = prev_close * (1.0 + params.price_limit) if (is_stock and prev_close) else None
    down_limit = prev_close * (1.0 - params.price_limit) if (is_stock and prev_close) else None

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
        s    = float(sig.iloc[i])
        m    = float(mid.iloc[i])
        ts   = mid.index[i]
        prev = float(mid.iloc[i - 1])
        row  = lob_df.iloc[i]   # full LOB snapshot for slippage/impact calc

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

            # T+1: shares bought today cannot be sold today — suppress all
            # intraday exits; the position is closed at EOD after the loop.
            if t1_mode:
                exit_on_flip = exit_on_hold = exit_on_stop = False

            if exit_on_flip or exit_on_hold or exit_on_stop:
                slip   = _slippage_cost(row, -pos.direction, pos.size, params)
                impact = _market_impact_cost(row, pos.size, params) \
                       + params.market_impact_perm_frac * pos.entry_impact
                cost   = _txn_cost(m, pos.size, params, is_sell=(pos.direction > 0))
                pnl.iloc[i] -= (slip + impact + cost)

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
                    "cost":        cost + slip + impact,
                    "slip_cost":   slip,
                    "impact_cost": impact,
                })
                pos = _Pos()

        # --- Entry (skipped when in regime-off) ---
        if pos.direction == 0 and not in_regime_off:
            want_long  = s >  params.entry_z
            want_short = s < -params.entry_z and params.instrument == "futures"

            # Price-limit non-tradability: a sealed up-limit has no ask side to
            # lift — buying is impossible, not merely expensive.
            if want_long and up_limit is not None and float(ask.iloc[i]) >= up_limit:
                want_long = False

            if want_long or want_short:
                direction = 1 if want_long else -1
                # Signal-proportional size: floor(|s|/entry_z), clamped [1, max_position_size]
                size      = int(np.clip(np.floor(abs(s) / params.entry_z), 1, params.max_position_size))
                # Toxicity/liquidity gate: block entry at extreme toxicity,
                # otherwise scale size down (round, min 1 lot)
                if exp_scale is not None:
                    scale = float(exp_scale.iloc[i])
                    if scale < params.gate_block_below:
                        continue
                    size = max(1, int(round(size * scale)))
                slip      = _slippage_cost(row, direction, size, params)
                impact    = _market_impact_cost(row, size, params)
                cost      = _txn_cost(m, size, params, is_sell=False)
                pnl.iloc[i] -= (slip + impact + cost)
                pos = _Pos(
                    direction    = direction,
                    entry_price  = m,
                    entry_idx    = i,
                    size         = size,
                    entry_signal = abs(s),
                    entry_impact = impact,
                )

    # --- Force-close any open position on the final snapshot ---
    # Without this, dangling positions accrued MTM PnL but never paid exit
    # costs and never appeared in the trade log. For T+1 stocks this close is
    # the next-day-open proxy; "eod_limit_down" flags a close that would have
    # been impossible in reality (bid sealed at the down-limit).
    if pos.direction != 0:
        i    = len(mid) - 1
        m    = float(mid.iloc[i])
        row  = lob_df.iloc[i]
        slip   = _slippage_cost(row, -pos.direction, pos.size, params)
        impact = _market_impact_cost(row, pos.size, params) \
               + params.market_impact_perm_frac * pos.entry_impact
        cost   = _txn_cost(m, pos.size, params, is_sell=(pos.direction > 0))
        pnl.iloc[i] -= (slip + impact + cost)

        reason = "eod_t1" if t1_mode else "eod"
        if (down_limit is not None and pos.direction > 0
                and float(bid.iloc[i]) <= down_limit):
            reason = "eod_limit_down"

        trades.append({
            "entry_time":  mid.index[pos.entry_idx],
            "exit_time":   mid.index[i],
            "direction":   pos.direction * pos.size,
            "entry_price": pos.entry_price,
            "exit_price":  m,
            "hold_ticks":  i - pos.entry_idx,
            "exit_reason": reason,
            "gross_pnl":   pos.direction * (m - pos.entry_price) * pos.size * LOT,
            "cost":        cost + slip + impact,
            "slip_cost":   slip,
            "impact_cost": impact,
        })
        pos = _Pos()

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "direction", "entry_price",
                 "exit_price", "hold_ticks", "exit_reason", "gross_pnl",
                 "cost", "slip_cost", "impact_cost"]
    )
    return pnl, trades_df
