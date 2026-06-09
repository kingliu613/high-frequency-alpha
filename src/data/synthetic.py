"""
Synthetic Chinese L2 market data generator.

Mimics the data format published by SSE/SZSE:
- 10-level order book (LOB) snapshots at 3-second intervals
- Opening call auction data (9:15–9:25)
- Trading sessions: 9:30–11:30, 13:00–15:00
- Price limits: ±10% (stocks), no limit for futures in simulation
- Tick size: 0.01 RMB
- Lot size: 100 shares

This module lets you prototype and unit-test signal code without
a Wind/Tushare subscription. Replace with real L2 feeds when live.
"""

import numpy as np
import pandas as pd

TICK = 0.01
LOT  = 100
N_LEVELS = 10


def _session_timestamps(date: str, freq_sec: int = 3) -> pd.DatetimeIndex:
    base = pd.Timestamp(date)
    morning = pd.date_range(
        base + pd.Timedelta(hours=9, minutes=30),
        base + pd.Timedelta(hours=11, minutes=30),
        freq=f"{freq_sec}s",
    )
    afternoon = pd.date_range(
        base + pd.Timedelta(hours=13),
        base + pd.Timedelta(hours=15),
        freq=f"{freq_sec}s",
    )
    return morning.append(afternoon)


def _vol_multiplier(ts: pd.Timestamp) -> float:
    """U-shaped intraday volatility: high at open/close, low midday."""
    t = ts.hour + ts.minute / 60.0
    if t < 10.0:
        return 2.0 + (10.0 - t) * 0.5
    elif t > 14.5:
        return 1.5 + (t - 14.5) * 1.5
    else:
        return 1.0


def simulate_lob_day(
    ticker: str = "IF2401.CFFEX",
    date: str = "2024-01-02",
    prev_close: float = 4000.0,
    daily_vol: float = 0.015,
    is_futures: bool = True,
    seed: int = 42,
    signal_strength: float = 0.05,
) -> pd.DataFrame:
    """
    Generate one day of synthetic Chinese L2 LOB snapshots.

    Returns DataFrame indexed by timestamp with columns:
        mid_price,
        bid_px_{1..10}, bid_vol_{1..10},
        ask_px_{1..10}, ask_vol_{1..10},
        last_price, last_volume,
        cum_buy_vol, cum_sell_vol

    Designed to match the Wind Level-2 / Tushare Pro snapshot format.

    signal_strength controls coupling between latent order-flow and price.
    Higher values produce stronger/more detectable alpha signals.
    """
    rng = np.random.default_rng(seed)
    tss = _session_timestamps(date)
    n   = len(tss)

    # Persistent latent order-flow pressure: AR(1), half-life ~30 ticks (90s).
    # Positive → buy pressure: price drifts up, bid side deeper, more buy trades.
    phi       = np.exp(-np.log(2) / 30)       # ≈ 0.977
    innov_std = np.sqrt(1.0 - phi ** 2)
    flow      = np.zeros(n)
    flow[0]   = rng.normal(0.0, 1.0)
    for i in range(1, n):
        flow[i] = phi * flow[i - 1] + rng.normal(0.0, innov_std)

    # Mid-price: GBM + flow-driven directional drift
    log_ret   = np.zeros(n)
    dt_frac   = 3.0 / (4.0 * 3600.0)
    price_cap = 0.10 if not is_futures else 0.15

    for i, ts in enumerate(tss):
        sigma      = daily_vol * _vol_multiplier(ts) * np.sqrt(dt_frac)
        drift      = flow[i] * sigma * signal_strength
        log_ret[i] = drift + rng.normal(0.0, sigma)

    prices = prev_close * np.exp(np.cumsum(log_ret))
    prices = np.clip(prices, prev_close * (1 - price_cap), prev_close * (1 + price_cap))
    prices = np.round(prices / TICK) * TICK

    vol_state = float(rng.lognormal(7.5, 0.8)) * LOT   # OU initial state
    records = []
    cum_buy  = 0
    cum_sell = 0

    for i, (ts, mid) in enumerate(zip(tss, prices)):
        vm = _vol_multiplier(ts)
        f  = float(flow[i])

        # Spread widens with volatility regime; 1–4 ticks
        n_spread_ticks = rng.integers(1, max(2, int(4 * vm)))
        half = n_spread_ticks * TICK / 2.0
        b1 = np.round((mid - half) / TICK) * TICK
        a1 = np.round((mid + half) / TICK) * TICK

        row = {"timestamp": ts, "mid_price": mid}

        # Build 10-level book; volume decays with depth.
        # Flow tilts bid/ask depth: positive flow → deeper bid, shallower ask.
        # OU mean-reversion for LOB base volume (replaces iid lognormal)
        vol_target = float(rng.lognormal(7.5, 0.8)) * LOT
        vol_state  = 0.7 * vol_state + 0.3 * vol_target
        base_vol   = max(LOT, int(vol_state))
        tilt     = np.exp(np.clip(f * 0.25, -2.0, 2.0))   # bid multiplier ∈ [e^-2, e^2]

        for lv in range(1, N_LEVELS + 1):
            decay = np.exp(-0.25 * (lv - 1))
            bv = max(LOT, int(base_vol * decay * tilt       * rng.lognormal(0.0, 0.4) / LOT) * LOT)
            av = max(LOT, int(base_vol * decay / tilt       * rng.lognormal(0.0, 0.4) / LOT) * LOT)
            row[f"bid_px_{lv}"]  = np.round((b1 - (lv - 1) * TICK) / TICK) * TICK
            row[f"bid_vol_{lv}"] = bv
            row[f"ask_px_{lv}"]  = np.round((a1 + (lv - 1) * TICK) / TICK) * TICK
            row[f"ask_vol_{lv}"] = av

        # Trade direction biased by flow: P(buy) ∈ [0.2, 0.8]
        p_buy    = 0.5 + 0.3 * np.tanh(f)
        is_buy   = rng.random() < p_buy
        trd_vol  = max(LOT, int(rng.lognormal(6.5, 0.8) / LOT) * LOT)
        row["last_price"]  = a1 if is_buy else b1
        row["last_volume"] = trd_vol

        if is_buy:
            cum_buy  += trd_vol
        else:
            cum_sell += trd_vol

        row["cum_buy_vol"]  = cum_buy
        row["cum_sell_vol"] = cum_sell
        records.append(row)

    df = pd.DataFrame(records).set_index("timestamp")
    df["ticker"] = ticker
    return df


def simulate_etf_series(lob_df: pd.DataFrame, seed: int = 0) -> pd.Series:
    """
    Simulate ETF price as NAV (LOB mid) plus AR(1) premium.

        premium[i] = 0.98 * premium[i-1] + N(0, σ_innov)
        σ_innov    = 0.001 * sqrt(1 - 0.98²)   ≈ 0.0002

    Models realistic ETF premium/discount to NAV (~10 bps persistence).
    """
    rng = np.random.default_rng(seed + 77777)
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]).astype(float) / 2.0
    n   = len(mid)

    phi       = 0.98
    sigma_inn = 0.001 * np.sqrt(1.0 - phi ** 2)
    premium   = np.zeros(n)
    premium[0] = rng.normal(0.0, 0.001)
    for i in range(1, n):
        premium[i] = phi * premium[i - 1] + rng.normal(0.0, sigma_inn)

    etf_px = mid.values * (1.0 + premium)
    return pd.Series(etf_px, index=lob_df.index, name="etf_px")


def simulate_auction_data(
    ticker: str = "IF2401.CFFEX",
    date: str = "2024-01-02",
    prev_close: float = 4000.0,
    daily_vol: float = 0.015,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """
    Generate opening call auction data (9:15–9:25).

    Chinese auction mechanics:
    - 9:15–9:20: orders placed, no cancellation allowed after 9:20
    - Indicative price published every 3s
    - At 9:25: market clears at volume-maximising price
    - 9:25–9:30: cooling period (no trading)

    Returns (auction_df, open_price)
    """
    rng = np.random.default_rng(seed + 9999)
    base = pd.Timestamp(date)
    tss  = pd.date_range(
        base + pd.Timedelta(hours=9, minutes=15),
        base + pd.Timedelta(hours=9, minutes=25),
        freq="3s",
    )
    n = len(tss)

    # True information gap (drives opening price)
    true_gap = np.clip(rng.normal(0.0, daily_vol * 0.8), -0.05, 0.05)
    open_price = np.round(prev_close * (1 + true_gap) / TICK) * TICK

    # Cumulative auction volumes grow through session
    fracs = np.linspace(0.1, 1.0, n)
    base_vol = 1_000_000

    cum_buy  = np.zeros(n, dtype=int)
    cum_sell = np.zeros(n, dtype=int)

    for i, frac in enumerate(fracs):
        noise = rng.lognormal(0, 0.3)
        total = int(base_vol * frac * noise)
        if true_gap >= 0:
            buy_frac  = 0.5 + 0.4 * abs(true_gap) / 0.05
        else:
            buy_frac  = 0.5 - 0.4 * abs(true_gap) / 0.05
        cum_buy[i]  = int(total * buy_frac)
        cum_sell[i] = int(total * (1 - buy_frac))

    indicative = prev_close + (open_price - prev_close) * np.linspace(0.2, 1.0, n)
    indicative += rng.normal(0, daily_vol * prev_close * 0.05, n)
    indicative  = np.round(np.clip(indicative,
                                   prev_close * 0.9,
                                   prev_close * 1.1) / TICK) * TICK
    indicative[-1] = open_price

    df = pd.DataFrame({
        "indicative_price": indicative,
        "cum_buy_vol":       cum_buy,
        "cum_sell_vol":      cum_sell,
        "prev_close":        prev_close,
        "ticker":            ticker,
    }, index=pd.DatetimeIndex(tss, name="timestamp"))

    return df, float(open_price)
