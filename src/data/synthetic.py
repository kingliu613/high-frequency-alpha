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

import re

import numpy as np
import pandas as pd

TICK = 0.01
LOT  = 100
N_LEVELS = 10

# ---------------------------------------------------------------------------
# Contract specifications for Chinese futures markets
# ---------------------------------------------------------------------------
# night_end: "HH:MM" end of night session; if hour < 21, it is next calendar
#            day (e.g. "01:00" = 01:00 on trading date). None = no night.
# day_sessions: list of ("HH:MM", "HH:MM") open/close pairs for day trading.

COMMODITY_SPECS: dict[str, dict] = {
    # SHFE — base metals (night 21:00 → 01:00 next day)
    "CU": dict(tick=10.0,  lot=5,    exchange="SHFE", daily_vol=0.012, ref_price=70000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="01:00"),
    "AL": dict(tick=5.0,   lot=5,    exchange="SHFE", daily_vol=0.010, ref_price=19000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="01:00"),
    "ZN": dict(tick=5.0,   lot=5,    exchange="SHFE", daily_vol=0.012, ref_price=22000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="01:00"),
    "NI": dict(tick=10.0,  lot=1,    exchange="SHFE", daily_vol=0.018, ref_price=130000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="01:00"),
    "SN": dict(tick=10.0,  lot=1,    exchange="SHFE", daily_vol=0.016, ref_price=250000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="01:00"),
    "PB": dict(tick=5.0,   lot=5,    exchange="SHFE", daily_vol=0.012, ref_price=16000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="01:00"),
    # SHFE — precious metals (night 21:00 → 02:30 next day)
    "AU": dict(tick=0.02,  lot=1000, exchange="SHFE", daily_vol=0.008, ref_price=520.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="02:30"),
    "AG": dict(tick=1.0,   lot=15,   exchange="SHFE", daily_vol=0.015, ref_price=6500.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="02:30"),
    # SHFE — ferrous / rubber (night 21:00 → 23:00 same evening)
    "RB": dict(tick=1.0,   lot=10,   exchange="SHFE", daily_vol=0.018, ref_price=3800.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "RU": dict(tick=5.0,   lot=10,   exchange="SHFE", daily_vol=0.015, ref_price=14000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "SP": dict(tick=2.0,   lot=10,   exchange="SHFE", daily_vol=0.012, ref_price=5500.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "SS": dict(tick=5.0,   lot=5,    exchange="SHFE", daily_vol=0.012, ref_price=14000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    # DCE — ferrous (night 21:00 → 23:00)
    "I":  dict(tick=0.5,   lot=100,  exchange="DCE",  daily_vol=0.020, ref_price=900.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "J":  dict(tick=0.5,   lot=100,  exchange="DCE",  daily_vol=0.022, ref_price=2000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "JM": dict(tick=0.5,   lot=60,   exchange="DCE",  daily_vol=0.020, ref_price=2400.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    # DCE — agricultural (night 21:00 → 23:00)
    "M":  dict(tick=1.0,   lot=10,   exchange="DCE",  daily_vol=0.015, ref_price=3500.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "Y":  dict(tick=2.0,   lot=10,   exchange="DCE",  daily_vol=0.012, ref_price=8000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "C":  dict(tick=1.0,   lot=10,   exchange="DCE",  daily_vol=0.008, ref_price=2700.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "A":  dict(tick=1.0,   lot=10,   exchange="DCE",  daily_vol=0.010, ref_price=4500.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "P":  dict(tick=2.0,   lot=10,   exchange="DCE",  daily_vol=0.012, ref_price=8000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "EG": dict(tick=1.0,   lot=10,   exchange="DCE",  daily_vol=0.015, ref_price=4800.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    # CZCE — chemicals / agri (night 21:00 → 23:00)
    "TA": dict(tick=2.0,   lot=5,    exchange="CZCE", daily_vol=0.015, ref_price=5500.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "MA": dict(tick=1.0,   lot=10,   exchange="CZCE", daily_vol=0.018, ref_price=2500.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "SR": dict(tick=1.0,   lot=10,   exchange="CZCE", daily_vol=0.012, ref_price=5500.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "CF": dict(tick=5.0,   lot=5,    exchange="CZCE", daily_vol=0.012, ref_price=14000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "ZC": dict(tick=0.2,   lot=200,  exchange="CZCE", daily_vol=0.020, ref_price=700.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "RM": dict(tick=1.0,   lot=10,   exchange="CZCE", daily_vol=0.015, ref_price=2800.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "OI": dict(tick=2.0,   lot=5,    exchange="CZCE", daily_vol=0.012, ref_price=9000.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "SA": dict(tick=1.0,   lot=20,   exchange="CZCE", daily_vol=0.015, ref_price=1800.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    "FG": dict(tick=1.0,   lot=20,   exchange="CZCE", daily_vol=0.015, ref_price=1600.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="23:00"),
    # INE — crude oil (night 21:00 → 02:30 next day)
    "SC": dict(tick=0.1,   lot=1000, exchange="INE",  daily_vol=0.022, ref_price=530.0,
               day_sessions=[("09:00","10:15"),("10:30","11:30"),("13:30","15:00")],
               night_end="02:30"),
    # CFFEX — index futures (no night session)
    "IF": dict(tick=0.2,   lot=300,  exchange="CFFEX", daily_vol=0.015, ref_price=4000.0,
               day_sessions=[("09:30","11:30"),("13:00","15:00")],
               night_end=None),
    "IH": dict(tick=0.2,   lot=300,  exchange="CFFEX", daily_vol=0.012, ref_price=2700.0,
               day_sessions=[("09:30","11:30"),("13:00","15:00")],
               night_end=None),
    "IC": dict(tick=0.2,   lot=200,  exchange="CFFEX", daily_vol=0.018, ref_price=6000.0,
               day_sessions=[("09:30","11:30"),("13:00","15:00")],
               night_end=None),
    "IM": dict(tick=0.2,   lot=200,  exchange="CFFEX", daily_vol=0.020, ref_price=6000.0,
               day_sessions=[("09:30","11:30"),("13:00","15:00")],
               night_end=None),
    # CFFEX — bond futures (no night session, close at 15:15)
    "T":  dict(tick=0.005, lot=10_000_000, exchange="CFFEX", daily_vol=0.003, ref_price=103.0,
               day_sessions=[("09:30","11:30"),("13:00","15:15")],
               night_end=None),
    "TF": dict(tick=0.005, lot=10_000_000, exchange="CFFEX", daily_vol=0.002, ref_price=101.0,
               day_sessions=[("09:30","11:30"),("13:00","15:15")],
               night_end=None),
    "TS": dict(tick=0.005, lot=2_000_000,  exchange="CFFEX", daily_vol=0.001, ref_price=101.0,
               day_sessions=[("09:30","11:30"),("13:00","15:15")],
               night_end=None),
    "TL": dict(tick=0.01,  lot=10_000_000, exchange="CFFEX", daily_vol=0.005, ref_price=106.0,
               day_sessions=[("09:30","11:30"),("13:00","15:15")],
               night_end=None),
}


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
    signal_strength: float = 0.01,
) -> pd.DataFrame:
    """
    Generate one day of synthetic Chinese L2 LOB snapshots.

    Returns DataFrame indexed by timestamp with columns:
        mid_price,
        bid_px_{1..10}, bid_vol_{1..10},
        ask_px_{1..10}, ask_vol_{1..10},
        last_price, last_volume,
        cum_buy_vol, cum_sell_vol,
        cum_buy_count, cum_sell_count,
        limit_buy_vol, limit_sell_vol,
        cancel_buy_vol, cancel_sell_vol,
        market_buy_vol, market_sell_vol,
        bid_depth, ask_depth

    Designed to match the Wind Level-2 / Tushare Pro snapshot format.

    signal_strength controls coupling between latent order-flow and price.
    Higher values produce stronger/more detectable alpha signals.

    WARNING — circular data bias: the same latent `flow` variable drives
    both LOB features (depth tilt, trade direction) AND price drift, so
    any signal derived from the LOB is guaranteed to have IC > 0. This
    inflates IC, Sharpe, and annualised return vs real OOS data.
    Use signal_strength=0.0 as a null baseline; use real L2 data for
    any claims about live performance.
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
    cum_buy_count = 0
    cum_sell_count = 0

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
        row["buy_count"] = 1 if is_buy else 0
        row["sell_count"] = 0 if is_buy else 1
        row["market_buy_vol"] = trd_vol if is_buy else 0
        row["market_sell_vol"] = 0 if is_buy else trd_vol

        passive_base = max(LOT, int(base_vol * 0.05 / LOT) * LOT)
        row["limit_buy_vol"] = max(
            LOT,
            int(passive_base * np.exp(np.clip(f * 0.25, -1.0, 1.0)) * rng.lognormal(0.0, 0.5) / LOT) * LOT,
        )
        row["limit_sell_vol"] = max(
            LOT,
            int(passive_base * np.exp(np.clip(-f * 0.25, -1.0, 1.0)) * rng.lognormal(0.0, 0.5) / LOT) * LOT,
        )
        row["cancel_buy_vol"] = max(
            0,
            int(passive_base * np.exp(np.clip(-f * 0.15, -1.0, 1.0)) * rng.lognormal(-0.6, 0.6) / LOT) * LOT,
        )
        row["cancel_sell_vol"] = max(
            0,
            int(passive_base * np.exp(np.clip(f * 0.15, -1.0, 1.0)) * rng.lognormal(-0.6, 0.6) / LOT) * LOT,
        )

        # Executed volume consumes the hit side's top-of-book depth, so depth
        # changes are mechanically consistent with trades (execution-rate and
        # cancellation estimators would otherwise see pure noise).
        if is_buy:
            row["ask_vol_1"] = max(LOT, row["ask_vol_1"] - trd_vol)
        else:
            row["bid_vol_1"] = max(LOT, row["bid_vol_1"] - trd_vol)

        if is_buy:
            cum_buy  += trd_vol
            cum_buy_count += 1
        else:
            cum_sell += trd_vol
            cum_sell_count += 1

        row["cum_buy_vol"]  = cum_buy
        row["cum_sell_vol"] = cum_sell
        row["cum_buy_count"] = cum_buy_count
        row["cum_sell_count"] = cum_sell_count
        row["bid_depth"] = sum(float(row[f"bid_vol_{lv}"]) for lv in range(1, N_LEVELS + 1))
        row["ask_depth"] = sum(float(row[f"ask_vol_{lv}"]) for lv in range(1, N_LEVELS + 1))
        records.append(row)

    df = pd.DataFrame(records).set_index("timestamp")
    df["ticker"] = ticker
    return df


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
    cum_buy_count  = np.zeros(n, dtype=int)
    cum_sell_count = np.zeros(n, dtype=int)

    for i, frac in enumerate(fracs):
        noise = rng.lognormal(0, 0.3)
        total = int(base_vol * frac * noise)
        if true_gap >= 0:
            buy_frac  = 0.5 + 0.4 * abs(true_gap) / 0.05
        else:
            buy_frac  = 0.5 - 0.4 * abs(true_gap) / 0.05
        cum_buy[i]  = int(total * buy_frac)
        cum_sell[i] = int(total * (1 - buy_frac))
        cum_buy_count[i] = max(0, int(cum_buy[i] / LOT))
        cum_sell_count[i] = max(0, int(cum_sell[i] / LOT))

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
        "cum_buy_count":     cum_buy_count,
        "cum_sell_count":    cum_sell_count,
        "prev_close":        prev_close,
        "ticker":            ticker,
    }, index=pd.DatetimeIndex(tss, name="timestamp"))

    return df, float(open_price)


# ---------------------------------------------------------------------------
# Commodity futures simulation
# ---------------------------------------------------------------------------

def _resolve_product(ticker: str) -> str:
    """'CU2401.SHFE' → 'CU',  'SC2403.INE' → 'SC',  'MA409.CZCE' → 'MA'."""
    base = ticker.split(".")[0]
    return re.sub(r"\d+$", "", base).upper()


def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def _commodity_session_timestamps(
    date: str,
    day_sessions: list[tuple[str, str]],
    night_end: str | None = None,
    include_night: bool = False,
    freq_sec: int = 3,
) -> pd.DatetimeIndex:
    """
    Build timestamp index for commodity futures sessions.

    Night session starts at 21:00 of the *previous* calendar day.
    If night_end hour < 21 (e.g. "01:00", "02:30"), it ends on `date`
    (early morning). If night_end hour >= 21, it ends the same evening.
    """
    base     = pd.Timestamp(date)
    segments: list[pd.DatetimeIndex] = []

    if include_night and night_end is not None:
        eh, em      = _parse_hhmm(night_end)
        night_start = base - pd.Timedelta(days=1) + pd.Timedelta(hours=21)
        night_end_ts = (
            base + pd.Timedelta(hours=eh, minutes=em)
            if eh < 21
            else base - pd.Timedelta(days=1) + pd.Timedelta(hours=eh, minutes=em)
        )
        segments.append(pd.date_range(night_start, night_end_ts, freq=f"{freq_sec}s"))

    for start_str, end_str in day_sessions:
        sh, sm = _parse_hhmm(start_str)
        eh, em = _parse_hhmm(end_str)
        segments.append(pd.date_range(
            base + pd.Timedelta(hours=sh, minutes=sm),
            base + pd.Timedelta(hours=eh, minutes=em),
            freq=f"{freq_sec}s",
        ))

    if not segments:
        return pd.DatetimeIndex([])
    result = segments[0]
    for seg in segments[1:]:
        result = result.append(seg)
    return result


def simulate_commodity_lob_day(
    ticker: str = "CU2401.SHFE",
    date: str = "2024-01-02",
    prev_close: float | None = None,
    include_night: bool = False,
    seed: int = 42,
    signal_strength: float = 0.01,
) -> pd.DataFrame:
    """
    Generate synthetic L2 LOB data for Chinese commodity / financial futures.

    Resolves contract specs (tick, lot, sessions, daily_vol) automatically
    from COMMODITY_SPECS. Supported tickers (product codes):
      SHFE metals: CU AL ZN NI SN PB
      SHFE precious: AU AG
      SHFE ferrous/rubber: RB RU SP SS
      DCE ferrous: I J JM
      DCE agri: M Y C A P EG
      CZCE: TA MA SR CF ZC RM OI SA FG
      INE: SC
      CFFEX index: IF IH IC IM
      CFFEX bond: T TF TS TL

    include_night=True prepends the prior evening's night session
    (commodity futures only; CFFEX has no night session).

    Returns same schema as simulate_lob_day() — compatible with all signals.
    """
    product = _resolve_product(ticker)
    if product not in COMMODITY_SPECS:
        raise ValueError(
            f"Unknown product '{product}'. Known: {sorted(COMMODITY_SPECS)}. "
            "Use simulate_lob_day() directly for unlisted contracts."
        )

    spec      = COMMODITY_SPECS[product]
    tick      = spec["tick"]
    lot       = spec["lot"]
    daily_vol = spec["daily_vol"]
    ref_price = spec["ref_price"] if prev_close is None else prev_close

    tss = _commodity_session_timestamps(
        date,
        day_sessions=spec["day_sessions"],
        night_end=spec["night_end"],
        include_night=include_night,
    )
    n   = len(tss)
    rng = np.random.default_rng(seed)

    # AR(1) latent order-flow, half-life 30 ticks (~90s at 3s cadence)
    phi       = np.exp(-np.log(2) / 30)
    innov_std = np.sqrt(1.0 - phi ** 2)
    flow      = np.zeros(n)
    flow[0]   = rng.normal(0.0, 1.0)
    for i in range(1, n):
        flow[i] = phi * flow[i - 1] + rng.normal(0.0, innov_std)

    # GBM price path — no daily price limit for futures
    log_ret = np.zeros(n)
    dt_frac = 3.0 / (4.0 * 3600.0)
    for i, ts in enumerate(tss):
        sigma      = daily_vol * _vol_multiplier(ts) * np.sqrt(dt_frac)
        log_ret[i] = flow[i] * sigma * signal_strength + rng.normal(0.0, sigma)

    prices = ref_price * np.exp(np.cumsum(log_ret))
    prices = np.round(prices / tick) * tick

    # Typical base volume per level, scaled to contract lot size
    vol_mu    = np.log(lot * 50)
    vol_state = float(rng.lognormal(vol_mu, 0.8))
    records   = []
    cum_buy   = 0
    cum_sell  = 0
    cum_buy_count = 0
    cum_sell_count = 0

    for i, (ts, mid) in enumerate(zip(tss, prices)):
        vm = _vol_multiplier(ts)
        f  = float(flow[i])

        n_spread_ticks = rng.integers(1, max(2, int(4 * vm)))
        half = n_spread_ticks * tick / 2.0
        b1   = np.round((mid - half) / tick) * tick
        a1   = np.round((mid + half) / tick) * tick

        row = {"timestamp": ts, "mid_price": mid}

        vol_target = float(rng.lognormal(vol_mu, 0.8))
        vol_state  = 0.7 * vol_state + 0.3 * vol_target
        base_vol   = max(lot, int(round(vol_state / lot)) * lot)
        tilt       = np.exp(np.clip(f * 0.25, -2.0, 2.0))

        for lv in range(1, N_LEVELS + 1):
            decay = np.exp(-0.25 * (lv - 1))
            bv = max(lot, int(round(base_vol * decay * tilt * rng.lognormal(0.0, 0.4) / lot)) * lot)
            av = max(lot, int(round(base_vol * decay / tilt * rng.lognormal(0.0, 0.4) / lot)) * lot)
            row[f"bid_px_{lv}"]  = np.round((b1 - (lv - 1) * tick) / tick) * tick
            row[f"bid_vol_{lv}"] = bv
            row[f"ask_px_{lv}"]  = np.round((a1 + (lv - 1) * tick) / tick) * tick
            row[f"ask_vol_{lv}"] = av

        p_buy   = 0.5 + 0.3 * np.tanh(f)
        is_buy  = rng.random() < p_buy
        trd_vol = max(lot, int(round(rng.lognormal(np.log(lot * 5), 0.8) / lot)) * lot)
        row["last_price"]  = a1 if is_buy else b1
        row["last_volume"] = trd_vol
        row["buy_count"] = 1 if is_buy else 0
        row["sell_count"] = 0 if is_buy else 1
        row["market_buy_vol"] = trd_vol if is_buy else 0
        row["market_sell_vol"] = 0 if is_buy else trd_vol

        passive_base = max(lot, int(round(base_vol * 0.05 / lot)) * lot)
        row["limit_buy_vol"] = max(
            lot,
            int(round(passive_base * np.exp(np.clip(f * 0.25, -1.0, 1.0)) * rng.lognormal(0.0, 0.5) / lot)) * lot,
        )
        row["limit_sell_vol"] = max(
            lot,
            int(round(passive_base * np.exp(np.clip(-f * 0.25, -1.0, 1.0)) * rng.lognormal(0.0, 0.5) / lot)) * lot,
        )
        row["cancel_buy_vol"] = max(
            0,
            int(round(passive_base * np.exp(np.clip(-f * 0.15, -1.0, 1.0)) * rng.lognormal(-0.6, 0.6) / lot)) * lot,
        )
        row["cancel_sell_vol"] = max(
            0,
            int(round(passive_base * np.exp(np.clip(f * 0.15, -1.0, 1.0)) * rng.lognormal(-0.6, 0.6) / lot)) * lot,
        )

        if is_buy:
            row["ask_vol_1"] = max(lot, row["ask_vol_1"] - trd_vol)
            cum_buy  += trd_vol
            cum_buy_count += 1
        else:
            row["bid_vol_1"] = max(lot, row["bid_vol_1"] - trd_vol)
            cum_sell += trd_vol
            cum_sell_count += 1

        row["cum_buy_vol"]  = cum_buy
        row["cum_sell_vol"] = cum_sell
        row["cum_buy_count"] = cum_buy_count
        row["cum_sell_count"] = cum_sell_count
        row["bid_depth"] = sum(float(row[f"bid_vol_{lv}"]) for lv in range(1, N_LEVELS + 1))
        row["ask_depth"] = sum(float(row[f"ask_vol_{lv}"]) for lv in range(1, N_LEVELS + 1))
        records.append(row)

    df = pd.DataFrame(records).set_index("timestamp")
    df["ticker"] = ticker
    return df
