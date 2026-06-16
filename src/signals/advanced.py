"""
Advanced China-specific HFT alpha factors.

Tier-1 (behavioural / China-unique) and Tier-2 (microstructure) factors that
extend the base set in ofi.py / features.py / auction.py. See
research/hft_factors_china.md for the full catalogue and source list.

Factors implemented here
------------------------
  Tier 1
    big_order_flow            大单净额 — institutional footprint via size-banded flow
    herding_intensity         intraday LSV-style order-flow clustering (retail herding)
  Tier 2
    aggressive_passive_imbalance (API)  aggression-vs-provision spread driver
    order_execution_imbalance    (OEI)  bid/ask execution-rate asymmetry
    order_book_slope                    cumulative-depth-per-tick asymmetry
    book_resiliency                     post-aggression top-of-book recovery speed
    signed_jump_reversal                BNS jump direction → short-horizon reversal
    realized_vol                        helper for sizing/regime (not a directional factor)
  Tier 3 (gates / regime — not directional alpha)
    vpin                      order-flow toxicity (volume-synchronized)
    kyle_lambda               tick-level price impact per unit signed flow
    cancel_spike_imbalance    one-sided cancellation bursts (veto/confirmation)
    exposure_gate             VPIN + λ → position-size multiplier ∈ [0, 1]

(Tier-1 close-auction imbalance lives in auction.py; sealing-strength lives in
features.py-style stock-mode logic but is exposed here as sealing_strength.)

Robustness
----------
All trade-direction factors derive per-tick buy/sell volume from the cumulative
`cum_buy_vol` / `cum_sell_vol` columns (present in synthetic data AND the AKShare
tick loader), NOT from `last_price` — real loaders set last_price to the mid and
carry direction only in the cumulative columns. Factors degrade to ~0 (not NaN)
when trade columns are missing.

References
----------
  API      — HF-liquidity study, SZSE tick data 2019–2021 (J. Int. Fin. Markets, 2025)
  OEI      — "Price Impact of Order Book Events from a Dimension of Time", CN L2 (2021)
  Herding  — "Intraday Herding Drivers in China's A-Share Market: CSI 500" (FRL, 2025);
             classic LSV: Lakonishok, Shleifer, Vishny (1992)
  大单净额 — domestic 主力资金 practice; size-banded active flow
  Jumps    — Barndorff-Nielsen & Shephard bipower variation
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _trade_flows(lob_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Per-tick executed buy / sell volume.

    Primary source: diffs of cumulative `cum_buy_vol` / `cum_sell_vol`
    (robust; present in synthetic and AKShare tick data).

    Fallback: classify `last_volume` by `last_price` vs mid when cumulative
    columns are absent or constant.

    Returns (dv_buy, dv_sell), both >= 0, aligned to lob_df.index.
    """
    idx = lob_df.index
    if "cum_buy_vol" in lob_df.columns and "cum_sell_vol" in lob_df.columns:
        cb = lob_df["cum_buy_vol"].astype(float)
        cs = lob_df["cum_sell_vol"].astype(float)
        if cb.nunique() > 1 or cs.nunique() > 1:
            dvb = cb.diff().clip(lower=0).fillna(0.0)
            dvs = cs.diff().clip(lower=0).fillna(0.0)
            return dvb, dvs

    # Fallback: direction from last_price relative to mid
    if "last_price" in lob_df.columns and "last_volume" in lob_df.columns:
        mid = (lob_df["bid_px_1"].astype(float) + lob_df["ask_px_1"].astype(float)) / 2.0
        lp  = lob_df["last_price"].astype(float)
        lv  = lob_df["last_volume"].astype(float).clip(lower=0)
        is_buy = lp >= mid
        dvb = lv.where(is_buy, 0.0)
        dvs = lv.where(~is_buy, 0.0)
        return dvb.fillna(0.0), dvs.fillna(0.0)

    zero = pd.Series(0.0, index=idx)
    return zero, zero.copy()


def _available_levels(lob_df: pd.DataFrame, requested: int) -> int:
    """Clamp requested LOB depth to what the frame actually carries (5 or 10)."""
    n = 0
    for lv in range(1, requested + 1):
        if f"bid_vol_{lv}" in lob_df.columns and f"ask_vol_{lv}" in lob_df.columns:
            n = lv
        else:
            break
    return max(1, n)


def _zscore(s: pd.Series, window: int = 200, min_periods: int = 50) -> pd.Series:
    rstd = s.rolling(window, min_periods=min_periods).std()
    return (s / rstd.replace(0.0, np.nan)).fillna(0.0)


def _cancel_estimates(lob_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Per-tick estimated cancellation volume at the top of book, per side.

    A drop in level-1 depth is either execution or cancellation:

        cancel_bid ≈ max( −ΔV_bid_1 − dv_sell, 0 )
        cancel_ask ≈ max( −ΔV_ask_1 − dv_buy , 0 )

    Returns (cancel_bid, cancel_ask), both ≥ 0, aligned to lob_df.index.
    Shared by cancel_spike_imbalance and spoof_filtered_qi.
    """
    dvb, dvs = _trade_flows(lob_df)
    db = lob_df["bid_vol_1"].astype(float).diff()
    da = lob_df["ask_vol_1"].astype(float).diff()
    cancel_bid = ((-db).clip(lower=0.0) - dvs).clip(lower=0.0).fillna(0.0)
    cancel_ask = ((-da).clip(lower=0.0) - dvb).clip(lower=0.0).fillna(0.0)
    return cancel_bid, cancel_ask


# ---------------------------------------------------------------------------
# Tier 1 — Smart-money large-order net flow  (大单净额)
# ---------------------------------------------------------------------------

def big_order_flow(
    lob_df: pd.DataFrame,
    window: int = 40,
    big_quantile: float = 0.8,
    threshold_lookback: int = 200,
) -> pd.Series:
    """
    Net signed flow from *large* trades only — institutional footprint.

    Round-lot retail noise dominates small trade bands; the large band isolates
    the 主力 (main force). Per tick, traded volume tv = dv_buy + dv_sell is
    flagged "large" when it exceeds a rolling `big_quantile` threshold; the net
    signed large-trade volume is then accumulated over `window` and scaled by
    total traded volume.

        big_t   = 1[ tv_t >= Q_{big_quantile}(tv; lookback) ]
        signed  = (dv_buy - dv_sell) · big_t
        factor  = Σ_window signed / Σ_window tv

    Positive → large trades are net buys → institutional accumulation → bullish.
    Horizon: minutes → next-day. China-specific (round-lot structure makes the
    size split clean). Heuristic when only one trade/tick is observed.
    """
    dvb, dvs = _trade_flows(lob_df)
    tv     = (dvb + dvs)
    signed = (dvb - dvs)

    thr    = tv.rolling(threshold_lookback, min_periods=20).quantile(big_quantile)
    # NaN threshold during warm-up → comparison False → factor silent (causal;
    # never backfill with full-series stats, that's look-ahead)
    is_big = tv >= thr

    big_signed = signed.where(is_big, 0.0)
    num = big_signed.rolling(window, min_periods=max(1, window // 2)).sum()
    den = tv.rolling(window, min_periods=max(1, window // 2)).sum().replace(0.0, np.nan)

    return _zscore((num / den).fillna(0.0)).rename("big_flow")


# ---------------------------------------------------------------------------
# Tier 1 — Intraday herding intensity
# ---------------------------------------------------------------------------

def herding_intensity(lob_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    LSV-style intraday herding: excess directional clustering of trades.

    Over a rolling `window`, p = fraction of trading ticks that are buy-initiated.
    The LSV adjustment factor removes the clustering expected by chance under a
    binomial(window, 0.5) null (normal approximation):

        AF      = sqrt( 2 / (π · window) ) · 0.5
        H_excess= |p - 0.5| - AF                     (herding strength, ≥0 when real)
        signal  = sign(p - 0.5) · max(H_excess, 0)

    Positive → crowd herding into buys → very-short continuation. NOTE: herding
    spikes mean-revert at a slightly longer horizon — the backtest exit logic
    (or a negative optimizer weight) captures the reversal leg; this factor is
    the continuation leg only. Retail-dominated A-shares show far stronger,
    more persistent herding than developed markets.

    Single-instrument proxy for the cross-sectional CSI-500 measure (FRL 2025).
    """
    dvb, dvs = _trade_flows(lob_df)
    has_trade = ((dvb + dvs) > 0).astype(float)
    buy_ind   = (dvb > dvs).astype(float) * has_trade

    n_trade = has_trade.rolling(window, min_periods=max(2, window // 2)).sum()
    n_buy   = buy_ind.rolling(window, min_periods=max(2, window // 2)).sum()
    p       = (n_buy / n_trade.replace(0.0, np.nan)).fillna(0.5)

    af       = np.sqrt(2.0 / (np.pi * window)) * 0.5
    h_excess = (p - 0.5).abs() - af
    signal   = np.sign(p - 0.5) * h_excess.clip(lower=0.0)

    return _zscore(signal.fillna(0.0)).rename("herding")


# ---------------------------------------------------------------------------
# Tier 2 — Aggressive-Passive Imbalance  (API)
# ---------------------------------------------------------------------------

def aggressive_passive_imbalance(lob_df: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Directional aggression normalised by total order activity (aggr + passive).

    Aggressive = executed (liquidity-taking) volume: dv_buy, dv_sell.
    Passive    = newly-added resting depth at the top of book (liquidity-providing):
                 positive increments of bid_vol_1 and ask_vol_1.

        API = Σ_window (dv_buy − dv_sell)
            / Σ_window (dv_buy + dv_sell + passive_added)

    Distinct from trade_imbalance (ignores the passive side) and from OFI (mixes
    aggressive + passive into one signed quantity). High |API| → aggression
    dominates provision → spread about to move / directional pressure. The
    order-based spread-change driver identified on SZSE tick data (2025).
    """
    dvb, dvs = _trade_flows(lob_df)

    bv1 = lob_df["bid_vol_1"].astype(float)
    av1 = lob_df["ask_vol_1"].astype(float)
    passive = bv1.diff().clip(lower=0).fillna(0.0) + av1.diff().clip(lower=0).fillna(0.0)

    num = (dvb - dvs).rolling(window, min_periods=max(1, window // 2)).sum()
    den = (dvb + dvs + passive).rolling(window, min_periods=max(1, window // 2)).sum()
    den = den.replace(0.0, np.nan)

    return (num / den).fillna(0.0).rename("api")


# ---------------------------------------------------------------------------
# Tier 2 — Order Execution Imbalance  (OEI)
# ---------------------------------------------------------------------------

def order_execution_imbalance(
    lob_df: pd.DataFrame,
    window: int = 10,
    depth_levels: int = 3,
) -> pd.Series:
    """
    Asymmetry in how fast each side's resting depth is being executed.

    Buys lift the ask (consume ask depth); sells hit the bid (consume bid depth).

        exec_ask = dv_buy(t)  / ask_depth_topN(t-1)
        exec_bid = dv_sell(t) / bid_depth_topN(t-1)
        OEI      = mean_window( exec_ask − exec_bid )

    Flow during (t-1, t] is scaled by the depth that was STANDING at t-1 —
    depth at t is post-consumption, which would understate the eaten side.

    Positive → ask book being eaten faster than bid → buy-side aggression → bullish.
    Robust because Chinese books have a low cancellation ratio, so execution rate
    is a stable quantity. Improves the Cont (2014) OFI model R² on CN L2 (2021).
    """
    n = min(depth_levels, _available_levels(lob_df, depth_levels))
    dvb, dvs = _trade_flows(lob_df)

    ask_depth = sum(lob_df[f"ask_vol_{lv}"].astype(float) for lv in range(1, n + 1)).shift(1)
    bid_depth = sum(lob_df[f"bid_vol_{lv}"].astype(float) for lv in range(1, n + 1)).shift(1)

    exec_ask = dvb / ask_depth.replace(0.0, np.nan)
    exec_bid = dvs / bid_depth.replace(0.0, np.nan)

    oei = (exec_ask - exec_bid).fillna(0.0)
    return _zscore(oei.rolling(window, min_periods=max(1, window // 2)).mean().fillna(0.0)).rename("oei")


# ---------------------------------------------------------------------------
# Tier 2 — Order-book slope / shape
# ---------------------------------------------------------------------------

def order_book_slope(lob_df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """
    Cumulative-depth-per-price asymmetry (book steepness).

        slope_bid = Σ_l V_bid_l / ( P_bid_1 − P_bid_n + tick )
        slope_ask = Σ_l V_ask_l / ( P_ask_n − P_ask_1 + tick )
        signal    = (slope_bid − slope_ask) / (slope_bid + slope_ask)

    A steeper side packs more volume close to the touch → stronger, more
    committed support/resistance. Positive → steeper bid → bullish. Distinct
    from depth_tilt (per-level inverse-distance weights) and queue_imbalance
    (raw level depth). Informative in China's large-tick regime where book
    shape carries real signal rather than noise.
    """
    n = min(n_levels, _available_levels(lob_df, n_levels))
    tick = 0.01

    bid_depth = sum(lob_df[f"bid_vol_{lv}"].astype(float) for lv in range(1, n + 1))
    ask_depth = sum(lob_df[f"ask_vol_{lv}"].astype(float) for lv in range(1, n + 1))

    bid_range = (lob_df["bid_px_1"].astype(float) - lob_df[f"bid_px_{n}"].astype(float)).clip(lower=0) + tick
    ask_range = (lob_df[f"ask_px_{n}"].astype(float) - lob_df["ask_px_1"].astype(float)).clip(lower=0) + tick

    slope_bid = bid_depth / bid_range
    slope_ask = ask_depth / ask_range

    total = (slope_bid + slope_ask).replace(0.0, np.nan)
    return ((slope_bid - slope_ask) / total).fillna(0.0).rename("book_slope")


# ---------------------------------------------------------------------------
# Tier 2 — Book resiliency
# ---------------------------------------------------------------------------

def book_resiliency(lob_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Top-of-book replenishment-speed asymmetry after depletion.

        refill_side  = max( ΔV_side_1, 0 )      (depth added back)
        deplete_side = max( −ΔV_side_1, 0 )     (depth consumed)
        resil_side   = Σ_window refill_side / (Σ_window deplete_side + ε)
        signal       = (resil_bid − resil_ask) / (resil_bid + resil_ask)

    High bid resiliency → fast dip-buying / strong latent demand → bullish.
    Low resiliency on a side → that side is weak → continuation away from it.
    Captures hidden liquidity dynamics not visible in static depth snapshots.
    """
    bv1 = lob_df["bid_vol_1"].astype(float)
    av1 = lob_df["ask_vol_1"].astype(float)

    db, da = bv1.diff(), av1.diff()
    refill_b, deplete_b = db.clip(lower=0).fillna(0.0), (-db).clip(lower=0).fillna(0.0)
    refill_a, deplete_a = da.clip(lower=0).fillna(0.0), (-da).clip(lower=0).fillna(0.0)

    mp = max(1, window // 2)
    resil_b = refill_b.rolling(window, min_periods=mp).sum() / (deplete_b.rolling(window, min_periods=mp).sum() + 1.0)
    resil_a = refill_a.rolling(window, min_periods=mp).sum() / (deplete_a.rolling(window, min_periods=mp).sum() + 1.0)

    total = (resil_b + resil_a).replace(0.0, np.nan)
    return ((resil_b - resil_a) / total).fillna(0.0).rename("resiliency")


# ---------------------------------------------------------------------------
# Tier 2 — Signed-jump reversal + realized vol
# ---------------------------------------------------------------------------

def realized_vol(lob_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Rolling realized volatility of mid-price log-returns (√RV).

    NOT a directional factor — use it for position sizing / regime gating
    (scale exposure down when RV spikes). Returned separately from the composite.
    """
    mid = (lob_df["bid_px_1"].astype(float) + lob_df["ask_px_1"].astype(float)) / 2.0
    r   = np.log(mid.replace(0.0, np.nan)).diff()
    rv  = (r ** 2).rolling(window, min_periods=max(2, window // 2)).sum()
    return np.sqrt(rv).fillna(0.0).rename("realized_vol")


def signed_jump_reversal(lob_df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Barndorff-Nielsen–Shephard jump component, signed, as a reversal signal.

        RV   = Σ_window r²
        BV   = (π/2) · Σ_window |r_t|·|r_{t-1}|        (jump-robust)
        jump = max(RV − BV, 0)
        dir  = sign( Σ_window r )                       (direction of the move)
        signal = − dir · √jump                          (overreaction reverses)

    Retail-driven names overreact to jumps and revert at the 5–30 min horizon,
    so the signal is the *negative* of the signed jump: a recent up-jump →
    negative signal (expect pullback). z-scored.
    """
    mid = (lob_df["bid_px_1"].astype(float) + lob_df["ask_px_1"].astype(float)) / 2.0
    r   = np.log(mid.replace(0.0, np.nan)).diff().fillna(0.0)

    mp  = max(2, window // 2)
    rv  = (r ** 2).rolling(window, min_periods=mp).sum()
    bv  = (np.pi / 2.0) * (r.abs() * r.abs().shift(1)).rolling(window, min_periods=mp).sum()
    jump = (rv - bv).clip(lower=0.0).fillna(0.0)

    direction = np.sign(r.rolling(window, min_periods=mp).sum().fillna(0.0))
    signed    = -direction * np.sqrt(jump)

    return _zscore(signed.fillna(0.0)).rename("signed_jump")


# ---------------------------------------------------------------------------
# Tier 1 — Limit-up/down sealing strength  (涨停/跌停封单)  — stock mode only
# ---------------------------------------------------------------------------

def sealing_strength(
    lob_df: pd.DataFrame,
    prev_close: float,
    limit_pct: float = 0.10,
    proximity: float = 0.001,
    turn_window: int = 40,
) -> pd.Series:
    """
    Strength of the sealing queue when a stock is at / hugging its price limit.

    At the up-limit the book has buyers queued at the limit price and no sellers
    (the board is "sealed", 封板). Sealing strength = sealing-queue volume scaled
    by recent turnover; a large, well-funded seal predicts next-day continuation
    (the 打板 momentum game). Symmetric at the down-limit.

        up_limit   = prev_close · (1 + limit_pct)
        SealUp     = bid_vol_1 / (turnover_window + ε)   when bid_px_1 ≈ up_limit
        SealDown   = ask_vol_1 / (turnover_window + ε)   when ask_px_1 ≈ down_limit
        signal     = SealUp − SealDown        (0 away from limits)

    Positive → strong up-limit seal → bullish (next-day, T+1). Stock-mode only;
    futures have no fixed daily price limit.
    """
    bv1 = lob_df["bid_vol_1"].astype(float)
    av1 = lob_df["ask_vol_1"].astype(float)
    bpx = lob_df["bid_px_1"].astype(float)
    apx = lob_df["ask_px_1"].astype(float)

    dvb, dvs = _trade_flows(lob_df)
    turnover = (dvb + dvs).rolling(turn_window, min_periods=max(1, turn_window // 2)).sum()

    up_limit   = prev_close * (1.0 + limit_pct)
    down_limit = prev_close * (1.0 - limit_pct)

    at_up   = bpx >= up_limit   * (1.0 - proximity)
    at_down = apx <= down_limit * (1.0 + proximity)

    seal_up   = (bv1 / (turnover + 1.0)).where(at_up,   0.0)
    seal_down = (av1 / (turnover + 1.0)).where(at_down, 0.0)

    return (seal_up - seal_down).fillna(0.0).rename("sealing")


# ---------------------------------------------------------------------------
# Tier 3 — VPIN  (order-flow toxicity; regime gate, NOT directional)
# ---------------------------------------------------------------------------

def vpin(
    lob_df: pd.DataFrame,
    n_buckets_day: int = 50,
    smooth_buckets: int = 10,
) -> pd.Series:
    """
    Volume-Synchronized Probability of Informed Trading.

    Easley, López de Prado, O'Hara (RFS 2012). Trade volume is partitioned into
    equal-volume buckets (volume time, not clock time); per bucket the absolute
    buy/sell imbalance is computed and averaged over the last `smooth_buckets`:

        VPIN = (1/n) Σ_buckets |V_buy − V_sell| / V_bucket            ∈ [0, 1]

    Buy/sell classification comes directly from the exchange-tagged cumulative
    flows (better than the BVC approximation the original paper needed).

    CAUSALITY: bucket boundaries are determined sequentially. The bucket-volume
    target at tick t uses only the expanding mean of per-tick volume observed
    up to t (estimated ticks/day from the data cadence), so no end-of-day
    information leaks into intraday values. The first bucket warm-starts after
    `warmup_ticks` observations.

    High VPIN → flow dominated by one side → adverse-selection risk for any
    resting order → spreads about to widen, one-sided moves likely.
    USE AS A GATE (scale exposure down when high), not as a direction signal.
    Validated as a risk-warning signal on Chinese index futures (2012–13 data).
    """
    dvb_s, dvs_s = _trade_flows(lob_df)
    dvb = dvb_s.to_numpy()
    dvs = dvs_s.to_numpy()
    tv  = dvb + dvs
    n   = len(tv)
    if n == 0 or tv.sum() <= 0.0:
        return pd.Series(0.0, index=lob_df.index, name="vpin")

    # Estimated snapshots per full session from observed cadence (fallback 4800)
    if isinstance(lob_df.index, pd.DatetimeIndex) and n > 1:
        med_dt = float(np.median(np.diff(lob_df.index.values).astype("timedelta64[s]").astype(float)))
        est_ticks_day = (4.0 * 3600.0) / max(med_dt, 1e-9)
    else:
        est_ticks_day = 4800.0

    warmup_ticks = max(10, int(est_ticks_day / n_buckets_day))

    # Sequential causal bucketing: target = expanding mean tick volume × ticks/bucket
    cum_tv  = 0.0
    buck_b  = 0.0   # buy volume in current bucket
    buck_s  = 0.0   # sell volume in current bucket
    buck_v  = 0.0   # total volume in current bucket
    imbs: list[float] = []          # completed-bucket imbalances
    out = np.zeros(n)
    last = 0.0

    for i in range(n):
        cum_tv += tv[i]
        buck_b += dvb[i]
        buck_s += dvs[i]
        buck_v += tv[i]

        mean_tick_vol = cum_tv / (i + 1)
        target = mean_tick_vol * (est_ticks_day / n_buckets_day)

        if i >= warmup_ticks and buck_v >= target and buck_v > 0.0:
            imbs.append(abs(buck_b - buck_s) / buck_v)
            if len(imbs) >= 2:
                last = float(np.mean(imbs[-smooth_buckets:]))
            buck_b = buck_s = buck_v = 0.0

        out[i] = last

    return pd.Series(out, index=lob_df.index, name="vpin")


# ---------------------------------------------------------------------------
# Tier 3 — Kyle's λ  (tick-level price impact; liquidity-state gate)
# ---------------------------------------------------------------------------

def kyle_lambda(
    lob_df: pd.DataFrame,
    window: int = 120,
) -> pd.Series:
    """
    Rolling tick-level price impact: return per unit of signed flow.

        Δmid_t = λ · q_t + ε_t      →     λ = Cov(Δmid, q) / Var(q)

    where q_t = (dv_buy − dv_sell)_t is per-tick signed executed volume and
    Δmid is the per-tick log-mid change. λ is clipped at 0 (negative impact
    estimates are noise) and z-scored against its own rolling history so the
    output is a stationary liquidity-STATE indicator:

        high λ → thin market, flow moves price → impact/reversal regime, cut size
        low  λ → deep market absorbs flow      → momentum can run, full size

    Use for sizing/turnover gating (Amihud-at-tick-scale); only weakly
    predictive directionally — don't put it in the directional composite.
    """
    dvb, dvs = _trade_flows(lob_df)
    q = dvb - dvs

    mid  = (lob_df["bid_px_1"].astype(float) + lob_df["ask_px_1"].astype(float)) / 2.0
    dmid = np.log(mid.replace(0.0, np.nan)).diff().fillna(0.0)

    mp  = max(20, window // 4)
    cov = dmid.rolling(window, min_periods=mp).cov(q)
    var = q.rolling(window, min_periods=mp).var()

    lam = (cov / var.replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)

    # z-score vs own rolling history → stationary state indicator
    mean = lam.rolling(400, min_periods=100).mean()
    std  = lam.rolling(400, min_periods=100).std()
    z    = ((lam - mean) / std.replace(0.0, np.nan)).fillna(0.0)

    return z.rename("kyle_lambda")


# ---------------------------------------------------------------------------
# Tier 3 — Cancellation-spike imbalance  (veto / confirmation signal)
# ---------------------------------------------------------------------------

def cancel_spike_imbalance(
    lob_df: pd.DataFrame,
    window: int = 10,
    spike_quantile: float = 0.9,
    spike_lookback: int = 300,
) -> pd.Series:
    """
    One-sided cancellation bursts, estimated from unexplained depth drops.

    A decrease in top-of-book depth is either execution or cancellation:

        cancel_bid ≈ max( −ΔV_bid_1 − dv_sell, 0 )     (drop not explained by sells)
        cancel_ask ≈ max( −ΔV_ask_1 − dv_buy , 0 )     (drop not explained by buys)

    Rolling sums over `window`; the imbalance is only emitted when total
    cancellation activity SPIKES above its rolling `spike_quantile` (Chinese
    books have a low baseline cancel ratio, so bursts are rare, low-noise
    events — the whole point of this factor):

        signal = (cancel_ask − cancel_bid) / total_cancels    when spiking, else 0

    Positive → sellers withdrawing quotes → bullish.
    Use as confirmation/veto on queue_imbalance (which a resting spoof can game):
    same sign → confirms; opposite sign → the visible depth is evaporating.
    """
    cancel_bid, cancel_ask = _cancel_estimates(lob_df)

    mp = max(1, window // 2)
    cb = cancel_bid.rolling(window, min_periods=mp).sum()
    ca = cancel_ask.rolling(window, min_periods=mp).sum()
    total = (ca + cb)

    thr      = total.rolling(spike_lookback, min_periods=30).quantile(spike_quantile)
    is_spike = total >= thr.fillna(np.inf)

    imb = ((ca - cb) / total.replace(0.0, np.nan)).fillna(0.0)
    return imb.where(is_spike, 0.0).rename("cancel_spike")


# ---------------------------------------------------------------------------
# Interaction — Spoof-filtered queue imbalance
# ---------------------------------------------------------------------------

def spoof_filtered_qi(
    lob_df: pd.DataFrame,
    n_levels: int = 5,
    window: int = 10,
    beta: float = 1.0,
) -> pd.Series:
    """
    Queue imbalance, trust-weighted by cancellation behaviour on the
    supporting side. Interaction factor: both inputs are public; the
    conditioning is the edge.

    Plain queue imbalance is spoofable: park a fat bid wall, lean the signal
    bullish, cancel before it trades. The tell is that the wall is being
    CANCELLED, not executed. Per tick:

        contra = cancel volume on the side that supports the QI sign,
                 over `window`, scaled by that side's standing depth
        trust  = exp( −beta · z⁺(contra) )            ∈ (0, 1]
        signal = QI · trust

    QI > 0 with a bid-side cancel burst → trust → 0 → fake support muted.
    QI backed by resting (uncancelled) depth passes through unchanged.
    Conservative by design: contradiction shrinks the signal toward zero,
    never flips it.
    """
    from .features import queue_imbalance as _qi

    qi = _qi(lob_df, n_levels=n_levels)
    cancel_bid, cancel_ask = _cancel_estimates(lob_df)

    mp = max(1, window // 2)
    cb = cancel_bid.rolling(window, min_periods=mp).sum()
    ca = cancel_ask.rolling(window, min_periods=mp).sum()

    bid_depth = lob_df["bid_vol_1"].astype(float).rolling(window, min_periods=mp).mean()
    ask_depth = lob_df["ask_vol_1"].astype(float).rolling(window, min_periods=mp).mean()

    cb_rate = (cb / bid_depth.replace(0.0, np.nan)).fillna(0.0)
    ca_rate = (ca / ask_depth.replace(0.0, np.nan)).fillna(0.0)

    # Cancels on the side the signal leans on = contradiction
    contra = cb_rate.where(qi > 0, ca_rate.where(qi < 0, 0.0))
    contra_z = _zscore(contra).clip(lower=0.0)

    trust = np.exp(-beta * contra_z)
    return (qi * trust).rename("qi_filtered")


# ---------------------------------------------------------------------------
# Interaction — Institutional-backed seal  (机构封单)
# ---------------------------------------------------------------------------

def institutional_seal(
    lob_df: pd.DataFrame,
    prev_close: float,
    limit_pct: float = 0.10,
    big_quantile: float = 0.8,
    flow_window: int = 40,
    threshold_lookback: int = 200,
) -> pd.Series:
    """
    Sealing strength weighted by large-trade participation — splits the 打板
    signal into institution-backed boards vs pure retail boards.

    A limit-up seal funded by institutional-size flow holds and continues
    next day far more reliably than a retail momentum pile-on, which breaks
    (炸板) and dumps. Per tick:

        big_frac = Σ_window big-trade volume / Σ_window total volume   ∈ [0, 1]
        signal   = sealing_strength · big_frac

    where "big" uses the same causal rolling-quantile threshold as
    big_order_flow. Retail-only seal (big_frac ≈ 0) → muted; fully
    institution-backed seal → passes at full strength. Zero away from limits
    (inherits sealing_strength support). Stock mode only.

    Horizon: next-day (T+1). Few quantify the seal queue by participant size —
    this interaction is the proprietary layer on a public factor.
    """
    seal = sealing_strength(lob_df, prev_close, limit_pct=limit_pct)

    dvb, dvs = _trade_flows(lob_df)
    tv = dvb + dvs

    thr    = tv.rolling(threshold_lookback, min_periods=20).quantile(big_quantile)
    is_big = tv >= thr   # NaN warm-up → False (causal)

    mp  = max(1, flow_window // 2)
    big = tv.where(is_big, 0.0).rolling(flow_window, min_periods=mp).sum()
    tot = tv.rolling(flow_window, min_periods=mp).sum()

    big_frac = (big / tot.replace(0.0, np.nan)).fillna(0.0).clip(0.0, 1.0)
    return (seal * big_frac).rename("seal_inst")


# ---------------------------------------------------------------------------
# Tier 3 — Exposure gate  (VPIN + λ → position-size multiplier)
# ---------------------------------------------------------------------------

def exposure_gate(
    lob_df: pd.DataFrame,
    vpin_weight: float = 0.6,
    lambda_weight: float = 0.4,
    steepness: float = 1.5,
    floor: float = 0.2,
) -> pd.Series:
    """
    Combine VPIN and Kyle-λ into a multiplicative exposure scale ∈ [floor, 1].

        risk  = w_v · z(VPIN) + w_λ · z(λ)
        scale = floor + (1 − floor) · sigmoid( −steepness · risk )

    risk ≈ 0 (normal conditions)  → scale ≈ midpoint ~0.6
    toxic + thin (risk >> 0)      → scale → floor   (cut exposure hard)
    benign + deep (risk << 0)     → scale → 1       (full size)

    Feed to run_backtest(..., exposure_scale=...): entry size is multiplied by
    the scale; entries are skipped entirely when the scaled size floors to zero.
    This implements the §4 regime-gate design from research/hft_factors_china.md.
    """
    v = vpin(lob_df)
    # Causal centering: expanding mean (NOT full-series mean — look-ahead)
    v_z = _zscore(v - v.expanding(min_periods=1).mean())
    l_z = kyle_lambda(lob_df)

    risk  = vpin_weight * v_z + lambda_weight * l_z
    scale = floor + (1.0 - floor) / (1.0 + np.exp(steepness * risk))

    return pd.Series(scale, index=lob_df.index, name="exposure_scale").clip(floor, 1.0)
