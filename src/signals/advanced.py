"""
Advanced strict factors, gates, and diagnostics.

Default directional alpha uses only formula-level factors declared in
`src.signals.composite.FACTOR_REGISTRY`: API and OEI from explicit order-event
fields live here. VPIN and Kyle lambda are gates, not directional votes.
Herding and cancellation bursts remain explicit diagnostics until their full
paper processes are implemented.

Missing paper-required event/count columns raise clear errors; callers should
use the registry availability helpers before requesting a factor on partial
real-data feeds.
"""

import math

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
    if "market_buy_vol" in lob_df.columns and "market_sell_vol" in lob_df.columns:
        return (
            lob_df["market_buy_vol"].astype(float).clip(lower=0).fillna(0.0),
            lob_df["market_sell_vol"].astype(float).clip(lower=0).fillna(0.0),
        )

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


def _trade_counts(lob_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-snapshot buyer/seller counts for LSV-style herding."""
    if "cum_buy_count" in lob_df.columns and "cum_sell_count" in lob_df.columns:
        return (
            lob_df["cum_buy_count"].astype(float).diff().clip(lower=0).fillna(0.0),
            lob_df["cum_sell_count"].astype(float).diff().clip(lower=0).fillna(0.0),
        )
    if "buy_count" in lob_df.columns and "sell_count" in lob_df.columns:
        return (
            lob_df["buy_count"].astype(float).clip(lower=0).fillna(0.0),
            lob_df["sell_count"].astype(float).clip(lower=0).fillna(0.0),
        )
    raise ValueError(
        "herding_intensity requires buy/sell count columns for the LSV "
        "method: cum_buy_count/cum_sell_count or buy_count/sell_count."
    )


def _market_event_columns(lob_df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Market-order and limit-order volumes for the api formula.

    The api formula only needs market (active) and limit (passive) volumes —
    cancel volumes are NOT part of the Σ(M_buy-M_sell)/Σ(M+L) formula.
    Using the smaller required set lets api run on Wind L2 3-second bars
    where active/passive breakdowns are available but cancel data is not.
    """
    required = ["limit_buy_vol", "limit_sell_vol", "market_buy_vol", "market_sell_vol"]
    missing = [c for c in required if c not in lob_df.columns]
    if missing:
        raise ValueError(
            "api requires active/passive order-event columns: "
            + ", ".join(required)
            + f". Missing: {', '.join(missing)}."
        )
    return {c: lob_df[c].astype(float).clip(lower=0).fillna(0.0) for c in required}


def _order_event_columns(
    lob_df: pd.DataFrame,
    cancel_fallback: bool = False,
) -> dict[str, pd.Series]:
    """
    Full order-event volumes for oei / cancel_spike (L/C/M notation, Chi 2021).

    When `cancel_fallback=True` and explicit cancel columns are absent, cancel
    volumes are estimated from level-1 depth changes minus attributed executions:

        cancel_bid ≈ max( −ΔV_bid_1 − dv_sell, 0 )
        cancel_ask ≈ max( −ΔV_ask_1 − dv_buy,  0 )

    This is an approximation (level-1 only; sub-tick cancels are unobservable),
    but it is strictly causal and substantially better than failing outright on
    Wind L2 3-second snapshot data where cancel feeds are not exposed.
    """
    required = [
        "limit_buy_vol",
        "limit_sell_vol",
        "cancel_buy_vol",
        "cancel_sell_vol",
        "market_buy_vol",
        "market_sell_vol",
    ]
    missing = [c for c in required if c not in lob_df.columns]

    if missing and not cancel_fallback:
        raise ValueError(
            "Exact order-event factors require L2 order-event columns: "
            + ", ".join(required)
            + f". Missing: {', '.join(missing)}."
        )

    out: dict[str, pd.Series] = {}

    # Market + limit volumes — required even in fallback mode
    market_limit = ["limit_buy_vol", "limit_sell_vol", "market_buy_vol", "market_sell_vol"]
    ml_missing = [c for c in market_limit if c not in lob_df.columns]
    if ml_missing:
        raise ValueError(
            "oei/cancel_spike require at minimum active+passive order-event columns: "
            + ", ".join(market_limit)
            + f". Missing: {', '.join(ml_missing)}."
        )
    for c in market_limit:
        out[c] = lob_df[c].astype(float).clip(lower=0).fillna(0.0)

    # Cancel volumes — exact if present, estimated if fallback
    if "cancel_buy_vol" in lob_df.columns and "cancel_sell_vol" in lob_df.columns:
        out["cancel_buy_vol"]  = lob_df["cancel_buy_vol"].astype(float).clip(lower=0).fillna(0.0)
        out["cancel_sell_vol"] = lob_df["cancel_sell_vol"].astype(float).clip(lower=0).fillna(0.0)
    elif cancel_fallback:
        out["cancel_buy_vol"], out["cancel_sell_vol"] = _cancel_estimates(lob_df)
    else:
        raise ValueError("cancel_buy_vol / cancel_sell_vol missing and cancel_fallback=False")

    if "bid_depth" in lob_df.columns:
        out["bid_depth"] = lob_df["bid_depth"].astype(float)
    else:
        n = _available_levels(lob_df, 10)
        out["bid_depth"] = sum(lob_df[f"bid_vol_{lv}"].astype(float) for lv in range(1, n + 1))
    if "ask_depth" in lob_df.columns:
        out["ask_depth"] = lob_df["ask_depth"].astype(float)
    else:
        n = _available_levels(lob_df, 10)
        out["ask_depth"] = sum(lob_df[f"ask_vol_{lv}"].astype(float) for lv in range(1, n + 1))
    return out


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


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

    Returns (cancel_bid, cancel_ask), both >= 0, aligned to lob_df.index.
    Used by cancel_spike_imbalance.
    """
    dvb, dvs = _trade_flows(lob_df)
    db = lob_df["bid_vol_1"].astype(float).diff()
    da = lob_df["ask_vol_1"].astype(float).diff()
    cancel_bid = ((-db).clip(lower=0.0) - dvs).clip(lower=0.0).fillna(0.0)
    cancel_ask = ((-da).clip(lower=0.0) - dvb).clip(lower=0.0).fillna(0.0)
    return cancel_bid, cancel_ask


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

    This is the LSV formula applied to intraday buyer/seller counts for one
    instrument. Cross-sectional panel construction belongs outside this
    per-instrument signal.
    """
    nb, ns = _trade_counts(lob_df)
    n_trade = (nb + ns).rolling(window, min_periods=max(2, window // 2)).sum()
    n_buy = nb.rolling(window, min_periods=max(2, window // 2)).sum()
    p       = (n_buy / n_trade.replace(0.0, np.nan)).fillna(0.5)

    af       = np.sqrt(2.0 / (np.pi * window)) * 0.5
    h_excess = (p - 0.5).abs() - af
    signal   = np.sign(p - 0.5) * h_excess.clip(lower=0.0)

    return _zscore(signal.fillna(0.0)).rename("herding")


# ---------------------------------------------------------------------------
# Tier 2 — Aggressive-Passive Imbalance  (API)
# ---------------------------------------------------------------------------

def aggressive_passive_imbalance(
    lob_df: pd.DataFrame,
    window: int = 10,
    k_levels: int = 5,
    cancel_fallback: bool = False,
) -> pd.Series:
    """
    Spread-pressure metric from Zhao et al. (2025) Definition 7.

    Paper: "High-frequency liquidity in the Chinese stock market"
           Zhao, Chen, Wu, Dai, Chen, Wu, Zhang — Pacific-Basin Finance Journal 90 (2025).

    For interval [t, t+Δt]:
        API_{t,t+Δt} = (A^b + A^a + C^b + C^a − P^b − P^a) / D̃_t

    where:
      A^b / A^a = aggressive buy / sell (market-order) volumes
      C^b / C^a = cancellation buy / sell volumes
      P^b / P^a = passive buy / sell (new limit-order submission) volumes
      D̃_t       = (1/2K) Σ_{i=1}^{K} [n^b_i(t) + n^a_i(t)]  (avg level size, K=5)

    Positive API → aggressive+cancel exceeds passive submissions → spread widens.

    This is a GATE / liquidity-condition signal, NOT a directional price predictor.
    High API → adverse spread-widening → reduce position size.

    cancel_fallback: estimate cancel volumes from LOB depth changes when explicit
        cancel columns are absent (same approximation used by oei/cancel_spike).
    """
    e = _order_event_columns(lob_df, cancel_fallback=cancel_fallback)

    aggressive = e["market_buy_vol"] + e["market_sell_vol"]
    cancel_vol = e["cancel_buy_vol"] + e["cancel_sell_vol"]
    passive    = e["limit_buy_vol"]  + e["limit_sell_vol"]

    # D̃_t: Zhao eq. 7 — average of top K=5 LOB level sizes on both sides
    n = min(_available_levels(lob_df, k_levels), k_levels)
    bid_sum = pd.Series(0.0, index=lob_df.index)
    ask_sum = pd.Series(0.0, index=lob_df.index)
    for lv in range(1, n + 1):
        if f"bid_vol_{lv}" in lob_df.columns:
            bid_sum = bid_sum + lob_df[f"bid_vol_{lv}"].astype(float).fillna(0.0)
        if f"ask_vol_{lv}" in lob_df.columns:
            ask_sum = ask_sum + lob_df[f"ask_vol_{lv}"].astype(float).fillna(0.0)
    depth = ((bid_sum + ask_sum) / (2 * n)).replace(0.0, np.nan)

    raw = ((aggressive + cancel_vol - passive) / depth).fillna(0.0)
    signal = raw.rolling(window, min_periods=max(1, window // 2)).mean().fillna(0.0)
    return _zscore(signal).rename("api")


# ---------------------------------------------------------------------------
# Tier 2 — Order Execution Imbalance  (OEI)
# ---------------------------------------------------------------------------

def order_execution_imbalance(
    lob_df: pd.DataFrame,
    window: int = 10,
    depth_levels: int = 3,
    cancel_fallback: bool = False,
) -> pd.Series:
    """
    Depth-normalized OFI from Cont, Kukanov, Stoikov (2014) eq. 4.

    NOTE ON ATTRIBUTION: Chi (2021) "Order Execution Imbalance" (OEI) is a
    different quantity — it measures the ratio of limit orders executed within
    10 seconds on each side, requiring per-order execution-duration data that
    is NOT available from LOB snapshots. That signal cannot be computed here.

    What this function computes is Cont et al. (2014) OFI with explicit depth
    normalization — i.e. eq. 4 from Chi (2021) which Chi attributes to Cont:

        (L^b_k - C^b_k - M^s_k) / D^b_k  −  (L^s_k - C^s_k - M^b_k) / D^s_k

    where L = limit-order volume, C = cancellation volume, M = market-order
    volume, D = total book depth on that side. Positive → bid side net
    replenished relative to depth; ask side net depleted → bullish.

    cancel_fallback : when True and explicit cancel columns are absent, cancel
        volumes are estimated from level-1 LOB depth changes.
    """
    e = _order_event_columns(lob_df, cancel_fallback=cancel_fallback)
    bid_depth = e["bid_depth"].shift(1).replace(0.0, np.nan)
    ask_depth = e["ask_depth"].shift(1).replace(0.0, np.nan)

    bid_term = (e["limit_buy_vol"] - e["cancel_buy_vol"] - e["market_sell_vol"]) / bid_depth
    ask_term = (e["limit_sell_vol"] - e["cancel_sell_vol"] - e["market_buy_vol"]) / ask_depth
    oei = (bid_term - ask_term).fillna(0.0)
    return _zscore(oei.rolling(window, min_periods=max(1, window // 2)).mean().fillna(0.0)).rename("oei")


# ---------------------------------------------------------------------------
# Tier 2 — Realized volatility helper
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

    Buy/sell classification follows the paper's Bulk Volume Classification
    (BVC):

        V_buy = V * Phi(delta_p / sigma_delta_p)
        V_sell = V - V_buy

    Trade volume is then split into equal-volume buckets. Bucket boundaries are
    determined sequentially with an expanding estimate of daily volume, so the
    implementation keeps the paper mechanics without using future ticks.

    High VPIN → flow dominated by one side → adverse-selection risk for any
    resting order → spreads about to widen, one-sided moves likely.
    USE AS A GATE (scale exposure down when high), not as a direction signal.
    Validated as a risk-warning signal on Chinese index futures (2012–13 data).
    """
    if "last_volume" in lob_df.columns:
        tv_s = lob_df["last_volume"].astype(float).clip(lower=0).fillna(0.0)
    elif "cum_buy_vol" in lob_df.columns and "cum_sell_vol" in lob_df.columns:
        tv_s = (
            lob_df["cum_buy_vol"].astype(float).diff().clip(lower=0).fillna(0.0)
            + lob_df["cum_sell_vol"].astype(float).diff().clip(lower=0).fillna(0.0)
        )
    else:
        raise ValueError("vpin requires last_volume or cumulative buy/sell volume columns")

    if "last_price" in lob_df.columns:
        price = lob_df["last_price"].astype(float)
    elif "bid_px_1" in lob_df.columns and "ask_px_1" in lob_df.columns:
        price = (lob_df["bid_px_1"].astype(float) + lob_df["ask_px_1"].astype(float)) / 2.0
    else:
        raise ValueError("vpin requires last_price or bid_px_1/ask_px_1 for BVC classification")

    dp = price.diff().fillna(0.0)
    sigma = dp.rolling(50, min_periods=10).std()
    sigma = sigma.fillna(dp.expanding(min_periods=2).std()).replace(0.0, np.nan)
    z = (dp / sigma).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    buy_frac = _normal_cdf(z)

    tv = tv_s.to_numpy()
    dvb = tv * buy_frac
    dvs = tv - dvb
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

    # Sequential causal bucketing. Large trades are split across bucket
    # boundaries, as in volume-time VPIN.
    cum_tv  = 0.0
    buck_b  = 0.0   # buy volume in current bucket
    buck_s  = 0.0   # sell volume in current bucket
    buck_v  = 0.0   # total volume in current bucket
    imbs: list[float] = []          # completed-bucket imbalances
    out = np.zeros(n)
    last = 0.0

    for i in range(n):
        cum_tv += tv[i]

        mean_tick_vol = cum_tv / (i + 1)
        target = mean_tick_vol * (est_ticks_day / n_buckets_day)
        rem_b = dvb[i]
        rem_s = dvs[i]
        rem_v = tv[i]

        while rem_v > 0.0 and target > 0.0:
            room = max(target - buck_v, 0.0)
            take = min(rem_v, room if room > 0.0 else target)
            frac = take / rem_v
            buck_b += rem_b * frac
            buck_s += rem_s * frac
            buck_v += take
            rem_b -= rem_b * frac
            rem_s -= rem_s * frac
            rem_v -= take

            if i >= warmup_ticks and buck_v >= target and buck_v > 0.0:
                imbs.append(abs(buck_b - buck_s) / buck_v)
                if imbs:
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
    Δmid is the per-tick log-mid change. This function returns the raw rolling
    paper estimate. Use `kyle_lambda_state()` when a stationary gate input is
    needed.
    """
    dvb, dvs = _trade_flows(lob_df)
    q = dvb - dvs

    mid  = (lob_df["bid_px_1"].astype(float) + lob_df["ask_px_1"].astype(float)) / 2.0
    dmid = np.log(mid.replace(0.0, np.nan)).diff().fillna(0.0)

    mp  = max(20, window // 4)
    cov = dmid.rolling(window, min_periods=mp).cov(q)
    var = q.rolling(window, min_periods=mp).var()

    lam = (cov / var.replace(0.0, np.nan)).fillna(0.0)
    return lam.rename("kyle_lambda")


def kyle_lambda_state(
    lob_df: pd.DataFrame,
    window: int = 120,
) -> pd.Series:
    """
    Stationary gate transform of raw Kyle lambda.

    The raw lambda is clipped at zero for risk-state interpretation and
    z-scored against its own rolling history. This is intentionally separate
    from `kyle_lambda()` so the paper formula remains inspectable.
    """
    lam = kyle_lambda(lob_df, window=window).clip(lower=0.0)
    mean = lam.rolling(400, min_periods=100).mean()
    std  = lam.rolling(400, min_periods=100).std()
    z    = ((lam - mean) / std.replace(0.0, np.nan)).fillna(0.0)

    return z.rename("kyle_lambda_state")


# ---------------------------------------------------------------------------
# Tier 3 — Cancellation-spike imbalance  (veto / confirmation signal)
# ---------------------------------------------------------------------------

def cancel_spike_imbalance(
    lob_df: pd.DataFrame,
    window: int = 10,
    spike_quantile: float = 0.9,
    spike_lookback: int = 300,
    cancel_fallback: bool = False,
) -> pd.Series:
    """
    One-sided cancellation bursts from explicit cancellation events.

    Uses the event volumes required by the paper setup rather than inferring
    cancels from snapshot depth drops:

        signal = (C_ask - C_bid) / (C_ask + C_bid)

    The spike filter is retained as the factor-selection rule; the cancellation
    inputs themselves are exact event fields.

    Positive → sellers withdrawing quotes → bullish.
    Use as confirmation/veto on queue_imbalance (which a resting spoof can game):
    same sign → confirms; opposite sign → the visible depth is evaporating.

    cancel_fallback : when True and explicit cancel columns are absent, cancel
        volumes are estimated from LOB depth changes (see _cancel_estimates).
        Useful on Wind L2 snapshots; note the estimate is level-1 only.
    """
    if cancel_fallback and (
        "cancel_buy_vol" not in lob_df.columns
        or "cancel_sell_vol" not in lob_df.columns
    ):
        cancel_bid, cancel_ask = _cancel_estimates(lob_df)
    else:
        e = _order_event_columns(lob_df, cancel_fallback=False)
        cancel_bid = e["cancel_buy_vol"]
        cancel_ask = e["cancel_sell_vol"]

    mp = max(1, window // 2)
    cb = cancel_bid.rolling(window, min_periods=mp).sum()
    ca = cancel_ask.rolling(window, min_periods=mp).sum()
    total = (ca + cb)

    min_spike_periods = min(max(1, spike_lookback), 30)
    thr      = total.rolling(spike_lookback, min_periods=min_spike_periods).quantile(spike_quantile)
    is_spike = total >= thr.fillna(np.inf)

    imb = ((ca - cb) / total.replace(0.0, np.nan)).fillna(0.0)
    return imb.where(is_spike, 0.0).rename("cancel_spike")


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
    l_z = kyle_lambda_state(lob_df)

    risk  = vpin_weight * v_z + lambda_weight * l_z
    scale = floor + (1.0 - floor) / (1.0 + np.exp(steepness * risk))

    return pd.Series(scale, index=lob_df.index, name="exposure_scale").clip(floor, 1.0)
