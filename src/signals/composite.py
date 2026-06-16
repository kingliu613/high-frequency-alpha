"""
Composite alpha signal combining all LOB features.

Signal hierarchy (by empirical importance for Chinese index futures):
  1. mlofi          – primary directional signal         weight 0.20
  2. agg_ofi        – sustained flow pressure            weight 0.14
  3. trade_imbalance– executed order confirmation        weight 0.24
  4. micro_price_dev– price-weighted imbalance           weight 0.18
  5. queue_imbalance– depth-side imbalance               weight 0.13
  6. depth_tilt     – shape asymmetry                    weight 0.09
  7. mom_5          – very-short momentum (5 ticks)      weight 0.02
  8. auction_signal – decaying auction prior             (if available)
  9. price_limit    – price-limit approach signal        (stock mode only)
 10. etf_basis      – ETF premium/discount mean-reversion (if etf_series)

Weights are soft defaults; override via `weights` argument.
All features are independently z-scored before combination to
prevent high-variance features (mlofi) from dominating.
"""

import numpy as np
import pandas as pd
from typing import Optional

from .ofi      import mlofi, aggregated_ofi, trade_imbalance
from .features import (
    micro_price_dev,
    queue_imbalance,
    depth_tilt,
    short_term_momentum,
)
from .auction  import auction_signal_series, close_auction_signal_series
from .advanced import (
    big_order_flow,
    herding_intensity,
    aggressive_passive_imbalance,
    order_execution_imbalance,
    order_book_slope,
    book_resiliency,
    signed_jump_reversal,
    sealing_strength,
    cancel_spike_imbalance,
    spoof_filtered_qi,
    institutional_seal,
)


DEFAULT_WEIGHTS = {
    # --- base LOB / flow signals ---
    "trade_imbalance": 0.12,   # execution-confirmed; robust to spoofing
    "micro_price_dev": 0.10,   # Stoikov; clean level-1 asymmetry
    "mlofi":           0.12,   # primary LOB flow signal
    "queue_imbalance": 0.04,   # depth-side imbalance (raw; spoofable)
    "qi_filtered":     0.03,   # spoof-filtered QI (interaction)
    "agg_ofi":         0.09,   # sustained flow pressure
    "depth_tilt":      0.05,   # LOB shape
    "mom_5":           0.02,   # very-short momentum (mean-reversion dominant)
    # --- advanced Tier-1/Tier-2 factors (research/hft_factors_china.md) ---
    "api":             0.09,   # aggressive-passive imbalance (spread driver)
    "oei":             0.08,   # order-execution imbalance (low-cancel CN edge)
    "big_flow":        0.09,   # 大单净额 institutional footprint
    "herding":         0.05,   # intraday retail herding (continuation leg)
    "book_slope":      0.05,   # cumulative-depth-per-tick asymmetry
    "resiliency":      0.04,   # post-aggression top-of-book recovery
    "signed_jump":     0.02,   # BNS jump → short-horizon reversal
    "cancel_spike":    0.01,   # rare one-sided cancel bursts (veto/confirm)
}
# Optional signal weights — drawn proportionally from DEFAULT_WEIGHTS when active
AUCTION_WEIGHT       = 0.10
PRICE_LIMIT_WEIGHT   = 0.08   # stock mode only
ETF_BASIS_WEIGHT     = 0.07   # when etf_series provided
CLOSE_AUCTION_WEIGHT = 0.08   # when close_auction_value provided
SEAL_WEIGHT          = 0.04   # 涨停/跌停封单 raw; stock mode only
SEAL_INST_WEIGHT     = 0.05   # institution-backed seal (interaction); stock mode


# ---------------------------------------------------------------------------
# Modular factor selection
# ---------------------------------------------------------------------------
# Pass factors=["flow", "auction"] (group names), factors=["mlofi", "api"]
# (factor names), or a mix, to build_feature_matrix. None = everything.
# build_composite_alpha automatically renormalises weights over whatever
# columns are present, so any subset forms a valid standalone strategy.

FACTOR_GROUPS: dict[str, list[str]] = {
    # order-flow factors: who is trading and which way
    "flow":        ["mlofi", "agg_ofi", "trade_imbalance", "api", "oei", "big_flow"],
    # standing-book structure: where the volume sits
    "book":        ["micro_price_dev", "queue_imbalance", "depth_tilt",
                    "book_slope", "resiliency"],
    # behavioural / retail dynamics
    "behavior":    ["herding", "mom_5", "signed_jump"],
    # call-auction information windows (need auction scalars)
    "auction":     ["auction_signal", "close_auction"],
    # price-limit regime (stock mode + prev_close required)
    "limit":       ["price_limit", "sealing", "seal_inst"],
    # proprietary interaction layer
    "interaction": ["qi_filtered", "seal_inst", "cancel_spike"],
    # ETF premium/discount (needs etf_series)
    "etf":         ["etf_basis"],
}

ALL_FACTORS: set[str] = set().union(*FACTOR_GROUPS.values())


def expand_factor_selection(factors) -> Optional[set[str]]:
    """
    Expand a mixed list of group names / factor names into a factor set.
    None passes through (= all factors). Unknown names raise ValueError.
    """
    if factors is None:
        return None
    out: set[str] = set()
    for f in factors:
        if f in FACTOR_GROUPS:
            out.update(FACTOR_GROUPS[f])
        elif f in ALL_FACTORS:
            out.add(f)
        else:
            raise ValueError(
                f"Unknown factor or group '{f}'. "
                f"Groups: {sorted(FACTOR_GROUPS)}; factors: {sorted(ALL_FACTORS)}"
            )
    return out


def build_feature_matrix(
    lob_df: pd.DataFrame,
    auction_value: Optional[float] = None,
    ofi_levels: int = 5,
    ofi_window: int = 10,
    qi_levels:  int = 5,
    half_life_auction_min: float = 20.0,
    prev_close: Optional[float] = None,
    instrument: str = "futures",
    etf_series=None,
    close_auction_value: Optional[float] = None,
    big_window: int = 40,
    herding_window: int = 20,
    jump_window: int = 20,
    resil_window: int = 20,
    half_life_close_min: float = 30.0,
    factors=None,
) -> pd.DataFrame:
    """
    Compute features from a LOB snapshot DataFrame.

    Parameters
    ----------
    lob_df           : output of simulate_lob_day() or real L2 loader
    auction_value    : scalar from auction.auction_composite(); None = skip
    ofi_levels       : LOB depth levels for OFI (max 10)
    ofi_window       : rolling ticks for aggregated_ofi / API / OEI
    qi_levels        : LOB depth levels for queue_imbalance / book_slope
    half_life_auction_min : opening-auction decay half-life in minutes
    prev_close       : previous close; enables price_limit + sealing (stock mode)
    instrument       : "futures" or "stock"; enables price_limit/sealing when "stock"
    etf_series       : pd.Series of ETF prices aligned to lob_df; None = skip
    close_auction_value : prior-session close-auction imbalance scalar; None = skip
    big_window       : rolling ticks for 大单 net flow
    herding_window   : rolling ticks for intraday herding intensity
    jump_window      : rolling ticks for signed-jump reversal
    resil_window     : rolling ticks for book resiliency
    half_life_close_min : close-auction prior decay half-life in minutes
    factors          : modular strategy selection — list of group names
                       (see FACTOR_GROUPS) and/or factor names; None = all.
                       Optional factors still require their inputs (e.g.
                       "auction_signal" needs auction_value).
    """
    from .features import price_limit_signal, etf_basis_signal

    selected = expand_factor_selection(factors)

    def want(name: str) -> bool:
        return selected is None or name in selected

    feats: dict[str, pd.Series] = {}

    # --- base LOB / flow signals ---
    if want("mlofi"):
        feats["mlofi"] = mlofi(lob_df, n_levels=ofi_levels)
    if want("agg_ofi"):
        feats["agg_ofi"] = aggregated_ofi(lob_df, window=ofi_window, n_levels=ofi_levels)
    if want("trade_imbalance"):
        feats["trade_imbalance"] = trade_imbalance(lob_df, window=ofi_window)
    if want("micro_price_dev"):
        feats["micro_price_dev"] = micro_price_dev(lob_df)
    if want("queue_imbalance"):
        feats["queue_imbalance"] = queue_imbalance(lob_df, n_levels=qi_levels)
    if want("depth_tilt"):
        feats["depth_tilt"] = depth_tilt(lob_df)
    if want("mom_5"):
        feats["mom_5"] = short_term_momentum(lob_df, windows=[5])["mom_5"]

    # --- advanced Tier-1 / Tier-2 factors ---
    if want("api"):
        feats["api"] = aggressive_passive_imbalance(lob_df, window=ofi_window)
    if want("oei"):
        feats["oei"] = order_execution_imbalance(lob_df, window=ofi_window)
    if want("big_flow"):
        feats["big_flow"] = big_order_flow(lob_df, window=big_window)
    if want("herding"):
        feats["herding"] = herding_intensity(lob_df, window=herding_window)
    if want("book_slope"):
        feats["book_slope"] = order_book_slope(lob_df, n_levels=qi_levels)
    if want("resiliency"):
        feats["resiliency"] = book_resiliency(lob_df, window=resil_window)
    if want("signed_jump"):
        feats["signed_jump"] = signed_jump_reversal(lob_df, window=jump_window)
    if want("cancel_spike"):
        feats["cancel_spike"] = cancel_spike_imbalance(lob_df, window=ofi_window)

    # --- interaction layer ---
    if want("qi_filtered"):
        feats["qi_filtered"] = spoof_filtered_qi(lob_df, n_levels=qi_levels,
                                                 window=ofi_window)

    # --- optional priors / regime-specific ---
    if auction_value is not None and want("auction_signal"):
        feats["auction_signal"] = auction_signal_series(
            lob_df, auction_value, half_life_min=half_life_auction_min
        )

    if close_auction_value is not None and want("close_auction"):
        feats["close_auction"] = close_auction_signal_series(
            lob_df, close_auction_value, half_life_min=half_life_close_min
        )

    if instrument == "stock" and prev_close is not None:
        if want("price_limit"):
            feats["price_limit"] = price_limit_signal(lob_df, prev_close)
        if want("sealing"):
            feats["sealing"] = sealing_strength(lob_df, prev_close)
        if want("seal_inst"):
            feats["seal_inst"] = institutional_seal(lob_df, prev_close,
                                                    flow_window=big_window)

    if etf_series is not None and want("etf_basis"):
        feats["etf_basis"] = etf_basis_signal(lob_df, etf_series)

    if not feats:
        raise ValueError(
            "Factor selection produced an empty feature set — selected "
            "factors may all be optional ones whose inputs were not provided."
        )

    return pd.DataFrame(feats, index=lob_df.index)


def build_composite_alpha(
    feature_df: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
    clip_z: float = 4.0,
    norm_window: int = 200,
) -> pd.Series:
    """
    Combine features into a single composite alpha.

    Each feature is independently z-scored before weighting to
    equalise variance across signals. The final output is also
    z-scored so downstream threshold logic is scale-invariant.

    When optional signals (auction, price_limit, etf_basis) are present,
    their weights are drawn from the other signals proportionally so
    total weight remains 1.0.

    Parameters
    ----------
    feature_df  : output of build_feature_matrix
    weights     : override default weights (partial OK)
    clip_z      : clip individual features at ±clip_z before combining
    norm_window : rolling window for output normalisation
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Compute total optional weight for signals present in feature_df
    optional_w = 0.0
    has_auction       = "auction_signal" in feature_df.columns
    has_price_limit   = "price_limit"    in feature_df.columns
    has_etf_basis     = "etf_basis"      in feature_df.columns
    has_close_auction = "close_auction"  in feature_df.columns
    has_sealing       = "sealing"        in feature_df.columns
    has_seal_inst     = "seal_inst"      in feature_df.columns

    if has_auction:
        optional_w += AUCTION_WEIGHT
    if has_price_limit:
        optional_w += PRICE_LIMIT_WEIGHT
    if has_etf_basis:
        optional_w += ETF_BASIS_WEIGHT
    if has_close_auction:
        optional_w += CLOSE_AUCTION_WEIGHT
    if has_sealing:
        optional_w += SEAL_WEIGHT
    if has_seal_inst:
        optional_w += SEAL_INST_WEIGHT

    if optional_w > 0.0:
        scale = 1.0 - optional_w
        w = {k: v * scale for k, v in w.items()}
        if has_auction:
            w["auction_signal"] = AUCTION_WEIGHT
        if has_price_limit:
            w["price_limit"] = PRICE_LIMIT_WEIGHT
        if has_etf_basis:
            w["etf_basis"] = ETF_BASIS_WEIGHT
        if has_close_auction:
            w["close_auction"] = CLOSE_AUCTION_WEIGHT
        if has_sealing:
            w["sealing"] = SEAL_WEIGHT
        if has_seal_inst:
            w["seal_inst"] = SEAL_INST_WEIGHT

    total_w = sum(w.get(c, 0.0) for c in feature_df.columns)
    if total_w == 0.0:
        raise ValueError("No feature weights matched available columns")

    alpha = pd.Series(0.0, index=feature_df.index)

    for col in feature_df.columns:
        wt = w.get(col, 0.0)
        if wt == 0.0:
            continue

        raw  = feature_df[col].fillna(0.0)
        rstd = raw.rolling(200, min_periods=50).std()
        z    = (raw / rstd.replace(0.0, np.nan)).fillna(0.0).clip(-clip_z, clip_z)
        alpha += (wt / total_w) * z

    # Final z-score normalisation
    rstd  = alpha.rolling(norm_window, min_periods=50).std()
    alpha = (alpha / rstd.replace(0.0, np.nan)).fillna(0.0)

    return alpha.rename("composite_alpha")
