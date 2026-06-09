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
from .auction  import auction_signal_series


DEFAULT_WEIGHTS = {
    "trade_imbalance": 0.24,   # execution-confirmed; robust to spoofing
    "micro_price_dev": 0.18,   # Stoikov; clean level-1 asymmetry
    "mlofi":           0.20,   # primary LOB flow signal
    "queue_imbalance": 0.13,   # depth-side imbalance
    "agg_ofi":         0.14,   # sustained flow pressure
    "depth_tilt":      0.09,   # LOB shape; underweighted before
    "mom_5":           0.02,   # very-short momentum (mean-reversion dominant)
}
# Optional signal weights — drawn proportionally from DEFAULT_WEIGHTS when active
AUCTION_WEIGHT     = 0.10
PRICE_LIMIT_WEIGHT = 0.08   # stock mode only
ETF_BASIS_WEIGHT   = 0.07   # when etf_series provided


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
) -> pd.DataFrame:
    """
    Compute all features from a LOB snapshot DataFrame.

    Parameters
    ----------
    lob_df           : output of simulate_lob_day() or real L2 loader
    auction_value    : scalar from auction.auction_composite(); None = skip
    ofi_levels       : LOB depth levels for OFI (max 10)
    ofi_window       : rolling ticks for aggregated_ofi
    qi_levels        : LOB depth levels for queue_imbalance
    half_life_auction_min : exponential decay half-life in minutes
    prev_close       : previous close for price_limit_signal (stock mode)
    instrument       : "futures" or "stock"; enables price_limit when "stock"
    etf_series       : pd.Series of ETF prices aligned to lob_df; None = skip
    """
    from .features import price_limit_signal, etf_basis_signal

    feats: dict[str, pd.Series] = {}

    feats["mlofi"]           = mlofi(lob_df, n_levels=ofi_levels)
    feats["agg_ofi"]         = aggregated_ofi(lob_df, window=ofi_window, n_levels=ofi_levels)
    feats["trade_imbalance"] = trade_imbalance(lob_df, window=ofi_window)
    feats["micro_price_dev"] = micro_price_dev(lob_df)
    feats["queue_imbalance"] = queue_imbalance(lob_df, n_levels=qi_levels)
    feats["depth_tilt"]      = depth_tilt(lob_df)

    mom_df = short_term_momentum(lob_df, windows=[5])
    feats["mom_5"] = mom_df["mom_5"]

    if auction_value is not None:
        feats["auction_signal"] = auction_signal_series(
            lob_df, auction_value, half_life_min=half_life_auction_min
        )

    if instrument == "stock" and prev_close is not None:
        feats["price_limit"] = price_limit_signal(lob_df, prev_close)

    if etf_series is not None:
        feats["etf_basis"] = etf_basis_signal(lob_df, etf_series)

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
    has_auction     = "auction_signal" in feature_df.columns
    has_price_limit = "price_limit"    in feature_df.columns
    has_etf_basis   = "etf_basis"      in feature_df.columns

    if has_auction:
        optional_w += AUCTION_WEIGHT
    if has_price_limit:
        optional_w += PRICE_LIMIT_WEIGHT
    if has_etf_basis:
        optional_w += ETF_BASIS_WEIGHT

    if optional_w > 0.0:
        scale = 1.0 - optional_w
        w = {k: v * scale for k, v in w.items()}
        if has_auction:
            w["auction_signal"] = AUCTION_WEIGHT
        if has_price_limit:
            w["price_limit"] = PRICE_LIMIT_WEIGHT
        if has_etf_basis:
            w["etf_basis"] = ETF_BASIS_WEIGHT

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
