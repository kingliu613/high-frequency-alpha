"""
Pre-flight schema validator for Wind L2 LOB DataFrames.

Run validate_lob_schema(df) immediately after loading to understand
which signals can be computed before any signal code is called.

Usage:
    from src.data.validator import validate_lob_schema, print_validation_report

    df = load_or_fetch_wind("IF2401.CFFEX", "2024-01-02")
    report = validate_lob_schema(df)
    print_validation_report(report)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Required column sets per signal (mirrors composite.py FACTOR_REGISTRY)
# ---------------------------------------------------------------------------

_LOB_LEVELS_10 = tuple(
    col
    for lv in range(1, 11)
    for col in (f"bid_px_{lv}", f"bid_vol_{lv}", f"ask_px_{lv}", f"ask_vol_{lv}")
)

_LOB_LEVELS_5 = tuple(
    col
    for lv in range(1, 6)
    for col in (f"bid_px_{lv}", f"bid_vol_{lv}", f"ask_px_{lv}", f"ask_vol_{lv}")
)

# (signal_name, role, required_cols, alternative_required_cols, notes)
_SIGNAL_SPECS: list[tuple] = [
    # --- strict alpha ---
    (
        "mlofi", "alpha",
        _LOB_LEVELS_10,
        (_LOB_LEVELS_5,),
        "Uses n_levels LOB; 10 ideal, 5 minimum",
    ),
    (
        "agg_ofi", "alpha",
        _LOB_LEVELS_10,
        (_LOB_LEVELS_5,),
        "Same LOB requirement as mlofi",
    ),
    (
        "trade_imbalance", "alpha",
        ("cum_buy_count", "cum_sell_count"),
        (("buy_count", "sell_count"),),
        "Needs buy/sell transaction COUNTS (NOB/NOS), not volumes",
    ),
    (
        "api", "gate",
        ("limit_buy_vol", "limit_sell_vol", "cancel_buy_vol", "cancel_sell_vol",
         "market_buy_vol", "market_sell_vol", "bid_vol_1", "ask_vol_1"),
        (),
        "Zhao 2025 spread-pressure gate: (M+C-L)/D̃ — NON-directional, not strict alpha",
    ),
    (
        "oei", "diagnostic",
        ("limit_buy_vol", "limit_sell_vol", "cancel_buy_vol",
         "cancel_sell_vol", "market_buy_vol", "market_sell_vol",
         "bid_depth", "ask_depth"),
        (("limit_buy_vol", "limit_sell_vol", "market_buy_vol", "market_sell_vol",
          "bid_px_1", "bid_vol_1", "ask_px_1", "ask_vol_1"),),
        "Cont (2014) depth-normalized OFI — Chi 2021 OEI uncomputable from LOB snapshots",
    ),
    # --- gates ---
    (
        "vpin", "gate",
        ("last_price", "last_volume"),
        (),
        "BVC equal-volume bucket VPIN; needs per-bar volume",
    ),
    (
        "kyle_lambda", "gate",
        ("bid_px_1", "ask_px_1", "cum_buy_vol", "cum_sell_vol"),
        (("bid_px_1", "ask_px_1", "market_buy_vol", "market_sell_vol"),),
        "Price-impact estimate; needs cumulative signed flow",
    ),
    # --- diagnostics ---
    (
        "queue_imbalance", "diagnostic",
        ("bid_vol_1", "ask_vol_1"),
        (),
        "Best bid/ask depth ratio",
    ),
]


@dataclass
class SignalStatus:
    name: str
    role: str
    available: bool
    missing_columns: list[str]
    notes: str
    n_levels_available: int = 0  # for mlofi/agg_ofi


@dataclass
class ValidationReport:
    n_rows: int
    n_cols: int
    index_is_datetime: bool
    has_ticker_col: bool
    n_levels_complete: int   # highest N where all bid/ask px+vol_N present
    n_rows_any_nan_bid1: int
    signals: list[SignalStatus] = field(default_factory=list)

    @property
    def runnable_alpha(self) -> list[str]:
        return [s.name for s in self.signals if s.available and s.role == "alpha"]

    @property
    def runnable_gates(self) -> list[str]:
        return [s.name for s in self.signals if s.available and s.role == "gate"]

    @property
    def blocked(self) -> list[SignalStatus]:
        return [s for s in self.signals if not s.available]


def _count_complete_lob_levels(df: pd.DataFrame) -> int:
    for lv in range(10, 0, -1):
        needed = (f"bid_px_{lv}", f"bid_vol_{lv}", f"ask_px_{lv}", f"ask_vol_{lv}")
        if all(c in df.columns for c in needed):
            return lv
    return 0


def _check_signal(df: pd.DataFrame, spec: tuple) -> SignalStatus:
    name, role, required, alternatives, notes = spec

    def _missing(cols):
        return [c for c in cols if c not in df.columns]

    miss = _missing(required)
    if not miss:
        n_lv = _count_complete_lob_levels(df) if name in ("mlofi", "agg_ofi") else 0
        return SignalStatus(name, role, True, [], notes, n_lv)

    for alt in alternatives:
        if not _missing(alt):
            n_lv = _count_complete_lob_levels(df) if name in ("mlofi", "agg_ofi") else 0
            return SignalStatus(name, role, True, [], notes, n_lv)

    return SignalStatus(name, role, False, miss, notes)


def validate_lob_schema(df: pd.DataFrame) -> ValidationReport:
    """
    Inspect a LOB DataFrame and return a ValidationReport.

    Call immediately after load_lob_wind() / load_or_fetch_wind() to
    understand which signals can be computed before calling build_feature_matrix().
    """
    n_levels = _count_complete_lob_levels(df)

    nan_bid1 = int(df["bid_px_1"].isna().sum()) if "bid_px_1" in df.columns else len(df)

    report = ValidationReport(
        n_rows=len(df),
        n_cols=len(df.columns),
        index_is_datetime=isinstance(df.index, pd.DatetimeIndex),
        has_ticker_col="ticker" in df.columns,
        n_levels_complete=n_levels,
        n_rows_any_nan_bid1=nan_bid1,
    )

    for spec in _SIGNAL_SPECS:
        report.signals.append(_check_signal(df, spec))

    return report


def print_validation_report(report: ValidationReport, verbose: bool = True) -> None:
    """Pretty-print a ValidationReport to stdout."""
    SEP = "=" * 60

    print(f"\n{SEP}")
    print("  Wind LOB Schema Validation")
    print(SEP)
    print(f"  Rows        : {report.n_rows:,}")
    print(f"  Columns     : {report.n_cols}")
    print(f"  DatetimeIdx : {report.index_is_datetime}")
    print(f"  LOB levels  : {report.n_levels_complete} complete levels")
    if report.n_rows_any_nan_bid1:
        print(f"  NaN bid_px_1: {report.n_rows_any_nan_bid1} rows  ← check data quality")

    print(f"\n{'Signal':<22} {'Role':<12} {'Status'}")
    print("-" * 60)
    for s in report.signals:
        tick  = "✓" if s.available else "✗"
        extra = f"  ({s.n_levels_complete} levels)" if s.n_levels_available else ""
        print(f"  {s.name:<20} {s.role:<12} {tick}{extra}")
        if not s.available and verbose:
            for col in s.missing_columns[:6]:
                print(f"    missing: {col}")
            if len(s.missing_columns) > 6:
                print(f"    ... and {len(s.missing_columns) - 6} more")

    print(f"\n  Runnable alpha : {report.runnable_alpha or ['(none)']}")
    print(f"  Runnable gates : {report.runnable_gates or ['(none)']}")
    print(SEP)


def assert_minimum_viable(df: pd.DataFrame, min_alpha: int = 1) -> None:
    """
    Raise ValueError if fewer than `min_alpha` strict alpha signals can run.
    Use as a hard gate before kicking off a long research run.
    """
    report = validate_lob_schema(df)
    n = len(report.runnable_alpha)
    if n < min_alpha:
        msg = (
            f"Only {n} alpha signal(s) runnable on this DataFrame "
            f"(need ≥ {min_alpha}). "
            f"Blocked: {[s.name for s in report.blocked if s.role == 'alpha']}. "
            f"Run print_validation_report() for details."
        )
        raise ValueError(msg)
