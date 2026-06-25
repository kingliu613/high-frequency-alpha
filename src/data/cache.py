"""
Parquet cache for Wind L2 LOB data.

Avoids re-fetching from the Wind terminal on every research run.
Cache is keyed by (ticker, date) and stored under WIND_CACHE_DIR
(default: ./data/wind_cache/).

Usage:
    from src.data.cache import load_or_fetch_wind

    df = load_or_fetch_wind("IF2401.CFFEX", "2024-01-02")
    # First call: fetches from Wind and saves to Parquet.
    # Subsequent calls: loads from disk in ~milliseconds.

Environment:
    WIND_CACHE_DIR  — override default cache root (default: ./data/wind_cache)
"""

from __future__ import annotations

import os
import re
import pandas as pd
from pathlib import Path
from typing import Optional, Callable


def _default_cache_root() -> Path:
    env = os.environ.get("WIND_CACHE_DIR")
    if env:
        return Path(env)
    # Place next to the project root regardless of cwd
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "data" / "wind_cache"


def _cache_path(ticker: str, date: str, cache_root: Optional[Path] = None) -> Path:
    root = cache_root or _default_cache_root()
    safe_ticker = re.sub(r"[^\w\-]", "_", ticker)
    return root / f"{safe_ticker}_{date}.parquet"


def save_to_cache(
    df: pd.DataFrame,
    ticker: str,
    date: str,
    cache_root: Optional[Path] = None,
) -> Path:
    """Write a LOB DataFrame to a Parquet file; return the path."""
    path = _cache_path(ticker, date, cache_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    return path


def load_from_cache(
    ticker: str,
    date: str,
    cache_root: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Return cached DataFrame or None if no cache entry exists."""
    path = _cache_path(ticker, date, cache_root)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return None


def load_or_fetch_wind(
    ticker: str,
    date: str,
    include_night: bool = False,
    freq_sec: int = 3,
    cache_root: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return LOB DataFrame for (ticker, date), using cache when available.

    Parameters
    ----------
    ticker        : Wind format, e.g. "IF2401.CFFEX" or "000001.SZ"
    date          : "YYYY-MM-DD"
    include_night : pass through to load_lob_wind (commodity futures only)
    freq_sec      : snapshot cadence (Wind L2 native = 3 seconds)
    cache_root    : override default cache directory
    force_refresh : ignore existing cache entry and re-fetch from Wind
    """
    if not force_refresh:
        cached = load_from_cache(ticker, date, cache_root)
        if cached is not None:
            return cached

    from src.data.loader import load_lob_wind
    df = load_lob_wind(ticker, date, include_night=include_night, freq_sec=freq_sec)

    if not df.empty:
        save_to_cache(df, ticker, date, cache_root)

    return df


def list_cached(cache_root: Optional[Path] = None) -> list[dict]:
    """Return a list of {ticker, date, path, size_mb} for all cached files."""
    root = cache_root or _default_cache_root()
    if not root.exists():
        return []

    out = []
    for p in sorted(root.glob("*.parquet")):
        # Filename format: <ticker>_<date>.parquet  e.g. IF2401_CFFEX_2024-01-02.parquet
        name = p.stem
        # Split on last occurrence of _YYYY-MM-DD
        m = re.match(r"^(.+)_(\d{4}-\d{2}-\d{2})$", name)
        if m:
            safe_ticker, date = m.group(1), m.group(2)
            ticker = safe_ticker.replace("_", ".", 1)  # restore first dot
        else:
            ticker, date = name, "unknown"
        out.append({
            "ticker":   ticker,
            "date":     date,
            "path":     str(p),
            "size_mb":  round(p.stat().st_size / 1_048_576, 2),
        })
    return out


def clear_cache(
    ticker: Optional[str] = None,
    date: Optional[str] = None,
    cache_root: Optional[Path] = None,
) -> int:
    """
    Delete cache files matching the given ticker and/or date filter.
    Pass neither to wipe the entire cache. Returns number of files deleted.
    """
    root = cache_root or _default_cache_root()
    if not root.exists():
        return 0

    deleted = 0
    for entry in list_cached(cache_root):
        match = True
        if ticker and entry["ticker"] != ticker:
            match = False
        if date and entry["date"] != date:
            match = False
        if match:
            Path(entry["path"]).unlink(missing_ok=True)
            deleted += 1
    return deleted
