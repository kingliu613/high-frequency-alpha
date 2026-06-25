"""
Tushare Pro data loader for A-share cross-sectional factors.

Setup:
    pip install tushare
    export TUSHARE_TOKEN=your_token_here
    (or set ts.set_token() in your env)

CRSP → Tushare column mapping (monthly):
    permno      → ts_code
    date        → trade_date
    ret         → pct_chg / 100
    prc         → close
    shrout      → float_share
    vol         → vol (手, multiply by 100 for shares)
    me          → total_mv (市值, in 万元)

Compustat → CSMAR mapping (annual, quarterly):
    See research/cross_sectional/csmar_mapping.md (to be created)
"""

import os
import pandas as pd


def get_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "")
    if not token:
        raise EnvironmentError("Set TUSHARE_TOKEN environment variable")
    return token


def load_monthly_returns(
    ts_codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Returns monthly return panel.
    Columns: ts_code, trade_date, ret, close, float_share, total_mv, vol
    """
    try:
        import tushare as ts
        ts.set_token(get_token())
        pro = ts.pro_api()
    except ImportError:
        raise ImportError("pip install tushare")

    frames = []
    for code in ts_codes:
        df = pro.monthly(
            ts_code=code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,close,vol,amount,pct_chg",
        )
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result["ret"] = result["pct_chg"] / 100
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result.drop(columns=["pct_chg"])


def load_daily_returns(
    ts_codes: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Returns daily return panel.
    Columns: ts_code, trade_date, ret, close, vol, amount, turnover_rate
    """
    try:
        import tushare as ts
        ts.set_token(get_token())
        pro = ts.pro_api()
    except ImportError:
        raise ImportError("pip install tushare")

    frames = []
    for code in ts_codes:
        df = pro.daily(
            ts_code=code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,close,vol,amount,pct_chg,turnover_rate",
        )
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result["ret"] = result["pct_chg"] / 100
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result.drop(columns=["pct_chg"])


def load_index_components(index_code: str = "000300.SH") -> list[str]:
    """Returns current CSI 300 constituent ts_codes."""
    try:
        import tushare as ts
        ts.set_token(get_token())
        pro = ts.pro_api()
    except ImportError:
        raise ImportError("pip install tushare")

    df = pro.index_weight(index_code=index_code)
    return df["con_code"].tolist()
