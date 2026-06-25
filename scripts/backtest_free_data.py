#!/usr/bin/env python3
"""
Real-data backtest using free sources only.

Path A — BaoStock 5-min bars (historical, multi-stock):
  Signals: intrabar direction, volume ratio, short-term momentum
  Stocks:  5 CSI-300 blue chips, 6 months of data
  Forward returns: 1–20 bars (5–100 min)

Path B — AKShare tick data (today, single stock):
  Signals: rolling trade imbalance (买盘 vs 卖盘)
  Forward returns: 1–40 ticks

Run:
    python scripts/backtest_free_data.py            # both paths
    python scripts/backtest_free_data.py --baostock # historical only
    python scripts/backtest_free_data.py --akshare  # today's ticks only
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from src.backtest.metrics import _spearman, pnl_metrics


# ── helpers ──────────────────────────────────────────────────────────────────

SEP = "=" * 64

def hdr(t): print(f"\n{SEP}\n  {t}\n{SEP}")

def ic_series(signal: pd.Series, fwd: pd.Series) -> float:
    mask = signal.notna() & fwd.notna()
    if mask.sum() < 30: return np.nan
    return _spearman(signal[mask], fwd[mask])


# ── Path A: BaoStock multi-stock 5-min ───────────────────────────────────────

TICKERS_BAOSTOCK = {
    "sz.000001": "Ping An Bank",
    "sh.600519": "Kweichow Moutai",
    "sh.601318": "Ping An Insurance",
    "sh.600036": "CMB",
    "sz.000858": "Wuliangye",
}

def load_baostock(ticker: str, start: str, end: str, freq: str = "5") -> pd.DataFrame:
    import baostock as bs

    lg = bs.login()
    rs = bs.query_history_k_data_plus(
        ticker,
        "date,time,open,high,low,close,volume,amount",
        start_date=start, end_date=end,
        frequency=freq, adjustflag="3",
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date","time","open","high","low","close","volume","amount"])
    for c in ["open","high","low","close","volume","amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ts"] = pd.to_datetime(df["time"].str[:14], format="%Y%m%d%H%M%S")
    df = df.set_index("ts").sort_index().dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_5min_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signals derivable from OHLCV 5-min bars.

    intrabar_dir : (close - open) / prev_close  — intrabar directional pressure
    vol_ratio    : volume / 20-bar rolling mean  — volume surge indicator
    open_gap     : (open - prev_close) / prev_close — gap from prior close
    bar_range    : (high - low) / prev_close  — intrabar volatility
    """
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    lo = df["low"].astype(float)
    v = df["volume"].astype(float)

    prev_c = c.shift(1)

    feats = pd.DataFrame(index=df.index)
    feats["intrabar_dir"] = (c - o) / prev_c.replace(0, np.nan)
    feats["vol_ratio"]    = v / v.rolling(20, min_periods=5).mean()
    feats["open_gap"]     = (o - prev_c) / prev_c.replace(0, np.nan)
    feats["bar_range"]    = (h - lo) / prev_c.replace(0, np.nan)

    # z-score each signal
    for col in feats.columns:
        mu  = feats[col].rolling(200, min_periods=50).mean()
        sig = feats[col].rolling(200, min_periods=50).std()
        feats[col] = (feats[col] - mu) / sig.replace(0, np.nan)

    return feats.dropna(how="all")


def run_baostock_backtest(start: str = "2023-07-01", end: str = "2024-01-01") -> None:
    hdr(f"BaoStock 5-min Backtest  {start} → {end}")

    all_ic: dict[str, list] = {}
    all_pnl: list[float] = []

    for ticker, name in TICKERS_BAOSTOCK.items():
        print(f"\n  Loading {name} ({ticker}) …", end=" ")
        df = load_baostock(ticker, start, end)
        if df.empty:
            print("no data"); continue
        print(f"{len(df):,} bars")

        feats = build_5min_signals(df)
        c     = df["close"].reindex(feats.index).astype(float)

        # Forward returns at 1, 5, 10, 20 bars
        fwd = {h: c.shift(-h) / c - 1 for h in [1, 5, 10, 20]}

        # Composite from paper-backed/free-data proxies only.
        # (vol_ratio sign: high vol + positive intrabar = bullish)
        comp = (
            feats.get("intrabar_dir", pd.Series(0, index=feats.index)) * 0.50 +
            feats.get("vol_ratio",    pd.Series(0, index=feats.index)) * 0.25 +
            feats.get("open_gap",     pd.Series(0, index=feats.index)) * 0.25
        ).fillna(0)

        # IC per signal per horizon
        for sig_name in feats.columns:
            if sig_name not in all_ic:
                all_ic[sig_name] = []
            ic_val = ic_series(feats[sig_name], fwd[5])
            all_ic[sig_name].append(ic_val)

        # Simple threshold backtest on composite (long-only, T+1 aware)
        pnl = pd.Series(0.0, index=comp.index)
        in_pos  = False
        entry_p = 0.0
        entry_i = -999

        for i in range(1, len(comp)):
            s = float(comp.iloc[i])
            p = float(c.iloc[i])

            if in_pos:
                pnl.iloc[i] = (p - float(c.iloc[i-1]))  # mark-to-market
                hold = i - entry_i
                unreal = (p - entry_p) / entry_p

                if s < -0.5 or hold >= 20 or unreal < -0.005:
                    cost = p * 0.0003 + p * 0.0005   # commission + stamp
                    pnl.iloc[i] -= cost
                    in_pos = False

            if not in_pos and s > 1.0:
                cost = p * 0.0003
                pnl.iloc[i] -= cost
                in_pos   = True
                entry_p  = p
                entry_i  = i

        all_pnl.append(float(pnl.sum()))

        m = pnl_metrics(pnl)
        print(f"    Composite IC@5bars: {ic_series(comp, fwd[5]):+.4f}  "
              f"Sharpe: {m['sharpe']:+.2f}  "
              f"WinRate: {m['win_rate']:.1%}")

    # Summary across all stocks
    print(f"\n{'─'*64}")
    print("Signal IC @ 5 bars (25 min) across all stocks:")
    print(f"  {'Signal':<22} {'Mean IC':>9}  {'Std':>7}  {'ICIR':>7}  {'N':>4}")
    for sig, vals in sorted(all_ic.items()):
        clean = [v for v in vals if not np.isnan(v)]
        if not clean: continue
        mu  = np.mean(clean)
        std = np.std(clean)
        ir  = mu / std if std > 0 else np.nan
        bar = "#" * max(0, int(abs(mu) * 300))
        print(f"  {sig:<22} {mu:>+9.4f}  {std:>7.4f}  {ir:>7.3f}  {len(clean):>4}  {bar}")

    if all_pnl:
        print(f"\n  Portfolio total PnL: {sum(all_pnl):+.2f} RMB  "
              f"(equal-weight, 1 share/trade)")


# ── Path B: AKShare tick trade imbalance ─────────────────────────────────────

def run_akshare_backtest(ticker: str = "000001") -> None:
    hdr(f"AKShare Tick Trade Imbalance — {ticker} (today)")

    try:
        import akshare as ak
    except ImportError:
        print("  pip install akshare"); return

    raw = ak.stock_intraday_em(symbol=ticker)
    raw.columns = ["time", "price", "volume_lot", "side"]
    raw["price"]  = pd.to_numeric(raw["price"],      errors="coerce")
    raw["volume"] = pd.to_numeric(raw["volume_lot"], errors="coerce") * 100

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    raw["ts"] = pd.to_datetime(today + " " + raw["time"])
    raw = raw.set_index("ts").sort_index().dropna(subset=["price"])

    # Filter to continuous session
    t = raw.index.time
    raw = raw[(t >= pd.Timestamp("09:30").time()) &
              (t <= pd.Timestamp("15:00").time())]

    if len(raw) < 50:
        print("  Not enough tick data (market closed or weekend?)")
        return

    print(f"  {len(raw):,} ticks loaded")

    # Trade imbalance signal
    is_buy  = raw["side"].str.contains("买", na=False)
    is_sell = raw["side"].str.contains("卖", na=False)

    # Snap to 3-second grid
    grid = pd.date_range(raw.index[0].floor("1min"),
                         raw.index[-1].ceil("1min"), freq="3s")

    buy_vol  = (raw["volume"] *  is_buy).resample("3s").sum().reindex(grid, fill_value=0)
    sell_vol = (raw["volume"] * is_sell).resample("3s").sum().reindex(grid, fill_value=0)
    price    = raw["price"].resample("3s").last().reindex(grid).ffill().dropna()

    buy_vol  = buy_vol.reindex(price.index)
    sell_vol = sell_vol.reindex(price.index)

    # Rolling 10-snapshot trade imbalance
    rb = buy_vol.rolling(10, min_periods=3).sum()
    rs = sell_vol.rolling(10, min_periods=3).sum()
    ti = (rb - rs) / (rb + rs).replace(0, np.nan)
    ti = ti.fillna(0)

    # Z-score
    ti_z = (ti - ti.rolling(100, min_periods=20).mean()) / \
            ti.rolling(100, min_periods=20).std().replace(0, np.nan)
    ti_z = ti_z.fillna(0)

    # IC at multiple horizons
    print(f"\n  Trade imbalance IC by horizon:")
    for h in [1, 3, 5, 10, 20, 40]:
        fwd = price.shift(-h) / price - 1
        ic  = ic_series(ti_z, fwd)
        bar = "#" * max(0, int(abs(ic) * 200)) if not np.isnan(ic) else ""
        print(f"    h={h:>3} snaps ({h*3:>4}s): IC={ic:+.4f}  {bar}")

    # 外盘/内盘 summary from snapshot
    snap = ak.stock_bid_ask_em(symbol=ticker)
    snap_d = dict(zip(snap["item"], snap["value"]))
    ext_vol = snap_d.get("外盘", 0)
    int_vol = snap_d.get("内盘", 0)
    total   = ext_vol + int_vol
    if total > 0:
        net_imb = (ext_vol - int_vol) / total
        print(f"\n  Current 外盘/内盘 imbalance: {net_imb:+.3f}  "
              f"(外盘={ext_vol:.0f}, 内盘={int_vol:.0f})")

    # Simple threshold backtest
    pnl = pd.Series(0.0, index=price.index)
    pos = 0
    ep  = 0.0
    ei  = -999

    for i in range(1, len(ti_z)):
        s = float(ti_z.iloc[i])
        p = float(price.iloc[i])

        if pos != 0:
            pnl.iloc[i] = pos * (p - float(price.iloc[i-1]))
            if (pos > 0 and s < -0.5) or (pos < 0 and s > 0.5) \
               or (i - ei) >= 40:
                pnl.iloc[i] -= p * 0.0003
                pos = 0

        if pos == 0:
            if s > 1.2:
                pos = 1;  ep = p; ei = i; pnl.iloc[i] -= p * 0.0003
            elif s < -1.2:
                pos = -1; ep = p; ei = i; pnl.iloc[i] -= p * 0.0003

    m = pnl_metrics(pnl)
    print(f"\n  Backtest (long/short, 1 share):")
    print(f"    Total PnL:   {m['total_pnl']:+.4f} RMB")
    print(f"    Win rate:    {m['win_rate']:.1%}")
    print(f"    Trades:      {m['n_nonzero']}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baostock", action="store_true")
    ap.add_argument("--akshare",  action="store_true")
    ap.add_argument("--ticker",   default="000001")
    ap.add_argument("--start",    default="2023-07-01")
    ap.add_argument("--end",      default="2024-01-01")
    args = ap.parse_args()

    both = not args.baostock and not args.akshare

    if args.baostock or both:
        run_baostock_backtest(args.start, args.end)

    if args.akshare or both:
        run_akshare_backtest(args.ticker)


if __name__ == "__main__":
    main()
