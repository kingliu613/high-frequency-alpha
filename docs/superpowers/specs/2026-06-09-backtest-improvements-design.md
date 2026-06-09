# Backtest Improvements — Design Spec
**Date:** 2026-06-09
**Scope:** Noise calibration · Walk-forward optimization · Strategy logic · New signals

---

## 1. Goals

| Goal | Success Criteria |
|---|---|
| Realistic simulation | IC@30s in 0.03–0.08 range (vs current 0.22); Sharpe 1.5–3.5 |
| Walk-forward OOS validation | Positive OOS Sharpe across all 2-day OOS windows |
| Better strategy logic | Signal-prop sizing, dynamic hold, regime filter reduce drawdown |
| New alpha signals | price_limit and etf_basis signals show IC > 0.02 independently |

---

## 2. Architecture

Changes confined to five existing files. No new files.

```
src/data/synthetic.py          noise calibration + OU LOB volumes
src/signals/features.py        price_limit_signal(), etf_basis_signal()
src/signals/composite.py       wire new signals, update DEFAULT_WEIGHTS
src/backtest/engine.py         signal-prop sizing, dynamic hold, regime filter
scripts/run_alpha_research.py  walk-forward loop + OOS reporting
```

---

## 3. Noise Calibration (`synthetic.py`)

**Change:** `signal_strength` default 0.12 → 0.05.

Rationale: current IC@30s = 0.22 is 3–7× above real-market range.
`signal_strength=0.05` targets IC ~0.04–0.08 at h=10 ticks (30s).

**LOB volume persistence:** Replace independent lognormal base volume per snapshot with
OU mean-reversion:

```python
vol_base[i] = 0.7 * vol_base[i-1] + 0.3 * lognormal_target + noise
```

This makes consecutive snapshots correlated (realistic cancel/replace dynamics)
without changing the signal embedding mechanism.

---

## 4. Walk-Forward Optimization (`run_alpha_research.py`)

**Window structure:**
- In-sample (IS): 10 trading days
- Out-of-sample (OOS): 2 trading days
- Roll: 2 days (non-overlapping OOS)
- Total: 20 days → 5 OOS windows

**Parameter grid:**
```python
entry_z   = [0.8, 1.0, 1.2, 1.5, 2.0]
max_hold  = [10, 15, 20, 30, 40]        # ticks (×3s)
# 25 combinations per IS window
```

**Selection metric:** IS Sharpe (annualized via daily PnL std).

**OOS reporting per window:** Sharpe, mean IC@10, total PnL, best IS params used.

**Final summary:** OOS-concatenated Sharpe, fraction of profitable OOS days,
IC stability (mean ± std across windows).

---

## 5. Strategy Logic (`engine.py`)

### 5.1 Signal-Proportional Sizing

```python
size = int(clip(floor(abs(signal) / entry_z), 1, max_size))
# max_size default = 3 (configurable via MarketParams)
```

At `entry_z=1.5`: signal=1.5→size=1, signal=3.0→size=2, signal=4.5→size=3.

### 5.2 Dynamic Max Hold

```python
hold_cap = int(base_hold * (1.0 + 0.5 * abs(signal) / entry_z))
hold_cap = min(hold_cap, 40)  # hard cap at 40 ticks = 120s
```

Strong signals get more time to play out before timeout exit.

### 5.3 Regime Filter

Rolling Spearman IC computed over last 200 ticks vs `fwd_1` (3s return).
- Skip new entries when `rolling_ic < 0.0`
- Resume entries when `rolling_ic > 0.02`
- Hysteresis prevents rapid toggling

`MarketParams` gets new fields:
```python
max_position_size: int   = 3
use_regime_filter: bool  = True
regime_ic_off:     float = 0.0
regime_ic_on:      float = 0.02
```

---

## 6. New Signals (`features.py`)

### 6.1 `price_limit_signal` (stock mode only)

**Mechanism:** As stock mid approaches ±10% daily limit, trapped liquidity
creates momentum. Signal decays exponentially with distance from limit.

```python
def price_limit_signal(lob_df, prev_close, limit_pct=0.10):
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]) / 2.0
    up_limit   = prev_close * (1 + limit_pct)
    down_limit = prev_close * (1 - limit_pct)

    pct_to_up   = (up_limit - mid) / up_limit
    pct_to_down = (mid - down_limit) / down_limit

    sig_up   =  np.exp(-10 * pct_to_up.clip(lower=0))
    sig_down = -np.exp(-10 * pct_to_down.clip(lower=0))

    # Only active within 3% of limit; zero otherwise
    sig = sig_up.where(pct_to_up < 0.03, 0.0) + sig_down.where(pct_to_down < 0.03, 0.0)
    return sig.rename("price_limit")
```

**Weight:** 0.08 (stock mode only; futures mode: weight=0.0).

### 6.2 `etf_basis_signal`

**Mechanism:** ETF price deviates from NAV via AR(1) process. Signal = −basis
(mean-reversion). ETF is T+0 so position can be held and exited same day.

**Synthetic ETF NAV** (added to `simulate_lob_day` or passed externally):
```python
nav_px[i]  = mid_px[i]  # basket = underlying mid
etf_px[i]  = nav_px[i] * (1 + premium[i])
premium[i] = 0.98 * premium[i-1] + N(0, 0.001)   # AR(1), σ=10bps
```

**Signal:**
```python
def etf_basis_signal(lob_df, etf_series):
    mid  = (lob_df["bid_px_1"] + lob_df["ask_px_1"]) / 2.0
    basis = etf_series / mid - 1.0          # ETF premium to NAV
    # mean-reversion: expensive ETF → short ETF (negative signal)
    sig = -basis
    roll_std = sig.rolling(200, min_periods=50).std()
    return (sig / roll_std.replace(0, np.nan)).fillna(0.0).rename("etf_basis")
```

**Weight:** 0.07. Activated only when `etf_series` passed to `build_feature_matrix`.

---

## 7. Updated Default Weights

Weights must sum to 1.0. ETF basis and price-limit weights are conditional
(activated only when the respective signal is present).

**Futures mode (no price_limit, no etf_basis):**
```python
trade_imbalance: 0.24
micro_price_dev: 0.18
mlofi:           0.20
queue_imbalance: 0.13
agg_ofi:         0.14
depth_tilt:      0.09
mom_5:           0.02
# total = 1.00  (mom_20 dropped — negligible IC at HFT timescales)
```

**Stock mode (price_limit active):**
Same as futures but price_limit: 0.08 drawn proportionally from others.

**ETF mode (etf_basis active):**
etf_basis: 0.07 drawn proportionally from others when present.

The existing `AUCTION_WEIGHT = 0.10` redistribution logic handles this cleanly —
same pattern applies for new optional signals.

---

## 8. Walk-Forward Results Schema

```
┌─────────────────────────────────────────────────────────────┐
│ Walk-Forward Summary                                        │
│ IS=10d OOS=2d  Grid: entry_z×max_hold (25 combos)          │
├──────────┬──────────┬──────────────┬────────────┬──────────┤
│ OOS Dates│ IS Params│ OOS IC@10    │ OOS PnL    │ Sharpe   │
│          │ ez / mh  │ mean ± std   │            │          │
├──────────┼──────────┼──────────────┼────────────┼──────────┤
│ d11–d12  │ 1.2 / 20 │ +0.0XX±0.0XX│ +XXXXX RMB │ X.XX     │
│ d13–d14  │ ...      │ ...          │ ...        │ ...      │
│ ...      │          │              │            │          │
├──────────┴──────────┴──────────────┴────────────┴──────────┤
│ OOS-concatenated Sharpe: X.XX                               │
│ Profitable OOS days: XX/10                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Out-of-Scope

- Real data integration (Tushare/Wind) — separate milestone
- Live order management (CTP API) — separate milestone
- Multi-asset portfolio (cross-stock signals) — future work
- ML-based signal combination (adaptive IC weighting) — Approach C, deferred
