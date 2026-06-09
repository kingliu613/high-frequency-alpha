# Backtest Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add noise calibration, walk-forward optimization, signal-proportional sizing, dynamic hold, regime filter, and two new signals (price-limit, ETF basis) to the Chinese HFT backtest.

**Architecture:** Five existing files modified; no new source files. Tests live in `tests/`. New signals added to `features.py`, wired into `composite.py`. Engine gets three new behaviours via two new `MarketParams` fields. Walk-forward loop added to `run_alpha_research.py`.

**Tech Stack:** Python 3.11+, numpy, pandas, pytest

---

## File Map

| File | Change |
|---|---|
| `src/data/synthetic.py` | `signal_strength` 0.12→0.05; OU persistence on LOB vol; add `simulate_etf_series()` |
| `src/signals/features.py` | Add `price_limit_signal()`, `etf_basis_signal()` |
| `src/signals/composite.py` | Update `DEFAULT_WEIGHTS`; wire optional signals in `build_feature_matrix` |
| `src/backtest/engine.py` | New `MarketParams` fields; signal-prop sizing; dynamic hold; regime filter |
| `scripts/run_alpha_research.py` | `run_day()` accepts `params`/`use_etf`; add `run_walkforward()`; `--walkforward` flag |
| `tests/__init__.py` | (new, empty) |
| `tests/test_synthetic.py` | (new) |
| `tests/test_signals.py` | (new) |
| `tests/test_engine.py` | (new) |

---

## Task 1: Test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create test directory and conftest with shared fixture**

```python
# tests/__init__.py
# (empty)
```

```python
# tests/conftest.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd
from src.data.synthetic import simulate_lob_day

@pytest.fixture
def lob_day():
    return simulate_lob_day(seed=0, date="2024-01-02")
```

- [ ] **Step 2: Verify pytest can discover tests**

```bash
cd /Users/yichanliu/Documents/high-frequency-alpha
python3 -m pytest tests/ --collect-only
```

Expected: `no tests ran` (no tests yet) with 0 errors.

- [ ] **Step 3: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add pytest infrastructure and shared lob_day fixture"
```

---

## Task 2: Noise calibration

**Files:**
- Modify: `src/data/synthetic.py`
- Create: `tests/test_synthetic.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_synthetic.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import simulate_lob_day, simulate_etf_series


def test_signal_strength_default_is_005():
    import inspect
    sig = inspect.signature(simulate_lob_day)
    assert sig.parameters["signal_strength"].default == 0.05


def test_lob_vol_has_persistence():
    """Consecutive bid_vol_1 values should be correlated (OU process)."""
    df = simulate_lob_day(seed=42)
    v = df["bid_vol_1"].astype(float)
    lag1_corr = v.corr(v.shift(1))
    assert lag1_corr > 0.3, f"Expected lag-1 autocorr > 0.3, got {lag1_corr:.3f}"


def test_simulate_etf_series_returns_series():
    df = simulate_lob_day(seed=0)
    etf = simulate_etf_series(df, seed=0)
    assert isinstance(etf, pd.Series)
    assert len(etf) == len(df)
    assert etf.index.equals(df.index)


def test_simulate_etf_series_premium_is_small():
    """ETF should trade within ±3% of mid (AR(1) σ=10bps)."""
    df = simulate_lob_day(seed=0)
    mid = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
    etf = simulate_etf_series(df, seed=0)
    premium = (etf / mid - 1.0).abs()
    assert premium.max() < 0.03, f"Max premium {premium.max():.4f} too large"
```

- [ ] **Step 2: Run to confirm failures**

```bash
python3 -m pytest tests/test_synthetic.py -v
```

Expected: 3 failures (`simulate_etf_series` not defined, default still 0.12).

- [ ] **Step 3: Apply noise calibration changes to `synthetic.py`**

Change the function signature default:
```python
# src/data/synthetic.py  line ~50
def simulate_lob_day(
    ...
    signal_strength: float = 0.05,   # was 0.12
) -> pd.DataFrame:
```

Replace the LOB volume section inside the main loop (find the block that builds base_vol and calls rng.lognormal):

```python
        # OU mean-reversion for LOB base volume (replaces iid lognormal)
        # vol_state is declared before the loop; see initialisation below
        vol_target = float(rng.lognormal(7.5, 0.8)) * LOT
        vol_state  = 0.7 * vol_state + 0.3 * vol_target
        base_vol   = max(LOT, int(vol_state))
```

Before the main records loop (after the `prices` array is built), add:

```python
    vol_state = float(rng.lognormal(7.5, 0.8)) * LOT   # OU initial state
    records   = []
    cum_buy   = 0
    cum_sell  = 0
```

(Remove the old `base_vol = max(LOT, int(rng.lognormal(7.5, 0.8)) * LOT)` line that was inside the loop.)

Add `simulate_etf_series` function at the bottom of `synthetic.py`:

```python
def simulate_etf_series(lob_df: pd.DataFrame, seed: int = 0) -> pd.Series:
    """
    Simulate ETF price as NAV (LOB mid) plus AR(1) premium.

        premium[i] = 0.98 * premium[i-1] + N(0, σ_innov)
        σ_innov    = 0.001 * sqrt(1 - 0.98²)   ≈ 0.0002

    Models realistic ETF premium/discount to NAV (~10 bps persistence).
    """
    rng = np.random.default_rng(seed + 77777)
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]).astype(float) / 2.0
    n   = len(mid)

    phi       = 0.98
    sigma_inn = 0.001 * np.sqrt(1.0 - phi ** 2)
    premium   = np.zeros(n)
    premium[0] = rng.normal(0.0, 0.001)
    for i in range(1, n):
        premium[i] = phi * premium[i - 1] + rng.normal(0.0, sigma_inn)

    etf_px = mid.values * (1.0 + premium)
    return pd.Series(etf_px, index=lob_df.index, name="etf_px")
```

- [ ] **Step 4: Run tests — all pass**

```bash
python3 -m pytest tests/test_synthetic.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/data/synthetic.py tests/test_synthetic.py
git commit -m "feat: calibrate signal_strength to 0.05, add OU LOB volumes, simulate_etf_series"
```

---

## Task 3: price_limit_signal

**Files:**
- Modify: `src/signals/features.py`
- Create: `tests/test_signals.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_signals.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import simulate_lob_day, simulate_etf_series
from src.signals.features import price_limit_signal, etf_basis_signal


class TestPriceLimitSignal:

    def test_zero_far_from_limit(self):
        """Signal is 0 when mid is more than 3% from either limit."""
        df = simulate_lob_day(seed=0, prev_close=100.0)
        sig = price_limit_signal(df, prev_close=100.0)
        # mid will be near 100; 3% activation zone starts at 107.3 (up) or 92.7 (down)
        # synthetic data won't go that far with signal_strength=0.05
        assert (sig.abs() < 1e-6).mean() > 0.90, "Expected mostly zeros far from limit"

    def test_positive_near_up_limit(self):
        """Approaching up-limit produces positive signal."""
        # Construct a LOB snapshot where mid ≈ 109 (within 3% of 110 up-limit)
        mid_px = 109.5
        spread = 0.02
        row = {"bid_px_1": mid_px - spread/2, "ask_px_1": mid_px + spread/2}
        for lv in range(2, 11):
            row[f"bid_px_{lv}"] = row["bid_px_1"] - lv * 0.01
            row[f"ask_px_{lv}"] = row["ask_px_1"] + lv * 0.01
            row[f"bid_vol_{lv}"] = 100
            row[f"ask_vol_{lv}"] = 100
        row["bid_vol_1"] = 100
        row["ask_vol_1"] = 100
        row["cum_buy_vol"] = 0
        row["cum_sell_vol"] = 0
        df = pd.DataFrame([row], index=pd.DatetimeIndex(["2024-01-02 10:00:00"]))
        sig = price_limit_signal(df, prev_close=100.0, limit_pct=0.10)
        assert float(sig.iloc[0]) > 0.5, f"Expected > 0.5 near up-limit, got {float(sig.iloc[0]):.4f}"

    def test_negative_near_down_limit(self):
        """Approaching down-limit produces negative signal."""
        mid_px = 90.5
        spread = 0.02
        row = {"bid_px_1": mid_px - spread/2, "ask_px_1": mid_px + spread/2}
        for lv in range(2, 11):
            row[f"bid_px_{lv}"] = row["bid_px_1"] - lv * 0.01
            row[f"ask_px_{lv}"] = row["ask_px_1"] + lv * 0.01
            row[f"bid_vol_{lv}"] = 100
            row[f"ask_vol_{lv}"] = 100
        row["bid_vol_1"] = 100
        row["ask_vol_1"] = 100
        row["cum_buy_vol"] = 0
        row["cum_sell_vol"] = 0
        df = pd.DataFrame([row], index=pd.DatetimeIndex(["2024-01-02 10:00:00"]))
        sig = price_limit_signal(df, prev_close=100.0, limit_pct=0.10)
        assert float(sig.iloc[0]) < -0.5, f"Expected < -0.5 near down-limit, got {float(sig.iloc[0]):.4f}"

    def test_returns_series_same_index(self):
        df = simulate_lob_day(seed=0)
        sig = price_limit_signal(df, prev_close=4000.0)
        assert isinstance(sig, pd.Series)
        assert sig.index.equals(df.index)
        assert sig.name == "price_limit"
```

- [ ] **Step 2: Run to confirm failures**

```bash
python3 -m pytest tests/test_signals.py::TestPriceLimitSignal -v
```

Expected: ImportError (`price_limit_signal` not defined).

- [ ] **Step 3: Implement `price_limit_signal` in `features.py`**

Add at the bottom of `src/signals/features.py`:

```python
# ---------------------------------------------------------------------------
# Price-limit approach signal  (stock mode only)
# ---------------------------------------------------------------------------

def price_limit_signal(
    lob_df: pd.DataFrame,
    prev_close: float,
    limit_pct: float = 0.10,
    activation_pct: float = 0.03,
) -> pd.Series:
    """
    Exponential signal as mid-price approaches daily ±10% price limit.

    Positive → approaching up-limit (涨停 momentum).
    Negative → approaching down-limit (跌停 momentum).

    Only active within `activation_pct` of the limit; zero otherwise.
    Stock-mode only — futures have no fixed daily price limits.

    Reference: PMC4395215 (2015) price continuation probability 0.68 after
    up-limit hit.
    """
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]).astype(float) / 2.0

    up_limit   = prev_close * (1.0 + limit_pct)
    down_limit = prev_close * (1.0 - limit_pct)

    pct_to_up   = (up_limit   - mid) / up_limit
    pct_to_down = (mid - down_limit) / down_limit

    sig_up   = np.exp(-10.0 * pct_to_up.clip(lower=0.0))
    sig_down = -np.exp(-10.0 * pct_to_down.clip(lower=0.0))

    mask_up   = pct_to_up   < activation_pct
    mask_down = pct_to_down < activation_pct

    sig = sig_up.where(mask_up, 0.0) + sig_down.where(mask_down, 0.0)
    return sig.rename("price_limit")
```

- [ ] **Step 4: Run tests — all pass**

```bash
python3 -m pytest tests/test_signals.py::TestPriceLimitSignal -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/signals/features.py tests/test_signals.py
git commit -m "feat: add price_limit_signal for stock mode"
```

---

## Task 4: etf_basis_signal

**Files:**
- Modify: `src/signals/features.py`
- Modify: `tests/test_signals.py` (add class)

- [ ] **Step 1: Write failing tests — append to `tests/test_signals.py`**

Add this class to the existing `tests/test_signals.py`:

```python
class TestEtfBasisSignal:

    def test_returns_series_same_index(self):
        df  = simulate_lob_day(seed=0)
        etf = simulate_etf_series(df, seed=0)
        sig = etf_basis_signal(df, etf)
        assert isinstance(sig, pd.Series)
        assert sig.index.equals(df.index)
        assert sig.name == "etf_basis"

    def test_sign_is_mean_reverting(self):
        """Positive ETF premium → negative signal (sell expensive ETF)."""
        df  = simulate_lob_day(seed=0)
        mid = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
        # Force ETF price 2% above mid (expensive ETF)
        etf_expensive = mid * 1.02
        sig = etf_basis_signal(df, etf_expensive)
        # After burn-in (200 ticks), signal should be negative
        assert float(sig.iloc[250:].mean()) < 0, "Expensive ETF should give negative signal"

    def test_zero_signal_at_par(self):
        """ETF at exact NAV → basis = 0 → signal = 0."""
        df  = simulate_lob_day(seed=0)
        mid = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
        sig = etf_basis_signal(df, mid)   # ETF = mid exactly
        # rolling std of a zero series is 0, so fillna(0) → all zeros
        assert (sig.abs() < 1e-9).all()
```

- [ ] **Step 2: Run to confirm failures**

```bash
python3 -m pytest tests/test_signals.py::TestEtfBasisSignal -v
```

Expected: ImportError (`etf_basis_signal` not defined).

- [ ] **Step 3: Implement `etf_basis_signal` in `features.py`**

Add after `price_limit_signal` in `src/signals/features.py`:

```python
# ---------------------------------------------------------------------------
# ETF basis signal  (mean-reversion vs NAV)
# ---------------------------------------------------------------------------

def etf_basis_signal(lob_df: pd.DataFrame, etf_series: pd.Series) -> pd.Series:
    """
    ETF premium/discount to NAV (underlying mid-price) as mean-reversion signal.

        basis(t) = ETF_price(t) / mid(t) - 1

    Signal = -basis: if ETF trades expensively vs basket, expect reversion down.

    Reference: ALPHA_NOTES §2.5 — ETF NAV spread arbitrage.
    Requires T+0 instrument (ETF 510300 / 510500 is T+0 in China).
    """
    mid = (lob_df["bid_px_1"] + lob_df["ask_px_1"]).astype(float) / 2.0
    etf = etf_series.reindex(lob_df.index).ffill()

    basis = etf / mid.replace(0.0, np.nan) - 1.0
    sig   = -basis  # mean-reversion: expensive ETF → negative signal

    roll_std = sig.rolling(200, min_periods=50).std()
    return (sig / roll_std.replace(0.0, np.nan)).fillna(0.0).rename("etf_basis")
```

- [ ] **Step 4: Run all signal tests**

```bash
python3 -m pytest tests/test_signals.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/signals/features.py tests/test_signals.py
git commit -m "feat: add etf_basis_signal with mean-reversion logic"
```

---

## Task 5: Update composite weights and wire new signals

**Files:**
- Modify: `src/signals/composite.py`

- [ ] **Step 1: Replace `DEFAULT_WEIGHTS` and add weight constants**

In `src/signals/composite.py`, replace the `DEFAULT_WEIGHTS` block and `AUCTION_WEIGHT` line:

```python
DEFAULT_WEIGHTS = {
    "trade_imbalance": 0.24,
    "micro_price_dev": 0.18,
    "mlofi":           0.20,
    "queue_imbalance": 0.13,
    "agg_ofi":         0.14,
    "depth_tilt":      0.09,
    "mom_5":           0.02,
}
# Optional signal weights — drawn proportionally from DEFAULT_WEIGHTS when active
AUCTION_WEIGHT     = 0.10
PRICE_LIMIT_WEIGHT = 0.08   # stock mode only
ETF_BASIS_WEIGHT   = 0.07   # when etf_series provided
```

- [ ] **Step 2: Update `build_feature_matrix` signature and body**

Replace the entire `build_feature_matrix` function:

```python
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
```

- [ ] **Step 3: Update `build_composite_alpha` to handle all optional weights**

Replace the optional-weight block in `build_composite_alpha` (the block that currently handles `has_auction`):

```python
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
```

- [ ] **Step 4: Run full backtest to confirm no regressions**

```bash
python3 scripts/run_alpha_research.py 2>&1 | tail -30
```

Expected: single-day run completes without error, IC values present.

- [ ] **Step 5: Commit**

```bash
git add src/signals/composite.py
git commit -m "feat: update composite weights, wire price_limit and etf_basis signals"
```

---

## Task 6: Engine improvements

**Files:**
- Modify: `src/backtest/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_engine.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data.synthetic import simulate_lob_day
from src.backtest.engine import MarketParams, run_backtest
from src.signals.composite import build_feature_matrix, build_composite_alpha


@pytest.fixture
def base_lob():
    return simulate_lob_day(seed=7, date="2024-01-02")

@pytest.fixture
def base_signal(base_lob):
    feat = build_feature_matrix(base_lob)
    return build_composite_alpha(feat)


class TestMarketParams:
    def test_new_fields_have_defaults(self):
        p = MarketParams()
        assert p.max_position_size == 3
        assert p.use_regime_filter is True
        assert p.regime_ic_off == 0.0
        assert p.regime_ic_on  == 0.02


class TestSignalPropSizing:
    def test_size_one_at_entry_z(self, base_lob, base_signal):
        p = MarketParams(entry_z=1.5, use_regime_filter=False)
        _, trades = run_backtest(base_lob, base_signal, params=p)
        if len(trades):
            # Minimum size is 1
            assert trades["direction"].abs().min() >= 1

    def test_size_scales_with_signal(self, base_lob):
        """Force a constant strong signal and check sizes > 1 appear."""
        p = MarketParams(entry_z=1.0, max_position_size=3, use_regime_filter=False)
        mid = (base_lob["bid_px_1"] + base_lob["ask_px_1"]) / 2.0
        # Constant signal at 2.5 × entry_z → expected size = floor(2.5) = 2
        strong_sig = pd.Series(2.5, index=base_lob.index)
        _, trades = run_backtest(base_lob, strong_sig, params=p)
        if len(trades):
            # All entries should be size 2 (floor(2.5/1.0) = 2)
            assert (trades["direction"].abs() == 2).all()


class TestDynamicHold:
    def test_strong_signal_allows_longer_hold(self, base_lob):
        """Trades entered on a strong constant signal should hold > base_hold ticks."""
        p = MarketParams(entry_z=1.0, max_hold=10, use_regime_filter=False)
        strong_sig = pd.Series(3.5, index=base_lob.index)
        _, trades = run_backtest(base_lob, strong_sig, params=p)
        timeout_trades = trades[trades["exit_reason"] == "timeout"]
        if len(timeout_trades):
            # Dynamic hold at signal=3.5, entry_z=1.0: cap = 10*(1+0.5*3.5/1.0)=27.5→27
            assert timeout_trades["hold_ticks"].max() > 10


class TestRegimeFilter:
    def test_no_entries_when_regime_off(self, base_lob):
        """Regime filter should block all entries when IC is persistently negative."""
        p = MarketParams(
            entry_z=0.1,           # very low threshold — many entries normally
            use_regime_filter=True,
            regime_ic_off=1.0,     # IC threshold so high regime is always OFF
            regime_ic_on=2.0,
        )
        mid = (base_lob["bid_px_1"] + base_lob["ask_px_1"]) / 2.0
        # Signal above entry_z almost everywhere
        sig = pd.Series(0.5, index=base_lob.index)
        _, trades = run_backtest(base_lob, sig, params=p)
        # With regime_ic_off=1.0, IC can never exceed 1.0 in 200 ticks, so always blocked
        assert len(trades) == 0, f"Expected 0 trades, got {len(trades)}"
```

- [ ] **Step 2: Run to confirm failures**

```bash
python3 -m pytest tests/test_engine.py -v
```

Expected: `TestMarketParams::test_new_fields_have_defaults` fails (fields not added yet).

- [ ] **Step 3: Add new fields to `MarketParams`**

In `src/backtest/engine.py`, update `MarketParams`:

```python
@dataclass
class MarketParams:
    """Configurable per-instrument parameters."""

    instrument: str   = "futures"
    price_limit: float = 0.10

    commission:   float = 0.000023
    stamp_duty:   float = 0.0005

    entry_z:      float = 1.5
    exit_z:       float = 0.3
    max_hold:     int   = 20
    stop_loss_bp: float = 15.0

    position_size:     int   = 1      # kept for backward compat; sizing now signal-driven
    max_position_size: int   = 3
    use_regime_filter: bool  = True
    regime_ic_off:     float = 0.0
    regime_ic_on:      float = 0.02
```

- [ ] **Step 4: Add `entry_signal` field to `_Pos` dataclass**

```python
@dataclass
class _Pos:
    direction:    int   = 0
    entry_price:  float = 0.0
    entry_idx:    int   = 0
    size:         int   = 0
    entry_signal: float = 0.0
```

- [ ] **Step 5: Implement signal-proportional sizing, dynamic hold, and regime filter**

Replace the entire body of `run_backtest` (the loop) with the following. Keep the function signature and the `mid`/`bid`/`ask`/`sig` setup lines unchanged:

```python
    pnl    = pd.Series(0.0, index=lob_df.index)
    trades = []
    pos    = _Pos()

    # Regime filter state
    from collections import deque
    regime_buf     = deque(maxlen=200)   # (signal_t-1, realized_return_t-1→t)
    regime_ic      = 0.0
    in_regime_off  = False

    for i in range(1, len(mid)):
        s           = float(sig.iloc[i])
        m           = float(mid.iloc[i])
        ts          = mid.index[i]
        prev        = float(mid.iloc[i - 1])
        half_spread = (float(ask.iloc[i]) - float(bid.iloc[i])) / 2.0

        # --- Regime filter update (no look-ahead: uses realized return at i-1→i) ---
        if i >= 2:
            realized = m / prev - 1.0 if prev != 0.0 else 0.0
            regime_buf.append((float(sig.iloc[i - 1]), realized))

        if params.use_regime_filter and len(regime_buf) >= 50 and i % 20 == 0:
            sigs_r = [x[0] for x in regime_buf]
            rets_r = [x[1] for x in regime_buf]
            ra = np.argsort(np.argsort(sigs_r)).astype(float)
            rb = np.argsort(np.argsort(rets_r)).astype(float)
            denom = ra.std() * rb.std()
            regime_ic = float(np.corrcoef(ra, rb)[0, 1]) if denom > 0 else 0.0
            if in_regime_off and regime_ic > params.regime_ic_on:
                in_regime_off = False
            elif not in_regime_off and regime_ic < params.regime_ic_off:
                in_regime_off = True

        # --- Mark-to-market (mid-to-mid) ---
        if pos.direction != 0:
            pnl.iloc[i] = pos.direction * pos.size * (m - prev) * LOT

            hold_ticks = i - pos.entry_idx
            unreal_ret = pos.direction * (m - pos.entry_price) / pos.entry_price if pos.entry_price != 0 else 0.0

            # Dynamic hold cap: base × (1 + 0.5 × |entry_signal| / entry_z), max 40
            hold_cap = int(params.max_hold * (1.0 + 0.5 * pos.entry_signal / params.entry_z))
            hold_cap = min(hold_cap, 40)

            exit_on_flip  = (pos.direction > 0 and s < -params.exit_z) or \
                            (pos.direction < 0 and s >  params.exit_z)
            exit_on_hold  = hold_ticks >= hold_cap
            exit_on_stop  = unreal_ret < -(params.stop_loss_bp / 10_000)

            if exit_on_flip or exit_on_hold or exit_on_stop:
                spread_cost = half_spread * pos.size * LOT
                cost        = _txn_cost(m, pos.size, params, is_sell=(pos.direction > 0))
                pnl.iloc[i] -= (spread_cost + cost)

                reason = (
                    "flip"    if exit_on_flip  else
                    "stop"    if exit_on_stop  else
                    "timeout"
                )
                trades.append({
                    "entry_time":  mid.index[pos.entry_idx],
                    "exit_time":   ts,
                    "direction":   pos.direction,
                    "entry_price": pos.entry_price,
                    "exit_price":  m,
                    "hold_ticks":  hold_ticks,
                    "exit_reason": reason,
                    "gross_pnl":   pos.direction * (m - pos.entry_price) * pos.size * LOT,
                    "cost":        cost + spread_cost,
                })
                pos = _Pos()

        # --- Entry (skipped when in regime-off) ---
        if pos.direction == 0 and not in_regime_off:
            want_long  = s >  params.entry_z
            want_short = s < -params.entry_z and params.instrument == "futures"

            if want_long or want_short:
                direction   = 1 if want_long else -1
                # Signal-proportional size: floor(|s|/entry_z), clamped [1, max_position_size]
                size        = int(np.clip(np.floor(abs(s) / params.entry_z), 1, params.max_position_size))
                spread_cost = half_spread * size * LOT
                cost        = _txn_cost(m, size, params, is_sell=False)
                pnl.iloc[i] -= (spread_cost + cost)
                pos = _Pos(
                    direction    = direction,
                    entry_price  = m,
                    entry_idx    = i,
                    size         = size,
                    entry_signal = abs(s),
                )
```

- [ ] **Step 6: Run engine tests**

```bash
python3 -m pytest tests/test_engine.py -v
```

Expected: all PASSED.

- [ ] **Step 7: Run full suite to confirm no regressions**

```bash
python3 -m pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 8: Commit**

```bash
git add src/backtest/engine.py tests/test_engine.py
git commit -m "feat: signal-proportional sizing, dynamic hold, regime filter in backtest engine"
```

---

## Task 7: Update run_day to accept params and use_etf

**Files:**
- Modify: `scripts/run_alpha_research.py`

- [ ] **Step 1: Update imports at top of `run_alpha_research.py`**

Add to the existing imports:

```python
from src.data.synthetic  import simulate_lob_day, simulate_auction_data, simulate_etf_series
```

(Replace the line `from src.data.synthetic  import simulate_lob_day, simulate_auction_data`)

- [ ] **Step 2: Update `run_day` signature and body**

Replace the `run_day` function signature and its params block:

```python
def run_day(
    date: str,
    ticker: str = "IF2401.CFFEX",
    is_futures: bool = True,
    seed: int = 42,
    verbose: bool = True,
    params: Optional[MarketParams] = None,
    use_etf: bool = False,
) -> dict:
```

Replace the params and `build_feature_matrix` call inside `run_day`:

```python
    if params is None:
        params = MarketParams(
            instrument = "futures" if is_futures else "stock",
            entry_z    = 1.5,
            exit_z     = 0.3,
            max_hold   = 20,
        )

    etf_series = simulate_etf_series(lob_df, seed=seed) if use_etf else None

    feat_df = build_feature_matrix(
        lob_df,
        auction_value         = auc_val,
        ofi_levels            = 5,
        ofi_window            = 10,
        prev_close            = float(open_price) if not is_futures else None,
        instrument            = params.instrument,
        etf_series            = etf_series,
    )
```

Replace the `run_backtest` call:

```python
    pnl, trades = run_backtest(lob_df, composite, params=params)
```

- [ ] **Step 3: Verify single-day still works**

```bash
python3 scripts/run_alpha_research.py 2>&1 | grep "IC\|Sharpe\|Peak"
```

Expected: prints IC and Sharpe values without error.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_alpha_research.py
git commit -m "feat: run_day accepts params and use_etf arguments"
```

---

## Task 8: Walk-forward optimizer

**Files:**
- Modify: `scripts/run_alpha_research.py`

- [ ] **Step 1: Add `run_walkforward` function**

Add this function after `run_multiday` in `run_alpha_research.py`:

```python
def run_walkforward(
    n_days:     int  = 20,
    ticker:     str  = "IF2401.CFFEX",
    is_futures: bool = True,
    is_window:  int  = 10,
    oos_window: int  = 2,
) -> pd.DataFrame:
    """
    Walk-forward parameter optimization.

    Grid-searches entry_z × max_hold on IS window, applies best params
    to next OOS window, rolls forward. Reports OOS-only metrics.
    """
    entry_z_grid  = [0.8, 1.0, 1.2, 1.5, 2.0]
    max_hold_grid = [10, 15, 20, 30, 40]

    dates    = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d").tolist()
    date_idx = {d: i for i, d in enumerate(dates)}
    oos_rows = []

    print(f"\n  Grid: entry_z×{entry_z_grid}  max_hold×{max_hold_grid}  "
          f"({len(entry_z_grid)*len(max_hold_grid)} combos per IS window)")

    for oos_start in range(is_window, n_days, oos_window):
        is_dates  = dates[oos_start - is_window : oos_start]
        oos_dates = dates[oos_start : oos_start + oos_window]
        if not oos_dates:
            break

        # ── IS grid search ──────────────────────────────────────────────
        best_sharpe = -np.inf
        best_ez, best_mh = 1.5, 20

        for ez in entry_z_grid:
            for mh in max_hold_grid:
                p = MarketParams(
                    instrument        = "futures" if is_futures else "stock",
                    entry_z           = ez,
                    exit_z            = 0.3,
                    max_hold          = mh,
                    use_regime_filter = False,   # skip for speed during grid search
                )
                daily_pnls = []
                for d in is_dates:
                    r = run_day(d, ticker=ticker, is_futures=is_futures,
                                seed=date_idx[d], verbose=False, params=p)
                    daily_pnls.append(float(r["pnl"].sum()))
                s = pd.Series(daily_pnls)
                sh = float(np.sqrt(252) * s.mean() / s.std()) if s.std() > 0 else 0.0
                if sh > best_sharpe:
                    best_sharpe = sh
                    best_ez, best_mh = ez, mh

        # ── OOS evaluation ───────────────────────────────────────────────
        p_oos = MarketParams(
            instrument = "futures" if is_futures else "stock",
            entry_z    = best_ez,
            exit_z     = 0.3,
            max_hold   = best_mh,
        )
        oos_pnls, oos_ics = [], []
        for d in oos_dates:
            r = run_day(d, ticker=ticker, is_futures=is_futures,
                        seed=date_idx[d], verbose=False, params=p_oos)
            oos_pnls.append(float(r["pnl"].sum()))
            oos_ics.append(float(r["ic_all"].get(10, np.nan)))

        row = {
            "oos_start":   oos_dates[0],
            "oos_end":     oos_dates[-1],
            "entry_z":     best_ez,
            "max_hold":    best_mh,
            "is_sharpe":   round(best_sharpe, 2),
            "oos_pnl":     round(sum(oos_pnls), 1),
            "oos_ic_mean": round(float(np.nanmean(oos_ics)), 4),
        }
        oos_rows.append(row)
        print(f"  OOS {row['oos_start']}–{row['oos_end']}  "
              f"params=(ez={best_ez}, mh={best_mh})  "
              f"IS_sh={best_sharpe:+.2f}  OOS_pnl={row['oos_pnl']:+.0f}  "
              f"IC@10={row['oos_ic_mean']:+.4f}")

    df = pd.DataFrame(oos_rows)

    hdr("Walk-Forward Summary  (IS=10d  OOS=2d)")
    print(df.to_string(index=False))

    all_pnl    = pd.Series([r["oos_pnl"] for r in oos_rows])
    oos_sharpe = float(np.sqrt(252) * all_pnl.mean() / all_pnl.std()) if all_pnl.std() > 0 else 0.0
    n_prof     = int((all_pnl > 0).sum())
    mean_ic    = float(np.nanmean([r["oos_ic_mean"] for r in oos_rows]))

    print(f"\n  OOS-concatenated Sharpe : {oos_sharpe:+.2f}")
    print(f"  Profitable OOS windows  : {n_prof}/{len(oos_rows)}")
    print(f"  Mean OOS IC@10          : {mean_ic:+.4f}")

    return df
```

- [ ] **Step 2: Add `--walkforward` flag to `main()`**

In the `main()` function, add the argument and call:

```python
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--etf",         action="store_true", help="include ETF basis signal")
```

And in the body of `main()`, after the existing `--multiday` block:

```python
    if args.walkforward:
        hdr("Walk-Forward Optimization")
        run_walkforward(n_days=args.days, ticker=ticker, is_futures=is_futures)
```

Also update the `run_day` call in `main()` to pass `use_etf`:

```python
    hdr("Single Day Deep-Dive")
    run_day(date=args.date, ticker=ticker, is_futures=is_futures,
            verbose=True, use_etf=args.etf)
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_alpha_research.py
git commit -m "feat: add walk-forward optimizer with IS grid search and OOS reporting"
```

---

## Task 9: Integration run

- [ ] **Step 1: Run full test suite**

```bash
python3 -m pytest tests/ -v
```

Expected: all tests PASSED.

- [ ] **Step 2: Single-day deep-dive with ETF signal**

```bash
python3 scripts/run_alpha_research.py --etf 2>&1 | grep -E "IC|Sharpe|Peak|etf_basis|price_limit"
```

Expected: output shows `etf_basis` IC entry; no errors.

- [ ] **Step 3: Full walk-forward run**

```bash
python3 scripts/run_alpha_research.py --walkforward --days 20 2>&1
```

Expected: OOS-concatenated Sharpe printed; 5 OOS windows shown; mean IC@10 in 0.02–0.12 range.

- [ ] **Step 4: Multi-day stability check**

```bash
python3 scripts/run_alpha_research.py --multiday --days 20 2>&1 | tail -20
```

Expected: mean IC@10 in 0.02–0.10 range (down from 0.22 due to signal_strength=0.05).

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete backtest improvements — noise calibration, walk-forward, new signals, engine v2"
```
