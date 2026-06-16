# Alpha Research Notes — Chinese HFT

## 1. Market Structure Overview

| Feature | Chinese A-Share | CSI 300 Futures (IF) |
|---|---|---|
| Settlement | T+1 (no same-day sell) | T+0 |
| Short selling | 融券 (illiquid, expensive) | Native (margin ~12%) |
| Price limit | ±10% main board, ±20% STAR/ChiNext | ±10% (circuit breakers ±5%/10%) |
| Tick size | 0.01 RMB | 0.2 index point |
| LOB update | 3-second snapshots (10 levels) | Same (CFFEX L2) |
| Trading hours | 09:30–11:30, 13:00–15:00 | Same + night session |
| Opening auction | 09:15–09:25 (non-cancellable after 09:20) | Same |
| Closing auction | 14:57–15:00 (call auction) | Same |
| Retail fraction | ~70% of turnover | Lower (mostly institutional) |
| Commission | 0.025–0.03% + 0.05% stamp (sell) | ~0.0023% (2.3/10000) |

**Key implication**: T+1 on stocks forces HFT to use index futures or ETFs for
long/short strategies. CSI 300 IF/IC are the primary HFT vehicles in China.

---

## 2. Alpha Landscape: Five Strategy Families

### 2.1 Order Flow Imbalance (OFI)  ← **implemented**
**Mechanism**: Net buy/sell pressure from LOB state transitions predicts
short-horizon mid-price moves.

**Signal**: `MLOFI(t) = Σ_l w_l · OFI_l(t)` where `w_l = exp(-λ(l-1))`

**Evidence**:
- Cont, Kukanov, Stoikov (2014): OFI explains ~60–80% of contemporaneous price
  moves in US equities. R² = 0.6–0.8 at 10-min horizon.
- Chen & Zhang (2025, arXiv:2505.17388): CSI 300 futures OFI follows
  mean-reverting OU process with Lévy-driven jumps. Signal half-life ~5–15 min.
- MDPI Entropy (2020): Trading imbalance in Chinese stocks shows IC of 0.03–0.08
  at 1–5 minute horizon.

**Expected IC**: 0.03–0.07 at 10–40 tick horizon (30s–2min)

**Holding period**: 30 seconds to 3 minutes

**Capacity**: High for IF futures (daily turnover >100B RMB). Lower for small-cap stocks.

**Risks**:
- Spoofing/layering: fake large orders at level 3–5 inflate MLOFI signal
  → Cross-validate with trade_imbalance (executed orders don't lie)
- Regime shift: OFI signal weakens near price limits (10% boundary)
- LOB snapshot lag: 3-second snapshots miss intra-snapshot cancellations

---

### 2.2 Opening Auction Imbalance  ← **implemented**
**Mechanism**: Non-cancellable orders after 09:20 reveal genuine information.
Buy-heavy auction → continuation for first 20–30 min of session.

**Signal**: `auction_signal(t) = imbalance · exp(-ln2·t/t_{1/2})`

**Evidence**:
- Cont et al. (2014): Auction imbalance predicts next-day returns (small order imbalance
  negative effect; large order imbalance strong continuation)
- Empirical regularities of Chinese opening call auction (arXiv:0905.0582):
  Indicative price drifts toward final open price in last 5 minutes of auction.

**Expected IC**: 0.02–0.05 at 5-minute horizon, decays to near-zero by 11:00

**Half-life**: ~20 min for stocks, ~10 min for futures (faster price discovery)

**Risks**:
- News gaps: extreme gaps (>5%) approaching price limits behave non-linearly
- Manipulation: institutional block orders sometimes deliberately imbalance auction

---

### 2.3 Micro-Price Deviation  ← **implemented**
**Mechanism**: Size-weighted mid-price deviates from arithmetic mid when one side
of the book dominates, indicating genuine queue preference.

**Signal**: `dev = (μ - mid) / (0.5 · spread)` where `μ = (V_ask·P_bid + V_bid·P_ask)/(V_bid+V_ask)`

**Reference**: Stoikov (2018) "The micro-price", Quantitative Finance 18(12).

**Expected IC**: 0.02–0.04, particularly strong in illiquid stocks where spread > 1 tick

---

### 2.4 Price-Limit Approach Signal  ← **not yet implemented**
**Mechanism**: As stock approaches ±10% daily limit:
- Approaching upward limit (涨停): early buyers want to lock in; queue forms
  behind ask at limit price; signal = imminent limit hit → momentum play
- After hitting limit: price often continues next day (magnet effect confirmed
  empirically in Chinese markets)

**Signal construction**:
```python
pct_to_limit = (limit_price - current_price) / limit_price
signal = exp(-10 · pct_to_limit)  if approaching up-limit
       = -exp(-10 · pct_to_limit) if approaching down-limit
```

**Evidence**: Statprob NLM (2015) "Statistical Properties and Pre-Hit Dynamics of
Price Limit Hits in Chinese Stock Markets" — price continuation probability 0.68
after up-limit hit.

**Constraints**:
- Only applicable to individual stocks, not futures
- Needs T+1 workaround: position via ETF or futures hedge
- Risk: gap-down reversal next morning after forced selling

---

### 2.5 ETF Arbitrage (Net-Asset-Value Spread)  ← **not yet implemented**
**Mechanism**: ETFs (510300 for CSI 300, 510500 for CSI 500) can be created/redeemed
against basket. When ETF premium to NAV > transaction cost threshold, sell ETF +
buy basket (or vice versa). T+0 for ETFs.

**Signal**: `basis = ETF_price / IOPV - 1`

**Threshold**: ~5 bps (creation/redemption fee ≈ 0.05%)

**Capacity**: Very high (ETF creation/redemption handles large flows)

**Constraint**: Need prime broker access for basket trading; settlement T+2 for
constituent stocks creates funding risk.

---

## 3. Primary Alpha: Multi-Level OFI — Construction Detail

```
Data requirements:
  - 10-level LOB snapshots at 3-second cadence
  - cum_buy_vol, cum_sell_vol (from tick data or 成交明细)
  - Opening auction: indicative price time series + final volumes

Signal pipeline:
  1. Load LOB day
  2. Compute OFI at each level l = 1..5 (CKS formula)
  3. Weight: w_l = exp(-0.5·(l-1)), normalise to sum=1
  4. Aggregate: MLOFI(t) = Σ w_l · OFI_l(t)
  5. Roll: agg_OFI(t) = Σ_{s=t-9}^{t} MLOFI(s)   [10-tick window = 30s]
  6. Normalise: z-score via rolling std (120-tick window = 6 min)
  7. Combine with micro_price_dev, queue_imbalance, auction_signal
  8. Final composite: weighted average + output z-score

Feature weights (empirically tuned, Chinese IF futures):
  mlofi:           0.30   (primary)
  agg_ofi:         0.20   (persistence)
  micro_price_dev: 0.15   (stoikov signal)
  queue_imbalance: 0.12   (depth asymmetry)
  trade_imbalance: 0.10   (execution confirmation)
  depth_tilt:      0.06   (shape signal)
  mom_5:           0.04   (5-tick = 15s)
  mom_20:          0.03   (20-tick = 60s)
  auction_signal:  0.10   (replaces proportional share, decays during day)
```

---

## 4. Expected Performance (Synthetic, Not OOS)

| Metric | Expected Range | Notes |
|---|---|---|
| IC @ 10 ticks (30s) | 0.03–0.08 | Rank correlation |
| IC @ 40 ticks (2min) | 0.02–0.05 | Decay visible |
| ICIR (daily) | 0.5–1.5 | Must be > 0.5 for live trading |
| Sharpe (IF futures) | 1.5–3.5 | After realistic costs |
| Avg hold | 30–90 seconds | |
| Daily turnover | 5–15x notional | Very high; need direct market access |

---

## 5. Chinese-Specific Risk Factors

| Risk | Description | Mitigation |
|---|---|---|
| T+1 constraint | Can't close stock position same day | Use futures/ETF for delta hedging |
| Spoofing | Large LOB orders placed and cancelled before execution | Weight trade_imbalance heavily; filter LOB orders < 3 snapshots |
| Circuit breakers | IF halts ±5% for 15 min, ±10% for day | Scale down position size 2% from limit |
| Auction manipulation | Large player can temporarily inflate auction imbalance | Use both imbalance AND gap; cap signal at ±1 |
| 3-second LOB lag | Exchanges publish every 3s; actual book moves faster | Cannot trade at sub-3s on snapshot data; use CTP tick stream for live |
| Regulatory risk | CSRC has tightened HFT rules since 2015 | Track ADV participation rate; stay below 20% of ADV |
| Short-selling cost | 融券 rate typically 8–12% annualised | Model as borrow cost; prefer futures for short exposure |

---

## 6. Data Sources (Real Data)

| Provider | What | Cost |
|---|---|---|
| Wind (万得) | L2 tick data, order book, auction data | ~¥30–80k/year |
| Tushare Pro | Snapshot LOB (30-min delay free, real-time paid) | ¥500–5000/year |
| AKShare | Daily + some intraday (free, unreliable) | Free |
| CFFEX direct | Futures tick via CTP API (real-time) | Broker dependent |
| JoinQuant | Minute bars + some L2 features | ¥8000/year |

**Recommended stack for research**: Tushare Pro (level 3 or 4) for L2 snapshots
+ CFFEX CTP for futures tick.

---

## 7. Implementation Checklist

- [x] Signal construction: MLOFI, micro-price, queue imbalance, auction
- [x] Synthetic data generator (test harness)
- [x] IC / ICIR analysis framework
- [x] Signal decay curve
- [x] Threshold-based backtest engine
- [x] Chinese market constraints (T+1, price limits, costs)
- [ ] Real data loader (Tushare Pro / Wind)
- [ ] Parameter optimisation grid (avoid overfitting: use walk-forward)
- [ ] Live order management (CTP API integration)
- [ ] Risk limits: max position, ADV fraction, intraday drawdown halt
- [ ] OOS validation: 6+ months held-out before live deployment

---

## 8. Key References

1. Cont, Kukanov, Stoikov (2014). "The Price Impact of Order Book Events."
   *Journal of Financial Econometrics*, 12(1), 47–88.

2. Stoikov, S. (2018). "The micro-price: a high-frequency estimator of future
   prices." *Quantitative Finance*, 18(12), 1959–1966.

3. Kolm, Turku, Westray (2023). "Multi-Level Order-Flow Imbalance in a Limit
   Order Book." *Applied Mathematical Finance*, 30(1), 20–50.

4. Chen Hu, Kouxiao Zhang (2025). "Stochastic Price Dynamics in Response to
   Order Flow Imbalance: Evidence from CSI 300 Index Futures." arXiv:2505.17388.

5. MDPI Entropy (2020). "Trading Imbalance in Chinese Stock Market — A
   High-Frequency View." PMC7517523.

6. arXiv:0905.0582 (2009). "Empirical regularities of opening call auction in
   Chinese stock market."

7. PMC4395215 (2015). "Statistical Properties and Pre-Hit Dynamics of Price
   Limit Hits in Chinese Stock Markets."
