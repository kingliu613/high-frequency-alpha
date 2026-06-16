# High-Frequency Alpha Factors for the Chinese A-Share Market

Reference doc for the upcoming algo build. Focuses on *newer / China-unique* factors
beyond what `src/signals/` already implements (OFI/MLOFI, trade imbalance, micro-price
deviation, queue imbalance, depth tilt, open-auction imbalance, price-limit, ETF-basis).

Last updated: 2026-06-11.

---

## 0. Self-prompt (run this to build the algo later)

> Build a high-frequency cross-sectional alpha strategy for Chinese A-shares (and
> IF/IC/IH futures + ETFs for intraday legs). Respect the market constraints in §1.
> Implement the factors in §3 that are NOT yet in `src/signals/` as new modules with
> docstrings citing sources. Each factor: stationary, z-scored, with a stated prediction
> horizon. Combine via the existing `composite.py` weighting scheme, re-fit weights by
> walk-forward IS/OOS (already have the optimizer). Backtest with realistic costs:
> stamp duty 0.05% on sells, ~0.02% commission, T+1 for stock legs, price-limit
> non-tradability. Report IC/ICIR per factor, turnover, OOS Sharpe net of cost.

---

## 1. Why China factors differ — microstructure constraints

These rules reshape which factors carry alpha. Every factor below is justified by one.

| Constraint | Value | Alpha implication |
|---|---|---|
| Settlement | **T+1** for stocks (buy today, sell tomorrow). Futures/ETF = T+0 | Stock HF signals must predict *next-day* return, or be traded via IF/IC/IH/ETF intraday. Pure intraday stock round-trips impossible. |
| Price limits | ±10% main board, **±20% STAR (科创板)/ChiNext (创业板)**, ±5% ST | Limit-up/down create a censored-return regime. "封单" (sealing queue) factors are tradeable; returns truncate. |
| Short selling | Restricted, hard-to-borrow, no naked short; securities-lending thin | Most signals are long-only or long-short via index futures. Reversal alpha hard to harvest short side directly. |
| Lot size | Buy in **100-share round lots**; sell odd allowed | Trade-size distribution clusters at round lots → size-classification factors (大单) are clean. |
| Auctions | Open 09:15–09:25, **Close 14:57–15:00 (added 2018)** | Two information-concentrated windows. You only use the open auction; close auction is unexploited. |
| Tick size | 0.01 RMB flat | **Large-tick regime** for most names → queue position dominates; spread mostly = 1 tick; book-shape/slope informative. |
| Participants | **Retail ≈ 80% of volume** | Behavioral alpha (herding, attention, lottery, limit-up chasing) persists far more than in US. |
| Cancellation | Very low cancel ratio vs US | LOB depth is *more* trustworthy (less spoofing) → queue/OEI factors stronger; cancellation *spikes* become a rare-but-real signal. |
| Northbound | Stock-Connect flow was real-time "smart money"; **real-time disclosure suspended Aug 2024**, now delayed/quarterly | Treat as a low-frequency overlay, not an HFT factor anymore. Honest caveat. |
| Session | 09:30–11:30, 13:00–15:00, 1.5h lunch gap | Volatility U-shape + a second open at 13:00. Two intraday-momentum windows, not one. |

---

## 2. Already in your repo (don't rebuild)

OFI level + MLOFI (Cont-Kukanov-Stoikov / Kolm 2023), aggregated OFI, trade imbalance,
micro-price + deviation (Stoikov 2018), multi-level queue imbalance, depth tilt,
short-term momentum, open call-auction imbalance + gap, price-limit approach signal,
ETF-basis mean-reversion. Composite z-scores + weights, walk-forward optimizer.

**Implementation status (2026-06-11): ALL factors in §3 are now implemented.**

| Factor | Where |
|---|---|
| §3.1 API | `src/signals/advanced.py::aggressive_passive_imbalance` |
| §3.2 OEI | `src/signals/advanced.py::order_execution_imbalance` |
| §3.3 VPIN | `src/signals/advanced.py::vpin` (gate, not directional) |
| §3.4 Close auction | `src/signals/auction.py::close_auction_imbalance` + `close_auction_signal_series`; generator `src/data/synthetic.py::simulate_close_auction_data` |
| §3.5 Herding | `src/signals/advanced.py::herding_intensity` |
| §3.6 大单净额 | `src/signals/advanced.py::big_order_flow` |
| §3.7 RV / signed jumps | `src/signals/advanced.py::realized_vol`, `signed_jump_reversal` |
| §3.8 Slope / resiliency | `src/signals/advanced.py::order_book_slope`, `book_resiliency` |
| §3.9 封单 sealing | `src/signals/advanced.py::sealing_strength` (stock mode) |
| §3.10 Cancel spikes | `src/signals/advanced.py::cancel_spike_imbalance` |
| §3.11 Kyle λ | `src/signals/advanced.py::kyle_lambda` (gate, not directional) |
| §4 regime gate | `src/signals/advanced.py::exposure_gate` → `run_backtest(..., exposure_scale=...)`; CLI `--gate` |

Directional factors are wired into `composite.py` DEFAULT_WEIGHTS; VPIN + λ feed
the exposure gate only. Tests: `tests/test_advanced.py`.

**Realism audit (2026-06-11):** look-ahead bias removed from VPIN (sequential causal
bucketing), exposure gate (expanding mean), big_order_flow (no warm-up backfill);
OEI now divides flow by *prior* depth. Engine enforces T+1 (stock positions held to
EOD, `exit_reason="eod_t1"`), price-limit non-tradability (buys blocked at sealed
up-limit; `"eod_limit_down"` flags unrealisable closes), EOD force-close with full
costs for all instruments, permanent market-impact charged at exit, and per-instrument
cost defaults via `MarketParams.default_for()` (stock ≈ 2.5 bp commission vs futures
0.23 bp). Synthetic trades now consume top-of-book depth. Causality regression tests
in `tests/test_advanced.py` (factor[t] invariant to future data).

Consequence worth knowing: under honest T+1, an intraday composite has almost no
stock-mode edge — one entry, held blind to the close. Stock alpha must target the
overnight/next-day horizon (close-auction, sealing, big_flow EOD snapshot), or trade
intraday through T+0 instruments (IF/IC/IH futures, ETFs).

---

## 3. New / unique factors to add

Each: definition, formula, China-specific rationale, prediction horizon.

### 3.1 Aggressive–Passive Imbalance (API)
**Source:** Chinese HF-liquidity study, SZSE 2019–2021 tick data, ~46.6B events (J. Int. Financial Markets, 2025).
Decompose order flow into **aggressive** (marketable / liquidity-taking) vs **passive**
(resting limit / liquidity-providing) on each side.

    API(t) = ( AggrBuy(t) − AggrSell(t) ) / ( AggrBuy(t) + AggrSell(t) )

Optionally a second axis: aggression *ratio* = Aggr / (Aggr + Passive) per side.
**Why China:** with low cancellation, the aggressive-vs-passive split is cleaner than in
spoof-heavy US books; API is the order-based driver of bid–ask spread changes.
**Horizon:** seconds–minutes; spread-change & next-tick mid predictor. Distinct from your
OFI (which mixes both) and trade_imbalance (which ignores the passive side).

### 3.2 Order Execution Imbalance (OEI)
**Source:** "Price Impact of Order Book Events from a Dimension of Time," Level-2 CN market (2021).
Measures *efficiency of execution* on bid vs ask limit books (how fast resting volume gets
filled), not just standing depth.

    OEI(t) = ExecRate_bid(t) − ExecRate_ask(t)

Improves the classic Cont (2014) OFI model R² on Chinese L2, especially when liquidity surges.
**Why China:** low cancel ratio → execution rate is a stable, informative quantity.
**Horizon:** one-tick-ahead price direction.

### 3.3 VPIN — order-flow toxicity (regime filter)
**Source:** Easley, López de Prado, O'Hara (2012); validated on CN index futures 2012–13.
Volume-bucketed; classify each bucket's buy/sell via bulk-volume (normal-CDF of price change).

    VPIN = Σ |V_buy,τ − V_sell,τ| / (n · V_bucket)

**Use:** not a directional alpha — a **toxicity/adverse-selection gate**. High VPIN → widen
sizing, expect spread blowout / one-sided moves. Gate the composite or scale position.
**Horizon:** minutes–hours regime.

### 3.4 Closing call-auction imbalance (尾盘集合竞价)  ★ high value
**Source:** China added a 14:57–15:00 close auction in 2018; analogous to your open-auction logic.
Index-rebalance flow, ETF creation/redemption hedges, and end-of-day informed positioning
concentrate here.

    CloseAucImb = (CloseBuyVol − CloseSellVol) / (CloseBuyVol + CloseSellVol)

**Why China:** mirror of your existing open-auction edge but on the *other* information window;
strong for index constituents on rebalance days. Predicts overnight + next-open gap.
**Horizon:** overnight → next-day open. Pairs naturally with your `auction.py`.

### 3.5 Intraday herding intensity
**Source:** "Intraday Herding Drivers in China's A-Share Market: Evidence from CSI 500," 2025
(tick data, 5-min, stock-specific LSV-style herding).
Build a 5-min herding statistic = degree to which the cross-section trades the same direction
beyond what volume alone implies.

    H_i(t) = |b_i(t) − E[b]| − E|b_i(t) − E[b]|,   b_i = BuyTrades/(Buy+Sell)

**Why China:** 80% retail → herding magnitude and predictability far exceed developed markets.
Herding spikes → very-short continuation then reversal.
**Horizon:** 5–30 min; sign flips at the reversal point — model both legs.

### 3.6 Smart-money large-order net flow (大单净额)  ★ very China-specific
Classify executed trades by notional into super-large / large / medium / small (Tushare &
exchange L2 expose 主动买/主动卖 by size band). Institutional footprint = net large-order flow.

    BigFlow_i = (ActiveBigBuy − ActiveBigSell) / TurnoverNotional

**Why China:** round-lot retail noise dominates small bands; the large/super-large band isolates
institutions. Widely used in domestic quant (主力资金) and genuinely predictive intraday→next-day.
**Horizon:** minutes → next-day. Note: vendor "main-force" tags are heuristic — prefer raw
size-banded active volume from L2 if available.

### 3.7 Realized-volatility & signed-jump factors
**Source:** standard HF econometrics (Barndorff-Nielsen–Shephard), applied intraday CN.
From 1-min (or finer) returns over a rolling window:

    RV = Σ r²,  BV = (π/2) Σ |r_t||r_{t−1}|,  Jump = max(RV − BV, 0),  signed by Σ sign(r)·r²

**Why China:** pronounced intraday vol U-shape + two sessions; signed jumps in retail-driven names
tend to short-horizon **reverse** (overreaction). Also feeds sizing.
**Horizon:** 5–30 min reversal; RV level as a sizing/regime input.

### 3.8 Order-book slope, shape & resiliency
**Source:** empirical LOB shape function, Chinese market; resiliency literature.
- **Slope** = depth accumulated per tick away from mid (steeper = more support).
- **Resiliency** = speed the book replenishes after an aggressive trade eats a level.

    Slope_bid = Σ_l V_bid,l / (P_bid,1 − P_bid,l)    (ditto ask)
    Resiliency = depth recovery fraction within Δt after an aggression

**Why China:** large-tick regime → slope/shape carry real info (not microstructure noise).
Low resiliency → impending continuation; high → mean-revert.
**Horizon:** seconds–minutes. Complements your static `depth_tilt`.

### 3.9 Limit-up sealing-strength factor (涨停封单)  ★ China-only retail game
For names approaching/at the +limit:

    SealRatio = SealingBidVol_at_limit / RecentTurnover
    TimeToLimit, OpenCount (how many times limit broke & resealed)

**Why China:** the limit-up board is a famous retail momentum game; strong sealing queue + early
seal + few re-opens → high prob of next-day continuation (打板 factor). Extends your basic
`price_limit_signal` from "approach" to "sealed-board quality."
**Horizon:** next-day (T+1 forces it). Long-only by construction.

### 3.10 Cancellation / fleeting-order spikes
Even with low baseline cancel ratio, *spikes* in cancellation on one side reveal informed
repositioning or failed iceberg refresh.

    CancelImb(t) = (CancelBid − CancelAsk) / (CancelBid + CancelAsk)   over short window

**Why China:** because cancels are rare, a burst is a strong, low-noise event.
**Horizon:** seconds. Use as a confirmation/veto on queue_imbalance (which you flagged as spoofable).

### 3.11 Tick-level price-impact / Kyle's λ (liquidity-state)
Regress short-horizon mid-return on signed order flow:

    Δmid = λ · SignedFlow + ε   →   λ = price impact per unit flow (Amihud-like, tick scale)

**Use:** liquidity-state factor; gates turnover and position size, and is itself weakly predictive
(low-λ names absorb flow → momentum; high-λ → impact/reversal).
**Horizon:** regime / sizing input.

---

## 4. Building the algo — practical notes

- **Horizon split:** stock-level factors → predict *next-day* (T+1). Want true intraday P&L →
  trade the signal through **IF/IC/IH futures or sector/index ETFs** (T+0), as your repo already
  hints with `use_etf` and `etf_basis`.
- **Cross-sectional construction:** z-score each factor within the universe per timestamp, neutralize
  vs size/industry if going long-short, rank-combine. IC/ICIR per factor for weighting.
- **Costs:** stamp duty **0.05% on sells** (one-sided), commission ≈0.02% + min, slippage = f(λ, spread).
  T+1 caps turnover for the stock book; cost discipline decides which HF factors survive.
- **Censoring:** drop/limit names at ±limit from the tradeable set except the dedicated sealing factor;
  treat limit returns as non-executable.
- **Regime gates:** VPIN (§3.3) and λ (§3.11) scale exposure; herding (§3.5) and jumps (§3.7) flip
  sign at their reversal horizon — don't treat as monotone.
- **Combine** into existing `composite.py`; re-fit weights with the walk-forward optimizer; report
  OOS Sharpe net of cost, turnover, per-factor decay.

---

## Sources

- Aggressive–Passive Imbalance / HF liquidity, SZSE tick data — *J. International Financial Markets, Institutions & Money*, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0927538X25000186
- Order Execution Imbalance, Level-2 CN — *Scientific Programming*, 2021. https://onlinelibrary.wiley.com/doi/10.1155/2021/9949565
- VPIN / order-flow toxicity — Easley, López de Prado, O'Hara, *RFS* 2012. https://academic.oup.com/rfs/article-abstract/25/5/1457/1569929 ; PIN→VPIN intro https://www.sciencedirect.com/science/article/abs/pii/S2173126812000344
- Trading Imbalance in Chinese Stock Market (open auction) — *Entropy* 2020. https://www.mdpi.com/1099-4300/22/8/897
- Intraday herding, CSI 500 — *Finance Research Letters*, 2025. https://www.sciencedirect.com/science/article/abs/pii/S1544612325015466
- Retail-herding price discovery breakdown — arXiv 2026. https://arxiv.org/pdf/2601.11602
- OFI on CSI 300 futures — Hu & Zhang, arXiv:2505.17388. https://arxiv.org/pdf/2505.17388
- Price-limit hit pre-dynamics, CN — *PLoS ONE / PMC* 2015. https://pmc.ncbi.nlm.nih.gov/articles/PMC4395215/
- Spoofing detection in HFT — arXiv:2009.14818. https://arxiv.org/pdf/2009.14818
- Smart money / northbound — Liao 2024, *Int. J. Finance & Economics*. https://onlinelibrary.wiley.com/doi/10.1002/ijfe.2751
- HFT risk simulation, CN multi-layer networks — *Frontiers in Physics* 2025. https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2025.1733200/full
