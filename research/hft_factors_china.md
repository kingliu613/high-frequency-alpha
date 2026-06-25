# High-Frequency Alpha Factors for the Chinese Market

This document describes the current paper-backed factor bank. The bank was
cross-checked against the Caitong paper folder and trimmed to factors with clear
references. Unsupported factors were removed from both documentation and code.

Last updated: 2026-06-17.

---

## 1. Market Constraints

| Constraint | Alpha implication |
|---|---|
| T+1 for stocks | Stock signals must be next-session aware; true intraday round trips need futures. |
| Retail-heavy trading | Herding and imbalance effects can persist. |
| Low cancellation baseline | Cancellation bursts are informative when they happen. |
| Opening call auction | Non-cancellable orders after 09:20 reveal stronger commitment. |
| Price limits | Near-limit behavior is a distinct Chinese-market state. |
| 3-second L2 snapshots | Sub-snapshot behavior is unobservable; avoid pretending otherwise. |

---

## 2. Strict Default Directional Alpha

These are the only factors included when `build_feature_matrix(..., factors=None)`
is called. Each has a formula-level source and required data fields declared in
`src/signals/composite.py::FACTOR_REGISTRY`.

| Group | Factor | Implementation | Strict formula / required data |
|---|---|---|---|
| Flow | MLOFI | `src/signals/ofi.py::mlofi` | OFI state-transition formula over observed LOB levels. |
| Flow | Aggregated OFI | `src/signals/ofi.py::aggregated_ofi` | Rolling aggregation of paper OFI increments. |
| Flow | Trade imbalance | `src/signals/ofi.py::trade_imbalance` | Transaction polarity `(NOB-NOS)/(NOB+NOS)` from buy/sell counts. |
| Flow | API | `src/signals/advanced.py::aggressive_passive_imbalance` | Aggressive-passive event imbalance from market and limit-order volumes. |
| Flow | OEI | `src/signals/advanced.py::order_execution_imbalance` | Chi event formula `((Lb-Cb-Ms)/Db)-((Ls-Cs-Mb)/Ds)`. |

---

## 3. Gates, Labels, Diagnostics

These are not part of the default directional alpha. They must be requested
explicitly via `factors=[...]` or used through dedicated APIs.

| Role | Item | Implementation | Reason not default alpha |
|---|---|---|---|
| Gate | VPIN | `src/signals/advanced.py::vpin` | Toxicity/risk state, not direction. |
| Gate | Kyle lambda | `src/signals/advanced.py::kyle_lambda` | Raw price-impact estimate; gate transform is separate. |
| Gate | Exposure gate | `src/signals/advanced.py::exposure_gate` | Position-size multiplier only. |
| Label | Price-limit state | `src/signals/features.py::price_limit_signal` | Hit-state label/regime, not a paper-defined alpha score. |
| Diagnostic | Opening auction | `src/signals/auction.py::auction_signal_series` | Paper defines auction statistics, not an intraday alpha curve. |
| Diagnostic | Queue imbalance | `src/signals/features.py::queue_imbalance` | Useful book statistic, but not retained as Caitong strict alpha. |
| Diagnostic | Herding | `src/signals/advanced.py::herding_intensity` | Full paper construction needs cross-sectional/investor-group data. |
| Diagnostic | Cancel spike | `src/signals/advanced.py::cancel_spike_imbalance` | Full spoofing detector is not implemented. |

---

## 4. Paper Formula Matrix

| Paper | Codable object | Formula/process status | Default role |
|---|---|---|---|
| `Trading Imbalance in Chinese Stock Market—A High-Frequency View.pdf` | Transaction polarity | Implemented as `(NOB-NOS)/(NOB+NOS)` using buy/sell counts. | Alpha |
| `Stochastic Price Dynamics in Response to Order Flow Imbalance-Evidence from CSI300 Index Futures.pdf` | OFI / aggregated OFI | Implemented from LOB state-transition OFI, with rolling aggregation. | Alpha |
| `Scientific Programming - 2021 - Chi - The Price Impact of Order Book Events from a Dimension of Time.pdf` | Order-event imbalance | Implemented with explicit `L`, `C`, `M`, and side-depth fields. | Alpha |
| `high frequency liquidity in chinese stock market.pdf` | Aggressive-passive imbalance | Implemented only when explicit market/limit order-event fields exist. | Alpha |
| `Flow Toxicity and Liquidity in a High-frequency World.pdf` / `From PIN to VPIN.pdf` | VPIN | Implemented with BVC and equal-volume buckets. | Gate |
| `Empirical regularities of opening call auction in Chinese stock market.pdf` | Relative order price and auction imbalance statistics | Kept as auction scalar/diagnostic; no default intraday alpha curve. | Diagnostic |
| `Statistical Properties and Pre-Hit Dynamics of Price Limit Hits in the Chinese Stock Markets.pdf` | Limit-hit and pre-hit event dynamics | Kept as hit-state label; no exponential approach score. | Label |
| `ON DETECTING SPOOFING STRATEGIES IN HIGH FREQUENCY TRADING.pdf` | Spoofing detection framework | Current simple cancel burst is diagnostic only, not full spoofing replication. | Diagnostic |
| `The Physics of Price Discovery- Deconvolving Information, Volatility, and the Critical Breakdown of Signal during Retail Herding.pdf` | Herding/retail-flow regime | Needs panel/investor-group construction; single-name version is diagnostic. | Diagnostic |
| `Int J Fin Econ - 2022 - Liao - Smart money or chasing stars...pdf` | Northbound smart-money behavior | Not implemented as HFT factor. | Excluded |
| `Simulation of high-frequency trading risks and regulatory strategies...pdf` | Regulatory/risk simulation | Not a direct alpha formula. | Excluded |

---

## 5. Implementation Map

| File | Role |
|---|---|
| `src/signals/ofi.py` | OFI, aggregated OFI, trade imbalance. |
| `src/signals/features.py` | Queue imbalance and price-limit hit state. |
| `src/signals/auction.py` | Opening auction signal. |
| `src/signals/advanced.py` | API, OEI, herding, cancel spike, VPIN, Kyle lambda, exposure gate. |
| `src/signals/composite.py` | Factor registry, strict default selection, feature matrix, composite alpha. |
| `src/signals/daily.py` | Daily aggregation for cross-sectional evaluation. |

---

## 6. Sources Represented In The Current Bank

- Trading imbalance and OFI evidence in Chinese markets.
- Price impact of order-book events from a time-dimension view in Chinese L2 data.
- High-frequency liquidity evidence from Chinese stock market order books.
- VPIN and flow-toxicity literature.
- Price-limit, auction, spoofing, and herding papers are retained as labels or
  diagnostics unless their full paper process is implemented.

The current code intentionally excludes factors that only had thematic,
proprietary, or external-note support without a clear reference in the paper set.

---

## 7. Caitong Paper Bibliography Reviewed

These are the local source papers reviewed from `/Users/yichanliu/Documents/Caitong papers`.
They are retained here as provenance for the audit; inclusion in this bibliography
does not mean every concept in a paper is an active factor.

- `Empirical regularities of opening call auction in Chinese stock market.pdf`
- `Flow Toxicity and Liquidity in a High-frequency World.pdf`
- `From PIN to VPIN.pdf`
- `Int  J Fin Econ - 2022 - Liao - Smart money or chasing stars  Evidence from northbound trading in China.pdf`
- `ON DETECTING SPOOFING STRATEGIES IN HIGH FREQUENCY TRADING.pdf`
- `Scientific Programming - 2021 - Chi - The Price Impact of Order Book Events from a Dimension of Time.pdf`
- `Simulation of high-frequency trading risks and regulatory strategies in China’s financial market based on multi-layer complex networks.pdf`
- `Statistical Properties and Pre-Hit Dynamics of Price Limit Hits in the Chinese Stock Markets.pdf`
- `Stochastic Price Dynamics in Response to Order Flow Imbalance-Evidence from CSI300 Index Futures.pdf`
- `The Physics of Price Discovery- Deconvolving Information, Volatility, and the Critical Breakdown of Signal during Retail Herding.pdf`
- `Trading Imbalance in Chinese Stock Market—A High-Frequency View.pdf`
- `high frequency liquidity in chinese stock market.pdf`
