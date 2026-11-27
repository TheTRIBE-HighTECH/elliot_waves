# PETR4 · MQL5-style strategy family search (27 Nov 2026)

**Research status: completed, exploratory; not an audited MetaTrader EA backtest.**

## Data and reproducibility

Source: https://github.com/msbrasil/Series-Temporais-PETR4
Raw CSV: \`data/raw/petr4_yf_raw.csv\` (third-party Yahoo-derived).
PETR4 B3 daily candlesticks, one inconsistent 2014-04-02 OHLC excluded.
Deterministic reproduction: \`python research/petr4_mql_catalog.py --cost-bps 13\`.

2014–2023 calibration/training diagnostics, 2024-only validation ranking,
and exact locked 23 Jan 2025–04 Nov 2026 holdout, 445 sessions.
Execution at next session open after signal computed on current day's close.
Long or cash, no leverage, no short selling, 13 bps transaction cost
per buy or sell, capital initially R$10,000, interest on cash 0.

Analyzed 460 candidate parameter configurations in twelve public strategy
families: EMA crossover, SMA crossover, trend price/SMA filters, momentum,
Donchian breakout, RSI mean reversion, Bollinger mean reversion, MACD,
SuperTrend, Keltner channels, ATR Chandelier exit and RSI trend.
359 candidates passed the train/validation frequency and exposure filters.

This did NOT execute, obtain, or audit every proprietary or commercial
EA distributed in the MQL5 Market.

### Important selection result

* **Global champion chosen exclusively on 2024:** Bollinger(50,1,1.0);
  validation adjusted +48.71%; holdout only +5.07%; training drawdown
  reached 82.56%. Not robust.
* **SMA 20** was the winner WITHIN the trend-filter family on validation
  2024 (+19.35%). It earned **+74.15%** adjusted on holdout, vs
  adjusted buy-and-hold **+49.96%**, both net of 13 bps per side.
  Unadjusted raw-price returns were +68.85% and +26.61%, respectively.
  It made 22 roundtrips, spent 59.78% of sessions long, and had
  max drawdown 13.61%, versus 22.16% for buy-and-hold.
* **SMA 20 was highlighted after evaluating holdout across the 12
  families.** Therefore it is an exploratory discovery, not a clean
  global out-of-sample winner. Family selection consumed the holdout.
* **Best ex-post parameter across all 460:** TSMOM(10,-0.05) delivered
  +78.51% adjusted but validation 2024 was only +1.86% and training
  drawdown 72.31%. It is *not validated* and should not be selected
  on this performance.

### Robustness of SMA20 in the same (already inspected) holdout

| Period | Adjusted return |
|---|---:|
| SMA10 | +41.22% |
| SMA15 | +55.87% |
| SMA20 | +74.15% |
| SMA25 | +76.79% |
| SMA30 | +65.65% |
| SMA40 | +51.98% |
| SMA50 | +45.57% |
| Buy and hold | +49.96% |

The local neighborhood of SMA15–40 performs better than adjusted BH,
but all these values are **retrospective post-selection diagnostics**.

Transaction fee/slippage sensitivity, costs *per side*:
| Cost bps | SMA20 | Buy & hold |
|---:|---:|---:|
| 0 | +84.40% | +50.35% |
| 13 | +74.15% | +49.96% |
| 26 | +64.47% | +49.57% |
| 50 | +47.98% | +48.85% |
| 100 | +18.76% | +47.37% |

Higher costs remove the gross advantage. With 2025 data alone, SMA20
returned about −10.4% while most gains arose from approximately +94.35%
during the investigated 2026 subperiod (not annualized). This is a
regime-sensitive, concentrated finding.

A paired circular 10-session block bootstrap over log daily excess
returns (4,000 replicates) gave a 95% interval of about **−24.76 to
+60.39 log-percentage points** for SMA20 less buy-and-hold; includes
zero. This is an indicative autocorrelation-aware diagnostic, not an
adjusted p-value for testing 460 alternatives.

## Limitations and next protocol

* The **Adj Close** ratio is a post-hoc reinvested-proventos proxy; it
  is *not executable market quotations*, and dividends/payment tax
  treatment isn't modeled with a B3 ledger. Opening prices were
  converted by the same daily adjustment factor.
* Unadjusted prices may differ materially around dividends.
* No real MetaTrader terminal/ticks were used; the matching MQL EA
  \`mql5/PETR4_SMA20_LongOnly.mq5\` is educational and not compiled
  or independently verified on a B3 broker.
* Evaluate symbol contract size, board/odd lots, local spreads, true
  brokerage and exchange fees, cash CDI, taxes and tax on dividends,
  data vendor differences, publication lag and realistic fill liquidity.
* The current 2024 validation picked an **overfit Bollinger strategy**.
  Selecting the trend-filter family because it won 2025–2026 is
  retrospective family-level overfitting, even though SMA20 was chosen
  within that family on 2024 alone.
* Next genuine test: freeze the SMA20 family *now*, test 2027 forward,
  and evaluate on other Brazilian blue-chips with no retuning.
* Distinguish profitable price trend timing from *Elliott* labels:
  SMA20 is not an Elliott detector. Our Elliott/GNN work still needs
  independent human-wave annotations.

MQL5 references:
- https://www.mql5.com/en/articles/3146 (Donchian EA logic)
- https://www.mql5.com/en/articles/15509 (combining MA and RSI)
- https://www.mql5.com/en/articles/24095 (PBO/multiple strategy testing)
- https://www.metatrader5.com/en/terminal/help/algotrading/strategy_optimization (forward optimization)

All intermediate candidate results and equity curves are in the
GitHub Actions artifact of the completed MQL-style strategy sweep.
