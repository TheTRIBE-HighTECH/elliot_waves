# PETR4 intermarket combinatorial experiment — 28 Nov 2026

## Data and reproducibility

- Reproducible Python: \`research/petr4_intermarket_combinatorial.py\`
- Sources (commit pinned): [msbrasil/Series-Temporais-PETR4](https://github.com/msbrasil/Series-Temporais-PETR4/commit/30e18599e1283ccb77a152fce55d3d5dea1f646a), combined Yahoo PETR4, ^BVSP, BZ=F, BCB PTAX and Selic, as-of 17 Nov 2026.
- Repository CI run [46163910994](https://github.com/TheTRIBE-HighTECH/elliot_waves/actions/runs/46163910994), artifact **PETR4-intermarket-correlation-46163910994**.
- Local independent numerical reimplementation of the two selected MACD rule variants matches the pipeline to less than 1e-6 percentage points.
- 2014–2023 training diagnostics, 2024 validation, 2 Jan 2025–17 Nov 2026 locked evaluation. **Retrospective rule choice using the lockbox does NOT constitute unbiased validation**.
- Long/flat, next B3 daily open after close signal, R$ 10,000 initial capital, 13 bps/side, no leverage, zero interest on cash. Yahoo dividend-adjusted OHLC proxy for total return, **not executable stock prices or official dividend cash ledger**.

### Combinatorial universe

68 technical variants, 826 unique intermarket gate masks, **56,236 raw combinations**, **45,309** passing train/validation eligibility. Gates include individual factors, AND/OR pairs, two-of-three market interactions, 63-/126-session trailing correlations in **returns** rather than levels, oil in USD and BRL, Ibovespa, FX, lagged residual and volatility regimes. Each external variable is delayed at least one complete PETR4 session at signal time to avoid foreign market close lookahead.

12,246 post-screen cross-market strategies outperformed the 2025–2026 adjusted buy-and-hold, and 2,458 outperformed buy-and-hold in each year **if selected in hindsight**. Massive data mining increases the false discovery rate; these counts should never be construed as independent trading edges.

### Main exploratory 2025–2026 candidates

| Strategy | Validation 2024 | 2025 | 2026 to 17 Nov | Total | Max DD |
|---|---:|---:|---:|---:|---:|
| **MACD EMA(12,35,5) + (IBOV > SMA50 OR BrentBRL > SMA50)** | +35.06% | +4.63% | +65.06% | **+73.94%** | **13.04%** |
| MACD EMA(12,35,5) + (BrentBRL momentum20 >0 OR PETR4–BrentBRL corr63 >0.25) | +43.31% | +0.72% | +67.31% | +69.72% | 13.85% |
| SMA20 baseline, PETR4 only | +19.35% | -11.11% | +93.22% | +72.98% | 14.19% |
| Buy and hold (dividend-adjusted proxy) | — | -6.00% | +50.69% | +42.65% | 27.03% |

The primary rule entered 19 times, was long ~49.47% of sessions and had Sharpe ~1.64. Its unadjusted raw-price return was approximately +68.85% (see all_candidates.csv for exact values). Benchmark raw price-only return differs significantly from adjusted total-return proxy.

**Most important negative result:** The parameter combination selected solely by its highest risk-adjusted 2024 validation score was TSMOM(60,-0.05) with (IBOV r20<0 OR USD/BRL below SMA50). It gained +60.24% during 2024 validation but only +12.09% during 2025–2026. Thus, the robust-looking MACD rules above **were identified after inspecting 2025–2026 results among thousands of models**. It is untrue that a prospective, globally selected model was proven to beat buy-and-hold.

The absolute top in-hindsight model, TSMOM(10,-0.05) with (IBOV r20>0 OR Brent USD r20>0), returned +125.67% adjusted (+5.69% in 2025, +112.00% in 2026) but failed 2024 validation with -4.52%. It is less defensible for deployment than the MACD candidate.

### Market correlations from daily returns (not price levels)

| Asset vs PETR4 | Pearson same-day 2025 | Same-day 2026 | Previous-market-day lag 2025 | Lag 2026 |
|---|---:|---:|---:|---:|
| Ibovespa | +0.528 | +0.391 | -0.031 | +0.092 |
| Brent in USD | +0.527 | +0.542 | +0.075 | +0.025 |
| Brent in BRL | +0.487 | +0.456 | +0.035 | +0.015 |
| USD/BRL | -0.024 | -0.246 | -0.106 | -0.039 |

Cross-market contemporaneous correlation is substantial but **does not imply 1-day forward predictability**. Filters appear more useful for regime conditioning. All factors for trading are lagged and calculated with rolling historical windows.

### Stress / sensitivity

MACD(12,35,5) + IBOV/BrentBRL trend, holding all else constant:
- 0 bps/side: +82.75% adjusted vs +43.03% BH
- 13 bps/side: +73.94% vs +42.65%
- 26 bps/side: +65.55% vs +42.28%
- 40 bps/side: +56.98% vs +41.89%
- 50 bps/side: +51.12% vs +41.60%
- 100 bps/side: +24.97% vs +40.19%

Same-filter MACD neighbors exhibit **varying results**. On 2025–2026, EMA MACD(8,50,5) generated +70.37%, MACD(12,26,5) +59.32%, MACD(8,35,5) +56.62%, MACD(12,35,9) +47.33%, using the same IBOV/Brent SMA50 gate. Some other parameter combinations fail the year-by-year condition: the apparent alpha is still parameter-sensitive.

Paired circular block bootstrap with 10-session blocks and 5,000 repeats, comparing **log cumulative wealth differences** of selected primary MACD strategy vs BH:
- Observed +19.83 log-percentage points.
- 95% interval approximately [-40.24, +93.33] log-percentage points, includes zero.
- Two-sided exploratory p-value ~0.56 (not adjusted for screening 56,236 alternatives).

This is neither statistical proof of genuine alpha nor evidence of live profitability.

### MQL5 demonstration implementation

[\`mql5/PETR4_Intermarket_MACD_v1.mq5\`](../mql5/PETR4_Intermarket_MACD_v1.mq5)
implements both explanatory regimes, configurable MACD, IBOV/Brent/FX symbols and an optional rolling correlation gate. It depends on standard \`<Trade/Trade.mqh>\`, uses a custom EMA-smoothed MACD signal to match pandas; standard MetaTrader iMACD uses SMA smoothing of its signal line.

**MQL5 version has NOT been compiled, tested in a brokerage terminal, or shown to reproduce Python P&L.** Broker-specific quotes, foreign-market day boundaries, lot sizes, settlement, corporate actions and fills change the strategy substantially.

### Next falsification and audit

- Preserve independent 2025–2026 period as known, **freeze** filter/parameter family now; test the untouched 2027 PETR4 tape or other Brazilian instruments with no re-optimization.
- Use actual B3 price, proventos and execution ledger, including ex-dates, taxes, spread, board lots, Slippage and overnight cash CDI.
- Multi-window walk-forward with explicit nested selection and family-wise false discovery controls (White Reality Check / SPA, DSR, PBO) before any claim of reliable improvement.
- Correlation features should be measured with publication timestamps rather than trusting exchange close date alone. Investigate the role of lag and asynchronous calendars.
- Code with tens of thousands of retrospectives is not equal to validating every proprietary EA from MQL5.

MQL5/source references:
- [Formulating Dynamic Multi-Pair EA — correlation mapping](https://www.mql5.com/en/articles/21460)
- [MQL5 walk-forward optimization](https://www.mql5.com/en/articles/3279)
- [MQL5 multiple-testing validation](https://www.mql5.com/en/articles/21603)
- [MQL5 multistrategy risk correlation](https://www.mql5.com/en/articles/21955)
- [MQL5 CopyClose timeseries memory order](https://www.mql5.com/en/docs/series/copyclose)
- [MetaQuotes MACD indicator signal SMA vs custom EMA](https://www.mql5.com/en/code/35)
