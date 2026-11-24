# Elliott neural multiscale research v3

Research branch; the existing production app and `modelo.py` remain unchanged.

## Data source audit and label provenance

* GOX-TryElliottWave (<https://github.com/elliottwavealgorithm/GOX-TryElliottWave>):
  human drawing/capture architecture, NOT a downloadable verified historical
  annotation dataset. `gox_annotations.py` is a conservative compatibility
  adapter for JSON exported with its described anchors. This is not an official
  GOX interchange schema. Mark imported rows `human_unverified` until
  an independent reviewer signs them off.
* Elliott Wave Atlas (<https://elliottwaveatlas.com/agents>):
  publishes roughly 100 current market calls/levels as JSON. Under
  <https://elliottwaveatlas.com/terms>, give attribution, do not resell or
  repackage provider content as a paid feed, and fetch at a reasonable rate.
  Daily snapshots preserve retrieval time and `asof` independently. They are
  NOT retroactively available point-in-time labels for 2024.
* ElliottWavesEngine (<https://github.com/SmarterSystems/ElliottWavesEngine>):
  MIT code, useful rule baseline, but algorithmic output is *weak supervision*.
  Its test fixture bars are fetched separately; rule-engine results should
  never be promoted to expert human labels.
* HF `usamaahmedsh/elliott-wave-scorer-training`: 60m synthetic geometric
  labeled samples (NOT market-confirmed Elliott sequences).
* HF `usamaahmedsh/elliott-wave-market-data-complete`: OHLCV from multiple
  markets; its page gives inconsistent row totals and its parquet viewer may
  error. Audit physical files before using, never presume row count.
* Binance Vision: primary initial real spot OHLCV for BTCUSDT/ETHUSDT/SOLUSDT.
  From 2025 spot timestamps may be microseconds; the downloader handles both.

## Run locally

```bash
python -m pip install -r research/requirements.txt
python research/download_binance.py --symbols BTCUSDT ETHUSDT SOLUSDT --start 2024-01 --end 2025-12 --out research/datasets
for asset in BTCUSDT ETHUSDT SOLUSDT; do
  python research/compare_nn.py --csv research/datasets/${asset}_1h.csv --epochs 6 --out research/runs/${asset}
done
python research/rank_models.py research/runs/BTCUSDT/report.json research/runs/ETHUSDT/report.json research/runs/SOLUSDT/report.json --out research/runs/ranking.json
```

Model comparison: MLP, single-scale TCN, multiscale TCN (Fibonacci
scales 1,3,8), multiscale GRU; each has wave, reversal and confidence heads.
Each model learns from both calendar and volume-information time, and the same
single checkpoint evaluates both clocks. Fixed 2024 training, 2025-H1
validation, 2025-H2 holdout. The cross-asset champion is chosen ONLY from
validation macro-F1, never from test or trading profits.

**Important:** Without independent human-approved wave-level labels, the
training target is generated from retrospective zigzag-like SWING PSEUDO-LABELS,
which do not enforce all Elliott rules. This experiment then compares learning
on real prices with weak targets; it is NOT a validated neural Elliott detector.
The source is stamped `RETROSPECTIVE_WEAK_SWING_LABELS` throughout.

To use actual expert labels, supply a CSV with exact hourly timestamps,
wave label (`bull_1` ... `bear_C`), `annotator`, `known_at` and
`label_status=expert_verified`. Require >=50 training labels. For conflicts
or later confirmations, retain multiple versions; independent review is
necessary. Do not automatically accept or forward-fill tentative GOX labels.

Snapshot once daily via `.github/workflows/elliott-research.yml` at 03:17 UTC.
The snapshot job stores JSON and SHA-256 manifests as per-run GitHub Actions
artifacts with 30-day retention, **not commits into a public reseller feed**.
This schedule activates only after the workflow lands on the default branch.
Run manually to capture on demand; source `asof` may predate capture
(weekends, market holidays). For durable historical access, move snapshots
into an authorized private archive under the source terms and access policy.

## Research limitations

1. Not all world's markets or architectures can be exhaustively compared.
2. Labels from an Elliott provider are opinions, not independent ground truth.
3. Retrospective zigzag labels are not proof of Elliott correctness.
4. No source is allowed to leak into a period before it was published/known.
5. Volume clock is an order-activity proxy, not observed information entropy.
6. Backtest uses next open execution, 13 bps per side, spot long-only. No
   investment recommendation. Selection by validation macro-F1 can choose a
   classifier with poor monetary performance.
7. An advanced next phase needs true timestamped order-book data, independent
   analyst consensus and validation across asset classes, eras, and regimes.
