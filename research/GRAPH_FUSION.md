# Real temporal discussion-graph fusion (research v4)

This increment **integrates observed reply-tree edges** into the experimental
multiscale Elliott neural detector. It uses the public Hacker News Algolia
search and item API (original Hacker News records: <https://github.com/HackerNews/API>).
The source describes actual **parent→reply** relationships; it does **not**
measure who saw, believed, or traded on a post, and is not a complete market
information graph. It is a real empirical, narrow-scope graph pilot.

## Why these data?

* Links are **observed**: no synthetic graph topology is inferred.
* Each record has an original UTC publication timestamp and a source story.
* We do not collect text, usernames, or points/votes.
* Markets: BTCUSDT, ETHUSDT, SOLUSDT story topics; optional COVID stories to
  study March 2020 discussion cascades.
* Internet archive search is **retrospective**. The 2026 search index can differ
  from what traders saw in 2024, even though each comment timestamp is genuine.
  This introduces possible story-selection/survivorship bias. Claims of
  causal market influence would be scientifically unjustified.

## Execute

```bash
pip install -r research/requirements.txt

python research/download_binance.py \
  --symbols BTCUSDT --start 2024-01 --end 2025-12 \
  --out research/datasets

python research/hn_reply_graph.py \
  --assets BTCUSDT --start 2024-01 --end 2025-12 \
  --stories-month 3 --max-nodes-story 200 --out research/datasets/hn_graph

python research/graph_fusion.py \
  --csv research/datasets/BTCUSDT_1h.csv \
  --events research/datasets/hn_graph/hn_reply_events.csv \
  --asset BTCUSDT --epochs 8 --out research/runs/graph_fusion_BTC
```

COVID-19 (independent event-graph collection, not yet paired with equities):
```bash
python research/hn_reply_graph.py --assets COVID \
  --start 2020-02 --end 2020-06 --stories-month 5 \
  --out research/datasets/hn_covid_2020
```

On GitHub Actions the graph-fusion benchmark job retrieves and trains on
actual public OHLCV and HN event edges, produces a report and two model files
(and preserves source raw edges with manifest/hashes). No fabricated model
metrics are committed. Artifacts expire after 30 days.

## Neural architecture

```
                     Chronological or informational price windows (3 scales)
                                      |
                              causal TCN price tower
                                      |
                               latent P[64] ----+
                                                |
HN story/comment explicit reply edges      gated fusion ---> wave head 17 states
       |                       |                |---> reversal probability
   timestamped hourly node counts          graph G[64] ---> confidence head
       + directed parent-hour->reply-hour adjacencies
       |                       |
       +------ 2-layer directed temporal message passing ----+
```

At each decision candle, all HN events and edges are filtered on event
timestamps <= candle close. Parent edges flow from older/equal to newer bins;
self-loops avoid empty-row numerical problems. The 3 graphical scales follow
Fibonacci-like spacing (1,3,8 market/information hours). Source changes
arriving in a later candle must NEVER change an earlier graph feature.

The training benchmark uses two regimes with identical architecture, seed,
features and splits: (A) price tower only and (B) price + real reply graph.
For each trained model, both time clocks are evaluated using **identical
weights**. The graph-trained model is additionally ablated with its graph
tower disabled at test time (diagnostic only: distribution shift).

## Scientific protocol

* 2024: training; 2025-H1: validation; 2025-H2: untouched holdout.
* OHLCV genuine, but the initial Elliott wave labels are **retrospective swing
  pseudo-labels**, not expert-reviewed Elliott. DO NOT claim high confidence in
  wave identity from this weak supervision.
* Reversal target uses future prices only as a *training label*, never as model
  input or a backtest signal.
* Execution: decision from closed candle, order at next open, 13 basis points
  per side, no leverage.
* Hyperparameters are fixed in the pilot; model selection must be on
  validation and subsequently rerun across independent assets/time periods.
* Public HN news selection can be sparse or biased. Fail closed on zero actual
  graph reply edges. Better future data: chronological archives of cross-site
  news shares, Reddit reply trees, Bluesky repost chains, and market order-book
  messages, with provenance, licenses and as-of collection.

## Tests

```bash
python -m unittest discover -s research -p 'test_*.py' -v
```

Tests cover explicit parent links, time-window filtering, prefix-invariant
historical graph features, correct directed edge adjacency and NN fusion
ablation. They should pass before treating any backtest as research evidence.

## What this version does NOT prove

This HN graph is a small proxy for discussion structure, not real all-population
information propagation, social contagion, actual investor beliefs, or a
Fibonacci mechanism. It cannot by itself establish a relation between social
branching and price moves. That requires better event coverage, independent
wave labels, negative controls and statistical testing.
