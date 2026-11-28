#!/usr/bin/env python3
"""Compare three *predeclared* PETR4 daily long/flat algorithms:
A. Donchian price breakout with ATR buffer and close-based ATR trailing exit.
B. MACD EMA(12,35,5) with lagged Ibovespa OR Brent/BRL trend filter.
C. The same breakout as A, confirmed by an Elliott multiscale neural GRU.
   For C, evaluate chronological and volume-information clocks using the SAME
   GRU checkpoint (as a clock ablation; not a fourth strategy).

Source and calendar: pinned real PETR4/IBOV/BrentUSD/USDBRL daily dataset.
Training diagnostics 2014-2023, parameter selection 2024 only,
locked 2025-01-02 to 2026-11-17 test. Train one multiscale GRU, three heads
(wave, reversal, confidence), using weak retrospective swing targets. The
Elliott labels are NOT independently verified. The model's predictions in
2025-26 never receive future labels or candles as inputs.

Decision t close -> execute t+1 open, initial capital R$10,000, 13 bps/side.
Adjusted OHLC = dividend-adjusted TOTAL RETURN PROXY (not tradeable quotes).
No portfolio interest or tax. No intrabar ATR stop assumed: decisions are at
close and exits at next open. Trading universes are PETR4 long or cash, 1x.

WARNING: MACD rule was discovered examining this same 2025-26 sample in prior
research; thus its 2025-26 outperformance is NOT a pristine OOS discovery.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import itertools

import numpy as np
import pandas as pd
import torch

from compare_nn import (
    CLASSES, SCALES, LENGTH, seed_everything, factory, train, predict,
    features, information_clock, scores
)
from petr4_neural_backtest import weak_targets_by_split, reversal_labels
from petr4_mql_catalog import load_prices, indicators, run_sim
from petr4_intermarket_combinatorial import (
    fetch, URL_PRICES, URL_MACRO, SOURCE_COMMIT, merge_macro, derived_factors,
    technical_candidates, split_dates,
)

FEE_BPS=13.
SEED=31


def atr_close_breakout(adj, atr,lookback=20,buffer=.25,trailing=2.5):
    """Closed-day Donchian+ATR entry; dynamic trailing ATR exit at next open.

    The resistance is the maximum high of PREVIOUS n sessions. ATR is
    computed with candle t as of its close; the resulting decision is
    executed only on next session. Highest-high trail is updated with observed
    daily highs, not future prices. No intraday stop fill claims.
    """
    prices=np.asarray(adj,dtype=float)
    high=prices[:,1]
    close=prices[:,3]
    atr=np.asarray(atr,dtype=float)
    n=len(close)
    result=np.zeros(n,dtype=np.int8)
    in_position=False
    peak=0.
    for t in range(max(lookback,30),n):
        a=float(atr[t])
        if not np.isfinite(a) or a<=0:continue
        if in_position:
            peak=max(peak,float(high[t]))
            stop=peak-trailing*a
            if close[t]<stop:
                in_position=False
                peak=0.
        else:
            resistance=np.max(high[t-lookback:t])
            if close[t]>resistance+buffer*a:
                in_position=True
                peak=float(high[t])
        result[t]=int(in_position)
    return result


def validate_parameters(adj,ats,parts,fee):
    """ATR parameters selected SOLELY with pre-2025 training/validation."""
    records=[]
    for lookback,atr_n,buffer,trailing in itertools.product(
        (10,20,40,55),(7,14,21),(.0,.15,.35,.65),(1.5,2.5,3.5,4.5)):
        signal=atr_close_breakout(adj,ats[atr_n],lookback,buffer,trailing)
        tr=run_sim(adj,signal,*parts["train"],fee_bps=fee)
        va=run_sim(adj,signal,*parts["val"],fee_bps=fee)
        # Val-only objective. Require training/validation activity.
        eligible=tr["roundtrips"]>=8 and va["roundtrips"]>=2 and \
            4<=va["exposure_pct"]<=95
        objective=(va["return_pct"]-.4*va["drawdown_pct"]
                   -.08*va["roundtrips"]) if eligible else -1e12
        records.append({
            "lookback":lookback,"atr_period":atr_n,"buffer_atr":buffer,
            "trail_atr":trailing,"training_return_pct":tr["return_pct"],
            "validation_return_pct":va["return_pct"],
            "validation_drawdown_pct":va["drawdown_pct"],
            "validation_trades":va["roundtrips"],
            "validation_score":objective,
            "signal":signal,
        })
    valid=[r for r in records if r["validation_score"]>-1e11]
    if not valid:raise RuntimeError("No valid ATR breakout parameter selection")
    winner=max(valid,key=lambda r:(r["validation_score"],r["training_return_pct"]))
    return winner,records


def aligned_features(adjdf,axis,ids):
    x,used=features(adjdf,axis,ids)
    return x,used


def paired_windows(adjdf,axis_a,axis_b,ids):
    a,ia=aligned_features(adjdf,axis_a,ids)
    b,ib=aligned_features(adjdf,axis_b,ids)
    common=np.intersect1d(ia,ib)
    if len(common)<3:raise ValueError("No paired market windows")
    a=a[np.isin(ia,common)]
    b=b[np.isin(ib,common)]
    return a,b,common


def neural_head_gates(model,validation,testing,selected_breakout,
                      labels,rev,dates):
    """Probabilistic Elliott impulse confirmation calibrated without PnL.

    A fixed quantile (40th percentile) of validation *breakout* decisions
    determines an impulse-probability floor. Both clocks share the same
    cutoff. Reversal and reported-confidence thresholds are relaxed
    and predeclared; the neural head is weak, not investor-calibrated.
    """
    (a_val,b_val,ids_val)=validation
    (a_test,b_test,ids_test)=testing
    pval_a=predict(model,a_val)
    pval_b=predict(model,b_val)
    ptest_a=predict(model,a_test)
    ptest_b=predict(model,b_test)
    cls=[CLASSES.index(f"bull_{k}") for k in (1,3,5)]
    pulse_val=pval_a[0][:,cls].sum(axis=1)
    gate_pool=pulse_val[selected_breakout[ids_val]>0]
    if len(gate_pool)<3:gate_pool=pulse_val
    cutoff=float(np.quantile(gate_pool,.40))
    def mask(ids,pred):
        wave,reversal,conf=pred
        impulse=wave[:,cls].sum(axis=1)
        valid=(impulse>=cutoff)&(reversal<=.70)&(conf>=.10)
        # A sparse set of predictions is held for EXACTLY one session
        # because the model emits one prediction per trading-day close.
        values=np.zeros(len(dates),dtype=np.int8)
        values[ids]=valid.astype(np.int8)
        # Before any sampling, signal is false; here every test day
        # is included from test start onward, so no stale forward filling.
        return values,{"cutoff":cutoff,
          "mean_impulse_probability":float(np.mean(impulse)),
          "mean_reversal_probability":float(np.mean(reversal)),
          "mean_reported_confidence":float(np.mean(conf)),
          "confirmation_fraction":float(np.mean(valid)),
          "n_predictions":len(ids)}
    cal,st_a=mask(ids_test,ptest_a)
    info,st_b=mask(ids_test,ptest_b)
    val_scores={
       "calendar":scores(pval_a,labels[ids_val],rev[ids_val]),
       "information":scores(pval_b,labels[ids_val],rev[ids_val]),
    }
    test_scores={
       "calendar":scores(ptest_a,labels[ids_test],rev[ids_test]),
       "information":scores(ptest_b,labels[ids_test],rev[ids_test]),
    }
    return {"calendar":cal,"information":info},{
        "validation_quantile":.40,"cutoff":cutoff,
        "validation_wave_scores":val_scores,"test_wave_scores":test_scores,
        "calendar_stats":st_a,"information_stats":st_b,
        "wave_classes_for_gate":["bull_1","bull_3","bull_5"],
        "reversal_max":.70,"confidence_min":.10,
    },{"calendar":(ids_test,ptest_a),"information":(ids_test,ptest_b)}


def summarize(signal,adj,raw,parts,fee):
    adj_results={}
    for seg in ("val","test","y2025","y2026","previous_comparable"):
        adj_results[seg]=run_sim(adj,signal,*parts[seg],fee_bps=fee)
    rawtest=run_sim(raw,signal,*parts["test"],fee_bps=fee)
    def compact(x):
        return {k:float(v) if isinstance(v,(np.floating,np.integer))
                else v for k,v in x.items() if k!="equity"}
    return {"adjusted_proxy":{k:compact(v) for k,v in adj_results.items()},
            "raw_trade_prices_test":compact(rawtest)},adj_results["test"]


def block_bootstrap_diff(base,benchmark,reps=2000,block=10,seed=13):
    """Retrospective 10-session stationary circular block bootstrap."""
    a=np.diff(np.log(np.r_[10000,base["equity"]]))
    b=np.diff(np.log(np.r_[10000,benchmark["equity"]]))
    diff=a-b
    rng=np.random.default_rng(seed)
    n=len(diff)
    vals=np.empty(reps)
    for j in range(reps):
        chosen=[]
        while len(chosen)<n:
            pos=int(rng.integers(0,n))
            chosen.extend((pos+np.arange(block))%n)
        vals[j]=100*float(diff[np.asarray(chosen[:n])].sum())
    return {"observed_log_excess_pct":100*float(diff.sum()),
            "CI95_10day_blocks_log_excess_pct":np.quantile(vals,[.025,.975]).tolist(),
            "repetitions":reps,"block_sessions":block}


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    seed_everything(SEED)
    raw_file=args.prices or fetch(URL_PRICES,args.out/"source_petr4.csv")
    macro_file=args.macro or fetch(URL_MACRO,args.out/"source_macro.csv")
    df,raw,adj,bad=load_prices(raw_file)
    df,raw,adj,macro=merge_macro(df,raw,adj,macro_file)
    segments=split_dates(df.timestamp.dt.tz_localize(None))
    ts=indicators(df,adj)
    print("MARKET",len(df),"sessions,",
          df.timestamp.iloc[0].date(),"to",df.timestamp.iloc[-1].date(),
          "excluded",bad,flush=True)

    # A: historical ATR breakout: select on 2024 only
    atr_choice,atr_trials=validate_parameters(adj,ts["atr"],segments,args.fee)
    atr_signal=atr_choice["signal"].astype(bool)
    atr_cfg={k:v for k,v in atr_choice.items() if k!="signal"}
    print("ATR_WINNER_VALIDATION",json.dumps(atr_cfg),flush=True)

    # B: fixed, previously researched MACD + lagged IBOV/BrentBRL gate.
    # NOTE: MACD formula and gate were selected in research that viewed
    # 2025-26, so OOS status is contaminated for the B model.
    technical=technical_candidates(df,adj)
    name="MACD(12,35,5)"
    if name not in technical:raise ValueError("MACD baseline not available")
    factors,_,_=derived_factors(macro,adj)
    for key in ("ibov_close_sma50_up","brent_brl_sma50_up"):
        if key not in factors:raise ValueError("Missing intermarket gate "+key)
    macro_mask=factors["ibov_close_sma50_up"] | factors["brent_brl_sma50_up"]
    macd_signal=technical[name][1]&macro_mask

    # C: retrain one actual GRU multiscale, with all features causal.
    close_df=pd.DataFrame({
      "timestamp":df.timestamp,
      "open":adj[:,0],"high":adj[:,1],"low":adj[:,2],
      "close":adj[:,3],"volume":df.Volume.to_numpy(dtype=float)})
    i_train=segments["val"][0]
    i_test=segments["test"][0]
    labels=weak_targets_by_split(close_df,i_train,i_test)
    reversal=reversal_labels(close_df,i_train,i_test)
    chronological=np.arange(len(close_df),dtype=float)
    informational,speed=information_clock(close_df.volume.to_numpy(),alpha=.8)
    train_ids=np.arange(segments["train"][0]+170,segments["train"][1]-10,3)
    val_ids=np.arange(segments["val"][0],segments["val"][1]+1)
    test_ids=np.arange(segments["test"][0]-1,segments["test"][1]+1)
    x_train_a,x_train_b,idtr=paired_windows(close_df,chronological,informational,train_ids)
    x_val_a,x_val_b,idval=paired_windows(close_df,chronological,informational,val_ids)
    x_test_a,x_test_b,idte=paired_windows(close_df,chronological,informational,test_ids)
    model=factory("gru_multi")
    model=train(model,np.concatenate([x_train_a,x_train_b]),np.tile(labels[idtr],2),
        np.tile(reversal[idtr],2),np.concatenate([x_val_a,x_val_b]),
        np.tile(labels[idval],2),np.tile(reversal[idval],2),
        epochs=args.epochs,seed=SEED)
    ck=args.out/"petr4_multiscale_gru.pt"
    torch.save({"state_dict":model.state_dict(),"classes":CLASSES,
        "architecture":"gru_multi","train":"2014-2023","val":"2024",
        "label_status":"retrospective_swing_pseudolabels"},ck)
    weight_sha=hashlib.sha256(ck.read_bytes()).hexdigest()
    gates,nn_meta,forecasts=neural_head_gates(model,(x_val_a,x_val_b,idval),
        (x_test_a,x_test_b,idte),atr_signal,labels,reversal,df.timestamp)
    neural_calendar=atr_signal & gates["calendar"].astype(bool)
    neural_information=atr_signal & gates["information"].astype(bool)

    variants={
      "ATR_breakout":atr_signal,
      "MACD_intermarket":macd_signal,
      "ATR_breakout_plus_Elliott_neural_calendar":neural_calendar,
      "ATR_breakout_plus_Elliott_neural_information":neural_information,
      "buy_hold":np.ones(len(df),dtype=bool),
    }
    full={};paths={}
    for name,signal in variants.items():
        row,curve=summarize(signal,adj,raw,segments,args.fee)
        full[name]=row
        lo,hi=segments["test"]
        data=pd.DataFrame({"date":df.timestamp.iloc[lo:hi+1].dt.strftime("%Y-%m-%d").to_numpy(),
                           "equity":curve["equity"],
                           "signal_from_prior_close":signal[lo-1:hi].astype(int)})
        path=args.out/("equity_"+name+".csv")
        data.to_csv(path,index=False)
        paths[name]=str(path.name)
        print("RESULT",name,"total",row["adjusted_proxy"]["test"]["return_pct"],
              "2025",row["adjusted_proxy"]["y2025"]["return_pct"],
              "2026",row["adjusted_proxy"]["y2026"]["return_pct"],flush=True)

    # Pred export for dashboard. Crucial: source day D1 signal only available
    # after session i ends; available_at encoded as following session date.
    rows=[]
    for clock,(ids,pr) in forecasts.items():
        wave,rev,conf=pr
        for j,i in enumerate(ids):
            if i+1>=len(df):continue
            rows.append({"clock":clock,"timestamp":df.timestamp.iloc[i].isoformat(),
             "known_at":df.timestamp.iloc[i+1].isoformat(),
             "weight_sha256":weight_sha,
             "p_bull_3":float(wave[j,CLASSES.index("bull_3")]),
             "p_reversal":float(rev[j]),"confidence":float(conf[j]),
             "p_bull_impulse_135":float(wave[j,[CLASSES.index(f"bull_{k}") for k in (1,3,5)]].sum())})
    pd.DataFrame(rows).to_csv(args.out/"paired_neural_predictions.csv",index=False)

    bh=run_sim(adj,variants["buy_hold"],*segments["test"],fee_bps=args.fee)
    intervals={}
    for key in variants:
        if key=="buy_hold":continue
        curve=run_sim(adj,variants[key],*segments["test"],fee_bps=args.fee)
        intervals[key]=block_bootstrap_diff(curve,bh)
    report={
      "schema":"PETR4_THREE_MODELS_BREAKOUT_MACD_NEURAL_V1",
      "date_range":{
        k:[str(df.timestamp.iloc[lo].date()),str(df.timestamp.iloc[hi].date())]
        for k,(lo,hi) in segments.items()},
      "data_provenance":{"source_commit":SOURCE_COMMIT,
       "petrobras_sha256":hashlib.sha256(pathlib.Path(raw_file).read_bytes()).hexdigest(),
       "macro_sha256":hashlib.sha256(pathlib.Path(macro_file).read_bytes()).hexdigest(),
       "source_ohlc_excluded_dates":bad},
      "cost_bps_per_side":args.fee,"initial_capital_brl":10000,
      "source_neural":"retrained GRU multiscale on 2014-2023; validation 2024",
      "neural_checkpoint_sha256":weight_sha,
      "neural_weak_labels_not_expert_verified":True,
      "neural_gate":nn_meta,
      "atr_selection_by_2024":atr_cfg,
      "atr_candidates_by_2024":{
         "count":len(atr_trials),
         "top_10":[{k:v for k,v in rec.items() if k!="signal"}
          for rec in sorted(atr_trials,key=lambda z:z["validation_score"],reverse=True)[:10]]},
      "macd_parameters":{"fast_ema":12,"slow_ema":35,"signal_ema":5,
         "gate":"IBOV_close > trailing_sma50 OR BrentBRL > trailing_sma50",
         "exogenous_lag_trading_sessions":1,
         "selection_warning":"MACD params previously selected after looking at 2025-26; NOT fresh OOS"},
      "result":full,
      "bootstrap_versus_buy_hold":intervals,
      "equity_csvs":paths,
      "notes":[
        "ATR breakout and its trailing exit are decided on closing price, filled next B3 open; no intraday stop assumptions.",
        "Adj Close creates a total-return price PROXY; dividends not tracked as event cash flows, prices not executable.",
        "No lookahead in neural model inputs; label pseudo-truth is retrospective and used only train/validation.",
        "Only 2024 validation used for ATR parameter selection and neural quantile gate; no 2025-26 optimization in new ATR models.",
        "MACD previously highlighted due to 2025-26 retrospective search; future validity unproven.",
        "Cash earns 0% (no CDI), no taxes, spread/latency/order-book fills/odd-lot constraints.",
        "Forecast versions calendar and informational share exact same trained GRU checkpoint.",
        "A/B comparability is per model and identical fee/splits; neural information clock based on volume only."
      ]
    }
    (args.out/"report.json").write_text(
      json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    simplified={}
    for name,values in full.items():
        a=values["adjusted_proxy"]
        simplified[name]={"2025":a["y2025"]["return_pct"],
                          "2026":a["y2026"]["return_pct"],
                          "total":a["test"]["return_pct"],
                          "raw_total":values["raw_trade_prices_test"]["return_pct"],
                          "drawdown":a["test"]["drawdown_pct"],
                          "sharpe":a["test"]["sharpe"],
                          "trades":a["test"]["roundtrips"],
                          "exposure":a["test"]["exposure_pct"]}
    print("THREE_MODELS_RESULTS_BEGIN",flush=True)
    print(json.dumps({
      "period":report["date_range"]["test"],
      "n_sessions":segments["test"][1]-segments["test"][0]+1,
      "atr_cfg":atr_cfg,"neural_gate":nn_meta,"models":simplified,
      "CI95_vs_BH":intervals},ensure_ascii=False,indent=2),flush=True)
    print("THREE_MODELS_RESULTS_END",flush=True)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=pathlib.Path,default=pathlib.Path("research/three_model_runs"))
    p.add_argument("--prices",type=pathlib.Path,default=None)
    p.add_argument("--macro",type=pathlib.Path,default=None)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--fee",type=float,default=FEE_BPS)
    a=p.parse_args()
    if a.epochs<1 or a.fee<0:raise ValueError("invalid training epochs or fee")
    return a

if __name__=="__main__":
    main(parse_args())
