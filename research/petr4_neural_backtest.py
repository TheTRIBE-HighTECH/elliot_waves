#!/usr/bin/env python3
"""PETR4 B3 daily OHLCV: causal neural A/B with chronological vs volume clock.

Adapts the EXISTING repo's MLP/TCN/GRU, 17 Elliott state + reversal +
confidence heads, and multiscale features for irregular Brazilian sessions.

Source: https://github.com/msbrasil/Series-Temporais-PETR4
    data/raw/petr4_yf_raw.csv (Yahoo-derived data, third-party archive).
    Source includes Adj Close for dividend/corporate action adjustment.
Training labels are WEAK retrospective swings, NOT verified Elliott counts.
No actual PETR4 news-reply graph is available: the alternate clock is a
trading-volume proxy, never claim it measures observed news propagation.

2014-2023 train; 2024 validation; 2025-2026 holdout (to source end).
No optimization on holdout. Same fitted weights used for both clocks.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, urllib.request
import numpy as np
import pandas as pd
import torch

from compare_nn import (
    CLASSES, SCALES, LENGTH, seed_everything,
    factory, train, predict, features, information_clock,
    weak_swing_targets, f1_macro, scores,
)

SOURCE = ("https://raw.githubusercontent.com/msbrasil/"
          "Series-Temporais-PETR4/main/data/raw/petr4_yf_raw.csv")
TRAIN_START="2014-01-01"
TRAIN_END="2024-01-01"
VAL_END="2025-01-01"
COST_BPS=13.0

def download(out):
    out=pathlib.Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(SOURCE, headers={"User-Agent":"elliott-petr4-b3-research/1.0"})
    with urllib.request.urlopen(req,timeout=90) as response:raw=response.read()
    if len(raw)<10000:raise RuntimeError("Unexpectedly small public dataset")
    out.write_bytes(raw)
    return out

def load(path):
    df=pd.read_csv(path)
    required={"Date","Adj Close","Open","High","Low","Close","Volume"}
    if not required.issubset(df.columns):raise ValueError(f"missing {required-set(df.columns)}")
    df["timestamp"]=pd.to_datetime(df.Date,utc=True,errors="raise")
    df=df.sort_values("timestamp").drop_duplicates("timestamp")
    for col in ("Adj Close","Open","High","Low","Close","Volume"):
        df[col]=pd.to_numeric(df[col],errors="coerce")
    df=df.dropna(subset=["Adj Close","Open","High","Low","Close","Volume"])
    df=df[(df.timestamp>=pd.Timestamp("2013-01-01",tz="UTC")) &
          (df.timestamp<=pd.Timestamp("2026-11-27",tz="UTC"))]
    df=df.reset_index(drop=True)
    if len(df)<2600:raise ValueError(f"too few PETR4 rows: {len(df)}")
    if not df.timestamp.is_monotonic_increasing or df.timestamp.duplicated().any():
        raise ValueError("bad daily timestamps")
    numeric=df[["Adj Close","Open","High","Low","Close","Volume"]]
    if not np.isfinite(numeric.to_numpy()).all():raise ValueError("NaN or infinite")
    if (numeric[["Adj Close","Open","High","Low","Close"]]<=0).any().any():
        raise ValueError("invalid nonpositive prices")
    if (numeric.Volume<0).any():raise ValueError("invalid volume")
    if ((df.High+1e-6<df[["Open","Close","Low"]].max(axis=1))|
        (df.Low-1e-6>df[["Open","Close","High"]].min(axis=1))).any():
        raise ValueError("bad high-low consistency")
    # Two perspectives: unadjusted raw market trade prices and a total-return
    # adjusted proxy. Adjusted prices are not directly executable quotations.
    raw=pd.DataFrame({"timestamp":df.timestamp,"open":df.Open,
      "high":df.High,"low":df.Low,"close":df.Close,"volume":df.Volume})
    factor=df["Adj Close"].to_numpy()/df.Close.to_numpy()
    adjusted=raw.copy()
    for col in ("open","high","low","close"):
        adjusted[col]=raw[col].to_numpy()*factor
    return raw,adjusted,df

def weak_targets_by_split(df,t1,t2):
    out=np.zeros(len(df),dtype=np.int64)
    for a,b in ((0,t1),(t1,t2),(t2,len(df))):
        out[a:b]=weak_swing_targets(df.close.iloc[a:b].to_numpy(),threshold=.025)
    return out

def reversal_labels(df,t1,t2,lookback=8,forward=8,threshold=.012):
    p=np.log(df.close.to_numpy(dtype=float))
    out=np.full(len(df),-1,dtype=np.int64)
    for a,b in ((0,t1),(t1,t2),(t2,len(df))):
        for i in range(a+lookback,b-forward):
            direction=np.sign(p[i]-p[i-lookback])
            if direction:
                out[i]=int(np.any((p[i+1:i+forward+1]-p[i])*direction<-threshold))
    return out

def sample_features(df,axis,index):
    x,used=features(df,axis,index)
    return x,used

def score_pair(prob,labels,reversal):
    return scores(prob,labels,reversal)

def execute(df,idx,pred,fee_bps=COST_BPS,threshold=.20,rev_max=.55,confidence_min=.15):
    """Long only; signals at session i close, fill next trading session open.

    Predictions generated every 3rd session are held until the next sampled
    signal. No closing and reopening every intermediate session.
    No leverage, stop/take, dividends, taxes, or borrow.
    """
    wave,rev,conf=pred
    desired=((wave[:,CLASSES.index("bull_3")]>=threshold)&
             (rev<=rev_max)&(conf>=confidence_min))
    mapping={int(i):bool(s) for i,s in zip(idx,desired)}
    start,end=int(idx[0]),int(idx[-1])
    cash,qty,entry=10000.,0.,0.
    peak,dd=10000.,0.
    wins,losses,pos,neg=0,0,0.,0.
    trades=[];equity=[];exposure=0;prev_signal=False
    fee=fee_bps/10000.
    for i in range(start,end+1):
        if i-1 in mapping:prev_signal=mapping[i-1]
        need=prev_signal
        op=float(df.open.iloc[i])
        if qty and not need:
            value=qty*op*(1-fee);pnl=value-entry
            cash=value;qty=0.
            pos+=max(0,pnl);neg+=max(0,-pnl)
            wins+=int(pnl>0);losses+=int(pnl<=0)
            trades.append({"date":str(df.timestamp.iloc[i].date()),"pnl":pnl,"reason":"signal_exit"})
        elif not qty and need:
            entry=cash;qty=cash/(op*(1+fee));cash=0.
        val=cash+qty*float(df.close.iloc[i])
        peak=max(peak,val);dd=max(dd,1-val/peak)
        exposure+=int(qty>0)
        equity.append({"date":str(df.timestamp.iloc[i].date()),"value":val})
    if qty:
        value=qty*float(df.close.iloc[end])*(1-fee)
        pnl=value-entry
        pos+=max(0,pnl);neg+=max(0,-pnl)
        wins+=int(pnl>0);losses+=int(pnl<=0)
        cash=value;equity[-1]["value"]=cash
        trades.append({"date":str(df.timestamp.iloc[end].date()),"pnl":pnl,"reason":"end"})
    ret_pct=100*(cash/10000-1)
    day_equity=np.array([10000.]+[x["value"] for x in equity])
    r=np.diff(np.log(day_equity))
    sr=np.std(r,ddof=1) if len(r)>2 else 0.
    sharpe=math.sqrt(252)*np.mean(r)/sr if sr>1e-10 else None
    return {"ret_pct":float(ret_pct),"end_brl":float(cash),
        "max_drawdown_pct":float(100*dd),
        "sharpe":float(sharpe) if sharpe is not None else None,
        "round_trips":len(trades),"exposure_pct":100*exposure/len(equity),
        "win_rate_pct":100*wins/max(1,len(trades)),
        "profit_factor":float(pos/neg) if neg>0 else None,
        "trades":trades,"equity":equity}

def benchmark_hold(df,start,end,fee_bps=COST_BPS):
    # Buy on session start open, exit end close, one round trip, no dividends
    first=float(df.open.iloc[start]);last=float(df.close.iloc[end])
    q=10000/(first*(1+fee_bps/10000))
    return 100*(q*last*(1-fee_bps/10000)/10000-1)

def bootstrap_pair(a,b,reps=2000,block=10,seed=20261127):
    """Paired block bootstrap of DAILY equity log returns with both clocks."""
    da=np.diff(np.log(np.asarray([10000.]+[x["value"] for x in a["equity"]])))
    db=np.diff(np.log(np.asarray([10000.]+[x["value"] for x in b["equity"]])))
    if len(da)!=len(db):raise ValueError("paired equity dates differ")
    d=db-da;n=len(d);rng=np.random.default_rng(seed)
    bs=np.empty(reps)
    for j in range(reps):
        indices=[]
        while len(indices)<n:
            st=int(rng.integers(0,n))
            indices.extend((st+np.arange(block))%n)
        bs[j]=float(d[np.array(indices[:n],dtype=int)].sum())
    # Sum log-relative performance; scale to pp of log wealth.
    return {"log_return_difference_pct":100*float(d.sum()),
            "ci95_block_bootstrap_log_pct":(100*np.quantile(bs,[.025,.975])).tolist(),
            "block_sessions":block,"replicates":reps}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",default=None)
    p.add_argument("--out",type=pathlib.Path,default=pathlib.Path("research/petr4_runs"))
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--alpha",type=float,default=.8)
    p.add_argument("--cost-bps",type=float,default=13.)
    p.add_argument("--seed",type=int,default=31)
    args=p.parse_args()
    seed_everything(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    src=pathlib.Path(args.csv) if args.csv else download(args.out/"source_petr4_yf_raw.csv")
    market,adj,raw=load(src)
    # Use adjusted price shapes to avoid fitting dividend drops as wave crashes;
    # the test PnL is reported both raw and adjusted proxy. All features are
    # built from historical windows up to the given decision date.
    t=market.timestamp
    i_train=int(t.searchsorted(pd.Timestamp(TRAIN_END,tz="UTC")))
    i_val=int(t.searchsorted(pd.Timestamp(VAL_END,tz="UTC")))
    i_start=int(t.searchsorted(pd.Timestamp(TRAIN_START,tz="UTC")))
    if min(i_train-i_start,i_val-i_train,len(t)-i_val)<180:
        raise ValueError("bad time splits")
    labels=weak_targets_by_split(adj,i_train,i_val)
    reversal=reversal_labels(adj,i_train,i_val)
    calendar=np.arange(len(market),dtype=float)
    information,speed=information_clock(market.volume.to_numpy(),args.alpha)
    if not np.all(np.diff(information)>0):
        raise AssertionError("information clock non-monotone")
    index={
      "train":np.arange(i_start+400,i_train-9,3,dtype=int),
      "validation":np.arange(i_train+15,i_val-9,2,dtype=int),
      "test":np.arange(i_val+15,len(t)-9,2,dtype=int),
    }
    paired={}
    for name,idx in index.items():
        c,ci=sample_features(adj,calendar,idx)
        inf,ii=sample_features(adj,information,idx)
        common=np.intersect1d(ci,ii)
        paired[name]=(c[np.isin(ci,common)],inf[np.isin(ii,common)],common)
    ct,it,train_idx=paired["train"]
    cv,iv,val_idx=paired["validation"]
    co,io,test_idx=paired["test"]
    if min(map(len,(train_idx,val_idx,test_idx)))<60:
        raise ValueError("insufficient samples in split")
    xtrain=np.concatenate([ct,it],axis=0)
    ytrain=np.tile(labels[train_idx],2);rtrain=np.tile(reversal[train_idx],2)
    xval=np.concatenate([cv,iv],axis=0)
    yval=np.tile(labels[val_idx],2);rval=np.tile(reversal[val_idx],2)
    candidates={}
    trained={}
    for model_name in ("mlp","tcn_single","tcn_multi","gru_multi"):
        print("TRAIN",model_name,"samples",len(xtrain),flush=True)
        seed_everything(args.seed)
        model=factory(model_name)
        model=train(model,xtrain,ytrain,rtrain,xval,yval,rval,epochs=args.epochs,seed=args.seed)
        val_preds={"calendar":predict(model,cv),"information":predict(model,iv)}
        test_preds={"calendar":predict(model,co),"information":predict(model,io)}
        val_metrics={name:score_pair(pred,labels[val_idx],reversal[val_idx])
                     for name,pred in val_preds.items()}
        test_metrics={name:score_pair(pred,labels[test_idx],reversal[test_idx])
                      for name,pred in test_preds.items()}
        selection=float(np.mean([v["macro_f1_labeled"] or 0 for v in val_metrics.values()]))
        results={}
        for mode,pred in test_preds.items():
            results[mode]={
              "raw_trade_prices":execute(market,test_idx,pred,args.cost_bps),
              "dividend_adjusted_proxy":execute(adj,test_idx,pred,args.cost_bps)}
        candidates[model_name]={
           "validation":val_metrics,"test":test_metrics,
           "selection_validation_macro_f1":selection,
           "test_backtest":results}
        trained[model_name]=(model,test_preds)
    winner=max(candidates,key=lambda m:candidates[m]["selection_validation_macro_f1"])
    model,selected_pred=trained[winner]
    checkpoint=args.out/"petr4_best_by_validation.pt"
    torch.save({"state_dict":model.state_dict(),"architecture":winner,
                "classes":CLASSES,"scales":SCALES,
                "label_status":"retrospective_weak_swings"},checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    forecast_rows=[]
    for mode,pred in selected_pred.items():
        wave,rev,conf=pred
        for k,i in enumerate(test_idx):
            # Daily bar time convention: timestamp = SESSION DATE 00:00 UTC.
            # Known only from NEXT trading session. Execution then uses next open.
            forecast_rows.append({
             "clock":mode,"timestamp":t.iloc[i].isoformat(),
             "known_at":t.iloc[i+1].isoformat(),"weight_sha256":digest,
             "p_bull_3":float(wave[k,CLASSES.index("bull_3")]),
             "p_reversal":float(rev[k]),"confidence":float(conf[k])})
    pd.DataFrame(forecast_rows).to_csv(args.out/"petr4_neural_paired_forecasts.csv",index=False)
    selected=candidates[winner]["test_backtest"]
    for mode in ("calendar","information"):
        for price_mode in ("raw_trade_prices","dividend_adjusted_proxy"):
            pd.DataFrame(selected[mode][price_mode]["equity"]).to_csv(
                args.out/f"equity_{mode}_{price_mode}.csv",index=False)
            pd.DataFrame(selected[mode][price_mode]["trades"]).to_csv(
                args.out/f"trades_{mode}_{price_mode}.csv",index=False)
    ci=bootstrap_pair(selected["calendar"]["raw_trade_prices"],
                      selected["information"]["raw_trade_prices"])
    def strip(o):
        if isinstance(o,dict):return {k:strip(v) for k,v in o.items() if k not in ("trades","equity")}
        if isinstance(o,list):return [strip(x) for x in o]
        return o
    report={
      "asset":"PETR4","market":"B3 Brazil", "timeframe":"daily_trading_session",
      "source_url":SOURCE,
      "source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
      "raw_count":len(raw),"date_first":str(t.iloc[0].date()),
      "date_last":str(t.iloc[-1].date()),
      "train_window":"2014–2023","validation_window":"2024",
      "holdout_window":"2025–2026 to latest source date",
      "rows_train":len(train_idx),"rows_validation":len(val_idx),
      "rows_test":len(test_idx),
      "cost_bps_per_side":args.cost_bps,"alpha_information_clock":args.alpha,
      "clock":"volume trading ACTIVITY; not measured social diffusion or news graph",
      "label_status":"RETROSPECTIVE_WEAK_SWING_PSEUDO_LABELS_NOT_VERIFIED_ELLIOTT",
      "winner_by_validation_only":winner,"weight_sha256":digest,
      "baseline_buy_hold_raw_return_pct":benchmark_hold(market,int(test_idx[0]),int(test_idx[-1]),args.cost_bps),
      "baseline_buy_hold_adjusted_proxy_return_pct":benchmark_hold(adj,int(test_idx[0]),int(test_idx[-1]),args.cost_bps),
      "bootstrap_raw_log_relative_return":ci,
      "models":strip(candidates),
      "notes":[
         "Same model weight checkpoint for both clocks, fitted on both training domains.",
         "Architectural selection uses 2024 only; 2025-2026 untouched for fitting.",
         "Reference is a third-party Yahoo-derived file, not primary exchange feed.",
         "The OHLC adjusted proxy approximates cash-dividend reinvestment; its prices are not executable.",
         "Raw-trade-price PnL excludes cash dividends, borrow, taxes, Brazilian exchange cost schedules, and financing.",
         "No real Petrobras social reply graph was available; only volume clock is tested.",
         "Wave labels are retrospective swing heuristics, not verified expert Elliott counts.",
         "Historical stock market performance does not establish future profitability."
       ]
    }
    (args.out/"petr4_backtest_report.json").write_text(
      json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    print("PETR4 BACKTEST RESULT:")
    summary={"source_first_last":[report["date_first"],report["date_last"]],
      "winning_model":winner,"selection_validation_f1":
       candidates[winner]["selection_validation_macro_f1"],
      "test_classification":candidates[winner]["test"],
      "test_brokerage":strip(selected),
      "buy_and_hold_raw":report["baseline_buy_hold_raw_return_pct"],
      "buy_and_hold_adj_proxy":report["baseline_buy_hold_adjusted_proxy_return_pct"],
      "bootstrap":ci,"n_train_val_test":[len(train_idx),len(val_idx),len(test_idx)]}
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
