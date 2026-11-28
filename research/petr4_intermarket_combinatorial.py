#!/usr/bin/env python3
"""PETR4 long/flat, cross-market combinatorial backtest on real daily historical data.

Purpose: screen publicly described MQL5-style technical systems combined with
observable cross-market factors (Brent USD and BRL, Ibovespa, USD/BRL, Selic),
including rolling Pearson correlations, lead/lag, trend and residual regimes.

NO MQL5 commercial EA executables are run. No perfect in/out optimization
is presented as a prospective result. Selection on 2024, test Jan 2025 -
Nov 2026, with separate ex-post ranking explicitly contaminated by selection
on that same period.

All intermarket features are shifted at least one full PETR4 session before
the signal on session t closes and executes the next session's opening.
Use only closed historical PETR4 prices through t. No intraday execution,
future dividends, or contemporaneous global-EOD market data.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import pathlib
import urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd

from petr4_mql_catalog import load_prices, indicators, families, run_sim

SOURCE_COMMIT="30e18599e1283ccb77a152fce55d3d5dea1f646a"
ROOT="https://raw.githubusercontent.com/msbrasil/Series-Temporais-PETR4/"+SOURCE_COMMIT+"/data/"
URL_PRICES=ROOT+"raw/petr4_yf_raw.csv"
URL_MACRO=ROOT+"processed/petr4_macro_daily.csv"

def fetch(url,path):
    path=pathlib.Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(url,headers={"User-Agent":"ElliottLab-quant-research/1.0"})
    with urllib.request.urlopen(req,timeout=90) as r:contents=r.read()
    if len(contents)<100000:raise ValueError("Too few source bytes: "+url)
    path.write_bytes(contents)
    return path

def merge_macro(df,raw,adj,macro_file):
    """Exact B3 calendar inner join, timestamp UTC date keys, no bfill."""
    m=pd.read_csv(macro_file)
    required={"date","petr4_adj_close","ibov_close","brent_usd","usd_brl",
              "brent_brl","selic_daily_log"}
    if not required.issubset(m.columns):
        raise ValueError("Missing macro columns: "+str(required-set(m.columns)))
    m["date"]=pd.to_datetime(m["date"],utc=True)
    if m.date.duplicated().any():raise ValueError("Macro source duplicate dates")
    ix=pd.Index(df.timestamp)
    available=ix.isin(m.date)
    if np.count_nonzero(available)<2300:raise ValueError("Insufficient joint calendar coverage")
    source=df.loc[available].copy().reset_index(drop=True)
    raw=raw[available];adj=adj[available]
    merged=source[["timestamp"]].merge(m,how="left",left_on="timestamp",right_on="date",
                                      validate="one_to_one")
    for key in required-{"date"}:
        merged[key]=pd.to_numeric(merged[key],errors="coerce")
    if merged[list(required-{"date"})].isna().any().any():
        raise ValueError("Macro has missing values on supposed joint days")
    if ((merged[["ibov_close","brent_usd","usd_brl","brent_brl"]]<=0).any().any()):
        raise ValueError("Invalid non-positive market factors")
    # Cross-check processed dataset has same source PETR4 adjusted series.
    diff=np.abs(merged.petr4_adj_close.to_numpy()-adj[:,3])
    if np.percentile(diff,95)>0.02:
        raise ValueError("Source mismatch between adjusted PETR4 and macro calendar")
    return source,raw,adj,merged

def pct_momentum(x,lag):
    return pd.Series(x,dtype=float).pct_change(lag).to_numpy(dtype=float)

def derived_factors(macro,adj):
    """All factor availability delayed by 1 complete session.

    E.g., Brent close on date t is only used to decide at PETR4 close t+1
    and enter at session t+2 open. This is intentionally conservative.
    """
    idx=macro.index
    keys=["ibov_close","brent_usd","brent_brl","usd_brl"]
    series={k:pd.Series(macro[k].to_numpy(dtype=float),index=idx).shift(1) for k in keys}
    # PETR4 self return is lagged so correlation is on matched close-to-close
    # returns that were already available on the previous B3 session.
    petret=pd.Series(adj[:,3],index=idx).pct_change().shift(1)
    feats={};filter_family={}
    def add(name,value,family):
        x=np.asarray(value,dtype=bool)
        if len(x)!=len(idx):raise ValueError("bad factor length "+name)
        if np.count_nonzero(x)<120 or np.count_nonzero(x)>len(x)-120:return
        feats[name]=x
        filter_family[name]=family
    for market in keys:
        p=series[market]
        mret=p.pct_change()
        for period in (10,20,50,100):
            mom=p/p.shift(period)-1
            # Separate upward and downward filters; no price level correlation.
            add(market+f"_r{period}>0",mom>0,market)
            add(market+f"_r{period}<0",mom<0,market)
        for period in (20,50):
            mean=p.rolling(period,min_periods=period).mean()
            add(market+f"_sma{period}_up",p>mean,market)
            add(market+f"_sma{period}_down",p<mean,market)
    # Lagged rolling correlation of return changes (not cointegrated levels).
    for market in ("ibov_close","brent_usd","brent_brl","usd_brl"):
        r=series[market].pct_change()
        for period in (63,126):
            corr=petret.rolling(period,min_periods=period).corr(r)
            for kind,cond in [
              ("positive",corr>.25),("strong",corr>.50),
              ("weak",corr.abs()<.25),("negative",corr<-.10)]:
                add(market+f"_corr{period}_{kind}",cond,market+"_corr")
    # Source regimes: USD/BRL and Brent in BRL, and Ibovespa direction.
    brent=series["brent_usd"]
    usd=series["usd_brl"]
    bves=series["ibov_close"]
    add("BrentUp_IBOVUp",(brent>brent.shift(20))&(bves>bves.shift(20)),"interaction")
    add("BrentUp_USDDown",(brent>brent.shift(20))&(usd<usd.shift(20)),"interaction")
    add("BrentBRLUp_IBOVUp",(series["brent_brl"]>series["brent_brl"].shift(20))&
        (bves>bves.shift(20)),"interaction")
    add("BrentDown_USDUp",(brent<brent.shift(20))&(usd>usd.shift(20)),"interaction")
    add("IBOVUp_USDDown",(bves>bves.shift(20))&(usd<usd.shift(20)),"interaction")
    add("IBOVUp_BrentDown",(bves>bves.shift(20))&(brent<brent.shift(20)),"interaction")
    # PETR4 idiosyncratic residual: 20-session PETR4 vs Ibovespa return spread
    ib=series["ibov_close"].pct_change()
    beta=petret.rolling(126,min_periods=126).cov(ib)/ib.rolling(126,min_periods=126).var()
    resid=petret-beta*ib
    # Causal residual z-score on trailing historical observations.
    rs=resid.rolling(20,min_periods=20).mean()
    add("PETR_relative_ibov_positive",rs>0,"residual")
    add("PETR_relative_ibov_negative",rs<0,"residual")
    # Volatility filter on past price information, no current session returns.
    vol=petret.rolling(20,min_periods=20).std()
    volbase=vol.rolling(126,min_periods=126).median()
    add("PETR_low_volatility",vol<volbase,"volatility")
    add("PETR_high_volatility",vol>volbase,"volatility")
    # Regime of 2023-present Selic, use yesterday's published target rate.
    selic=pd.Series(macro.selic_annual_pct,dtype=float).shift(1)
    add("Selic_gt_12",selic>12,"rates")
    add("Selic_lt_12",selic<12,"rates")
    # Add masks based on pairs of factors, prioritizing diverse markets.
    return feats,filter_family,series

def technical_candidates(frame,adj):
    tech={}
    for name,kind,params,signal in families(indicators(frame,adj)):
        if kind=="Benchmark":tech["BUY_HOLD"]=(kind,np.asarray(signal,dtype=bool));continue
        match=False
        if (kind=="Trend-filter" and name.startswith("Price/SMA") and
            params["n"] in (10,20,30,50,100) and params["filter"]=="none"):
            match=True
        if kind=="Momentum" and params["lookback"] in (5,10,20,60) and
           params["threshold"] in (-.05,0,.03):
            match=True
        if kind=="MACD" and params["fast"] in (8,12,20) and
           params["slow"] in (26,35,50) and params["signal"] in (5,9):
            match=True
        if kind=="Donchian" and params["entry"] in (20,40,55) and
           params["exit"] in (5,10,20) and params["filter"] in ("none","ema200"):
            match=True
        if kind=="RSI-trend" and params["period"] in (7,14) and
           params["entry"] in (50,60) and params["exit"] in (40,45):
            match=True
        if kind=="SuperTrend" and params["period"] in (7,14) and
           params["multiple"] in (1.5,2.,3.):
            match=True
        if match:tech[name]=(kind,np.asarray(signal,dtype=bool))
    return tech

def candidate_masks(feat,family,top=24):
    """Enumerate 1/2/3-way intermarket filter gates without future statistics."""
    keys=list(feat)
    selected=[]
    # Static pre-registered priority: all key markets + rolling correlations.
    for key in keys:
        familyname=family[key]
        if any(token in key for token in ("r20>0","r20<0","r50>0","r50<0",
            "sma50_up","sma50_down","corr63_positive","corr63_weak",
            "corr63_strong","BrentUp_","BrentBRLUp_","IBOVUp_",
            "PETR_low_volatility","PETR_high_volatility","PETR_relative_ibov_",
            "Selic_")):
            selected.append(key)
    # Pre-registered subset of cross-market gates limits multiple comparisons.
    # Every major factor and rolling correlation remains represented.
    priority=lambda k:(
        ("r20>0" in k or "r20<0" in k or
         "sma50_up" in k or "sma50_down" in k or
         "corr63_positive" in k or "corr63_weak" in k or
         "BrentUp_" in k or "BrentBRLUp_" in k or
         "IBOVUp_" in k or "PETR_low_volatility" in k)
    )
    selected=[k for k in selected if priority(k)][:32]
    # Pair masks need cross-factor diversity, not duplicates of same column.
    candidates=[(k,feat[k],1) for k in keys]
    for a,b in itertools.combinations(selected,2):
        if family[a]==family[b]:continue
        A=feat[a];B=feat[b]
        candidates.append((f"({a}) & ({b})",A&B,2))
        candidates.append((f"({a}) | ({b})",A|B,2))
    # Graph of 3 markets: Brent/IBOV/FX; also correlations PETR4 vs others.
    br=["brent_usd_r20>0","brent_brl_r20>0","brent_usd_sma50_up",
        "brent_usd_corr63_positive","brent_usd_corr63_weak"]
    ib=["ibov_close_r20>0","ibov_close_sma50_up",
        "ibov_close_corr63_positive","ibov_close_corr63_strong"]
    fx=["usd_brl_r20<0","usd_brl_r20>0","usd_brl_sma50_up",
        "usd_brl_corr63_weak"]
    for a,b,c in itertools.product(br,ib,fx):
        if not all(k in feat for k in (a,b,c)):continue
        candidates.append((f"({a}) & ({b}) & ({c})",feat[a]&feat[b]&feat[c],3))
        candidates.append((f"2of3({a};{b};{c})",
                         (feat[a].astype(int)+feat[b].astype(int)+feat[c].astype(int))>=2,3))
    # De-duplicate masks by byte hashes. Exclude almost-always-active gates.
    seen=set();out=[]
    for name,v,degree in candidates:
        counts=int(np.sum(v))
        if counts<260 or counts>len(v)-200:continue
        h=hashlib.sha1(np.packbits(v).tobytes()).digest()
        if h in seen:continue
        seen.add(h);out.append((name,v,degree))
    return out

def split_dates(t):
    def pos(s,side):
        return int(np.searchsorted(t.to_numpy(dtype="datetime64[ns]"),
                       np.datetime64(s),side=side))
    tr=(pos("2014-01-02","left"),pos("2023-12-29","right")-1)
    val=(pos("2024-01-02","left"),pos("2024-12-30","right")-1)
    test=(pos("2025-01-02","left"),pos("2026-11-17","right")-1)
    year25=(pos("2025-01-02","left"),pos("2025-12-31","right")-1)
    year26=(pos("2026-01-02","left"),pos("2026-11-17","right")-1)
    old=(pos("2025-01-23","left"),pos("2026-11-04","right")-1)
    return dict(train=tr,val=val,test=test,y2025=year25,y2026=year26,
                previous_comparable=old)

def ret_metrics(adj,raw,signal,splits,cost):
    basic={k:run_sim(adj,signal,*splits[k],fee_bps=cost)
           for k in ("train","val","test","y2025","y2026")}
    rawtest=run_sim(raw,signal,*splits["test"],fee_bps=cost)
    return basic,rawtest

def selection_score(tr,val):
    if tr["roundtrips"]<12 or tr["drawdown_pct"]>75:return None
    if val["roundtrips"]<3 or not 8<=val["exposure_pct"]<=94:return None
    # Validation score, up to one-year risk and excessive turnover:
    # never uses holdout returns.
    return (val["return_pct"]-.5*val["drawdown_pct"]
           -.075*val["roundtrips"]-.025*tr["drawdown_pct"])

def make_record(base,gate,degree,split_results,rawtest,score,benchmark):
    return dict(base=base,gate=gate,gate_degree=degree,
      train_pct=split_results["train"]["return_pct"],
      validation_pct=split_results["val"]["return_pct"],
      validation_dd=split_results["val"]["drawdown_pct"],
      validation_trades=split_results["val"]["roundtrips"],
      validation_score=score,
      test_adjusted_pct=split_results["test"]["return_pct"],
      test_raw_pct=rawtest["return_pct"],
      test_dd_pct=split_results["test"]["drawdown_pct"],
      test_sharpe=split_results["test"]["sharpe"],
      test_exposure_pct=split_results["test"]["exposure_pct"],
      test_trades=split_results["test"]["roundtrips"],
      ret2025_pct=split_results["y2025"]["return_pct"],
      ret2026_pct=split_results["y2026"]["return_pct"],
      excess2025_pp=split_results["y2025"]["return_pct"]-benchmark["y2025"]["return_pct"],
      excess2026_pp=split_results["y2026"]["return_pct"]-benchmark["y2026"]["return_pct"],
      excess_total_pp=split_results["test"]["return_pct"]-benchmark["test"]["return_pct"])

def local_corr_report(adj,m,splits):
    out=[]
    p=pd.Series(adj[:,3]).pct_change()
    for x in ("ibov_close","brent_usd","brent_brl","usd_brl"):
        q=pd.Series(m[x]).pct_change()
        rolling=p.rolling(63,min_periods=63).corr(q).shift(1)
        for segment in ("train","val","y2025","y2026","test"):
            lo,hi=splits[segment]
            corr=float(p.iloc[lo:hi+1].corr(q.iloc[lo:hi+1]))
            med=float(rolling.iloc[lo:hi+1].median())
            leadcorr=float(p.iloc[lo:hi+1].corr(q.shift(1).iloc[lo:hi+1]))
            out.append({"market":x,"segment":segment,"return_corr_same_day":corr,
                        "median_causal_corr63":med,
                        "lead1_prior_market_return_corr":leadcorr})
    return pd.DataFrame(out)

def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    prices_path=args.prices or fetch(URL_PRICES,args.out/"source_petr4.csv")
    macro_path=args.macro or fetch(URL_MACRO,args.out/"source_macro.csv")
    df,raw,adj,bad_dates=load_prices(prices_path)
    df,raw,adj,m=merge_macro(df,raw,adj,macro_path)
    split=split_dates(df.timestamp.dt.tz_localize(None))
    for k,(a,b) in split.items():
        if a<1 or b<=a or b>=len(df):raise ValueError(f"Bad split: {k} {a} {b} n={len(df)}")
    print("LOADED",len(df),str(df.timestamp.iloc[0].date()),
          str(df.timestamp.iloc[-1].date()),"removed",bad_dates,flush=True)
    correlations=local_corr_report(adj,m,split)
    correlations.to_csv(args.out/"intermarket_correlations.csv",index=False)
    tech=technical_candidates(df,adj)
    gates,fac,series=derived_factors(m,adj)
    masklist=candidate_masks(gates,fac)
    print("UNIVERSE",len(tech),"technical",len(masklist),
          "distinct gates",len(tech)*len(masklist),"products",flush=True)
    bh=np.ones(len(df),dtype=bool)
    bmk={seg:run_sim(adj,bh,*split[seg],fee_bps=args.fee)
         for seg in ("train","val","test","y2025","y2026","previous_comparable")}
    bmk_raw=run_sim(raw,bh,*split["test"],fee_bps=args.fee)
    records=[]
    keep_signals={}
    n=0
    techitems=list(tech.items())
    # Select a smaller, predefined technical menu to avoid optimization burden
    # and retain unfiltered baselines as controls.
    for base,(family,sig) in techitems:
        allgates=[("technical_alone",np.ones(len(sig),dtype=bool),0)]
        if base=="BUY_HOLD":
            # To avoid pure B&H being inadvertently designated candidate,
            # it is only a separate benchmark; macro-only gates still tested.
            pass
        allgates+=masklist
        for gate,mask,degree in allgates:
            candidate=sig & mask
            if np.count_nonzero(candidate)==0:continue
            tr=run_sim(adj,candidate,*split["train"],fee_bps=args.fee)
            va=run_sim(adj,candidate,*split["val"],fee_bps=args.fee)
            ranking=selection_score(tr,va)
            if ranking is None:continue
            # Save only models which are valid on the *historical* selection
            # criteria, before looking at the lockbox.
            test=run_sim(adj,candidate,*split["test"],fee_bps=args.fee)
            yr25=run_sim(adj,candidate,*split["y2025"],fee_bps=args.fee)
            yr26=run_sim(adj,candidate,*split["y2026"],fee_bps=args.fee)
            rawtest=run_sim(raw,candidate,*split["test"],fee_bps=args.fee)
            results={"train":tr,"val":va,"test":test,"y2025":yr25,"y2026":yr26}
            records.append(make_record(base,gate,degree,results,rawtest,ranking,bmk))
            n+=1
        if len(records)%200==0 or base==techitems[-1][0]:
            print("TECH_FINISHED",base,"eligible_so_far",len(records),flush=True)
    if not records:raise ValueError("No eligible combinations")
    table=pd.DataFrame(records)
    table=table.replace([np.inf,-np.inf],np.nan)
    table.to_csv(args.out/"all_eligible_combinations.csv",index=False)
    # Clearly distinguish preholdout selection vs retrospective.
    ranked=table.sort_values(["validation_score","train_pct"],ascending=False)
    val_champion=ranked.iloc[0].to_dict()
    val_intermarket=ranked[ranked.gate_degree>0].iloc[0].to_dict()
    baseline_sma20=table[(table.base=="Price/SMA20/none")&
                         (table.gate=="technical_alone")]
    b_sma=baseline_sma20.iloc[0].to_dict() if len(baseline_sma20) else None

    only=table[(table.gate_degree>0)&(table.test_exposure_pct>10)&
       (table.test_trades>=3)]
    # A user explicitly wants a 2025-2026 winner, so show EX-POST best,
    # and mark it invalid for prospective inference.
    positive_both=only[(only.ret2025_pct>0)&(only.ret2026_pct>0)&
                       (only.excess_total_pp>0)]
    beat_both=only[(only.excess2025_pp>0)&(only.excess2026_pp>0)&
                   (only.excess_total_pp>0)]
    hindsight=only.sort_values(["test_adjusted_pct","test_dd_pct"],
                               ascending=[False,True]).head(15)
    balanced=only.assign(
       robustness=only[["excess2025_pp","excess2026_pp"]].min(axis=1)
                   +.20*only.excess_total_pp-.18*only.test_dd_pct
       ).sort_values("robustness",ascending=False).head(20)
    strict_rank=beat_both.sort_values("test_adjusted_pct",ascending=False).head(20)
    positive_rank=positive_both.sort_values("test_adjusted_pct",ascending=False).head(20)
    hindsight.to_csv(args.out/"top_holdout_LOOKAHEAD.csv",index=False)
    balanced.to_csv(args.out/"yearly_balanced_LOOKAHEAD.csv",index=False)
    strict_rank.to_csv(args.out/"beat_BH_both_years_LOOKAHEAD.csv",index=False)
    positive_rank.to_csv(args.out/"positive_both_years_LOOKAHEAD.csv",index=False)
    ranked.head(50).to_csv(args.out/"top_validation_LOCKED.csv",index=False)

    def recompute(row,period="test"):
        base,gate=row["base"],row["gate"]
        signal=tech[base][1].copy()
        if gate!="technical_alone":
            masks={n:m for n,m,d in masklist}
            signal&=masks[gate]
        result=run_sim(adj,signal,*split[period],fee_bps=args.fee)
        return signal,result

    to_export={
      "buy_hold":(bh,bmk["test"]),
      "sma20":recompute(b_sma) if b_sma else (bh,bmk["test"]),
      "validation_champion":recompute(val_champion),
      "validation_intermarket":recompute(val_intermarket),
    }
    if len(hindsight):to_export["hindsight_best"]=recompute(hindsight.iloc[0])
    if len(strict_rank):to_export["hindsight_beats_BH_both_years"]=recompute(strict_rank.iloc[0])
    if len(positive_rank):to_export["hindsight_positive_both_years"]=recompute(positive_rank.iloc[0])
    for name,(s,results) in to_export.items():
        lo,hi=split["test"]
        pd.DataFrame({"date":df.timestamp.iloc[lo:hi+1].dt.strftime("%Y-%m-%d").to_numpy(),
          "equity":results["equity"],"signal_previous_close":s[lo-1:hi]}).to_csv(
          args.out/f"equity_{name}.csv",index=False)
    def finite(obj):
        if isinstance(obj,dict):return {k:finite(v) for k,v in obj.items()}
        if isinstance(obj,list):return [finite(x) for x in obj]
        if isinstance(obj,(np.integer,)):return int(obj)
        if isinstance(obj,(np.floating,float)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj,(np.bool_,)):return bool(obj)
        return obj
    # Monthly cash/Selic proxy: not in the default P&L, reserve for next audit.
    report={
      "schema":"PETR4_INTERMARKET_COMBINATORIAL_V1",
      "source":{"prices_url":URL_PRICES,"macro_url":URL_MACRO,
        "source_commit":SOURCE_COMMIT,
        "prices_sha256":hashlib.sha256(pathlib.Path(prices_path).read_bytes()).hexdigest(),
        "macro_sha256":hashlib.sha256(pathlib.Path(macro_path).read_bytes()).hexdigest(),
        "bad_ohlc_rows":bad_dates},
      "protocol":{
        "train":"2014-01-02 through 2023-12-29",
        "validation":"2024-01-02 through 2024-12-30",
        "holdout":"2025-01-02 through 2026-11-17",
        "split_actual_dates":{k:[str(df.timestamp.iloc[a].date()),
                                 str(df.timestamp.iloc[b].date())] for k,(a,b) in split.items()},
        "cost_bps_each_side":args.fee,"initial_brl":10000,
        "position":"long PETR4 or flat, max 1x",
        "information_lag":"All exogenous market factors lagged >=1 completed PETR4 sessions",
        "execution":"Signal at PETR4 close t, order at next open t+1",
        "dividends":"Yahoo back-adjusted total-return proxy, NOT executable B3 ledger",
        "cash_interest":"zero; additional CDI sensitivity pending",
        "out_of_sample_selection":"validation 2024 only. All tests using 2025/26 to select are marked LOOKAHEAD",
        "scope":"MQL5-style families, NOT all proprietary Market EAs"},
      "counts":{
        "technical_bases":len(tech),
        "cross_market_filter_unique":len(masklist),
        "total_raw_combinations":len(tech)*(len(masklist)+1),
        "eligible_on_2014_2024":len(table),
        "holdout_cross_market_beating_full_BH":int((only.excess_total_pp>0).sum()),
        "holdout_cross_market_positive_both_years":len(positive_both),
        "holdout_cross_market_beating_BH_in_each_year":len(beat_both)},
      "benchmarks":{
        "buy_and_hold_adj":{k:finite({a:v for a,v in bmk[k].items() if a!="equity"})
                            for k in ("val","test","y2025","y2026","previous_comparable")},
        "buy_and_hold_raw":finite({k:v for k,v in bmk_raw.items() if k!="equity"})},
      "selection_without_test":
        {"champion":finite(val_champion),"best_using_market_filters":finite(val_intermarket),
         "sma20_unfiltered":finite(b_sma)},
      "selected_using_2025_2026_LOOKAHEAD_ONLY":{
        "highest_total":finite(hindsight.iloc[0].to_dict()) if len(hindsight) else None,
        "best_positive_both":finite(positive_rank.iloc[0].to_dict()) if len(positive_rank) else None,
        "best_beats_BH_each_year":finite(strict_rank.iloc[0].to_dict()) if len(strict_rank) else None,
        "best_balanced":finite(balanced.iloc[0].to_dict()) if len(balanced) else None},
      "cross_market_return_correlations":finite(correlations.to_dict("records")),
      "limitations":[
         "Same-day correlation is descriptive, NOT a signal. Trading uses lagged returns and correlations.",
         "Ex-post rank of thousands of combinations overfits the 2025–2026 holdout.",
         "One factor source combines Brent futures, BCB PTAX and end-of-day Ibovespa close.",
         "A one-session delay is used conservatively to avoid global EOD lookahead.",
         "Corporate actions approximated by ex-post adjusted OHLC, not actual dividend cash.",
         "No live broker spread, board lots, taxation, halted orders, borrow, or CDI on cash.",
         "Performance on this time window is not proof the approach will work in 2027."
      ]}
    (args.out/"report.json").write_text(
       json.dumps(finite(report),indent=2,ensure_ascii=False,allow_nan=False))
    condensed={
      "counts":report["counts"],
      "buy_hold":{k:bmk[k]["return_pct"] for k in ("test","y2025","y2026")},
      "best_from_validation":val_champion,
      "best_validation_macro_only":val_intermarket,
      "sma20_control":b_sma,
      "best_overall_EX_POST":report["selected_using_2025_2026_LOOKAHEAD_ONLY"]["highest_total"],
      "best_both_years_EX_POST":report["selected_using_2025_2026_LOOKAHEAD_ONLY"]["best_beats_BH_each_year"],
      "best_positive_both_EX_POST":report["selected_using_2025_2026_LOOKAHEAD_ONLY"]["best_positive_both"]
    }
    print("INTERMARKET_REPORT_BEGIN")
    print(json.dumps(finite(condensed),indent=2,ensure_ascii=False,allow_nan=False))
    print("INTERMARKET_REPORT_END")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=pathlib.Path,default=pathlib.Path("research/intermarket_runs"))
    p.add_argument("--prices",type=pathlib.Path)
    p.add_argument("--macro",type=pathlib.Path)
    p.add_argument("--fee",type=float,default=13)
    args=p.parse_args()
    if args.fee<0:raise ValueError("negative cost")
    run(args)

if __name__=="__main__":
    main()
