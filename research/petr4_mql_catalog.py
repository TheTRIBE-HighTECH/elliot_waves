#!/usr/bin/env python3
"""PETR4/B3 empirical MQL5-style strategy family benchmark, long-only, 1x.

DO NOT claim commercial MQL5 EAs have been executed. This reproduces common
public mathematical strategy families, not proprietary Expert Advisor code.
Train/parameter filter 2014-2023, rank on 2024, locked holdout 2025-01-23
to 2026-11-04 matching PETR4 neural study's window. A retrospective best
holdout search is included ONLY as EXPLORATORY, clearly marked contaminated.

OHLCV: public Yahoo-derived file via msbrasil/Series-Temporais-PETR4.
Historical 'Adj Close' ratio used for TOTAL-RETURN PROXY, NOT executable
stock quotation; raw price performance is reported separately.
"""
import argparse, hashlib, itertools, json, pathlib, urllib.request
import numpy as np
import pandas as pd

SOURCE=("https://raw.githubusercontent.com/msbrasil/"
        "Series-Temporais-PETR4/main/data/raw/petr4_yf_raw.csv")
TRAIN_START="2014-01-02"
TRAIN_END="2023-12-29"
VALIDATION_START="2024-01-02"
VALIDATION_END="2024-12-30"
TEST_START="2025-01-23"
TEST_END="2026-11-04"
COST_BPS=13.0

def acquire(path):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(SOURCE,headers={"User-Agent":"quant-petr4-strategy-research/1.0"})
    with urllib.request.urlopen(req,timeout=75) as resp:raw=resp.read()
    if len(raw)<400000:raise RuntimeError("Truncated price source")
    path.write_bytes(raw)
    return path

def load_prices(path):
    df=pd.read_csv(path)
    expected={"Date","Adj Close","Open","High","Low","Close","Volume"}
    if not expected.issubset(df.columns):raise ValueError("Source columns missing")
    df["timestamp"]=pd.to_datetime(df.Date,utc=True)
    df=df.drop_duplicates("timestamp").sort_values("timestamp")
    for col in ("Adj Close","Open","High","Low","Close","Volume"):
        df[col]=pd.to_numeric(df[col],errors="coerce")
    df=df.dropna(subset=list(expected-{"Date"})).reset_index(drop=True)
    df=df[(df.timestamp>=pd.Timestamp("2013-01-01",tz="UTC"))&
          (df.timestamp<=pd.Timestamp("2026-11-27",tz="UTC"))].copy().reset_index(drop=True)
    invalid=(df[["Adj Close","Open","High","Low","Close"]].le(0).any(axis=1)|
             (df.Volume<0)|
             (df.High+1e-6<df[["Open","Close","Low"]].max(axis=1))|
             (df.Low-1e-6>df[["Open","Close","High"]].min(axis=1)))
    dates=df.loc[invalid,"Date"].tolist()
    if len(dates)>5:raise ValueError(f"Unexpected data corruption: {dates[:10]}")
    print("EXCLUDED_OHLC",json.dumps(dates),flush=True)
    df=df.loc[~invalid].reset_index(drop=True)
    f=df["Adj Close"]/df["Close"]
    raw=df[["Open","High","Low","Close"]].to_numpy(dtype=float)
    adjusted=raw*f.to_numpy(dtype=float)[:,None]
    return df,raw,adjusted,dates

def indicators(df,adjusted):
    o,h,l,c=adjusted.T
    index=df.index
    close=pd.Series(c,index=index)
    high=pd.Series(h,index=index)
    low=pd.Series(l,index=index)
    volume=pd.Series(df.Volume.to_numpy(dtype=float),index=index)
    prev=close.shift(1)
    tr=pd.concat([high-low,(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
    cache={"close":close,"high":high,"low":low,"vol":volume,
           "ema":{},"sma":{},"std":{},"rsi":{},"atr":{},"highprev":{},"lowprev":{},"adx":{}}
    ema_lengths=[5,8,10,12,15,16,20,21,26,30,35,40,50,60,65,75,100,120,150,180,200,250,300]
    sma_lengths=[5,10,15,20,30,40,50,65,75,100,120,150,200,250,300]
    for n in ema_lengths:cache["ema"][n]=close.ewm(span=n,adjust=False).mean().to_numpy()
    for n in sma_lengths:cache["sma"][n]=close.rolling(n,min_periods=n).mean().to_numpy()
    for n in (10,20,30,50):
        cache["std"][n]=close.rolling(n,min_periods=n).std(ddof=0).to_numpy()
    for n in (7,10,14,21,28):
        d=close.diff();up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
        down=(-d).clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
        rs=up/down.replace(0,np.nan)
        cache["rsi"][n]=(100-100/(1+rs)).fillna(50).to_numpy()
    for n in (7,10,14,21,28):
        cache["atr"][n]=tr.ewm(alpha=1/n,adjust=False).mean().to_numpy()
    for n in (5,10,20,40,55,60,80,120,150):
        cache["highprev"][n]=high.shift(1).rolling(n,min_periods=n).max().to_numpy()
        cache["lowprev"][n]=low.shift(1).rolling(n,min_periods=n).min().to_numpy()
    # 14-day ADX: Wilder directional movement and smoothed True Range.
    highdiff=high.diff()
    lowdiff=-low.diff()
    plus=np.where((highdiff>lowdiff)&(highdiff>0),highdiff,0.)
    minus=np.where((lowdiff>highdiff)&(lowdiff>0),lowdiff,0.)
    sm=tr.ewm(alpha=1/14,adjust=False).mean().replace(0,np.nan)
    plusdi=100*pd.Series(plus).ewm(alpha=1/14,adjust=False).mean()/sm
    minusdi=100*pd.Series(minus).ewm(alpha=1/14,adjust=False).mean()/sm
    dx=100*(plusdi-minusdi).abs()/(plusdi+minusdi).replace(0,np.nan)
    cache["adx"][14]=dx.ewm(alpha=1/14,adjust=False).mean().fillna(0).to_numpy()
    return cache

def hold(buy,sell):
    """Causal decisions on current close. 1 means next session open long."""
    n=len(buy);result=np.zeros(n,dtype=np.int8);state=0
    for i in range(n):
        if state:
            if sell[i]:state=0
        elif buy[i]:state=1
        result[i]=state
    return result

def supertrend(c,h,l,atr,period,multiple):
    mid=(h+l)/2
    ub=mid+multiple*atr
    lb=mid-multiple*atr
    trend=np.zeros(len(c),dtype=np.int8)
    final_ub=np.copy(ub);final_lb=np.copy(lb)
    for i in range(1,len(c)):
        final_ub[i]=(ub[i] if ub[i]<final_ub[i-1] or c[i-1]>final_ub[i-1] else final_ub[i-1])
        final_lb[i]=(lb[i] if lb[i]>final_lb[i-1] or c[i-1]<final_lb[i-1] else final_lb[i-1])
        trend[i]=trend[i-1]
        if trend[i-1]==0 and c[i]>final_ub[i-1]:trend[i]=1
        elif trend[i-1]==1 and c[i]<final_lb[i-1]:trend[i]=0
    trend[:max(30,3*period)]=0
    return trend

def families(cache):
    c=cache["close"].to_numpy();h=cache["high"].to_numpy();l=cache["low"].to_numpy()
    ema=cache["ema"];sma=cache["sma"];rsi=cache["rsi"];atr=cache["atr"];adx=cache["adx"][14]
    zero=np.zeros(len(c),dtype=bool)
    for kind in ("EMA","SMA"):
        means=ema if kind=="EMA" else sma
        for fast in (5,10,15,20,30,40,50):
            for slow in (65,100,150,200,250):
                if fast not in means or slow not in means:continue
                name=f"{kind}({fast},{slow})"
                yield name,kind,{"fast":fast,"slow":slow},(means[fast]>means[slow]).astype(np.int8)
    for n in (10,20,30,50,75,100,150,200,250,300):
        for filt in ("none","ema200","ema200_slope"):
            base=c>sma[n]
            if filt!="none":base&=(c>ema[200])
            if filt=="ema200_slope":base&=(ema[200]>np.roll(ema[200],20))
            yield f"Price/SMA{n}/{filt}","Trend-filter",{"n":n,"filter":filt},base.astype(np.int8)
    for n in (5,10,20,40,60,100,150,200):
        for pct in (-.05,0,.03,.07):
            shift=np.roll(c,n);mom=(c/shift-1)
            base=mom>pct;base[:n]=False
            yield f"TSMOM({n},{pct})","Momentum",{"lookback":n,"threshold":pct},base.astype(np.int8)
    for entry in (10,20,40,55,80,120):
        for exit_ in (5,10,20,40):
            if exit_>entry:continue
            for filt in ("none","ema200","adx20"):
                buy=c>cache["highprev"][entry]
                if filt=="ema200":buy&=c>ema[200]
                if filt=="adx20":buy&=adx>20
                sell=c<cache["lowprev"][exit_]
                yield f"Donchian({entry},{exit_})/{filt}","Donchian",{"entry":entry,"exit":exit_,"filter":filt},hold(buy,sell)
    for period in (7,14,21,28):
        R=rsi[period]
        for buy_at in (20,25,30,35,40):
            for sell_at in (50,60,70,80):
                yield f"RSI({period},{buy_at},{sell_at})","RSI-reversion",{"period":period,"buy":buy_at,"sell":sell_at},hold(R<buy_at,R>sell_at)
    for n in (10,20,30,50):
        mid=sma[n];std=cache["std"][n]
        for entry_z in (1,1.5,2,2.5):
            for exit_z in (0,.5,1.):
                yield f"BB({n},{entry_z},{exit_z})","Bollinger-reversion",{"period":n,"z_entry":entry_z,"z_exit":exit_z},hold(c<mid-entry_z*std,c>mid+exit_z*std)
    for f in (8,12,16,20):
        for slow in (21,26,35,50):
            if f>=slow:continue
            raw=ema[f]-ema[slow]
            for sig in (5,9,13):
                smooth=pd.Series(raw).ewm(span=sig,adjust=False).mean().to_numpy()
                yield f"MACD({f},{slow},{sig})","MACD",{"fast":f,"slow":slow,"signal":sig},(raw>smooth).astype(np.int8)
    for period in (7,10,14,21):
        for factor in (1.5,2.,3.,4.):
            yield f"SuperTrend({period},{factor})","SuperTrend",{"period":period,"multiple":factor},supertrend(c,h,l,atr[period],period,factor)
    for length in (10,20,40,60):
        for factor in (1.,2.,3.):
            base=ema[length]+factor*atr[14]
            buy=c>base
            sell=c<ema[length]
            yield f"Keltner({length},{factor})","Keltner",{"period":length,"atr_multiple":factor},hold(buy,sell)
    for length in (10,20,40,60):
        for factor in (1.5,2.5,3.5):
            highprevious=cache["highprev"].get(length)
            if highprevious is None:continue
            # Close below ATR trailing stop after participating in trend.
            trailing=highprevious-factor*atr[14]
            buy=c>ema[50]
            sell=c<trailing
            yield f"Chandelier({length},{factor})","Chandelier",{"period":length,"atr_multiple":factor},hold(buy,sell)
    for period in (7,14,21):
        for entry in (50,55,60,65):
            for exit_ in (35,40,45,50):
                base=(rsi[period]>entry)&(c>ema[200])
                sell=(rsi[period]<exit_)|(c<ema[200])
                yield f"RSITrend({period},{entry},{exit_})","RSI-trend",{"period":period,"entry":entry,"exit":exit_},hold(base,sell)
    # Constant-position baseline for audit.
    yield "Buy-and-hold", "Benchmark", {},np.ones(len(c),dtype=np.int8)

def run_sim(prices,signal,lo,hi,fee_bps=COST_BPS):
    """All-in long/flat. Causal: yesterday's CLOSE controls today's OPEN.

    Splits and dividends use BACK-ADJUSTED OHLC as a total-return PROXY.
    A position held from yesterday's close earns close-to-open gap;
    today's position earns open-to-close. Costs on switching at the open.
    """
    op=prices[:,0];cl=prices[:,3]
    # Start independent flat, no validation or training holdings carried over.
    desired=signal[np.arange(lo,hi+1)-1].astype(bool)
    prev=np.r_[False,desired[:-1]]
    prior_close=np.r_[op[lo],cl[lo:hi]]
    prev_exposure=prev.astype(float)
    today_exposure=desired.astype(float)
    overnight=1+prev_exposure*(op[lo:hi+1]/prior_close-1)
    intraday=1+today_exposure*(cl[lo:hi+1]/op[lo:hi+1]-1)
    fee=fee_bps/10000
    transact=np.where(~prev&desired,1/(1+fee),
               np.where(prev&~desired,1-fee,1.))
    growth=overnight*transact*intraday
    if desired[-1]:growth[-1]*=1-fee # flat at final close
    equity=10000*np.cumprod(growth)
    peak=np.maximum.accumulate(np.r_[10000.,equity])
    dd=np.max(1-np.r_[10000.,equity]/peak)
    logret=np.log(growth)
    vol=np.std(logret,ddof=1)
    sharpe=float(np.mean(logret)/vol*np.sqrt(252)) if vol>1e-12 else None
    trades=int(np.sum(desired&~prev))
    return {
      "return_pct":float(100*(equity[-1]/10000-1)),
      "end_brl":float(equity[-1]),
      "drawdown_pct":float(dd*100),
      "sharpe":sharpe,
      "roundtrips":trades,
      "exposure_pct":float(100*np.mean(desired)),
      "equity":equity,
    }

def score(result):
    # Must have some trades and nonzero market exposure to avoid cash 'winners'.
    if result["roundtrips"]<2 or result["exposure_pct"]<5:return -1e10
    return result["return_pct"]-.35*result["drawdown_pct"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=pathlib.Path,default=pathlib.Path("research/mql_petr4_runs"))
    ap.add_argument("--cost-bps",type=float,default=13.)
    ap.add_argument("--csv",type=pathlib.Path,default=None)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    src=args.csv or acquire(args.out/"petr4_public_source.csv")
    df,raw,adj,bad_dates=load_prices(src)
    dates=df.timestamp.dt.tz_localize(None).to_numpy(dtype='datetime64[ns]')
    def begin_day(day):
        return int(np.searchsorted(dates,np.datetime64(day),side="left"))
    def end_day(day):
        return int(np.searchsorted(dates,np.datetime64(day),side="right")-1)
    # Exact same holdout window as previous PETR4 neural result.
    slices={
      "train":(begin_day(TRAIN_START),end_day(TRAIN_END)),
      "validation":(begin_day(VALIDATION_START),end_day(VALIDATION_END)),
      "test":(begin_day(TEST_START),end_day(TEST_END)),
    }
    for k,(lo,hi) in list(slices.items()):
        if hi>=len(df):raise ValueError("date outside dataset "+k)
        if lo>=hi:raise ValueError("invalid split "+k)
    source_ohlc=indicators(df,adj)
    catalogs=[];signals={}
    for k,(name,family,params,pos) in enumerate(families(source_ohlc)):
        pos=np.asarray(pos,dtype=np.int8)
        train=run_sim(adj,pos,*slices["train"],fee_bps=args.cost_bps)
        validation=run_sim(adj,pos,*slices["validation"],fee_bps=args.cost_bps)
        test=run_sim(adj,pos,*slices["test"],fee_bps=args.cost_bps)
        rawtest=run_sim(raw,pos,*slices["test"],fee_bps=args.cost_bps)
        row={"strategy":name,"family":family,"params":json.dumps(params,sort_keys=True),
             "train_return_pct":train["return_pct"],"train_dd_pct":train["drawdown_pct"],
             "train_trades":train["roundtrips"],
             "validation_return_pct":validation["return_pct"],
             "validation_dd_pct":validation["drawdown_pct"],
             "validation_trades":validation["roundtrips"],
             "validation_score":score(validation),
             "test_return_adjusted_pct":test["return_pct"],
             "test_return_raw_pct":rawtest["return_pct"],
             "test_dd_pct":test["drawdown_pct"],
             "test_sharpe":test["sharpe"],"test_trades":test["roundtrips"],
             "test_exposure_pct":test["exposure_pct"]}
        catalogs.append(row)
        signals[name]=pos
        if (k+1)%100==0:print("CANDIDATES",k+1,flush=True)
    matrix=pd.DataFrame(catalogs)
    baseline=matrix[matrix.family=="Benchmark"].iloc[0]
    available=matrix[(matrix.family!="Benchmark") & (matrix.train_trades>=6) &
                     (matrix.validation_trades>=2)&(matrix.validation_score>-1e9)].copy()
    if available.empty:raise ValueError("No candidate passed filters")
    # 2024 selection only. No peeking at 2025–2026 to choose the winner.
    family_winners=available.sort_values(
       ["validation_score","train_return_pct"],ascending=False).drop_duplicates("family")
    champ=available.sort_values(
       ["validation_score","train_return_pct"],ascending=False).iloc[0]
    # FOR TRANSPARENCY ONLY: ex-post top holdout is contaminated by 2025–26 knowledge.
    hindsight=matrix[matrix.family!="Benchmark"].sort_values(
       "test_return_adjusted_pct",ascending=False).head(12)
    above=int((matrix[matrix.family!="Benchmark"].test_return_adjusted_pct >
               baseline.test_return_adjusted_pct).sum())
    def reduce(row):
        result=row.to_dict() if hasattr(row,'to_dict') else dict(row)
        for key,value in result.items():
            if isinstance(value,(np.integer,)):result[key]=int(value)
            elif isinstance(value,(np.floating,float)):result[key]=float(value) if np.isfinite(value) else None
        return result
    def curve_export(name,label):
        sig=signals[name];lo,hi=slices["test"]
        result=run_sim(adj,sig,lo,hi,args.cost_bps)
        out=pd.DataFrame({"timestamp":df.timestamp.iloc[lo:hi+1].dt.strftime("%Y-%m-%d"),
                          "equity":result["equity"]})
        out.to_csv(args.out/f"equity_{label}.csv",index=False)
    for name,label in [(champ.strategy,"champion"),("Buy-and-hold","buy_hold"),(hindsight.iloc[0].strategy,"hindsight")]:
        curve_export(name,label)
    matrix.sort_values(["validation_score","train_return_pct"],ascending=False).to_csv(
        args.out/"all_candidates.csv",index=False)
    family_winners.to_csv(args.out/"family_validation_winners.csv",index=False)
    hindsight.to_csv(args.out/"hindsight_NOT_VALIDATED.csv",index=False)
    benchmark_raw=run_sim(raw,signals["Buy-and-hold"],*slices["test"],args.cost_bps)
    report={
       "asset":"PETR4.SA","timeframe":"1d_B3_sessions",
       "source_url":SOURCE,
       "source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
       "data_quality_excluded_ohlc_dates":bad_dates,
       "splits":{name:{"start":str(df.Date.iloc[lo]),"end":str(df.Date.iloc[hi]),
                        "candles":hi-lo+1} for name,(lo,hi) in slices.items()},
       "cost_bps_per_execution":args.cost_bps,
       "model_scope":"Reimplemented MQL5-style strategy families; NOT proprietary EA backtests.",
       "candidates_total":len(matrix),"candidates_eligible":len(available),
       "families_total":matrix.family.nunique()-1,
       "selection":"Best 2024 validation return minus 0.35 times drawdown; minimum 6 training and 2 validation round trips, >=5% validation exposure.",
       "benchmark_buy_hold_adjusted_proxy_pct":float(baseline.test_return_adjusted_pct),
       "benchmark_buy_hold_raw_price_pct":float(benchmark_raw["return_pct"]),
       "selected_champion":reduce(champ),
       "per_family_best_by_2024":list(map(reduce,family_winners.to_dict("records"))),
       "hindsight_best_2025_2026_NOT_PROSPECTIVELY_VALIDATED":list(map(reduce,hindsight.to_dict("records"))),
       "holdout_candidates_beating_buy_hold_count_EXPLORATORY":above,
       "notes":[
          "No candidate selected using holdout; ex-post ranking is exploratory, not evidence of edge.",
          "Adjusted OHLC is an ex-post total-return proxy, not executable market prices.",
          "PETR4 dividend/bonus events, bid-ask spreads and brokers' fees not modelled exactly.",
          "All-in long or cash with no leverage; cash earns zero CDI; fractional shares.",
          "Signals depend only on price data available at close i; execute open i+1.",
          "No real MetaTrader 5 terminal or proprietary MQL5 Expert Advisor executable was run.",
          "One 2024 validation year is insufficient to establish stable predictive advantage."
       ],
    }
    (args.out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    print("REPORT_SUMMARY_START",flush=True)
    print(json.dumps({k:report[k] for k in (
      "candidates_total","candidates_eligible","families_total",
      "benchmark_buy_hold_adjusted_proxy_pct","benchmark_buy_hold_raw_price_pct",
      "selected_champion","holdout_candidates_beating_buy_hold_count_EXPLORATORY",
      "hindsight_best_2025_2026_NOT_PROSPECTIVELY_VALIDATED")},indent=2,allow_nan=False),flush=True)
    print("REPORT_SUMMARY_END",flush=True)

if __name__=="__main__":main()
