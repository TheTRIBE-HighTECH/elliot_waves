#!/usr/bin/env python3
"""Fetch Binance Vision SPOT OHLCV months and normalize to UTC candle opens.

Not a library of human Elliott labels. See https://github.com/binance/binance-public-data .
"""
import argparse, datetime as dt, io, pathlib, urllib.request, zipfile
import pandas as pd

BASE="https://data.binance.vision/data/spot/monthly/klines"
COLS=["timestamp","open","high","low","close","volume","close_time",
      "quote_volume","num_trades","taker_buy_volume","taker_buy_quote_volume","ignore"]
def months(first,last):
    a=dt.date.fromisoformat(first+"-01");b=dt.date.fromisoformat(last+"-01")
    while a<=b:
        yield a.strftime("%Y-%m")
        a=(a.replace(day=28)+dt.timedelta(days=4)).replace(day=1)
def normalize(raw,symbol,interval):
    df=pd.read_csv(io.BytesIO(raw),header=None,names=COLS,low_memory=False)
    if len(df)==0:raise ValueError("empty kline file")
    first=pd.to_numeric(df.timestamp,errors="coerce")
    if first.isna().any():raise ValueError("non-numeric timestamps")
    unit="us" if first.median()>1e14 else "ms" if first.median()>1e11 else "s"
    df["timestamp"]=pd.to_datetime(first,unit=unit,utc=True)
    for x in COLS[1:11]:df[x]=pd.to_numeric(df[x],errors="coerce")
    df["symbol"]=symbol;df["interval"]=interval
    fields=["timestamp","open","high","low","close","volume","num_trades",
            "taker_buy_volume","quote_volume","symbol","interval"]
    df=df[fields]
    if df.isna().any().any():raise ValueError("missing OHLCV value")
    if not ((df.high>=df[['open','close','low']].max(axis=1)) &
            (df.low<=df[['open','close','high']].min(axis=1))).all():
        raise ValueError("OHLC inconsistent")
    if (df[['open','high','low','close']]<=0).any().any() or (df.volume<0).any():
        raise ValueError("negative or zero prices/volume")
    return df

def download(symbol,period,interval="1h",timeout=60):
    name=f"{symbol}-{interval}-{period}.zip"
    url=f"{BASE}/{symbol}/{interval}/{name}"
    req=urllib.request.Request(url,headers={"User-Agent":"elliott-research/1.0"})
    with urllib.request.urlopen(req,timeout=timeout) as resp:body=resp.read()
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        files=[p for p in z.namelist() if p.endswith(".csv") and not p.startswith("__MACOSX/")]
        if len(files)!=1:raise ValueError(f"unexpected zip members {files}")
        content=z.read(files[0])
    return normalize(content,symbol,interval)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--symbols",nargs="+",default=["BTCUSDT"])
    p.add_argument("--start",default="2024-01")
    p.add_argument("--end",default="2025-12")
    p.add_argument("--interval",default="1h")
    p.add_argument("--out",type=pathlib.Path,default=pathlib.Path("research/datasets"))
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    for symbol in args.symbols:
        parts=[]
        for period in months(args.start,args.end):
            print(f"Fetching {symbol} {period}",flush=True)
            parts.append(download(symbol,period,args.interval))
        df=pd.concat(parts,ignore_index=True).sort_values("timestamp")
        if df.timestamp.duplicated().any():raise ValueError("duplicated candles")
        delta=df.timestamp.diff().dt.total_seconds().dropna()
        if args.interval=="1h" and (delta!=3600).any():
            raise ValueError("hourly data gaps, stop training")
        result=args.out/f"{symbol}_{args.interval}.csv"
        df.to_csv(result,index=False)
        print(f"{result}: {len(df)} candles | {df.timestamp.iloc[0]} to {df.timestamp.iloc[-1]}")
if __name__=="__main__":main()
