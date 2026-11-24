#!/usr/bin/env python3
"""Multiscale Elliott neural benchmark on REAL historical OHLCV.

Four candidate architectures, shared weights per architecture for calendar and
volume-based information clocks. 2024 train / 2025H1 validation / 2025H2 test.
Without independently verified labels, uses retrospective SWING PSEUDO-LABELS;
these are NOT verified Elliott waves. No predictive claims from pseudo-labels.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

CLASSES=["unknown"]+[f"{side}_{w}" for side in ("bull","bear") for w in "12345ABC"]
SCALES=(1,3,8); LENGTH=48; NCHANNELS=4

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))

def read_candles(path):
    df=pd.read_csv(path);df.columns=[c.lower().strip() for c in df.columns]
    if "date" in df and "timestamp" not in df:
        df["timestamp"]=pd.to_datetime(df.date,format="%d-%m-%Y %H:%M",utc=True)
    else:
        x=df["timestamp"]
        if pd.api.types.is_numeric_dtype(x):
            med=x.abs().median();unit="us" if med>1e14 else "ms" if med>1e11 else "s"
            df["timestamp"]=pd.to_datetime(x,unit=unit,utc=True)
        else:df["timestamp"]=pd.to_datetime(x,utc=True)
    cols=["timestamp","open","high","low","close","volume"]
    if not set(cols).issubset(df.columns):raise ValueError("OHLCV missing")
    df=df.sort_values("timestamp").reset_index(drop=True)
    if df[cols].isna().any().any() or df.timestamp.duplicated().any():
        raise ValueError("missing/duplicated candles")
    if (df[cols[1:5]]<=0).any().any() or (df.volume<0).any():
        raise ValueError("invalid OHLCV")
    dif=df.timestamp.diff().dt.total_seconds().dropna()
    if (dif!=3600).any():raise ValueError("benchmark assumes gap-free 1h candles")
    return df

def information_clock(volume,alpha=.8):
    v=np.asarray(volume,dtype=float);n=len(v)
    if alpha==0:return np.arange(n,dtype=float),np.ones(n)
    log=pd.Series(np.log1p(v))
    mean=log.shift(1).rolling(168,min_periods=24).mean()
    sd=log.shift(1).rolling(168,min_periods=24).std(ddof=0).clip(lower=.15)
    z=((log-mean)/sd).fillna(0).clip(-2.5,2.5)
    raw=np.exp(alpha*z)
    norm=raw.shift(1).rolling(168,min_periods=24).mean().fillna(1).clip(lower=.1)
    speed=np.asarray((raw/norm).clip(.35,3.5),dtype=float)
    tau=np.zeros(n)
    tau[1:]=np.cumsum(speed[1:])
    return tau,speed

def features(df,axis,indices):
    c=np.log(df.close.to_numpy(dtype=float));v=np.log1p(df.volume.to_numpy(dtype=float))
    r=np.log(df.high.to_numpy()/df.low.to_numpy())
    X=[];keep=[]
    for i in indices:
        arr=[];valid=True
        for scale in SCALES:
            ticks=axis[i]-np.arange(LENGTH-1,-1,-1)*scale
            if ticks[0]<axis[0]:valid=False;break
            grid=axis[:i+1]
            p=np.interp(ticks,grid,c[:i+1]);vol=np.interp(ticks,grid,v[:i+1])
            spread=np.interp(ticks,grid,r[:i+1])
            delta=np.diff(p,prepend=p[0]);sd=max(np.std(delta),2e-5)
            shape=np.clip((p-p[-1])/(sd*math.sqrt(LENGTH)),-10,10)
            movement=np.clip(delta/sd,-8,8)
            activity=np.clip((vol-vol.mean())/max(vol.std(),.15),-5,5)
            range_feature=np.clip(spread/max(np.median(spread),1e-5),0,12)
            arr.append(np.stack([shape,movement,activity,range_feature]).astype("float32"))
        if valid:X.append(np.stack(arr));keep.append(i)
    if not X:raise ValueError("not enough historical data for multiscale features")
    return np.stack(X),np.asarray(keep,dtype=int)

def weak_swing_targets(price,threshold=.025):
    """Retrospective pivot labels, never pass labels into inference or trade logic."""
    p=np.asarray(price);y=np.zeros(len(p),dtype=np.int64)
    piv=[0];direction=0;ext=0;ref=p[0]
    for i in range(1,len(p)):
        q=p[i]
        if direction==0:
            if q>=ref*(1+threshold):direction=1;ref=q;ext=i
            elif q<=ref*(1-threshold):direction=-1;ref=q;ext=i
        elif direction==1:
            if q>=ref:ref=q;ext=i
            elif q<=ref*(1-threshold):
                if ext>piv[-1]:piv.append(ext)
                direction=-1;ref=q;ext=i
        else:
            if q<=ref:ref=q;ext=i
            elif q>=ref*(1+threshold):
                if ext>piv[-1]:piv.append(ext)
                direction=1;ref=q;ext=i
    for offset in range(0,(len(piv)-1)//8*8,8):
        bullish=p[piv[offset+1]]>p[piv[offset]]
        for s in range(8):
            y[piv[offset+s]+1:piv[offset+s+1]+1]=(1 if bullish else 9)+s
    return y

def label_blocks(df,t1,t2):
    price=df.close.to_numpy();out=np.zeros(len(df),dtype=np.int64)
    for a,b in ((0,t1),(t1,t2),(t2,len(df))):
        out[a:b]=weak_swing_targets(price[a:b])
    return out

def future_reversal(df,t1,t2,h=8):
    p=np.log(df.close.to_numpy());out=np.full(len(df),-1,dtype=np.int64)
    for a,b in ((0,t1),(t1,t2),(t2,len(p))):
        for i in range(a+8,b-h):
            trend=np.sign(p[i]-p[i-8])
            if trend:out[i]=int(np.any((p[i+1:i+h+1]-p[i])*trend<-.012))
    return out

def expert_labels(path,df):
    """Accept expert-reviewed per-candle labels only; never infer unseen intervals.
    CSV columns timestamp,label,label_status,annotator,known_at.
    """
    an=pd.read_csv(path)
    required={"timestamp","label","label_status","annotator","known_at"}
    if not required.issubset(an.columns):raise ValueError("need "+str(required))
    if not (an.label_status=="expert_verified").all():
        raise ValueError("expert CSV contains unverified annotations")
    if not an.label.isin(CLASSES).all():raise ValueError("unknown label")
    an["timestamp"]=pd.to_datetime(an.timestamp,utc=True)
    an["known_at"]=pd.to_datetime(an.known_at,utc=True)
    if (an.known_at<an.timestamp).any():raise ValueError("annotation predates candle")
    if an.timestamp.duplicated().any():raise ValueError("duplicate annotations")
    labels=dict(zip(an.timestamp,an.label))
    return np.array([CLASSES.index(labels.get(t,"unknown")) for t in df.timestamp],dtype=np.int64)

class Head(nn.Module):
    def __init__(self,dim):
        super().__init__();self.w=nn.Linear(dim,len(CLASSES))
        self.r=nn.Linear(dim,1);self.c=nn.Linear(dim,1)
    def forward(self,z):return self.w(z),self.r(z).flatten(),self.c(z).flatten()

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(len(SCALES)*NCHANNELS*LENGTH,96),
                                   nn.GELU(),nn.Dropout(.1),nn.Linear(96,64),nn.GELU())
        self.head=Head(64)
    def forward(self,x):return self.head(self.encoder(x))

class TCNBlock(nn.Module):
    def __init__(self,hidden,d):
        super().__init__();self.d=d;self.conv=nn.Conv1d(hidden,hidden,3,dilation=d)
        self.norm=nn.GroupNorm(4,hidden)
    def forward(self,x):
        z=self.conv(F.pad(x,(2*self.d,0)));return x+.35*F.gelu(self.norm(z))

class TCN(nn.Module):
    def __init__(self,multi=False):
        super().__init__();self.multi=multi
        count=len(SCALES) if multi else 1
        self.layers=nn.ModuleList([
            nn.Sequential(nn.Conv1d(NCHANNELS,16,1),
                          TCNBlock(16,1),TCNBlock(16,2),TCNBlock(16,4))
            for _ in range(count)])
        self.proj=nn.Sequential(nn.Linear(32*count,64),nn.GELU(),nn.Dropout(.1))
        self.head=Head(64)
    def forward(self,x):
        z=[self.layers[k](x[:,k]) for k in range(len(self.layers))]
        pooled=torch.cat([torch.cat([v[:,:,-1],v.mean(-1)],1) for v in z],1)
        return self.head(self.proj(pooled))

class GRUMulti(nn.Module):
    def __init__(self):
        super().__init__();self.rnns=nn.ModuleList([
            nn.GRU(NCHANNELS,24,batch_first=True) for _ in SCALES])
        self.proj=nn.Sequential(nn.Linear(24*len(SCALES),64),nn.GELU(),nn.Dropout(.1))
        self.head=Head(64)
    def forward(self,x):
        z=[net(x[:,k].transpose(1,2))[1][-1] for k,net in enumerate(self.rnns)]
        return self.head(self.proj(torch.cat(z,1)))

def factory(name):
    return {"mlp":MLP,"tcn_single":lambda:TCN(False),
            "tcn_multi":lambda:TCN(True),"gru_multi":GRUMulti}[name]()

def train(model,x,y,rev,xv,yv,rv,epochs=6,seed=31):
    seed_everything(seed);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.005)
    xb=torch.from_numpy(x);yb=torch.from_numpy(y)
    rb=torch.tensor(rev,dtype=torch.float32)
    vx=torch.from_numpy(xv);vy=torch.from_numpy(yv)
    vr=torch.tensor(rv,dtype=torch.float32)
    best=1e99;state=None
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(y))
        for start in range(0,len(y),96):
            idx=order[start:start+96]
            w,r,c=model(xb[idx]);known=yb[idx]!=0
            ce=F.cross_entropy(w,yb[idx],reduction="none")
            ce=torch.where(known,ce,.15*ce).mean()
            valid=rb[idx]>=0
            rloss=F.binary_cross_entropy_with_logits(r[valid],rb[idx][valid]) if valid.any() else 0*w.mean()
            correct=(w.detach().argmax(1)==yb[idx]).float()
            closs=F.binary_cross_entropy_with_logits(c,correct)
            loss=ce+.5*rloss+.15*closs
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        with torch.no_grad():
            vw,vrlog,_=model(vx)
            loss=F.cross_entropy(vw,vy,reduction="none")
            loss=torch.where(vy!=0,loss,.15*loss).mean()
            valid=vr>=0
            if valid.any():loss+=.5*F.binary_cross_entropy_with_logits(vrlog[valid],vr[valid])
            vl=float(loss)
        if vl<best:
            best=vl;state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    return model

def predict(model,x,temp=1.):
    model.eval();all_w=[];all_r=[];all_c=[]
    with torch.no_grad():
        for i in range(0,len(x),256):
            w,r,c=model(torch.from_numpy(x[i:i+256]))
            all_w.append(torch.softmax(w/temp,-1).numpy())
            all_r.append(torch.sigmoid(r).numpy())
            all_c.append(torch.sigmoid(c).numpy())
    return np.concatenate(all_w),np.concatenate(all_r),np.concatenate(all_c)

def f1_macro(y,yhat):
    labels=sorted(set(int(v) for v in y if v!=0))
    if not labels:return None
    fs=[]
    for cls in labels:
        tp=np.sum((y==cls)&(yhat==cls))
        fp=np.sum((y!=cls)&(yhat==cls))
        fn=np.sum((y==cls)&(yhat!=cls))
        fs.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    return float(np.mean(fs))

def scores(p,y,ry):
    w,r,c=p
    f=f1_macro(y,w.argmax(1))
    mask=ry>=0
    brier=float(np.mean((r[mask]-ry[mask])**2)) if mask.any() else None
    active=y!=0
    confidence=w.max(1)
    calibration=float(np.mean(abs((w.argmax(1)==y).astype(float)-confidence)))
    return {"macro_f1_labeled":f,"brier_reversal":brier,
            "wave_accuracy_labeled":float(np.mean(w.argmax(1)[active]==y[active])) if active.any() else None,
            "coverage_labeled":float(np.mean(active)),
            "confidence_error_abs":calibration}

def backtest(df,indices,probs,cost_bps=13.):
    w,rev,conf=probs;sig=(w[:,3]>=.20)&(rev<=.55)&(conf>=.15)
    mapping={int(idx):bool(v) for idx,v in zip(indices,sig)}
    cash=10000.;qty=0.;entry=0.;ntr=0;positive=negative=0.
    curve=[];peak=10000.;dd=0.
    start=int(indices[0]);end=int(indices[-1]);fee=cost_bps/10000.
    last_desire=False
    for i in range(start,end+1):
        desired=bool(mapping.get(i-1,False))
        # Hold the last decision until the next sampled signal: otherwise
        # non-sampled candles would force an exit and distort the benchmark.
        if i>start and i-1 not in mapping:desired=last_desire
        if i-1 in mapping:last_desire=desired
        px=float(df.open.iloc[i])
        if qty and not desired:
            cash=qty*px*(1-fee);pnl=cash-entry
            positive+=max(0,pnl);negative+=max(0,-pnl)
            qty=0;ntr+=1
        elif not qty and desired:
            entry=cash;qty=cash/(px*(1+fee));cash=0.
        value=cash+qty*float(df.close.iloc[i])
        peak=max(peak,value);dd=max(dd,1-value/peak);curve.append(value)
    if qty:
        cash=qty*float(df.close.iloc[end])*(1-fee)
        pnl=cash-entry;positive+=max(0,pnl);negative+=max(0,-pnl)
        ntr+=1;curve[-1]=cash
    return {"return_pct":float(100*(cash/10000-1)),
            "max_drawdown_pct":float(100*dd),"round_trips":ntr,
            "profit_factor":float(positive/negative) if negative else None,
            "n_signals":int(sig.sum())}

def run(args):
    seed_everything(args.seed);df=read_candles(args.csv)
    t=df.timestamp
    train_end=int(t.searchsorted(pd.Timestamp("2025-01-01",tz="UTC")))
    test=int(t.searchsorted(pd.Timestamp("2025-07-01",tz="UTC")))
    if train_end<600 or test-train_end<500 or len(df)-test<500:
        raise ValueError("Need complete 2024/2025 history for locked split")
    close=df.close.to_numpy()
    labels=expert_labels(args.labels,df) if args.labels else label_blocks(df,train_end,test)
    source="expert_verified" if args.labels else "RETROSPECTIVE_WEAK_SWING_LABELS"
    rev=future_reversal(df,train_end,test)
    cal=np.arange(len(df),dtype=float)
    info,speed=information_clock(df.volume.to_numpy(),args.alpha)
    assert np.all(np.diff(info)>0)
    indices={
        "train":np.arange(400,train_end-24,12),
        "validation":np.arange(train_end+48,test-24,12),
        "test":np.arange(test+48,len(df)-9,12),
    }
    paired={}
    for name,ids in indices.items():
        a,ia=features(df,cal,ids);b,ib=features(df,info,ids)
        common=np.intersect1d(ia,ib)
        a=a[np.isin(ia,common)];b=b[np.isin(ib,common)]
        paired[name]=(a,b,common)
    atr,btr,itr=paired["train"];av,bv,iv=paired["validation"]
    at,bt,it=paired["test"]
    if args.labels and sum(labels[itr]!=0)<50:raise ValueError("Insufficient expert-verified training labels (<50)")
    x=np.concatenate([atr,btr]);y=np.tile(labels[itr],2);ry=np.tile(rev[itr],2)
    xv=np.concatenate([av,bv]);yv=np.tile(labels[iv],2);rv=np.tile(rev[iv],2)
    reports={};candidates={}
    for name in ("mlp","tcn_single","tcn_multi","gru_multi"):
        print("training",name,flush=True)
        model=factory(name)
        model=train(model,x,y,ry,xv,yv,rv,epochs=args.epochs,seed=args.seed)
        avp=predict(model,av);bvp=predict(model,bv)
        am=predict(model,at);bm=predict(model,bt)
        vm={"calendar":scores(avp,labels[iv],rev[iv]),
            "information":scores(bvp,labels[iv],rev[iv])}
        tm={"calendar":scores(am,labels[it],rev[it]),
            "information":scores(bm,labels[it],rev[it])}
        validation_score=float(np.mean([z["macro_f1_labeled"] or 0 for z in vm.values()]))
        # Selection never sees test metrics or trading profits.
        reports[name]={"validation":vm,"test":tm,
                       "test_backtest":{"calendar":backtest(df,it,am,args.cost_bps),
                                        "information":backtest(df,it,bm,args.cost_bps)},
                       "validation_selection_score":validation_score}
        candidates[name]=model
    best=max(reports,key=lambda n:reports[n]["validation_selection_score"])
    args.out.mkdir(parents=True,exist_ok=True)
    torch.save({"model":candidates[best].state_dict(),"architecture":best,
                "input_scales":SCALES,"classes":CLASSES,"label_source":source},
               args.out/"best_by_validation.pt")
    report={"status":source,"data":"REAL OHLCV; Elliott waves not expert-confirmed" if not args.labels else "REAL OHLCV + expert-verified",
            "training_split":"2024","validation_split":"2025-H1","test_split":"2025-H2",
            "n_samples_per_clock":{k:len(v[2]) for k,v in paired.items()},
            "candidates":reports,
            "best_by_validation":best,"selection_rule":"Mean validation macro-F1 on non-unknown pseudo or expert wave classes",
            "alpha":args.alpha,"cost_bps_per_side":args.cost_bps,
            "limitations":["Best denotes selection on validation only; not universal optimum.",
                           "Proxy OHLCV volume is not direct informational surprise.",
                           "Without human consensus labels, benchmarks learn heuristic swings.",
                           "No data from test split was used for model fitting or architecture selection.",
                           "No train/test leakage via price features; offline retrospective targets allowed only for fitting."],
            "source_sha256":hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()}
    (args.out/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({"best_by_validation":best,"test":reports[best]["test"],
                      "test_backtest":reports[best]["test_backtest"],"status":source},
                     indent=2,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",type=Path,required=True)
    p.add_argument("--labels",type=Path,default=None)
    p.add_argument("--alpha",type=float,default=.8)
    p.add_argument("--epochs",type=int,default=6)
    p.add_argument("--cost-bps",type=float,default=13.)
    p.add_argument("--seed",type=int,default=31)
    p.add_argument("--out",type=Path,default=Path("research/runs"))
    args=p.parse_args()
    if args.epochs<1:raise ValueError("epochs must be positive")
    run(args)
if __name__=="__main__":main()
