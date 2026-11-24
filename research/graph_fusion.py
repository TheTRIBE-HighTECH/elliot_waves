#!/usr/bin/env python3
"""Fuse REAL timestamped reply graphs into the existing 3-head Elliott TCN.

Shared neural weights across four controlled inference modes:
 (1) chronological/no-graph, (2) chronological/graph features,
 (3) graph clock/no-graph, (4) graph clock/graph features.

Real OHLCV + real HN reply trees. Elliott targets remain WEAK retrospective
swing pseudo-labels, NOT verified human EW annotations. Dataset reconstructed
after the fact (not contemporaneous immutable 2024/25 availability).
"""
from __future__ import annotations
import argparse, dataclasses, json, pathlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from compare_nn import (
    CLASSES, SCALES, LENGTH, NCHANNELS, Head, TCNBlock, backtest, features,
    label_blocks, future_reversal, scores, train, predict, read_candles,
    seed_everything,
)

GRAPH_CHANNELS=10
MODES=("calendar_no_graph","calendar_graph","graphclock_no_graph","graphclock_graph")

def graph_features_hourly(candles,nodes):
    """Causal per-hour graph statistics. Edges are actual parent->reply IDs.
    A comment in hourly candle i is available at close of candle i, never at open.
    """
    df=candles
    out=np.zeros((len(df),GRAPH_CHANNELS),dtype=np.float32)
    cdate=pd.DatetimeIndex(df.timestamp)
    if not cdate.is_monotonic_increasing:raise ValueError("candles not ordered")
    edges=nodes.copy()
    required={"id","parent_id","story_id","time_utc","depth","kind"}
    if not required.issubset(edges.columns):raise ValueError("HN node fields missing")
    if edges["id"].duplicated().any():raise ValueError("duplicate node IDs")
    edges["time_utc"]=pd.to_datetime(edges.time_utc,utc=True)
    edges["hour"]=edges.time_utc.dt.floor("h")
    edges=edges[edges.hour.isin(cdate)].copy()
    if len(edges)==0:raise ValueError("No HN nodes overlap with candle timestamps")
    lookup={int(v):x for v,x in enumerate(cdate)}
    time_to_index={v:i for i,v in lookup.items()}
    global_id_time=dict(zip(nodes.id.astype(int),
                            pd.to_datetime(nodes.time_utc,utc=True)))
    for hour,grp in edges.groupby("hour",sort=True):
        i=time_to_index[pd.Timestamp(hour)]
        comments=grp[grp.kind=="comment"]
        depths=comments.depth.to_numpy(dtype=float)
        n=len(comments)
        by_parent=comments.parent_id.dropna().astype(str).value_counts()
        num_sibling_pairs=sum(v*(v-1)/2 for v in by_parent if v>1)
        max_pairs=n*(n-1)/2 if n>1 else 1
        delays=[]
        for r in comments.itertuples():
            try:pid=int(r.parent_id)
            except (ValueError,TypeError):continue
            if pid in global_id_time:
                d=(r.time_utc-global_id_time[pid]).total_seconds()/3600.
                if d>=0:delays.append(np.log1p(d))
        out[i]=np.array([
            np.log1p(len(grp)),np.log1p(n),
            np.log1p(int((grp.kind=="story").sum())),
            np.log1p(int(np.sum(depths>=2))), min(10.,max(depths,default=0))/5.,
            float(np.mean(depths)/5.) if n else 0.,
            float(num_sibling_pairs/max_pairs),
            float(np.mean(delays)/4.) if delays else 0.,
            np.log1p(grp.story_id.nunique()),
            float(((by_parent/n)**2).sum()) if n else 0.,
        ],dtype=np.float32)
    return out

def graph_clock(graph,alpha=.6):
    """Endogenous graph clock using ONLY observed end-of-bar cascade counts.

    No lookahead: previous 168 hourly records form the normalization; the
    current hour's posts/comments contribute only to that hour's close signal.
    """
    events=pd.Series(np.asarray(graph[:,0],dtype=float))
    prior_mean=events.shift(1).rolling(168,min_periods=24).mean()
    prior_sd=events.shift(1).rolling(168,min_periods=24).std(ddof=0).clip(lower=.2)
    z=((events-prior_mean)/prior_sd).fillna(0).clip(-2.5,2.5).to_numpy()
    velocity=np.clip(np.exp(alpha*z),.35,3.5)
    timeline=np.zeros(len(velocity),dtype=float)
    timeline[1:]=np.cumsum(velocity[1:])
    if not np.all(np.diff(timeline)>0):raise AssertionError("Time must advance")
    return timeline,velocity

def sample_graph(graph,clock,index):
    # Piecewise constant backward-looking sampling; NEVER interpolate into future.
    length=len(graph)
    xs=[]
    for i in index:
        channel=[]
        for scale in SCALES:
            targets=clock[i]-np.arange(LENGTH-1,-1,-1)*scale
            if targets[0]<clock[0]:raise ValueError("Not enough past graph history")
            j=np.searchsorted(clock[:i+1],targets,side="right")-1
            assert np.max(j)<=i
            channel.append(graph[j].T)
        xs.append(np.stack(channel).astype("float32"))
    return np.stack(xs).astype("float32")

def fuse(candles,clock,graph,indices,enable_graph):
    p,ids=features(candles,clock,indices)
    g=sample_graph(graph,clock,ids) if enable_graph else np.zeros(
        (len(ids),len(SCALES),GRAPH_CHANNELS,LENGTH),dtype="float32")
    return np.concatenate([p,g],axis=2).astype("float32"),ids

class TemporalGraphTCN(nn.Module):
    """Graph-conditioned causal TCN. Graph edge structure is encoded as
    past-only hourly topological features. This is NOT message-passing GNN.
    """
    def __init__(self):
        super().__init__()
        self.branches=nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(NCHANNELS+GRAPH_CHANNELS,24,1),
                TCNBlock(24,1),TCNBlock(24,2),TCNBlock(24,4),
                TCNBlock(24,8),
            ) for _ in SCALES])
        self.fuse=nn.Sequential(nn.Linear(48*len(SCALES),96),
                                nn.GELU(),nn.Dropout(.12),
                                nn.Linear(96,64),nn.GELU())
        self.head=Head(64)
    def forward(self,x):
        z=[layer(x[:,s]) for s,layer in enumerate(self.branches)]
        pooled=torch.cat([torch.cat([a[:,:,-1],a.mean(-1)],dim=1) for a in z],dim=1)
        return self.head(self.fuse(pooled))

def evaluate(df,nodes,epochs=8,alpha=.6,cost_bps=13.,seed=41,out=None):
    seed_everything(seed)
    times=df.timestamp
    t1=int(times.searchsorted(pd.Timestamp("2025-01-01",tz="UTC")))
    t2=int(times.searchsorted(pd.Timestamp("2025-07-01",tz="UTC")))
    if t1<500 or t2-t1<500 or len(df)-t2<500:raise ValueError("Need 2024/2025 history")
    g=graph_features_hourly(df,nodes)
    clock=np.arange(len(df),dtype=float)
    gclock,speed=graph_clock(g,alpha)
    labels=label_blocks(df,t1,t2)
    rev=future_reversal(df,t1,t2)
    ix={
      "train":np.arange(400,t1-24,12,dtype=int),
      "validation":np.arange(t1+48,t2-24,12,dtype=int),
      "test":np.arange(t2+48,len(df)-9,12,dtype=int),
    }
    modespec={
      "calendar_no_graph":(clock,False),
      "calendar_graph":(clock,True),
      "graphclock_no_graph":(gclock,False),
      "graphclock_graph":(gclock,True)
    }
    splits={}
    for split,ids in ix.items():
        allm={}
        for m,(axis,enabled) in modespec.items():
            x,ki=fuse(df,axis,g,ids,enabled)
            allm[m]=(x,ki)
        common=sorted(set.intersection(*(set(z[1]) for z in allm.values())))
        common=np.asarray(common,dtype=int)
        if len(common)<50:raise ValueError("too few common samples")
        splits[split]=(common,{m:z[0][np.isin(z[1],common)]
                               for m,z in allm.items()})
    ii,xs=splits["train"];iv,vs=splits["validation"];it,ts=splits["test"]
    xt=np.concatenate([xs[m] for m in MODES])
    yt=np.tile(labels[ii],len(MODES));rt=np.tile(rev[ii],len(MODES))
    xv=np.concatenate([vs[m] for m in MODES])
    yv=np.tile(labels[iv],len(MODES));rv=np.tile(rev[iv],len(MODES))
    model=TemporalGraphTCN()
    model=train(model,xt,yt,rt,xv,yv,rv,epochs=epochs,seed=seed)
    report={"label_status":"RETROSPECTIVE_WEAK_SWING_LABELS",
      "graph_source":"real Hacker News nested comment/reply graph (historical reconstruction)",
      "graph_nodes":int(len(nodes)),"graph_hour_count":int((g[:,0]>0).sum()),
      "clock_velocity_min":float(speed.min()),"clock_velocity_max":float(speed.max()),
      "time_splits":{"train":"2024","val":"2025-H1","test":"2025-H2"},
      "architecture":"causal graph-conditioned multiscale TCN with 3 heads",
      "heads":["17 wave states","8-hour future price reversal probability","classification-confidence surrogate"],
      "same_weights_all_four_modes":True,
      "sample_counts":{k:len(v[0]) for k,v in splits.items()},
      "modes":{},
      "limitations":[
        "Real graph edges and timestamps, but retrospective HN snapshot is not authenticated historical availability.",
        "HN is a biased sample of tech/crypto discourse, not full social transmission.",
        "Hourly conversation edges are replies, not causal influence proof.",
        "Elliott wave targets are heuristic and not manually verified.",
        "The graph encoder is trained on causal topology aggregates, not an end-to-end GNN.",
        "Graph clock and volume-clock comparisons must be evaluated separately.",
      ],
    }
    for m in MODES:
        vp=predict(model,vs[m]);tp=predict(model,ts[m])
        report["modes"][m]={
           "validation":scores(vp,labels[iv],rev[iv]),
           "test":scores(tp,labels[it],rev[it]),
           "backtest":backtest(df,it,tp,cost_bps)
        }
    if out:
        out=pathlib.Path(out);out.mkdir(exist_ok=True,parents=True)
        torch.save({"state_dict":model.state_dict(),"channels":NCHANNELS+GRAPH_CHANNELS,
                    "classes":CLASSES,"scales":SCALES,
                    "label_status":report["label_status"]},out/"graph_fusion_tcn.pt")
        (out/"comparison.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
        pd.DataFrame({"timestamp":df.timestamp,"hn_events_log":g[:,0],
                      "hn_comments_log":g[:,1],"graph_clock":gclock,
                      "graph_velocity":speed}).to_csv(out/"graph_clock.csv",index=False)
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",required=True)
    parser.add_argument("--nodes",required=True)
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--alpha",type=float,default=.6)
    parser.add_argument("--cost-bps",type=float,default=13.)
    parser.add_argument("--out",default="research/graph_runs")
    args=parser.parse_args()
    result=evaluate(read_candles(args.csv),pd.read_csv(args.nodes),epochs=args.epochs,
                    alpha=args.alpha,cost_bps=args.cost_bps,out=args.out)
    print(json.dumps({k:v for k,v in result.items() if k!="modes"},indent=2,ensure_ascii=False))
    print("Modes:",json.dumps({m:{"f1":v["test"]["macro_f1_labeled"],
                         "brier":v["test"]["brier_reversal"],
                         "return_pct":v["backtest"]["return_pct"]}
                         for m,v in result["modes"].items()},indent=2))
if __name__=="__main__":main()
