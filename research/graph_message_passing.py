#!/usr/bin/env python3
"""Price TCN + REAL temporal reply-graph encoder, both with three prediction heads.

Identical network weights evaluate calendar and volume-deformed informational
clocks. Records are HISTORICAL real OHLCV and explicit HN parent->reply edges.
Weak ZigZag proxy labels are NOT verified Elliott waves, nor causal information
flow or an out-of-sample investing edge.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from compare_nn import (CLASSES, Head, SCALES, LENGTH, TCNBlock, read_candles,
                        label_blocks, future_reversal, information_clock,
                        features, scores, backtest, seed_everything)
from temporal_graph_features import load_events, make_hourly_graph, graph_windows, GRAPH_DIM

class PriceTower(nn.Module):
    """Multiscale causal TCN on market data."""
    def __init__(self):
        super().__init__()
        self.branches=nn.ModuleList([
            nn.Sequential(nn.Conv1d(4,16,1),TCNBlock(16,1),
                          TCNBlock(16,2),TCNBlock(16,4))
            for _ in SCALES])
        self.to_latent=nn.Sequential(nn.Linear(32*len(SCALES),64),nn.GELU())
    def forward(self,price):
        z=[branch(price[:,k]) for k,branch in enumerate(self.branches)]
        combined=torch.cat([torch.cat([h[:,:,-1],h.mean(2)],1) for h in z],1)
        return self.to_latent(combined)

class TemporalGraphTower(nn.Module):
    """Learned directed message passing on hourly time-bins.

    A[b,k,dst,src] is >0 only if a real reply at dst responded to a source
    parent at src and both were posted before the decision candle. A is
    row-normalized and includes each bin's self loop.
    """
    def __init__(self):
        super().__init__()
        self.node_init=nn.Linear(GRAPH_DIM,16)
        self.self_m=nn.ModuleList([nn.Linear(16,16) for _ in range(2)])
        self.msg_m=nn.ModuleList([nn.Linear(16,16) for _ in range(2)])
        self.pool=nn.Sequential(nn.Linear(32*len(SCALES),64),nn.GELU())
    def forward(self,features,adj):
        out=[]
        for k in range(len(SCALES)):
            nodes=F.gelu(self.node_init(features[:,k].transpose(1,2)))
            for self_layer,message_layer in zip(self.self_m,self.msg_m):
                messages=torch.bmm(adj[:,k],nodes)
                nodes=F.gelu(self_layer(nodes)+message_layer(messages))
            out.append(torch.cat([nodes[:,-1],nodes.mean(1)],dim=1))
        return self.pool(torch.cat(out,dim=1))

class PriceGraphFusion(nn.Module):
    """Three-headed neural detector with explicit price and graph towers."""
    def __init__(self):
        super().__init__()
        self.price=PriceTower()
        self.graph=TemporalGraphTower()
        self.gate=nn.Linear(128,64)
        self.fuse=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Dropout(.10))
        self.head=Head(64)
    def forward(self,price,graph,adj,graph_enabled=True):
        p=self.price(price)
        if graph_enabled:
            g=self.graph(graph,adj)
            # If sample has no observed discussion events, no graph contribution.
            exists=(graph.abs().sum(dim=(1,2,3))>0).float()[:,None]
            gated=g*torch.sigmoid(self.gate(torch.cat([p,g],dim=1)))*exists
        else:
            gated=torch.zeros_like(p)
        z=self.fuse(torch.cat([p,gated],dim=1))
        return self.head(z)

def split_samples(df,axis,graph,ids):
    x,px_ids=features(df,axis,ids)
    g,a,g_ids=graph_windows(graph,axis,ids,SCALES,LENGTH)
    if not np.array_equal(px_ids,g_ids):
        common=np.intersect1d(px_ids,g_ids)
        x=x[np.isin(px_ids,common)];g=g[np.isin(g_ids,common)]
        a=a[np.isin(g_ids,common)];px_ids=common
    return x,g,a,px_ids

def tensorize(x):return torch.from_numpy(np.ascontiguousarray(x))

def batch_loss(pred,y,rev,weights):
    w,r,c=pred
    ce=F.cross_entropy(w,y,weight=weights)
    valid=rev>=0
    revloss=F.binary_cross_entropy_with_logits(r[valid],rev[valid]) if valid.any() else w.mean()*0
    # Empirical confidence target, calibrated subsequently on validation.
    confidence_target=(w.detach().argmax(1)==y).float()
    conf_loss=F.binary_cross_entropy_with_logits(c,confidence_target)
    return ce+.5*revloss+.15*conf_loss

def run_training(model,train_samples,train_labels,train_rev,
                 validation_samples,validation_labels,validation_rev,
                 epochs=8,seed=31,graph_enabled=True):
    seed_everything(seed)
    weights=np.bincount(train_labels,minlength=len(CLASSES)).astype(float)
    weights=np.where(weights>0,1./np.sqrt(np.maximum(weights,1)),0.)
    weights/=np.mean(weights[weights>0])
    weights=np.clip(weights,0,6)
    weights[0]=min(.20,weights[0])
    w_tensor=torch.tensor(weights,dtype=torch.float32)
    px=tensorize(train_samples[0]);gx=tensorize(train_samples[1]);adj=tensorize(train_samples[2])
    y=torch.tensor(train_labels,dtype=torch.long)
    r=torch.tensor(train_rev,dtype=torch.float32)
    vpx=tensorize(validation_samples[0]);vg=tensorize(validation_samples[1]);va=tensorize(validation_samples[2])
    vy=torch.tensor(validation_labels,dtype=torch.long)
    vr=torch.tensor(validation_rev,dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.005)
    best=float("inf");state=None
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(y))
        for step in range(0,len(y),64):
            sel=order[step:step+64]
            result=model(px[sel],gx[sel],adj[sel],graph_enabled=graph_enabled)
            loss=batch_loss(result,y[sel],r[sel],w_tensor)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        model.eval()
        with torch.no_grad():
            out=model(vpx,vg,va,graph_enabled=graph_enabled)
            validation_loss=float(batch_loss(out,vy,vr,w_tensor))
        print(f"epoch {epoch+1}/{epochs} validation_loss={validation_loss:.4f}",flush=True)
        if validation_loss<best:
            best=validation_loss
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    return model,best

def infer(model,samples,graph_enabled=True):
    px,g,adj=samples
    probabilities=[];reversals=[];confidence=[]
    model.eval()
    with torch.no_grad():
        for k in range(0,len(px),128):
            w,r,c=model(tensorize(px[k:k+128]),tensorize(g[k:k+128]),
                        tensorize(adj[k:k+128]),graph_enabled=graph_enabled)
            probabilities.append(torch.softmax(w,dim=1).numpy())
            reversals.append(torch.sigmoid(r).numpy())
            confidence.append(torch.sigmoid(c).numpy())
    return np.concatenate(probabilities),np.concatenate(reversals),np.concatenate(confidence)

def score_eval(model,segments,labels,rev,df,bps,graph_enabled):
    outcome={}
    for clock,(px,g,adj,idx) in segments.items():
        prob=infer(model,(px,g,adj),graph_enabled)
        outcome[clock]={"classification":scores(prob,labels[idx],rev[idx]),
                        "backtest":backtest(df,idx,prob,bps)}
    return outcome

def main_experiment(args):
    seed_everything(args.seed)
    df=read_candles(args.csv)
    ev=load_events(args.events,args.asset)
    if ev.empty:raise ValueError("Zero source events for asset; fail closed")
    graph=make_hourly_graph(df,ev)
    if graph.valid_edge_count==0:raise ValueError("No verified parent-child reply edges inside market window")
    train_end=int(df.timestamp.searchsorted(pd.Timestamp("2025-01-01",tz="UTC")))
    test_start=int(df.timestamp.searchsorted(pd.Timestamp("2025-07-01",tz="UTC")))
    if train_end<600 or test_start-train_end<500 or len(df)-test_start<500:
        raise ValueError("Need 2024/2025 complete, split locked")
    # Weak labels ONLY; the historical price data are genuine but Elliott truth
    # is NOT externally verified. This distinction is in every output report.
    labels=label_blocks(df,train_end,test_start)
    reversal=future_reversal(df,train_end,test_start)
    calendar=np.arange(len(df),dtype=float)
    clock_input=(np.expm1(graph.hourly[:,0]) if args.clock_source=="graph"
                 else df.volume.to_numpy())
    informational,speed=information_clock(clock_input,args.alpha)
    if not np.all(np.diff(informational)>0):raise AssertionError("Nonmonotone clock")

    ids={
      "train":np.arange(400,train_end-24,12),
      "validation":np.arange(train_end+48,test_start-24,12),
      "test":np.arange(test_start+48,len(df)-9,12)
    }
    splits={}
    for part,indices in ids.items():
        chronological=split_samples(df,calendar,graph,indices)
        deformed=split_samples(df,informational,graph,indices)
        common=np.intersect1d(chronological[3],deformed[3])
        def select(item):
            x,g,a,ix=item;which=np.isin(ix,common)
            return x[which],g[which],a[which],ix[which]
        splits[part]={"calendar":select(chronological),"information":select(deformed)}

    tr=splits["train"];val=splits["validation"];te=splits["test"]
    tri=tr["calendar"][3];vai=val["calendar"][3]
    def joined(seg):
        return tuple(np.concatenate([seg["calendar"][j],seg["information"][j]],axis=0)
                     for j in range(3))
    xtr=joined(tr);xv=joined(val)
    ytr=np.tile(labels[tri],2);rtr=np.tile(reversal[tri],2)
    yv=np.tile(labels[vai],2);rv=np.tile(reversal[vai],2)

    result={}
    for name,enabled in [("price_only",False),("price_plus_reply_graph",True)]:
        seed_everything(args.seed)
        model=PriceGraphFusion()
        model,vl=run_training(model,xtr,ytr,rtr,xv,yv,rv,
                              args.epochs,args.seed,graph_enabled=enabled)
        result[name]={
          "best_validation_loss":vl,
          "validation":score_eval(model,val,labels,reversal,df,args.bps,enabled),
          "test":score_eval(model,te,labels,reversal,df,args.bps,enabled)
        }
        args.out.mkdir(parents=True,exist_ok=True)
        torch.save({
          "state":model.state_dict(),"graph_enabled":enabled,
          "architecture":"multiscale_causal_TCN_plus_directed_reply_graph",
          "classes":CLASSES,"scales":SCALES,
          "label_status":"RETROSPECTIVE_SWING_PROXY"
        },args.out/(name+".pt"))
        if enabled:
            result[name]["ablation_no_graph_same_weights"]=score_eval(
                model,te,labels,reversal,df,args.bps,False)

    meta={
      "asset":args.asset,"events_count":graph.event_count,
      "observed_parent_child_edges":graph.valid_edge_count,
      "graph_hourly_coverage":graph.coverage,
      "source":"Hacker News reply-parent relationships, past posts only",
      "source_type":"Retrospective search with real source event timestamps",
      "label_status":"RETROSPECTIVE_SWING_PROXY_NOT_VERIFIED_ELLIOTT",
      "training_year":2024,"validation":"2025-H1","test":"2025-H2",
      "trained_weights_shared_across_clocks":True,
      "graph_windows_directed_and_causal":True,
      "cost_bps_per_side":args.bps,"alpha_clock":args.alpha,
      "clock_source":args.clock_source,
      "speed_min":float(speed.min()),"speed_max":float(speed.max()),
      "n_samples":{k:len(v["calendar"][3]) for k,v in splits.items()},
      "ohlcv_sha256":hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
      "graph_sha256":hashlib.sha256(Path(args.events).read_bytes()).hexdigest(),
      "caveats":[
        "HN replies represent real discussion structure, NOT direct measured information transmission.",
        "Story selection from today's search index can induce retrospective selection/survivorship bias.",
        "Without independent analyst labels, wave F1 only measures agreement with pseudo-labels.",
        "No information was obtained from future candles or future replies in inference windows.",
        "Graph-event and volume clocks are distinct; the clock source is recorded per run.",
        "Trade signal on candle close, execution next open; no lookahead fills, spot long only."
      ]
    }
    report={"metadata":meta,"models":result}
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/"graph_fusion_report.json").write_text(
         json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps({"metadata":meta,
       "holdout_calendar":{
          k:{"f1":v["test"]["calendar"]["classification"]["macro_f1_labeled"],
             "return_pct":v["test"]["calendar"]["backtest"]["return_pct"]}
          for k,v in result.items()},
       "holdout_information":{
          k:{"f1":v["test"]["information"]["classification"]["macro_f1_labeled"],
             "return_pct":v["test"]["information"]["backtest"]["return_pct"]}
          for k,v in result.items()}},indent=2,ensure_ascii=False))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--csv",required=True,type=Path)
    parser.add_argument("--events",required=True,type=Path)
    parser.add_argument("--asset",default="BTCUSDT")
    parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--alpha",type=float,default=.8)
    parser.add_argument("--clock-source",choices=["graph","volume"],default="graph")
    parser.add_argument("--bps",type=float,default=13.)
    parser.add_argument("--seed",type=int,default=31)
    parser.add_argument("--out",type=Path,default=Path("research/runs/hn_graph"))
    args=parser.parse_args()
    if args.epochs<1:raise ValueError("Positive epochs required")
    main_experiment(args)
if __name__=="__main__":main()
