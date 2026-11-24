"""Causal construction of graph features and explicit reply-edge adjacency.

Each candle i includes only events posted before candle i closes.
Edges represent actual discussion replies; no propagation/impact is assumed.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

GRAPH_DIM=8

@dataclass
class TemporalGraph:
    hourly: np.ndarray  # [n_candles, 8]
    edges: np.ndarray   # [n_edges, 2] parent_hour -> reply_hour
    event_count: int
    valid_edge_count: int
    coverage: float

def load_events(path,asset):
    data=pd.read_csv(path)
    required={"asset","node_id","parent_id","story_id","created_at","depth","node_type"}
    if not required.issubset(data.columns):
        raise ValueError("Missing graph fields: "+str(required-set(data.columns)))
    data=data.loc[data.asset==asset].copy()
    if data.empty:return data
    data["created_at"]=pd.to_datetime(data.created_at,utc=True,errors="raise")
    if data.node_id.duplicated().any():
        raise ValueError("duplicate graph nodes for asset")
    if not data.node_type.isin(["story","reply"]).all():
        raise ValueError("unexpected graph node type")
    if (data.depth<0).any():raise ValueError("negative graph depth")
    return data.sort_values(["created_at","node_id"]).reset_index(drop=True)

def make_hourly_graph(df,events):
    """Map source event time to its actual OHLCV bar; it is known after bar close.

    Event timestamps may be inside the candle; no feature is exposed until
    the end of that candle (trading model trades only on next bar).
    """
    n=len(df)
    feat=np.zeros((n,GRAPH_DIM),dtype="float32")
    empty=TemporalGraph(feat,np.empty((0,2),dtype=np.int32),0,0,0.)
    if events.empty:return empty
    times=pd.DatetimeIndex(pd.to_datetime(df.timestamp,utc=True))
    start=times.as_unit("ns").asi8  # explicit nanosecond precision across pandas 2/3
    end=int(start[-1]+3_600_000_000_000)
    index={}
    valid=[]
    for row in events.itertuples():
        ns=int(row.created_at.value)
        if ns<start[0] or ns>=end:continue
        candle=int(np.searchsorted(start,ns,side="right")-1)
        if candle<0 or candle>=n:continue
        if int(row.node_id) in index:raise ValueError("duplicate node_id")
        data={"id":int(row.node_id),"parent":int(row.parent_id) if pd.notna(row.parent_id) and str(row.parent_id).strip()!="" else None,
              "candle":candle,"depth":int(row.depth),
              "created_ns":ns,"kind":str(row.node_type),
              "story":int(row.story_id)}
        index[data["id"]]=data;valid.append(data)
    parents=[set() for _ in range(n)]
    stories=[set() for _ in range(n)]
    depths=np.zeros(n,dtype=float)
    lags=np.zeros(n,dtype=float)
    edges=[]
    rapid=np.zeros(n,dtype=float)
    for node in valid:
        i=node["candle"];feat[i,0]+=1
        if node["kind"]=="story":feat[i,2]+=1
        else:feat[i,1]+=1
        stories[i].add(node["story"])
        depths[i]+=node["depth"]
        p=index.get(node["parent"])
        if p is None:continue
        if p["created_ns"]>node["created_ns"]:
            raise ValueError("reply precedes its parent")
        edges.append((p["candle"],i))
        feat[i,3]+=1
        parents[i].add(p["id"])
        delay=(node["created_ns"]-p["created_ns"])/60_000_000_000
        lags[i]+=np.log1p(delay)
        rapid[i]+=float(delay<=60)
    for i in range(n):
        feat[i,4]=len(parents[i])
        feat[i,5]=depths[i]/max(1,feat[i,0])
        feat[i,6]=lags[i]/max(1,feat[i,3])
        feat[i,7]=rapid[i]/max(1,feat[i,3])
        feat[i,0:5]=np.log1p(feat[i,0:5])
        feat[i,5]=min(feat[i,5],20)/10
        feat[i,6]=min(feat[i,6],12)/12
    arr=np.asarray(edges,dtype=np.int32).reshape(-1,2)
    if len(arr):arr=arr[np.argsort(arr[:,1],kind="stable")]
    return TemporalGraph(feat,arr,len(valid),len(arr),float(np.mean(feat[:,0]>0)))

def graph_windows(graph,axis,indices,scales=(1,3,8),length=48):
    """Return [B,S,C,T] graph features and [B,S,T,T] real directed adjacency.

    Input graph derives only from observed past messages; transformed clock is
    strictly increasing. No edge whose reply occurs after the current i.
    """
    axis=np.asarray(axis,dtype=float)
    if len(axis)!=len(graph.hourly):raise ValueError("market/graph length mismatch")
    if np.any(np.diff(axis)<=0):raise ValueError("clock not monotone")
    edges=graph.edges
    source=edges[:,0] if len(edges) else np.array([],dtype=int)
    dest=edges[:,1] if len(edges) else np.array([],dtype=int)
    sample_g=[];sample_a=[];kept=[]
    for i in indices:
        values=[];adjs=[];valid=True
        for scale in scales:
            ticks=axis[i]-np.arange(length-1,-1,-1)*scale
            if ticks[0]<axis[0]:valid=False;break
            # All interpolated timestamps are <= the sample's last candle.
            hist=axis[:i+1]
            historical_index=np.searchsorted(hist,ticks,side="right")-1
            # Discrete comments are not linearly interpolated backwards in time.
            features=graph.hourly[historical_index].T.astype("float32")
            adjacency=np.eye(length,dtype="float32")
            if len(edges):
                lo=np.searchsorted(dest,0)   # edges are sorted by destination candle
                hi=np.searchsorted(dest,i,side="right")
                if hi>lo:
                    src=source[lo:hi];dst=dest[lo:hi]
                    relevant=(axis[src]>=ticks[0]) & (axis[dst]>=ticks[0])
                    src=src[relevant];dst=dst[relevant]
                    if len(src):
                        a=np.searchsorted(ticks,axis[src],side="right")-1
                        b=np.searchsorted(ticks,axis[dst],side="right")-1
                        a=np.clip(a,0,length-1);b=np.clip(b,0,length-1)
                        np.add.at(adjacency,(b,a),1.)
            # Row-normalize incoming messages; self-loops permit no-event slots.
            adjacency/=np.maximum(adjacency.sum(1,keepdims=True),1)
            values.append(features);adjs.append(adjacency)
        if valid:
            kept.append(i)
            sample_g.append(np.stack(values));sample_a.append(np.stack(adjs))
    if not kept:
        raise ValueError("empty graph feature windows")
    return np.stack(sample_g),np.stack(sample_a),np.asarray(kept,dtype=int)
