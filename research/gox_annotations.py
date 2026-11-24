#!/usr/bin/env python3
"""Import GOX-style human-drawn waves into a provenance-aware event schema.

GOX-TryElliottWave is a drawing research MVP, NOT an open verified annotation
dataset. Export schema varies; this adapter accepts 'waves' objects with
start_date/end_date and prices, or normalized 'anchors'/'points'.
Unknown/ambiguous records are rejected, never guessed.
"""
import argparse, csv, datetime as dt, json, pathlib
from collections import defaultdict

FIELDS=("asset","timeframe","degree","wave","direction","extreme_time",
        "extreme_price","confirmation_time","known_at","annotator",
        "source","source_url","label_status","annotation_version")

def timestamp(v):
    if isinstance(v,(int,float)):
        if v>1e14: v/=1e6
        elif v>1e11: v/=1e3
        return dt.datetime.fromtimestamp(v,tz=dt.timezone.utc).isoformat()
    if not v: return None
    s=str(v).replace("Z","+00:00")
    if len(s)==10:s+="T00:00:00+00:00"
    d=dt.datetime.fromisoformat(s)
    if d.tzinfo is None:raise ValueError("Timezone required for times")
    return d.astimezone(dt.timezone.utc).isoformat()

def convert(obj):
    if isinstance(obj,list):obj={"drawings":obj}
    if not isinstance(obj,dict):raise ValueError("expected JSON object")
    asset=obj.get("symbol") or obj.get("asset")
    timeframe=obj.get("timeframe") or obj.get("interval")
    annotator=obj.get("annotator")
    source_url=obj.get("source_url")
    if not all([asset,timeframe,annotator]):raise ValueError("asset/timeframe/annotator required")
    records=[]
    drawings=obj.get("drawings") or obj.get("waves") or []
    if not isinstance(drawings,list):raise ValueError("drawings must be an array")
    for drawing in drawings:
        label=str(drawing.get("wave") or drawing.get("label") or "")
        if label not in "12345ABC" or len(label)!=1:
            continue
        degree=drawing.get("degree")
        if not degree:raise ValueError("wave degree is required")
        direction=str(drawing.get("direction") or obj.get("direction") or "")
        if direction not in ("bull","bear"):raise ValueError("direction bull/bear required")
        points=drawing.get("points") or drawing.get("anchors")
        if points:
            if len(points)<2:raise ValueError("two anchor points required")
            endpoint=points[-1]
            extreme=endpoint.get("time") or endpoint.get("timestamp") or endpoint.get("date")
            price=endpoint.get("price")
        else:
            extreme=drawing.get("end_date") or drawing.get("end_time")
            price=drawing.get("end_price")
        if not extreme or price is None:raise ValueError("missing wave end anchor")
        confirmed=(drawing.get("confirmed_at") or drawing.get("confirmation_time") or
                   obj.get("confirmed_at"))
        known=(drawing.get("known_at") or obj.get("published_at") or
               obj.get("observed_at"))
        # Keep unknown confirmation as unknown. Never substitute pivot time.
        t=timestamp(extreme)
        cf=timestamp(confirmed) if confirmed else None
        kn=timestamp(known) if known else None
        if cf and cf<t: raise ValueError("confirmation before extreme")
        if kn and cf and kn<cf:raise ValueError("known_at before confirmation")
        records.append(dict(asset=asset,timeframe=timeframe,degree=degree,wave=label,
                            direction=direction,extreme_time=t,extreme_price=float(price),
                            confirmation_time=cf,known_at=kn,
                            annotator=annotator,source="GOX-compatible human drawing",
                            source_url=source_url,label_status="human_unverified",
                            annotation_version=str(obj.get("version") or "1")))
    return records

def main():
    p=argparse.ArgumentParser()
    p.add_argument("input",help="Human-approved export JSON")
    p.add_argument("--out",default="research/gox_normalized.csv")
    args=p.parse_args()
    records=convert(json.loads(pathlib.Path(args.input).read_text()))
    out=pathlib.Path(args.out);out.parent.mkdir(exist_ok=True,parents=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(records)
    print(f"{len(records)} human-unverified waves -> {out}")
if __name__=="__main__":main()
