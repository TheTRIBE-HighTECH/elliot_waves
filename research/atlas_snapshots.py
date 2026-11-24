#!/usr/bin/env python3
"""Elliott Wave Atlas snapshots: one dated collection, attributable and immutable.

Source https://elliottwaveatlas.com/terms: attribution, no paid redistribution.
NO backdating. These snapshots are contemporaneous provider opinions, not
independent truth labels, and not expert-verified historical Elliott pivots.
"""
import argparse, datetime as dt, hashlib, json, pathlib, time, urllib.request

BASE="https://elliottwaveatlas.com"
CREDIT="Elliott Wave Atlas"
def read_json(path,timeout=30):
    request=urllib.request.Request(BASE+path,headers={"User-Agent":"ElliottResearch/1.0 contact: research; snapshots once/day","Accept":"application/json"})
    last=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request,timeout=timeout) as response:
                body=response.read(8_000_000)
            return json.loads(body)
        except Exception as exc:
            last=exc
            if attempt<2: time.sleep(2**attempt)
    raise RuntimeError(f"source unavailable: {path} ({last})")
def collect(folder, max_markets=100, spacing=1., include_full=True):
    folder=pathlib.Path(folder)
    now=dt.datetime.now(dt.timezone.utc)
    day=now.strftime("%Y-%m-%d")
    target=folder/day
    target.mkdir(parents=True,exist_ok=True)
    index=read_json("/data/index.json")
    markets=index.get("markets",[])
    if not isinstance(markets,list) or not markets: raise ValueError("Empty or unexpected index")
    markets=sorted([m for m in markets if isinstance(m,dict) and m.get("slug")],
                   key=lambda m:str(m["slug"]))
    if max_markets: markets=markets[:max_markets]
    statuses=[]; errors=[]
    for m in markets:
        slug=str(m["slug"])
        if not all(c.isalnum() or c in "-_" for c in slug): continue
        path="/data/markets/"+slug+".json"
        try:
            full=read_json(path) if include_full else m
            if full.get("symbol")!=m.get("symbol"):
                raise ValueError("index/market symbol mismatch")
            raw=json.dumps(full,sort_keys=True,ensure_ascii=False,indent=2).encode("utf-8")
            digest=hashlib.sha256(raw).hexdigest()
            # One file per capture UTC day, no overwriting an existing same-day capture.
            dest=target/(slug+".json")
            if not dest.exists(): dest.write_bytes(raw)
            statuses.append({
              "symbol":full.get("symbol"),"slug":slug,"asof":full.get("asof"),
              "retrieved_utc":now.isoformat(),"sha256":digest,
              "source_url":BASE+path,"license_attribution":CREDIT,
              "wave_count_is_opinion":True,
              "file":dest.name,
            })
        except Exception as e: errors.append({"slug":slug,"error":str(e)})
        time.sleep(spacing)
    report={
      "capture_utc":now.isoformat(),"provider":CREDIT,
      "source":BASE+"/data/index.json","terms":BASE+"/terms",
      "note":"Provider count snapshots. Not historical backfills or expert-confirmed labels.",
      "market_count":len(statuses),"errors":errors,"snapshots":statuses,
    }
    (target/"manifest.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    if not statuses: raise RuntimeError("No snapshots captured")
    return report
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default="research/snapshots")
    parser.add_argument("--max-markets",type=int,default=100)
    parser.add_argument("--spacing",type=float,default=1.)
    args=parser.parse_args()
    report=collect(args.out,args.max_markets,max(.5,args.spacing))
    print(json.dumps({k:v for k,v in report.items() if k!="snapshots"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
