#!/usr/bin/env python3
"""Real temporal information diffusion graph from HN Search/Algolia.

Edges are explicit parent->reply relationships, never inferred from price.
Backfill is reconstructed in 2026, not a certified point-in-time archive:
timestamps reflect reported creation times, not an immutable as-of snapshot.
Only graph structure is kept (not comment text, author names, or user profiles).

Source/API: https://hn.algolia.com/api
"""
from __future__ import annotations
import argparse, calendar, csv, datetime as dt, hashlib, json
import pathlib, time, urllib.parse, urllib.request

ROOT="https://hn.algolia.com/api/v1"
TOPICS=("bitcoin","ethereum","crypto")
FIELDS=("id","parent_id","story_id","time_utc","depth","kind","topic")

def utc_epoch(value):
    if isinstance(value,(int,float)):return int(value)
    if isinstance(value,str):
        d=dt.datetime.fromisoformat(value.replace("Z","+00:00"))
        if d.tzinfo is None:raise ValueError("source timestamp missing UTC offset")
        return int(d.timestamp())
    raise ValueError("No timestamp found")

def request_json(url,timeout=35,retries=3):
    req=urllib.request.Request(url,headers={"User-Agent":"ElliottFractalResearch/1.0 (non-commercial, scientific)","Accept":"application/json"})
    last=None
    for k in range(retries):
        try:
            with urllib.request.urlopen(req,timeout=timeout) as res:
                raw=res.read(15_000_000)
            return json.loads(raw)
        except Exception as err:
            last=err
            if k<retries-1:time.sleep(1.5**k)
    raise RuntimeError(f"request failed after {retries} attempts: {url} : {last}")

def iter_months(start,end):
    y,m=map(int,start.split("-"));yy,mm=map(int,end.split("-"))
    while (y,m)<=(yy,mm):
        yield y,m
        m+=1
        if m==13:y,m=y+1,1

def range_epoch(y,m):
    lo=calendar.timegm((y,m,1,0,0,0))
    hi=calendar.timegm((y+1,1,1,0,0,0)) if m==12 else calendar.timegm((y,m+1,1,0,0,0))
    return lo,hi

def find_stories(query,y,m,limit=10):
    lo,hi=range_epoch(y,m)
    params=urllib.parse.urlencode({
        "query":query,"tags":"story","numericFilters":f"created_at_i>={lo},created_at_i<{hi}",
        "hitsPerPage":min(100,max(1,limit)),"page":0,
    })
    result=request_json(ROOT+"/search_by_date?"+params)
    hits=result.get("hits",[])
    # Require a topical story title at posting; don't use popularity today.
    token=query.lower()
    arr=[]
    for hit in hits:
        title=str(hit.get("title") or "").lower()
        if token not in title:continue
        if not hit.get("objectID") or not hit.get("created_at_i"):continue
        t=int(hit["created_at_i"])
        if lo<=t<hi:arr.append({"id":int(hit["objectID"]),"topic":token,"published":t})
    return arr[:limit]

def extract_thread(root,topic,limit_nodes=10000):
    """Preorder is source retrieval order, timestamps checked independently.
    Child edge uses parent from the actual nested comment tree.
    """
    sid=int(root["id"])
    stack=[(root,None,0)]
    records=[]
    seen=set()
    while stack:
        node,parent,depth=stack.pop()
        ident=node.get("id") or node.get("objectID")
        if ident is None:continue
        ident=int(ident)
        if ident in seen:continue
        seen.add(ident)
        ts=node.get("created_at_i",node.get("created_at"))
        if ts is None:continue
        epoch=utc_epoch(ts)
        actual_parent=node.get("parent_id")
        if parent is not None and actual_parent is not None:
            if int(actual_parent)!=int(parent):raise ValueError("Parent ID inconsistent with thread nesting")
        records.append({
            "id":ident,"parent_id":int(parent) if parent is not None else "",
            "story_id":sid,"time_utc":dt.datetime.fromtimestamp(epoch,dt.timezone.utc).isoformat(),
            "depth":depth,"kind":"story" if depth==0 else "comment",
            "topic":topic,
        })
        if len(records)>limit_nodes:raise ValueError("Story exceeds event cap; skip, never silently truncate")
        children=node.get("children") or []
        if not isinstance(children,list):raise ValueError("Unexpected children schema")
        for child in reversed(children):stack.append((child,ident,depth+1))
    # Validation uses only nodes that survived metadata checks.
    indexed={r["id"]:r for r in records}
    cleaned=[]
    for r in records:
        pid=r["parent_id"]
        if pid=="" or (pid in indexed and r["time_utc"]>=indexed[pid]["time_utc"]):
            cleaned.append(r)
    if not cleaned or cleaned[0]["id"]!=sid:
        raise ValueError("Root missing or mismatched")
    return cleaned

def collect(start,end,out,per_topic_per_month=10,topics=TOPICS,pause=.12,limit_nodes=10000):
    out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    jobs={}
    search_errors=[];errors=[]
    for y,m in iter_months(start,end):
        for topic in topics:
            try:
                matches=find_stories(topic,y,m,per_topic_per_month)
                for match in matches:
                    sid=match["id"]
                    if sid not in jobs:jobs[sid]=dict(match)
                    else:
                        prev=jobs[sid]["topic"].split("|")
                        if topic not in prev:jobs[sid]["topic"]+="|"+topic
            except Exception as exc:
                search_errors.append({"month":f"{y}-{m:02d}","topic":topic,"error":str(exc)})
            time.sleep(max(.08,pause))
    all_events=[];manifest=[];seen=set()
    for j,(sid,hit) in enumerate(sorted(jobs.items(),key=lambda pair:pair[1]["published"])):
        try:
            raw=request_json(ROOT+f"/items/{sid}")
            if int(raw.get("id") or raw.get("objectID") or -1)!=sid:
                raise ValueError("Story ID mismatch")
            nodes=extract_thread(raw,hit["topic"],limit_nodes=limit_nodes)
            for x in nodes:
                if x["id"] in seen:continue
                seen.add(x["id"]);all_events.append(x)
            manifest.append({
                "story_id":sid,"url":f"https://news.ycombinator.com/item?id={sid}",
                "api_url":ROOT+f"/items/{sid}","topic":hit["topic"],
                "published_utc":dt.datetime.fromtimestamp(hit["published"],dt.timezone.utc).isoformat(),
                "nodes":len(nodes),
                "sha256_structure":hashlib.sha256(
                    json.dumps(nodes,sort_keys=True).encode()).hexdigest(),
            })
        except Exception as exc:errors.append({"story_id":sid,"error":str(exc)})
        if (j+1)%25==0:print(f"fetched {j+1}/{len(jobs)} threads; events={len(all_events)}",flush=True)
        time.sleep(max(.08,pause))
    if not all_events:raise RuntimeError("No real social-network edges retrieved")
    all_events.sort(key=lambda r:(r["time_utc"],r["id"]))
    with (out/"hn_nodes.csv").open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=FIELDS);wr.writeheader();wr.writerows(all_events)
    metadata={
       "retrieved_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
       "start":start,"end":end,"topics":list(topics),"per_topic_per_month":per_topic_per_month,
       "source":"Hacker News Search by Algolia (public historical reconstruction)",
       "api_documentation":"https://hn.algolia.com/api",
       "limitations":["Not an independent point-in-time snapshot; source edits/deletions may bias past reconstruction",
                      "Only public Hacker News threads whose STORY TITLE matches the chosen keywords",
                      "Thread selection is incomplete; search results sorted by date and capped per month",
                      "Observed replies are conversation edges, not proof of actual influence or audience reach",
                      "No private user data or raw comment text persisted"],
       "stories":len(manifest),"events":len(all_events),
       "edges":sum(bool(x["parent_id"]!="") for x in all_events),
       "search_errors":search_errors,"thread_errors":errors,
       "threads":manifest,
    }
    (out/"provenance.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in metadata.items() if k!="threads"},indent=2,ensure_ascii=False))
    return metadata

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--start",default="2024-01");p.add_argument("--end",default="2025-12")
    p.add_argument("--topics",nargs="+",default=list(TOPICS))
    p.add_argument("--per-topic-per-month",type=int,default=10)
    p.add_argument("--out",default="research/graph_data")
    args=p.parse_args()
    collect(args.start,args.end,args.out,args.per_topic_per_month,tuple(args.topics))
if __name__=="__main__":main()
