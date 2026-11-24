#!/usr/bin/env python3
"""Import real, timestamped Hacker News *reply trees* for market-event research.

Edges connect explicit reply->parent records, not inferred human influence.
Algolia stories/search https://hn.algolia.com/api/
Algolia item comments https://hn.algolia.com/api/v1/items/<ID>
HN native schema https://github.com/HackerNews/API

Only structural IDs and times are stored (no usernames, message content, votes).
Retrospective API reads are NOT point-in-time snapshots; publication timestamps
are retained and current rank/comment counts are NEVER used as historical features.
"""
import argparse
import calendar
import csv
import datetime as dt
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

SOURCE="https://hn.algolia.com/api/v1"
TOPICS={"BTCUSDT":"bitcoin","ETHUSDT":"ethereum","SOLUSDT":"solana",
        "COVID":"coronavirus"}
FIELDS=("asset","topic","story_id","node_id","parent_id","created_at",
        "node_type","depth","source_url")

def api(path,query=None,timeout=30):
    url=SOURCE+path
    if query:url+="?"+urllib.parse.urlencode(query)
    req=urllib.request.Request(url,headers={"Accept":"application/json",
        "User-Agent":"TheTRIBE-HighTECH-elliott-research/1.0"})
    last=None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req,timeout=timeout) as response:
                content=response.read(8_000_000)
            return json.loads(content)
        except (urllib.error.HTTPError,urllib.error.URLError,TimeoutError) as e:
            last=e
            if attempt==3:break
            time.sleep(2**attempt)
    raise RuntimeError(f"Provider query failed: {url}: {last}")

def utc_time(event):
    value=event.get("created_at_i")
    if value is None:value=event.get("time")
    if value is not None:
        return dt.datetime.fromtimestamp(int(value),tz=dt.timezone.utc)
    raw=event.get("created_at")
    if raw is None:return None
    parsed=dt.datetime.fromisoformat(str(raw).replace("Z","+00:00"))
    if parsed.tzinfo is None:parsed=parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)

def flatten_story(tree,asset,topic,start,end,max_nodes=300):
    """Traverse reported parent-child links; emit valid nodes and parent edges."""
    if not isinstance(tree,dict):return []
    root_id=tree.get("id")
    if root_id is None:return []
    # Future comment count/rank is irrelevant; use only event timestamps and links.
    pending=[(tree,None,0)];seen=set();records=[]
    while pending:
        item,parent,depth=pending.pop()
        item_id=item.get("id")
        if item_id is None or item_id in seen:continue
        seen.add(item_id)
        ts=utc_time(item)
        if ts is not None and start<=ts<end and item.get("deleted") is not True:
            records.append({
               "asset":asset,"topic":topic,"story_id":int(root_id),
               "node_id":int(item_id),"parent_id":int(parent) if parent else "",
               "created_at":ts.isoformat(),"node_type":"story" if parent is None else "reply",
               "depth":depth,"source_url":f"https://news.ycombinator.com/item?id={int(root_id)}",
            })
        # The parent is metadata from the nesting; no inferred social edges.
        for child in reversed(item.get("children") or []):
            if isinstance(child,dict):
                pending.append((child,int(item_id),depth+1))
    # No future-based selection: earliest N events, not sorted by votes.
    records.sort(key=lambda r:(r["created_at"],r["node_id"]))
    return records[:max_nodes]

def month_ranges(start_ym,end_ym):
    y,m=map(int,start_ym.split("-"));Y,M=map(int,end_ym.split("-"))
    while (y,m)<=(Y,M):
        first=dt.datetime(y,m,1,tzinfo=dt.timezone.utc)
        last=dt.datetime(y+int(m==12),(m%12)+1,1,tzinfo=dt.timezone.utc)
        yield first,last
        y,m=last.year,last.month

def harvest(assets,start_ym,end_ym,stories_per_month=4,
            max_nodes_story=300,sleep_s=.2,source=api):
    records=[];ledger=[];ids=set()
    for asset in assets:
        if asset not in TOPICS:raise ValueError(f"Unsupported topic {asset}")
        term=TOPICS[asset]
        for start,end in month_ranges(start_ym,end_ym):
            query={"query":term,"tags":"story",
                   "numericFilters":f"created_at_i>={int(start.timestamp())},created_at_i<{int(end.timestamp())}",
                   "hitsPerPage":100,"page":0}
            search=source("/search_by_date",query)
            hits=search.get("hits") or []
            # Search_by_date newest-first: choose EARLIEST in each month;
            # never select by present-day score/descendant count.
            matching=[]
            for hit in hits:
                try:
                    when=utc_time(hit)
                    if when is not None and start<=when<end:
                        matching.append((when,int(hit.get("objectID") or hit.get("id"))))
                except (ValueError,TypeError):continue
            matching.sort()
            used=0;nodes=0;edges=0
            for _,story_id in matching[:stories_per_month]:
                tree=source(f"/items/{story_id}")
                part=flatten_story(tree,asset,term,start,end,max_nodes_story)
                for item in part:
                    key=(asset,item["node_id"])
                    if key not in ids:
                        ids.add(key);records.append(item)
                used+=1;nodes+=len(part);edges+=sum(bool(p["parent_id"]) for p in part)
                if sleep_s:time.sleep(sleep_s)
            ledger.append({"asset":asset,"month":start.strftime("%Y-%m"),
                           "search_hits_page":len(hits),"stories_fetched":used,
                           "nodes_extracted":nodes,"reply_rows":edges})
            print(f"{asset} {start:%Y-%m}: {used} stories, {nodes} events, {edges} reply rows",
                  flush=True)
            if sleep_s:time.sleep(sleep_s)
    records.sort(key=lambda x:(x["asset"],x["created_at"],x["node_id"]))
    return records,ledger

def save(records,ledger,directory):
    directory=pathlib.Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/"hn_reply_events.csv"
    with path.open("w",newline="",encoding="utf8") as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(records)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={
      "retrieved_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
      "provider":"Hacker News Algolia search + item reply trees",
      "api_base":SOURCE,
      "relation":"Reply-to-parent (discussion topology, not observed information transmission)",
      "historical_availability":"Retrospective extraction; does not prove time-specific index membership",
      "no_authors_or_text_collected":True,"records":len(records),
      "reply_rows":sum(bool(x["parent_id"]) for x in records),
      "sha256_csv":digest,"queries":ledger
    }
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--assets",nargs="+",choices=list(TOPICS),default=["BTCUSDT","ETHUSDT","SOLUSDT"])
    p.add_argument("--start",default="2024-01")
    p.add_argument("--end",default="2025-12")
    p.add_argument("--stories-month",type=int,default=4)
    p.add_argument("--max-nodes-story",type=int,default=300)
    p.add_argument("--spacing",type=float,default=.2)
    p.add_argument("--out",default="research/datasets/hn_graph")
    a=p.parse_args()
    if a.stories_month<1 or a.max_nodes_story<1:raise ValueError("Positive limits required")
    rows,ledger=harvest(a.assets,a.start,a.end,a.stories_month,
                        a.max_nodes_story,max(0.,a.spacing))
    manifest=save(rows,ledger,a.out)
    if len(rows)<1:raise ValueError("No public discussion graph events found")
    print(json.dumps({k:v for k,v in manifest.items() if k!="queries"},indent=2))
if __name__=="__main__":main()
