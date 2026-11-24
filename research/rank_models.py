#!/usr/bin/env python3
"""Select neural architecture across assets ONLY by validation F1.

Test periods are reported after model selection, not used to select champions.
No model is a verified Elliott detector without independently reviewed labels.
"""
import argparse, json, pathlib, statistics
NAMES=("mlp","tcn_single","tcn_multi","gru_multi")

def rank(files):
    reports=[json.loads(pathlib.Path(p).read_text()) for p in files]
    if not reports:raise ValueError("need at least one model report")
    if len(set(x.get("status") for x in reports))!=1:
        raise ValueError("cannot mix expert labels with weak/pseudo supervision")
    order=[]
    for name in NAMES:
        valid=[]
        for x in reports:
            for domain in ("calendar","information"):
                f=x["candidates"][name]["validation"][domain]["macro_f1_labeled"]
                if f is not None:valid.append(float(f))
        order.append({"model":name,"validation_mean_f1":statistics.mean(valid) if valid else None,
                      "n_asset_clock_splits":len(valid)})
    if not all(x["validation_mean_f1"] is not None for x in order):
        raise ValueError("insufficient labeled validation coverage")
    order.sort(key=lambda x:(-x["validation_mean_f1"],x["model"]))
    winner=order[0]["model"]
    holdout=[]
    for path,x in zip(files,reports):
        cand=x["candidates"][winner]
        holdout.append({
          "report":str(path),
          "test_f1_calendar":cand["test"]["calendar"]["macro_f1_labeled"],
          "test_f1_information":cand["test"]["information"]["macro_f1_labeled"],
          "return_calendar_pct":cand["test_backtest"]["calendar"]["return_pct"],
          "return_information_pct":cand["test_backtest"]["information"]["return_pct"],
        })
    return {"best_architecture_by_validation":winner,"validation_ranking":order,
            "locked_holdout_per_asset":holdout,"n_assets":len(files),
            "label_quality":reports[0]["status"],
            "caveat":"Best among four candidates only. Weak labels do not verify Elliott truth."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("reports",nargs="+")
    p.add_argument("--out",type=pathlib.Path,default=pathlib.Path("research/runs/ranking.json"))
    a=p.parse_args();r=rank(a.reports);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False))
    print(json.dumps(r,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
