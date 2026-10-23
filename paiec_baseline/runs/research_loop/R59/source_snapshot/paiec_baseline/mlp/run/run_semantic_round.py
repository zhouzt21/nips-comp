"""R7 same-token-window semantic versus lexical local adaptation."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from mlp.evaluate import _load_directory,_stable_seed,BUDGETS,ALC_WEIGHTS
from mlp.semantic_features import SemanticFeatures,MODEL_REVISION
from mlp.local_ridge import kernel_predict
from mlp.reporting.summarize_candidate import summarize

ROOT=Path("runs/research_loop/R7");RIDGES=[None,16.,4.,1.,.25]


def episodes(directory,seed,max_pairs,engine):
    key=hashlib.sha256(f"{directory}|{seed}|{max_pairs}|64|{engine.fingerprint}".encode()+Path(__file__).read_bytes()).hexdigest()
    path=Path("D:/paiec_research_cache/semantic_episodes")/(key+".npz")
    names=["k_semantic","c_semantic","k_text_hash","c_text_hash","a","y"]
    if path.exists():
        with np.load(path,allow_pickle=False) as pack:return [{k:pack[f"{i}_{k}"] for k in names} for i in range(int(pack["count"]))]
    responses,subjects,items=_load_directory(Path("D:/paiec_payload/data"),directory)
    groups=[(s,g) for s,g in responses.groupby("subject_id",sort=False) if g["item_id"].nunique()>=80 and s in subjects]
    if len(groups)>max_pairs:
        rng=np.random.default_rng(_stable_seed(2026,directory,"pair_sample"))
        groups=[groups[i] for i in np.sort(rng.choice(len(groups),max_pairs,replace=False))]
    pending=[];texts={}
    for sid,group in groups:
        ids=pd.unique(group["item_id"]).copy();rng=np.random.default_rng(_stable_seed(seed,directory,sid));rng.shuffle(ids)
        cut=len(ids)//2;acquired_ids=ids[:31];evaluation_ids=ids[cut:cut+64]
        acquired=group[group["item_id"].isin(acquired_ids)].sample(frac=1,random_state=int(rng.integers(2**31)))
        acquired=acquired.drop_duplicates("item_id").set_index("item_id").loc[acquired_ids].reset_index()
        evaluation=group[group["item_id"].isin(evaluation_ids)]
        item_ids=[*acquired["item_id"],*evaluation["item_id"]]
        for item_id in item_ids:texts[item_id]=items[item_id]["content"]
        pending.append((item_ids,acquired["response"].to_numpy(float),evaluation["response"].to_numpy(float)))
    vectors=engine.encode_many(list(texts.values())) if texts else {}
    keys={item_id:engine.key(text) for item_id,text in texts.items()};out=[]
    for ids,a,y in pending:
        row={"a":a,"y":y}
        for mode in ["semantic","text_hash"]:
            x=np.stack([vectors[keys[item_id]][mode] for item_id in ids]);support=x[:31]
            row["k_"+mode]=support@support.T;row["c_"+mode]=x[31:]@support.T
        out.append(row)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,count=len(out),**{f"{i}_{k}":v for i,row in enumerate(out) for k,v in row.items()})
    print(f"encoded {directory} seed={seed} pairs={len(out)} new_embeddings_total={engine.encoded}",flush=True)
    return out


def scores(row,mode,ridge):
    return np.array([np.mean((kernel_predict(row['k_'+mode],row['c_'+mode],row['a'][:n],ridge)-row['y'])**2) for n in BUDGETS])


def main():
    ROOT.mkdir(parents=True,exist_ok=True);start=time.time();engine=SemanticFeatures();selections=[]
    for fold in range(3):
        split=json.loads(Path(f"runs/mlp_cv/fold{fold}_split.json").read_text());sums={k:np.zeros(len(RIDGES)) for k in ["semantic","text_hash"]};count=0
        for directory in split["train_benchmarks"]:
            for row in episodes(directory,73,25,engine):
                for mode in sums:sums[mode]+=[scores(row,mode,r)@ALC_WEIGHTS for r in RIDGES]
                count+=1
        selected={"fold":fold,"train_benchmarks":split["train_benchmarks"],"heldout_benchmarks":split["development_benchmarks"],
                  "selection_pairs":count,"selection_seed":73,"ridge_grid":RIDGES,"model_revision":MODEL_REVISION,
                  "training_alc":{k:(v/count).tolist() for k,v in sums.items()},"selected_ridge":{k:RIDGES[int(np.argmin(v))] for k,v in sums.items()}}
        selections.append(selected);(ROOT/f"fold{fold}_selection.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
        print(f"selected fold{fold}: {selected['selected_ridge']}",flush=True)
    for fold,sel in enumerate(selections):
        historical=json.loads(Path(f"runs/research_loop/R3/fold{fold}_evaluation.json").read_text())
        names=["semantic","text_hash","smoothed_mean_2","training_selected_smoothed","inner_selected_smoothed","local_ridge","original:residual_2","empirical_mean"]
        totals={k:np.zeros(6) for k in names};by_benchmark={};count=0
        for directory in sel["heldout_benchmarks"]:
            rows=episodes(directory,0,100,engine)
            if not rows:continue
            old=historical["by_benchmark"][directory];sums={k:np.zeros(6) for k in names}
            assert len(rows)==old["pairs_per_seed"][0]
            for row in rows:
                for mode in ["semantic","text_hash"]:sums[mode]+=scores(row,mode,sel["selected_ridge"][mode])
                sums["smoothed_mean_2"]+=scores(row,"semantic",None)
            np.testing.assert_allclose(sums["smoothed_mean_2"]/len(rows),old["brier_mean"]["smoothed_mean_2"],atol=1e-7,rtol=0)
            for name in names[3:]:sums[name]=np.array(old["brier_mean"][name])*len(rows)
            by_benchmark[directory]={"pairs_per_seed":[len(rows)],"brier_mean":{k:(v/len(rows)).tolist() for k,v in sums.items()},"alc_mean":{k:float(v/len(rows)@ALC_WEIGHTS) for k,v in sums.items()}}
            count+=len(rows)
            for k,v in sums.items():totals[k]+=v
        out={"selection":sel,"pairs_per_seed":[count],"by_benchmark":by_benchmark,"results":{k:{"brier_mean":(v/count).tolist(),"alc_mean":float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (ROOT/f"fold{fold}_evaluation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    summary=summarize([ROOT/f"fold{i}_evaluation.json" for i in range(3)],"semantic",["smoothed_mean_2","training_selected_smoothed","inner_selected_smoothed","text_hash","local_ridge","original:residual_2","empirical_mean"])
    summary["elapsed_seconds"]=time.time()-start;summary["new_embeddings"]=engine.encoded
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["pair_equal_alc","comparisons"]},indent=2))


if __name__=="__main__":main()
