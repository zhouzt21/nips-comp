"""R9 explicit cohort evidence diagnostic; not a claim about hosted sampling."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .evaluate import _load_directory,_stable_seed,BUDGETS,ALC_WEIGHTS
from .model import _SUBJECT_FIELDS
from .summarize_candidate import summarize
from shared_mean.model import from_counts

ROOT=Path("runs/research_loop/R9");KAPPAS=[2.,8.,32.];ALPHAS=[1.,2.,4.,8.,16.]


def records(directory,seed,cap):
    key=hashlib.sha256(f"{directory}|{seed}|{cap}|54074453da7b9911be0f48e4c560be547bc84d21".encode()+Path(__file__).read_bytes()).hexdigest()
    path=Path("D:/paiec_research_cache/shared")/(key+".json")
    if path.exists():return json.loads(path.read_text())
    responses,subjects,items=_load_directory(Path("D:/paiec_payload/data"),directory)
    groups=[(s,g) for s,g in responses.groupby("subject_id",sort=False) if g["item_id"].nunique()>=80 and s in subjects]
    if len(groups)>cap:
        rng=np.random.default_rng(_stable_seed(2026,directory,"pair_sample"));groups=[groups[i] for i in np.sort(rng.choice(len(groups),cap,replace=False))]
    out=[]
    for sid,group in groups:
        ids=pd.unique(group["item_id"]).copy();rng=np.random.default_rng(_stable_seed(seed,directory,sid));rng.shuffle(ids)
        cut=len(ids)//2;acquired_ids=ids[:31];evaluation_ids=ids[cut:cut+64]
        acquired=group[group["item_id"].isin(acquired_ids)].sample(frac=1,random_state=int(rng.integers(2**31)))
        acquired=acquired.drop_duplicates("item_id").set_index("item_id").loc[acquired_ids].reset_index()
        evaluation=group[group["item_id"].isin(evaluation_ids)]
        def plain(value):
            if value is None or pd.isna(value):return ""
            text=str(value).strip()
            return "" if text.lower() in {"nan","none","<na>","nat"} else text
        subject={field:plain(subjects[sid].get(field)) for field in _SUBJECT_FIELDS}
        out.append({"subject":subject,"key":json.dumps(subject,sort_keys=True),"a":acquired["response"].astype(int).tolist(),
                    "evaluation_mean":float(evaluation["response"].mean()),"evaluation_responses":len(evaluation),
                    "context_input_items":min(cut,64)+len(evaluation_ids)})
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out),encoding="utf-8")
    return out


def score_rows(rows,directory,cohort_size,kappa,alpha):
    order=np.arange(len(rows));np.random.default_rng(_stable_seed(193,directory,"cohort")).shuffle(order)
    scores={k:[] for k in ["shared_prior","smoothed_mean_2","selected_smoothed","empirical_mean"]}
    for offset in range(0,len(order),cohort_size):
        cohort=[rows[i] for i in order[offset:offset+cohort_size]]
        assert sum(r["context_input_items"] for r in cohort)<=1000
        for target in cohort:
            values={k:[] for k in scores}
            for n in BUDGETS:
                own_s=own_n=peer_s=peer_n=0
                for row in cohort:
                    if row["key"]==target["key"]:own_s+=sum(row["a"][:n]);own_n+=n
                    else:peer_s+=sum(row["a"][:n]);peer_n+=n
                predictions={"shared_prior":from_counts(own_s,own_n,peer_s,peer_n,kappa),
                    "smoothed_mean_2":(1+own_s)/(2+own_n),"selected_smoothed":(alpha*.5+own_s)/(alpha+own_n),
                    "empirical_mean":own_s/own_n if own_n else .5}
                q=target["evaluation_mean"]
                for name,p in predictions.items():values[name].append(p*p-2*p*q+q)
            for name,v in values.items():scores[name].append(v)
    return {k:np.array(v) for k,v in scores.items()}


def main():
    ROOT.mkdir(parents=True,exist_ok=True);start=time.time();selections=[]
    for fold in range(3):
        split=json.loads(Path(f"runs/mlp_cv/fold{fold}_split.json").read_text());sums=np.zeros(len(KAPPAS));baseline=np.zeros(len(ALPHAS));count=0
        for directory in split["train_benchmarks"]:
            rows=records(directory,73,25)
            if not rows:continue
            sums+=[np.sum(score_rows(rows,directory,4,k,2)["shared_prior"]@ALC_WEIGHTS) for k in KAPPAS]
            baseline+=[np.sum(score_rows(rows,directory,4,2,a)["selected_smoothed"]@ALC_WEIGHTS) for a in ALPHAS];count+=len(rows)
        sel={"fold":fold,"train_benchmarks":split["train_benchmarks"],"heldout_benchmarks":split["development_benchmarks"],
             "kappa_grid":KAPPAS,"alpha_grid":ALPHAS,"training_alc":(sums/count).tolist(),"baseline_training_alc":(baseline/count).tolist(),
             "selected_kappa":KAPPAS[int(np.argmin(sums))],"selected_alpha":ALPHAS[int(np.argmin(baseline))],"selection_pairs":count}
        selections.append(sel);(ROOT/f"fold{fold}_selection.json").write_text(json.dumps(sel,indent=2),encoding="utf-8")
        print(f"selected fold{fold}: kappa={sel['selected_kappa']} alpha={sel['selected_alpha']}",flush=True)
    for size in [1,2,4,6]:
        folder=ROOT/f"cohort{size}";folder.mkdir(exist_ok=True)
        for fold,sel in enumerate(selections):
            totals={k:np.zeros(6) for k in ["shared_prior","smoothed_mean_2","selected_smoothed","empirical_mean"]};by_benchmark={};count=0;duplicates=0
            old=json.loads(Path(f"runs/research_loop/R3/fold{fold}_evaluation.json").read_text())
            for directory in sel["heldout_benchmarks"]:
                rows=records(directory,0,100)
                if not rows:continue
                output=score_rows(rows,directory,size,sel["selected_kappa"],sel["selected_alpha"])
                duplicates+=len(rows)-len({r["key"] for r in rows})
                if size==1:
                    np.testing.assert_allclose(output["smoothed_mean_2"].mean(0),old["by_benchmark"][directory]["brier_mean"]["smoothed_mean_2"],atol=1e-7,rtol=0)
                    np.testing.assert_allclose(output["shared_prior"],output["smoothed_mean_2"],atol=1e-12,rtol=0)
                by_benchmark[directory]={"pairs_per_seed":[len(rows)],"brier_mean":{k:v.mean(0).tolist() for k,v in output.items()},
                    "alc_mean":{k:float(v.mean(0)@ALC_WEIGHTS) for k,v in output.items()}}
                count+=len(rows)
                for k,v in output.items():totals[k]+=v.sum(0)
            result={"evidence_scope":"conditional same-benchmark shared-label cohort; organizer cohort policy unverified","cohort_size":size,
                "selection":sel,"pairs_per_seed":[count],"duplicate_visible_configuration_excess":duplicates,"by_benchmark":by_benchmark,
                "results":{k:{"brier_mean":(v/count).tolist(),"alc_mean":float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
            (folder/f"fold{fold}_evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        summary=summarize([folder/f"fold{i}_evaluation.json" for i in range(3)],"shared_prior",["smoothed_mean_2","selected_smoothed","empirical_mean"])
        summary["scope"]="conditional shared evidence; does not replace isolated-pair gate; hosted policy unverified"
        summary["cohort_size"]=size
        (folder/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(f"cohort{size}: {summary['pair_equal_alc']}",flush=True)
    (ROOT/"runtime.json").write_text(json.dumps({"elapsed_seconds":time.time()-start}),encoding="utf-8")


if __name__=="__main__":main()
