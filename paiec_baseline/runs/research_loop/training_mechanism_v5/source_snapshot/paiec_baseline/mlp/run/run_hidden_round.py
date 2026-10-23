"""R4 frozen learned features versus matched raw-input local ridge control."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.embedding_episodes import episodes
from mlp.evaluate import _load_weights, ALC_WEIGHTS, BUDGETS
from mlp.local_ridge import kernel_predict
from mlp.reporting.summarize_candidate import summarize

ROOT=Path("runs/research_loop/R4")
RIDGES=[None,10.,1.,.1,.01]


def kernels(row,weights,mode):
    x=row["x"]
    if mode=="hidden":
        w1,b1,w2,b2,_,_=weights
        x=np.maximum(np.maximum(x@w1+b1,0)@w2+b2,0)
    x=x.astype(float)
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return x[:31]@x[:31].T,x[31:]@x[:31].T


def scores(row,k,c,ridge,alpha=2.):
    return np.array([np.mean((kernel_predict(k,c,row["a"][:n],ridge,alpha)-row["y"])**2) for n in BUDGETS])


def main():
    ROOT.mkdir(parents=True,exist_ok=True); start=time.time(); selections=[]
    for fold in range(3):
        path=Path(f"runs/mlp_cv/fold{fold}/weights.npz")
        weights=_load_weights(path); meta=json.loads(path.with_suffix(".json").read_text())
        split=json.loads(Path(f"runs/mlp_cv/fold{fold}_split.json").read_text())
        assert not set(split["development_benchmarks"])&set(meta["train_benchmarks"]+meta["validation_benchmarks"])
        sums={mode:np.zeros(len(RIDGES)) for mode in ["raw_full","hidden"]}; count=0
        for directory in meta["validation_benchmarks"]:
            for row in episodes(directory,73,25):
                for mode in sums:
                    k,c=kernels(row,weights,mode)
                    sums[mode]+=[scores(row,k,c,r)@ALC_WEIGHTS for r in RIDGES]
                count+=1
        selection={"fold":fold,"selection_benchmarks":meta["validation_benchmarks"],"selection_pairs":count,
                   "heldout_benchmarks":split["development_benchmarks"],"ridge_grid":RIDGES,
                   "inner_alc":{k:(v/count).tolist() for k,v in sums.items()},
                   "selected_ridge":{k:RIDGES[int(np.argmin(v))] for k,v in sums.items()},
                   "weight_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        selections.append(selection)
        (ROOT/f"fold{fold}_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
        print(f"selected fold{fold}: {selection['selected_ridge']}; pairs={count}",flush=True)
    for fold,selection in enumerate(selections):
        weights=_load_weights(Path(f"runs/mlp_cv/fold{fold}/weights.npz"))
        historical=json.loads(Path(f"runs/research_loop/R3/fold{fold}_evaluation.json").read_text())
        all_names=["raw_full","hidden","smoothed_mean_2","training_selected_smoothed","inner_selected_smoothed","local_ridge","original:residual_2","empirical_mean"]
        totals={k:np.zeros(6) for k in all_names}; by_benchmark={}; count=0
        for directory in selection["heldout_benchmarks"]:
            rows=episodes(directory,0,100)
            if not rows:continue
            old=historical["by_benchmark"][directory]; sums={k:np.zeros(6) for k in all_names}
            assert len(rows)==old["pairs_per_seed"][0]
            for row in rows:
                for mode in ["raw_full","hidden"]:
                    k,c=kernels(row,weights,mode)
                    sums[mode]+=scores(row,k,c,selection["selected_ridge"][mode])
                sums["smoothed_mean_2"]+=scores(row,k,c,None)
            np.testing.assert_allclose(sums["smoothed_mean_2"]/len(rows),old["brier_mean"]["smoothed_mean_2"],atol=1e-7,rtol=0)
            for method in all_names[3:]:sums[method]=np.array(old["brier_mean"][method])*len(rows)
            by_benchmark[directory]={"pairs_per_seed":[len(rows)],"brier_mean":{k:(v/len(rows)).tolist() for k,v in sums.items()},
                "alc_mean":{k:float(v/len(rows)@ALC_WEIGHTS) for k,v in sums.items()}}
            count+=len(rows)
            for k,v in sums.items():totals[k]+=v
            print(f"outer fold{fold} {directory}",flush=True)
        out={"selection":selection,"pairs_per_seed":[count],"by_benchmark":by_benchmark,
             "results":{k:{"brier_mean":(v/count).tolist(),"alc_mean":float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (ROOT/f"fold{fold}_evaluation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    files=[ROOT/f"fold{i}_evaluation.json" for i in range(3)]
    summary=summarize(files,"hidden",["smoothed_mean_2","training_selected_smoothed","inner_selected_smoothed","raw_full","local_ridge","original:residual_2","empirical_mean"])
    summary["elapsed_seconds"]=time.time()-start
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["pair_equal_alc","comparisons"]},indent=2))


if __name__=="__main__":main()
