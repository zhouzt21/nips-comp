"""Frozen R3 selection, three acquisition trajectories; no parameter updates."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .run_local_round import episodes, score, DATA, ALPHAS
from .evaluate import BUDGETS, ALC_WEIGHTS
from .summarize_candidate import summarize


def main():
    root=Path("runs/research_loop/R3_verify")
    root.mkdir(parents=True,exist_ok=True)
    start=time.time()
    revision=json.loads((DATA/"corpus_manifest.json").read_text())["revision"]
    names=["mlp/model.py","mlp/local_ridge.py","mlp/evaluate.py","mlp/run_local_round.py"]
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
    encoder_hash=hashlib.sha256(json.dumps({"code":hashes,"revision":revision},sort_keys=True).encode()).hexdigest()
    for fold in range(3):
        selection=json.loads(Path(f"runs/research_loop/R3/fold{fold}_selection.json").read_text())
        totals={name:np.zeros((3,6)) for name in ["local_ridge","smoothed_mean_2","inner_selected_smoothed","training_selected_smoothed"]}
        counts=np.zeros(3,int); by_benchmark={}
        for directory in selection["heldout_benchmarks"]:
            sums={name:np.zeros((3,6)) for name in totals}; ns=[]
            for seed in range(3):
                rows=episodes(directory,seed,100,encoder_hash)
                ns.append(len(rows)); counts[seed]+=len(rows)
                for row in rows:
                    sums["local_ridge"][seed]+=score(row,selection["selected_ridge"])
                    for name,alpha in [("smoothed_mean_2",2.),("inner_selected_smoothed",selection["selected_alpha"]),("training_selected_smoothed",[2.,2.,4.][fold])]:
                        sums[name][seed]+=score(row,None,alpha)
            if not ns[0]:continue
            by_benchmark[directory]={"pairs_per_seed":ns,
                "alc_mean":{name:float(np.mean((v/np.array(ns)[:,None])@ALC_WEIGHTS)) for name,v in sums.items()},
                "brier_mean":{name:(v/np.array(ns)[:,None]).mean(0).tolist() for name,v in sums.items()}}
            for name,v in sums.items():totals[name]+=v
            print(f"verified fold{fold} {directory}",flush=True)
        results={name:{"brier_mean":(v/counts[:,None]).mean(0).tolist(),"brier_by_seed":(v/counts[:,None]).tolist(),
                       "alc_mean":float(np.mean((v/counts[:,None])@ALC_WEIGHTS)),"alc_by_seed":((v/counts[:,None])@ALC_WEIGHTS).tolist()} for name,v in totals.items()}
        out={"source_revision":revision,"selection":selection,"seeds":3,"pairs_per_seed":counts.tolist(),
             "results":results,"by_benchmark":by_benchmark,"code_sha256":hashes,"budgets":BUDGETS}
        (root/f"fold{fold}_evaluation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    summary=summarize([root/f"fold{i}_evaluation.json" for i in range(3)],"local_ridge",["smoothed_mean_2","inner_selected_smoothed","training_selected_smoothed"])
    summary["elapsed_seconds"]=time.time()-start
    summary["seeds"]=[0,1,2]
    summary["bootstrap_unit"]="benchmark; average item-split seeds within benchmark before resampling"
    (root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["pair_equal_alc","comparisons"]},indent=2))


if __name__ == "__main__":main()
