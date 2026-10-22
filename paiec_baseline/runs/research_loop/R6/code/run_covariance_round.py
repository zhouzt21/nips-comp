"""R6 covariance selection without outer-label tuning; reuses R3 cache."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .run_local_round import episodes,score
from .kernel_covariance import predict_kernel
from .evaluate import BUDGETS,ALC_WEIGHTS
from .summarize_candidate import summarize

ROOT=Path("runs/research_loop/R6");ETAS=[0.,.25,1.,4.]


def scores(row,eta):
    return np.array([np.mean((predict_kernel(row["k"],row["c"],row["a"][:n],eta)-row["y"])**2) for n in BUDGETS])


def main():
    ROOT.mkdir(parents=True,exist_ok=True);start=time.time();selections=[]
    source=json.loads(Path("runs/research_loop/R3/fold0_evaluation.json").read_text())
    for name,sha in source["code_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:raise ValueError("R3 cache source changed")
    encoder_hash=hashlib.sha256(json.dumps({"code":source["code_sha256"],"revision":source["source_revision"]},sort_keys=True).encode()).hexdigest()
    for fold in range(3):
        old=json.loads(Path(f"runs/research_loop/R3/fold{fold}_selection.json").read_text())
        sums=np.zeros(4);controls=np.zeros(4);count=0
        for directory in old["train_benchmarks"]:
            for row in episodes(directory,73,25,encoder_hash):
                sums+=[scores(row,e)@ALC_WEIGHTS for e in ETAS]
                controls+=[np.array(score(row,None,2+e))@ALC_WEIGHTS for e in ETAS];count+=1
        selected={"fold":fold,"train_benchmarks":old["train_benchmarks"],"heldout_benchmarks":old["heldout_benchmarks"],
                  "eta_grid":ETAS,"training_alc":(sums/count).tolist(),"training_control_alc":(controls/count).tolist(),
                  "selected_eta":ETAS[int(np.argmin(sums))],"selected_control_alpha":2+ETAS[int(np.argmin(controls))],
                  "selection_pairs":count,"selection_seed":73,"length_scale":.5}
        selections.append(selected);(ROOT/f"fold{fold}_selection.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
        print(f"selected fold{fold}: {selected['selected_eta']} control-alpha={selected['selected_control_alpha']}",flush=True)
    for fold,sel in enumerate(selections):
        historical=json.loads(Path(f"runs/research_loop/R3/fold{fold}_evaluation.json").read_text())
        names=["covariance","matched_smoothing","selected_smoothing","smoothed_mean_2","training_selected_smoothed","inner_selected_smoothed","local_ridge","original:residual_2","empirical_mean"]
        totals={k:np.zeros(6) for k in names};by_benchmark={};count=0
        for directory in sel["heldout_benchmarks"]:
            rows=episodes(directory,0,100,encoder_hash)
            if not rows:continue
            sums={k:np.zeros(6) for k in names};old=historical["by_benchmark"][directory]
            for row in rows:
                sums["covariance"]+=scores(row,sel["selected_eta"])
                sums["matched_smoothing"]+=score(row,None,2+sel["selected_eta"])
                sums["selected_smoothing"]+=score(row,None,sel["selected_control_alpha"])
            for name in names[3:]:sums[name]=np.array(old["brier_mean"][name])*len(rows)
            by_benchmark[directory]={"pairs_per_seed":[len(rows)],"brier_mean":{k:(v/len(rows)).tolist() for k,v in sums.items()},"alc_mean":{k:float(v/len(rows)@ALC_WEIGHTS) for k,v in sums.items()}}
            count+=len(rows)
            for k,v in sums.items():totals[k]+=v
        out={"selection":sel,"pairs_per_seed":[count],"by_benchmark":by_benchmark,"results":{k:{"brier_mean":(v/count).tolist(),"alc_mean":float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (ROOT/f"fold{fold}_evaluation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    summary=summarize([ROOT/f"fold{i}_evaluation.json" for i in range(3)],"covariance",["smoothed_mean_2","training_selected_smoothed","inner_selected_smoothed","matched_smoothing","selected_smoothing","local_ridge","original:residual_2","empirical_mean"])
    summary["elapsed_seconds"]=time.time()-start
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["pair_equal_alc","comparisons"]},indent=2))


if __name__=="__main__":main()
