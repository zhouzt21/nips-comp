"""R7 frozen three-seed verification; no candidate or parameter reselection."""
import json
from pathlib import Path
import time
import numpy as np
from mlp.run.run_semantic_round import episodes,scores
from mlp.semantic_features import SemanticFeatures
from mlp.evaluate import ALC_WEIGHTS
from mlp.reporting.summarize_candidate import summarize


def main():
    root=Path("runs/research_loop/R7_verify");root.mkdir(parents=True,exist_ok=True)
    start=time.time();engine=SemanticFeatures()
    for fold in range(3):
        sel=json.loads(Path(f"runs/research_loop/R7/fold{fold}_selection.json").read_text())
        historical=json.loads(Path(f"runs/research_loop/R3_verify/fold{fold}_evaluation.json").read_text())
        names=["semantic","text_hash","smoothed_mean_2","training_selected_smoothed","inner_selected_smoothed","local_ridge"]
        totals={k:np.zeros(6) for k in names};by_benchmark={};count=0
        for directory in sel["heldout_benchmarks"]:
            sums={k:np.zeros((3,6)) for k in names[:3]};ns=[]
            for seed in range(3):
                rows=episodes(directory,seed,100,engine);ns.append(len(rows))
                for row in rows:
                    for mode in ["semantic","text_hash"]:sums[mode][seed]+=scores(row,mode,sel["selected_ridge"][mode])
                    sums["smoothed_mean_2"][seed]+=scores(row,"semantic",None)
            if not ns[0]:continue
            assert len(set(ns))==1
            old=historical["by_benchmark"][directory];assert ns==old["pairs_per_seed"]
            averaged={k:(v/ns[0]).mean(0) for k,v in sums.items()}
            np.testing.assert_allclose(averaged["smoothed_mean_2"],old["brier_mean"]["smoothed_mean_2"],atol=1e-7,rtol=0)
            for name in names[3:]:averaged[name]=np.array(old["brier_mean"][name])
            by_benchmark[directory]={"pairs_per_seed":ns,"brier_mean":{k:v.tolist() for k,v in averaged.items()},"alc_mean":{k:float(v@ALC_WEIGHTS) for k,v in averaged.items()}}
            count+=ns[0]
            for k,v in averaged.items():totals[k]+=v*ns[0]
        out={"selection":sel,"seeds":[0,1,2],"pairs_per_seed":[count]*3,"by_benchmark":by_benchmark,
             "results":{k:{"brier_mean":(v/count).tolist(),"alc_mean":float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (root/f"fold{fold}_evaluation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    summary=summarize([root/f"fold{i}_evaluation.json" for i in range(3)],"semantic",names[1:])
    summary["elapsed_seconds"]=time.time()-start;summary["new_embeddings"]=engine.encoded
    summary["seeds"]=[0,1,2];summary["bootstrap_unit"]="benchmark after within-benchmark seed averaging"
    (root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["pair_equal_alc","comparisons"]},indent=2))


if __name__=="__main__":main()
