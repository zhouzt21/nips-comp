"""R3: training-only selection of a local ridge adaptation rule.

Caches input-derived kernels, support labels and separate evaluation labels.
The predictor is called with support labels only. No model sees eval labels.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .evaluate import _load_directory, _stable_seed, BUDGETS, ALC_WEIGHTS
from .local_ridge import item_vector, kernel_predict
from .summarize_candidate import summarize

DATA = Path("D:/paiec_payload/data")
ROOT = Path("runs/research_loop/R3")
CACHE = Path("D:/paiec_research_cache/R3")
RIDGES = [None, 16., 4., 1., .25]
ALPHAS = [2., 1., 4., 8., 16.]


def episodes(directory, seed, max_pairs, encoder_hash):
    key = hashlib.sha256(f"{directory}|{seed}|{max_pairs}|64|{encoder_hash}".encode()).hexdigest()
    cache = CACHE / (key + ".npz")
    if cache.exists():
        with np.load(cache, allow_pickle=False) as pack:
            return [{name:pack[f"{i}_{name}"] for name in ["k", "c", "a", "y"]}
                    for i in range(int(pack["count"]))]
    responses, subjects, items = _load_directory(DATA, directory)
    groups = [(s,g) for s,g in responses.groupby("subject_id", sort=False)
              if g["item_id"].nunique() >= 80 and s in subjects]
    if len(groups) > max_pairs:
        rng = np.random.default_rng(_stable_seed(2026,directory,"pair_sample"))
        groups = [groups[i] for i in np.sort(rng.choice(len(groups), max_pairs, replace=False))]
    @lru_cache(maxsize=4096)
    def vector(item_id, interactors):
        item = items[item_id]
        return item_vector({"item_content":item["content"],"item_features":item["item_features"],"interactors":interactors})
    output = []
    for subject_id, group in groups:
        ids = pd.unique(group["item_id"]).copy()
        rng = np.random.default_rng(_stable_seed(seed,directory,subject_id))
        rng.shuffle(ids)
        cut = len(ids)//2
        acquired_ids, evaluation_ids = ids[:31],ids[cut:cut+64]
        acquired = group[group["item_id"].isin(acquired_ids)].sample(frac=1, random_state=int(rng.integers(2**31)))
        acquired = acquired.drop_duplicates("item_id").set_index("item_id").loc[acquired_ids].reset_index()
        evaluation = group[group["item_id"].isin(evaluation_ids)]
        def vectors(rows):
            return np.stack([vector(item, str(interactors)) for item,interactors in
                             rows[["item_id","interactors"]].itertuples(index=False,name=None)])
        a, q = vectors(acquired), vectors(evaluation)
        output.append({"k":a@a.T,"c":q@a.T,"a":acquired["response"].to_numpy(float),
                       "y":evaluation["response"].to_numpy(float)})
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = {f"{i}_{name}":array for i, row in enumerate(output) for name,array in row.items()}
    np.savez_compressed(cache, count=len(output), **arrays)
    return output


def score(episode, ridge, alpha=2.):
    return [float(np.mean((kernel_predict(episode["k"],episode["c"],episode["a"][:n],ridge,alpha)-episode["y"])**2))
            for n in BUDGETS]


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    manifest = json.loads((DATA/"corpus_manifest.json").read_text())
    code_files = ["mlp/model.py","mlp/local_ridge.py","mlp/evaluate.py","mlp/run_local_round.py"]
    hashes = {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in code_files}
    encoder_hash = hashlib.sha256(json.dumps({"code":hashes,"revision":manifest["revision"]},sort_keys=True).encode()).hexdigest()
    (ROOT/"code").mkdir(exist_ok=True)
    for name in code_files:
        (ROOT/"code"/Path(name).name).write_bytes(Path(name).read_bytes())
    selections = []
    # Finish every training-only selection before computing any outer result.
    for fold in range(3):
        split = json.loads(Path(f"runs/mlp_cv/fold{fold}_split.json").read_text())
        sums = np.zeros(len(RIDGES)); smooth_sums = np.zeros(len(ALPHAS)); count=0
        for directory in split["train_benchmarks"]:
            for row in episodes(directory,73,25,encoder_hash):
                sums += [np.array(score(row,r))@ALC_WEIGHTS for r in RIDGES]
                smooth_sums += [np.array(score(row,None,alpha))@ALC_WEIGHTS for alpha in ALPHAS]
                count += 1
        best = int(np.argmin(sums)); smooth_best = int(np.argmin(smooth_sums))
        selection = {"fold":fold,"train_benchmarks":split["train_benchmarks"],
                     "heldout_benchmarks":split["development_benchmarks"],
                     "selection_seed":73,"selection_max_pairs_per_benchmark":25,"selection_pairs":count,
                     "ridges":RIDGES,"training_alc":(sums/count).tolist(),"selected_ridge":RIDGES[best],
                     "alphas":ALPHAS,"training_smoothed_alc":(smooth_sums/count).tolist(),
                     "selected_alpha":ALPHAS[smooth_best],"source_revision":manifest["revision"]}
        selections.append(selection)
        (ROOT/f"fold{fold}_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
        print(f"selected fold{fold}: ridge={RIDGES[best]} alpha={ALPHAS[smooth_best]} pairs={count}",flush=True)
    for fold, selection in enumerate(selections):
        totals=defaultdict(lambda:np.zeros(6)); by_benchmark={}; count=0; responses=0
        historical=json.loads(Path(f"runs/research_loop/R1/fold{fold}_evaluation.json").read_text())
        for directory in selection["heldout_benchmarks"]:
            rows=episodes(directory,0,100,encoder_hash)
            if not rows:
                continue
            sums=defaultdict(lambda:np.zeros(6))
            for row in rows:
                sums["local_ridge"] += score(row,selection["selected_ridge"])
                for name,alpha in [("smoothed_mean_2",2.),("inner_selected_smoothed",selection["selected_alpha"]),
                                   ("training_selected_smoothed",[2.,2.,4.][fold])]:
                    sums[name] += score(row,None,alpha)
                sums["empirical_mean"] += [np.mean(((row["a"][:n].mean() if n else .5)-row["y"])**2) for n in BUDGETS]
                responses += len(row["y"])
            old=historical["by_benchmark"][directory]
            if old["pairs_per_seed"][0] != len(rows):
                raise ValueError("Pair set changed")
            np.testing.assert_allclose(sums["smoothed_mean_2"]/len(rows),old["brier_mean"]["smoothed_mean_2"],atol=1e-7,rtol=0)
            sums["original:residual_2"] = np.array(old["brier_mean"]["original:residual_2"])*len(rows)
            by_benchmark[directory]={"pairs_per_seed":[len(rows)],
                "brier_mean":{k:(v/len(rows)).tolist() for k,v in sums.items()},
                "alc_mean":{k:float(v/len(rows)@ALC_WEIGHTS) for k,v in sums.items()}}
            count += len(rows)
            for k,v in sums.items(): totals[k]+=v
            print(f"outer fold{fold} {directory}: {len(rows)}",flush=True)
        results={name:{"brier_mean":(v/count).tolist(),"brier_by_seed":[(v/count).tolist()],
                       "alc_mean":float(v/count@ALC_WEIGHTS)} for name,v in totals.items()}
        output={"source_revision":manifest["revision"],"selection":selection,"code_sha256":hashes,
                "pairs_per_seed":[count],"evaluation_responses_per_seed":[responses],"seeds":1,
                "budgets":BUDGETS,"alc_weights":ALC_WEIGHTS.tolist(),"results":results,"by_benchmark":by_benchmark,
                "evidence_scope":"reused public outer folds; train-only hyperparameter selection"}
        (ROOT/f"fold{fold}_evaluation.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    files=[ROOT/f"fold{fold}_evaluation.json" for fold in range(3)]
    summary=summarize(files,"local_ridge",["smoothed_mean_2","inner_selected_smoothed","training_selected_smoothed","empirical_mean","original:residual_2"])
    summary["elapsed_seconds"]=time.time()-start
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["pair_equal_alc","comparisons","elapsed_seconds"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
