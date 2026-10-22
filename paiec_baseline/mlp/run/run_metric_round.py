"""R5 train, export, evaluate: no outer label participates in model selection."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from mlp.embedding_episodes import episodes
from mlp.episodic_metric import probabilities,torch_probabilities
from mlp.evaluate import BUDGETS,ALC_WEIGHTS
from mlp.reporting.summarize_candidate import summarize

ROOT=Path("runs/research_loop/R5")


def tensor_rows(directories):
    rows=[]
    for directory in directories:
        loaded=episodes(directory,73,25)
        rows.extend({k:torch.from_numpy(v) for k,v in row.items()} for row in loaded)
        print(f"load {directory}: {len(loaded)}",flush=True)
    return rows


def loss(row,weight):
    return sum(float(w)*(torch_probabilities(row["x"],row["a"],weight,n)-row["y"]).square().mean()
               for n,w in zip(BUDGETS,ALC_WEIGHTS))


def evaluate_inner(rows,weight):
    with torch.no_grad():return float(np.mean([loss(row,weight).item() for row in rows]))


def train_fold(fold):
    meta=json.loads(Path(f"runs/mlp_cv/fold{fold}/weights.json").read_text())
    train=tensor_rows(meta["train_benchmarks"]); val=tensor_rows(meta["validation_benchmarks"])
    torch.manual_seed(2026); rng=np.random.default_rng(2026)
    weight=torch.nn.Parameter(torch.zeros(2048)); optimizer=torch.optim.Adam([weight],lr=.01)
    best=evaluate_inner(val,weight); best_epoch=0; best_weight=weight.detach().clone(); patience=0
    history=[{"epoch":0,"inner_alc":best}]
    for epoch in range(1,21):
        order=rng.permutation(len(train)); epoch_losses=[]
        for offset in range(0,len(order),16):
            indices=order[offset:offset+16]; optimizer.zero_grad()
            for index in indices:
                value=loss(train[index],weight)
                (value/len(indices)).backward(); epoch_losses.append(value.item())
            (.01*weight.square().mean()).backward(); optimizer.step()
            with torch.no_grad():weight.clamp_(-2,2)
        score=evaluate_inner(val,weight)
        history.append({"epoch":epoch,"training_alc":float(np.mean(epoch_losses)),"inner_alc":score})
        print(f"fold={fold} epoch={epoch} inner_alc={score:.7f}",flush=True)
        if score<best-1e-6:
            best=score;best_epoch=epoch;best_weight=weight.detach().clone();patience=0
        else:
            patience+=1
            if patience>=4:break
    exported=best_weight.numpy()
    errors=[]
    for budget in BUDGETS:
        row=val[0]
        reference=torch_probabilities(row["x"],row["a"],best_weight,budget).numpy()
        actual=probabilities(row["x"].numpy(),row["a"].numpy(),exported,budget)
        errors.append(float(np.max(np.abs(reference-actual))))
    if max(errors)>2e-6:raise ValueError(f"Export disagreement: {errors}")
    out=ROOT/f"fold{fold}_metric.npz"; np.savez_compressed(out,log_weight=exported)
    report={"fold":fold,"source_revision":meta["source_revision"],"train_benchmarks":meta["train_benchmarks"],
            "validation_benchmarks":meta["validation_benchmarks"],"training_episodes":len(train),"validation_episodes":len(val),
            "best_epoch":best_epoch,"inner_alc":best,"history":history,"export_max_abs_error":max(errors),
            "ridge":1.,"alpha":2.,"learning_rate":.01,"log_weight_l2":.01,"seed":2026,
            "episode_seed":73,"max_pairs_per_benchmark":25,"weight_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
            "log_weight_std":float(exported.std()),"log_weight_min":float(exported.min()),"log_weight_max":float(exported.max())}
    out.with_suffix(".json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return exported


def main():
    ROOT.mkdir(parents=True,exist_ok=True);start=time.time();torch.set_num_threads(4)
    trained=[train_fold(fold) for fold in range(3)]
    for fold,weight in enumerate(trained):
        historical=json.loads(Path(f"runs/research_loop/R3/fold{fold}_evaluation.json").read_text())
        split=json.loads(Path(f"runs/mlp_cv/fold{fold}_split.json").read_text())
        meta=json.loads((ROOT/f"fold{fold}_metric.json").read_text())
        assert not set(split["development_benchmarks"])&set(meta["train_benchmarks"]+meta["validation_benchmarks"])
        names=["episodic_metric","matched_metric","smoothed_mean_2","training_selected_smoothed","inner_selected_smoothed","local_ridge","original:residual_2","empirical_mean"]
        totals={k:np.zeros(6) for k in names};by_benchmark={};count=0
        for directory in split["development_benchmarks"]:
            rows=episodes(directory,0,100)
            if not rows:continue
            old=historical["by_benchmark"][directory];sums={k:np.zeros(6) for k in names}
            assert len(rows)==old["pairs_per_seed"][0]
            for row in rows:
                for name,v in [("episodic_metric",weight),("matched_metric",np.zeros(2048))]:
                    sums[name]+=[np.mean((probabilities(row["x"],row["a"],v,n)-row["y"])**2) for n in BUDGETS]
            for name in names[2:]:sums[name]=np.array(old["brier_mean"][name])*len(rows)
            by_benchmark[directory]={"pairs_per_seed":[len(rows)],"brier_mean":{k:(v/len(rows)).tolist() for k,v in sums.items()},
                "alc_mean":{k:float(v/len(rows)@ALC_WEIGHTS) for k,v in sums.items()}}
            count+=len(rows)
            for k,v in sums.items():totals[k]+=v
            print(f"outer fold{fold} {directory}",flush=True)
        out={"training":meta,"pairs_per_seed":[count],"by_benchmark":by_benchmark,"results":{k:{"brier_mean":(v/count).tolist(),"alc_mean":float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (ROOT/f"fold{fold}_evaluation.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    files=[ROOT/f"fold{i}_evaluation.json" for i in range(3)]
    summary=summarize(files,"episodic_metric",["smoothed_mean_2","training_selected_smoothed","inner_selected_smoothed","matched_metric","local_ridge","original:residual_2","empirical_mean"])
    summary["elapsed_seconds"]=time.time()-start
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ["pair_equal_alc","comparisons"]},indent=2))


if __name__=="__main__":main()
