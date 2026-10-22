"""R24 isolated-pair metadata ablation and current-support permutation control."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .metadata_episodes import episodes
from .cohort_semantic import CachedSemantic
from .evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from .summarize_candidate import summarize

RIDGES=[None,16.,4.,1.,.25]


def prediction(support,query,labels,ridge):
    n=len(labels);base=(1+sum(labels))/(2+n)
    if n<2 or ridge is None:return np.full(len(query),base)
    center=support.mean(0);x=support-center
    beta=x.T@np.linalg.solve(x@x.T+ridge*np.eye(n),labels-labels.mean())
    return np.clip(base+(query-center)@beta,0,1)


def score(row,mode,ridge,permutations=0,rng=None):
    values=[]
    for n in BUDGETS:
        labels=row['a'][:n];meta=row['support_meta'][:n]
        losses=[]
        for repeat in range(permutations or 1):
            m=meta[rng.permutation(n)] if permutations else meta
            if mode=='metadata':support=m;query=row['query_meta']
            elif mode=='combined':support=np.concatenate([row['support'][:n],m],axis=1);query=np.concatenate([row['query'],row['query_meta']],axis=1)
            else:support=row['support'][:n];query=row['query']
            losses.append(np.mean((prediction(support,query,labels,ridge)-row['y'])**2))
        values.append(np.mean(losses))
    return np.asarray(values)


def main():
    root=Path('runs/research_loop/R24');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic();start=time.time();selections=[]
    sources=['mlp/metadata_features.py','mlp/metadata_episodes.py','mlp/run_metadata_round.py','research_loop/R24b_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text());sums={k:np.zeros(len(RIDGES)) for k in ['metadata','combined']};count=0
        for directory in split['train_benchmarks']:
            for row in episodes(directory,73,25,engine):
                for mode in sums:sums[mode]+=[score(row,mode,r)@ALC_WEIGHTS for r in RIDGES]
                count+=1
        selection={'fold':fold,'train_benchmarks':split['train_benchmarks'],'heldout_benchmarks':split['development_benchmarks'],
            'training_pairs':count,'training_alc':{k:(v/count).tolist() for k,v in sums.items()},'ridges':RIDGES,
            'selected_ridge':{k:RIDGES[int(np.argmin(v))] for k,v in sums.items()}}
        selections.append(selection);(root/f'fold{fold}_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('selected',fold,selection['selected_ridge'],flush=True)
    names=['combined','metadata','metadata_permuted','semantic','text_hash','smoothed_mean_2','training_selected_smoothed']
    for fold,selection in enumerate(selections):
        old=json.loads(Path(f'runs/research_loop/R7/fold{fold}_evaluation.json').read_text());total={k:np.zeros(6) for k in names};count=0;by={}
        for directory in selection['heldout_benchmarks']:
            rows=episodes(directory,0,100,engine)
            if not rows:continue
            rng=np.random.default_rng(_stable_seed(871,directory,'metadata'))
            avg={m:np.mean([score(r,m,selection['selected_ridge'][m]) for r in rows],axis=0) for m in ['metadata','combined']}
            avg['metadata_permuted']=np.mean([score(r,'combined',selection['selected_ridge']['combined'],8,rng) for r in rows],axis=0)
            historic=old['by_benchmark'][directory];n=len(rows);assert historic['pairs_per_seed'][0]==n;count+=n
            for k in names[3:]:avg[k]=np.array(historic['brier_mean'][k])
            np.testing.assert_allclose(np.mean([score(r,'semantic',None) for r in rows],axis=0),avg['smoothed_mean_2'],atol=1e-7,rtol=0)
            by[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():total[k]+=n*v
            print('outer',fold,directory,flush=True)
        result={'selection':selection,'pairs_per_seed':[count],'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'combined',names[1:]);d['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
