"""R25 block penalties with exact originalsemantic fallback."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .metadata_episodes import episodes
from .cohort_semantic import CachedSemantic
from .run_metadata_round import prediction
from .evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from .summarize_candidate import summarize

RIDGES=[None,16.,4.,1.,.25]


def score(row,semantic_ridge,metadata_ridge,permutations=0,rng=None):
    values=[]
    for n in BUDGETS:
        s=row['support'][:n]/np.sqrt(semantic_ridge);q=row['query']/np.sqrt(semantic_ridge);losses=[]
        for repeat in range(permutations or 1):
            support=s;query=q
            if metadata_ridge is not None:
                m=row['support_meta'][:n]/np.sqrt(metadata_ridge)
                if permutations:m=m[rng.permutation(n)]
                support=np.concatenate([s,m],axis=1);query=np.concatenate([q,row['query_meta']/np.sqrt(metadata_ridge)],axis=1)
            p=prediction(support,query,row['a'][:n],1.)
            losses.append(np.mean((p-row['y'])**2))
        values.append(np.mean(losses))
    return np.asarray(values)


def main():
    root=Path('runs/research_loop/R25');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic();start=time.time();selections=[]
    sources=['mlp/run_metadata_penalty_round.py','mlp/run_metadata_round.py','mlp/metadata_features.py','research_loop/R25b_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        old=json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text());sem=old['selected_ridge']['semantic'];sums=np.zeros(len(RIDGES));count=0
        for directory in old['train_benchmarks']:
            for row in episodes(directory,73,25,engine):sums+=[score(row,sem,r)@ALC_WEIGHTS for r in RIDGES];count+=1
        selection={'fold':fold,'train_benchmarks':old['train_benchmarks'],'heldout_benchmarks':old['heldout_benchmarks'],
            'semantic_ridge':sem,'metadata_ridge':RIDGES[int(np.argmin(sums))],'training_pairs':count,'training_alc':(sums/count).tolist(),'metadata_grid':RIDGES}
        selections.append(selection);(root/f'fold{fold}_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('selected',fold,selection['metadata_ridge'],flush=True)
    names=['separate','metadata_permuted','semantic','uniform','smoothed_mean_2','training_selected_smoothed']
    for fold,selection in enumerate(selections):
        old=json.loads(Path(f'runs/research_loop/R24/fold{fold}_evaluation.json').read_text());total={k:np.zeros(6) for k in names};count=0;by={}
        for directory in selection['heldout_benchmarks']:
            rows=episodes(directory,0,100,engine)
            if not rows:continue
            rng=np.random.default_rng(_stable_seed(119,directory,'separate_metadata'));sem=selection['semantic_ridge'];meta=selection['metadata_ridge']
            avg={'separate':np.mean([score(r,sem,meta) for r in rows],axis=0),
                 'metadata_permuted':np.mean([score(r,sem,meta,8,rng) for r in rows],axis=0)}
            historical=old['by_benchmark'][directory];n=len(rows);assert historical['pairs_per_seed'][0]==n;count+=n
            for k,source in [('semantic','semantic'),('uniform','combined'),('smoothed_mean_2','smoothed_mean_2'),('training_selected_smoothed','training_selected_smoothed')]:avg[k]=np.asarray(historical['brier_mean'][source])
            np.testing.assert_allclose(np.mean([score(r,sem,None) for r in rows],axis=0),avg['semantic'],atol=1e-7,rtol=0)
            by[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():total[k]+=n*v
        result={'selection':selection,'pairs_per_seed':[count],'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'separate',names[1:]);d['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
