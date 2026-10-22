"""R36 conditional cohort4: prevalence versus item-effect sharing at full pair coverage."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .cohort_semantic import CachedSemantic
from .conditional_components import components
from .measurement_episodes import episodes, weighted_brier
from .model import _SUBJECT_FIELDS
from .visible_item_evidence import plain, cohorts
from .evaluate import BUDGETS, ALC_WEIGHTS, _stable_seed
from .summarize_candidate import summarize

NAMES=['own_pooled','peer_pooled','own_selected','peer_selected','own_original','peer_original','peer','base','permuted_own_pooled']


def evaluate(directory,fold,selection,original,engine,root):
    target=root/(directory.split('/')[0]+'_evaluation.json')
    if target.exists():return json.loads(target.read_text())
    subjects=pd.read_parquet(Path('D:/paiec_payload/data')/directory/'subjects.parquet').set_index('subject_id').to_dict('index')
    keys={str(s):json.dumps({f:plain(row.get(f)) for f in _SUBJECT_FIELDS},sort_keys=True) for s,row in subjects.items()}
    sums={k:np.zeros(6) for k in NAMES};oldsum={k:np.zeros(6) for k in NAMES};nrows=0;oldrows=0;seed_results={};duplicates=0;assignment=None
    with (root/(directory.split('/')[0]+'_pair_brier.csv')).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['subject_id','seed','budget','old_subset',*NAMES])
        for seed in [0,1,2]:
            rows=[]
            for r in episodes(directory,engine,1.,16.,seeds=(seed,)):
                rows.append({'subject_id':r['subject_id'],'key':keys[r['subject_id']],
                             'a':r['labels'],'support':r['support'][:,:384].copy(),'query':r['query'][:,:384].copy(),
                             'counts':r['counts'],'successes':r['successes']})
            n=len(rows);selected=np.arange(n)
            if n>100:selected=np.sort(np.random.default_rng(_stable_seed(2026,directory,'pair_sample')).choice(n,100,replace=False))
            subset=set(selected.tolist());partitions=[[rows[i] for i in selected],[r for i,r in enumerate(rows) if i not in subset]]
            current={k:np.zeros(6) for k in NAMES};members=[]
            for oldflag,partition in [(True,partitions[0]),(False,partitions[1])]:
                for cohort in cohorts(partition,directory):
                    members.append([r['subject_id'] for r in cohort])
                    duplicates+=len(cohort)-len({r['key'] for r in cohort})
                    losses={k:np.empty((len(cohort),6)) for k in NAMES}
                    for j,budget in enumerate(BUDGETS):
                        context=[{'key':r['key'],'labels':r['a'][:budget],'vectors':r['support'][:budget]} for r in cohort]
                        rng=np.random.default_rng(_stable_seed(3601,directory,f'{seed}:{budget}:{cohort[0]["subject_id"]}'))
                        effects=components(context,[r['query'] for r in cohort],selection['selected_ridge']['pooled'],
                                           selection['selected_ridge']['own_selected'],original,rng)
                        for i,(row,effect) in enumerate(zip(cohort,effects)):
                            predictions={'own_pooled':effect['base']+effect['pooled'],'peer_pooled':effect['peer']+effect['pooled'],
                                'own_selected':effect['base']+effect['own'],'peer_selected':effect['peer']+effect['own'],
                                'own_original':effect['base']+effect['original'],'peer_original':effect['peer']+effect['original'],
                                'peer':effect['peer'],'base':effect['base']}
                            for name,p in predictions.items():losses[name][i,j]=weighted_brier(np.clip(p,0,1),row['counts'],row['successes'])
                            losses['permuted_own_pooled'][i,j]=np.mean([weighted_brier(np.clip(effect['base']+p,0,1),row['counts'],row['successes']) for p in effect['permuted']])
                            writer.writerow([row['subject_id'],seed,budget,int(oldflag),*[losses[k][i,j] for k in NAMES]])
                    for name,values in losses.items():
                        current[name]+=values.sum(0)
                        if oldflag:oldsum[name]+=values.sum(0)
                    nrows+=len(cohort)
                    if oldflag:oldrows+=len(cohort)
            if assignment is None:assignment=members
            else:assert assignment==members
            for k in NAMES:sums[k]+=current[k]
            seed_results[str(seed)]={k:float(v/n@ALC_WEIGHTS) for k,v in current.items()}
            print(directory,'seed',seed,'pairs',n,flush=True)
    avg={k:v/nrows for k,v in sums.items()};sub={k:v/oldrows for k,v in oldsum.items()}
    historic=json.loads(Path(f'runs/research_loop/R20_verify/fold{fold}_evaluation.json').read_text())['by_benchmark'][directory]
    for name,old in [('peer_pooled','pooled'),('peer_selected','own_selected'),('peer_original','own_original'),('peer','peer'),('base','base')]:
        np.testing.assert_allclose(sub[name],historic['brier_mean'][old],atol=1e-7,rtol=0)
    d={'pairs_per_seed':[nrows//3]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},
       'alc_per_seed':seed_results,'old_subset_pairs_per_seed':oldrows//3,'original_subset_brier':{k:v.tolist() for k,v in sub.items()},
       'duplicate_visible_key_extra_rows_per_seed':duplicates//3,'cohort_members':assignment}
    target.write_text(json.dumps(d,indent=2),encoding='utf-8');return d


def main():
    root=Path('runs/research_loop/R36');root.mkdir(parents=True,exist_ok=True);start=time.time();engine=CachedSemantic()
    sources=['mlp/run_conditional_factorial.py','mlp/conditional_components.py','mlp/measurement_episodes.py',
             'mlp/pooled_item_slope.py','mlp/peer_item_predictor.py','research_loop/R36_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        selection=json.loads(Path(f'runs/research_loop/R20/fold{fold}_selection.json').read_text())
        original=json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text())['selected_ridge']['semantic']
        population=json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text())
        count=0;total={k:np.zeros(6) for k in NAMES};by={}
        for directory,reference in population['by_benchmark'].items():
            d=evaluate(directory,fold,selection,original,engine,root);n=d['pairs_per_seed'][0]
            assert n==reference['pairs_per_seed'][0];count+=n;by[directory]=d
            for k in NAMES:total[k]+=n*np.asarray(d['brier_mean'][k])
        result={'selection':selection,'pairs_per_seed':[count]*3,'by_benchmark':by,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],NAMES[0],NAMES[1:])
    d.update(scope='conditional cohort4,all5242pairs,three seeds,64queryitems; preserved old cohorts plus separate added-pair cohorts; not isolated-pair or verified hosted protocol',elapsed_seconds=time.time()-start)
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    for p in sources:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
