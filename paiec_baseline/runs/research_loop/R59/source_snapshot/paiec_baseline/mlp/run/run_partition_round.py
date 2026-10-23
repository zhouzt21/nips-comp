"""R37 exact current-count partition posterior, with unchanged own semantic slopes."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from shared_mean.partition import predict_counts
from shared_mean.hierarchical import HierarchicalCounts
from mlp.cohort_semantic import CachedSemantic
from mlp.source_semantic_episodes import episodes as source_episodes
from mlp.measurement_episodes import episodes, weighted_brier
from mlp.pooled_item_slope import prepare
from mlp.peer_item_predictor import means
from mlp.visible_item_evidence import cohorts, plain
from mlp.model import _SUBJECT_FIELDS
from mlp.evaluate import BUDGETS, ALC_WEIGHTS
from mlp.reporting.summarize_candidate import summarize

ALPHAS=[None,1.,4.,16.]
NAMES=['partition_item','fixed_partition_item','hierarchy_item','own_item','peer_item','partition_mean','hierarchy_mean','base','peer']


def evidence(cohort,n,ridge):
    context=[{'key':r['key'],'labels':r['a'][:n],'vectors':r['support'][:n]} for r in cohort]
    groups=prepare(context);keys=list(groups);countmap={key:[0.,0] for key in keys}
    for row in context:countmap[row['key']][0]+=row['labels'].sum();countmap[row['key']][1]+=len(row['labels'])
    effects={}
    for key,(center,x,y) in groups.items():
        effects[key]=x.T@np.linalg.solve(x@x.T+ridge*np.eye(len(y)),y) if len(y) and ridge is not None else np.zeros(x.shape[1])
    slopes=[(r['query']-groups[r['key']][0])@effects[r['key']] for r in cohort]
    return keys,[countmap[k] for k in keys],slopes,means(context)


def loss(row,p):
    if 'counts' in row:return weighted_brier(np.clip(p,0,1),row['counts'],row['successes'])
    return float(np.mean((np.clip(p,0,1)-row['y'])**2))


def outer_rows(directory,seed,engine):
    table=pd.read_parquet(Path('D:/paiec_payload/data')/directory/'subjects.parquet').set_index('subject_id').to_dict('index')
    keys={str(s):json.dumps({f:plain(r.get(f)) for f in _SUBJECT_FIELDS},sort_keys=True) for s,r in table.items()}
    rows={}
    for r in episodes(directory,engine,1.,16.,seeds=(seed,)):
        rows[r['subject_id']]={'subject_id':r['subject_id'],'key':keys[r['subject_id']],'a':r['labels'],
            'support':r['support'][:,:384].copy(),'query':r['query'][:,:384].copy(),'counts':r['counts'],'successes':r['successes']}
    return rows


def main():
    start=time.time();root=Path('runs/research_loop/R37');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic()
    sources=['mlp/run/run_partition_round.py','shared_mean/partition.py','shared_mean/hierarchical.py','mlp/measurement_episodes.py','../research_loop/R37_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    selections=[]
    for fold in range(3):
        dest=root/f'fold{fold}_selection.json'
        if dest.exists():selections.append(json.loads(dest.read_text()));continue
        old=json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text());ridge=old['selected_ridge']['semantic']
        sums=np.zeros((2,len(ALPHAS)));count=0
        for directory in old['train_benchmarks']:
            rows=source_episodes(directory,engine)
            for cohort in cohorts(rows,directory):
                for j,n in enumerate(BUDGETS):
                    keys,counts,slopes,rates=evidence(cohort,n,ridge);positions={k:i for i,k in enumerate(keys)}
                    for mode,update in enumerate([True,False]):
                        for i,alpha in enumerate(ALPHAS):
                            probabilities,_=predict_counts(counts,alpha,update)
                            sums[mode,i]+=ALC_WEIGHTS[j]*sum(loss(r,probabilities[positions[r['key']]]+slope) for r,slope in zip(cohort,slopes))
                count+=len(cohort)
        selected={'fold':fold,'source_benchmarks':old['train_benchmarks'],'heldout_benchmarks':old['heldout_benchmarks'],
                  'source_seed':73,'max_source_pairs':25,'source_pairs':count,'ridge':ridge,'alphas':ALPHAS,
                  'source_alc':(sums/count).tolist(),'posterior_alpha':ALPHAS[int(np.argmin(sums[0]))],
                  'fixed_alpha':ALPHAS[int(np.argmin(sums[1]))]}
        selections.append(selected);dest.write_text(json.dumps(selected,indent=2),encoding='utf-8')
        print('selected',fold,'posterior',selected['posterior_alpha'],'fixed',selected['fixed_alpha'],flush=True)
    for fold,selection in enumerate(selections):
        historical=json.loads(Path(f'runs/research_loop/R36/fold{fold}_evaluation.json').read_text())
        hierarchy_name=json.loads(Path(f'runs/research_loop/R11/fold{fold}_selection.json').read_text())['hierarchy_selected']['adaptive']
        hierarchy=HierarchicalCounts(float(hierarchy_name.split('_')[1]),(1.,4.,16.))
        total={k:np.zeros(6) for k in NAMES};by={};count=0
        for directory,reference in historical['by_benchmark'].items():
            target=root/(directory.split('/')[0]+'_evaluation.json')
            if target.exists():current=json.loads(target.read_text())
            else:
                sums={k:np.zeros(6) for k in NAMES};posterior_independence=np.zeros(6);nrows=0;seed_scores={}
                with (root/(directory.split('/')[0]+'_pair_brier.csv')).open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.writer(stream);writer.writerow(['subject_id','seed','budget',*NAMES,'posterior_all_independent'])
                    for seed in [0,1,2]:
                        rows=outer_rows(directory,seed,engine);seed_sums={k:np.zeros(6) for k in NAMES}
                        for members in reference['cohort_members']:
                            cohort=[rows[s] for s in members]
                            for j,n in enumerate(BUDGETS):
                                keys,counts,slopes,rates=evidence(cohort,n,selection['ridge']);positions={k:i for i,k in enumerate(keys)}
                                posterior,weight=predict_counts(counts,selection['posterior_alpha'],True)
                                fixed,_=predict_counts(counts,selection['fixed_alpha'],False)
                                old_mean,_=hierarchy.predict_counts(counts)
                                for row,slope in zip(cohort,slopes):
                                    index=positions[row['key']];base,peer=rates[row['key']]
                                    predictions={'partition_item':posterior[index]+slope,'fixed_partition_item':fixed[index]+slope,
                                        'hierarchy_item':old_mean[index]+slope,'own_item':base+slope,'peer_item':peer+slope,
                                        'partition_mean':posterior[index],'hierarchy_mean':old_mean[index],'base':base,'peer':peer}
                                    values={k:loss(row,p) for k,p in predictions.items()}
                                    for k,v in values.items():seed_sums[k][j]+=v
                                    posterior_independence[j]+=weight
                                    writer.writerow([row['subject_id'],seed,n,*[values[k] for k in NAMES],weight])
                            nrows+=len(cohort)
                        for k in NAMES:sums[k]+=seed_sums[k]
                        seed_scores[str(seed)]={k:float(v/len(rows)@ALC_WEIGHTS) for k,v in seed_sums.items()}
                        print(directory,'seed',seed,flush=True)
                assert nrows==3*reference['pairs_per_seed'][0]
                avg={k:v/nrows for k,v in sums.items()}
                for k,oldk in [('own_item','own_original'),('peer_item','peer_original'),('base','base'),('peer','peer')]:
                    np.testing.assert_allclose(avg[k],reference['brier_mean'][oldk],atol=1e-7,rtol=0)
                current={'pairs_per_seed':[nrows//3]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},
                         'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},'alc_per_seed':seed_scores,
                         'posterior_independence_mean':(posterior_independence/nrows).tolist()}
                target.write_text(json.dumps(current,indent=2),encoding='utf-8')
            pairs=current['pairs_per_seed'][0];count+=pairs;by[directory]=current
            for k in NAMES:total[k]+=pairs*np.asarray(current['brier_mean'][k])
        result={'selection':selection,'pairs_per_seed':[count]*3,'by_benchmark':by,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],NAMES[0],NAMES[1:])
    d.update(scope='conditional cohort4/full5242pairs/three seeds/64queryitems; source-only partition priors; hosted evidence scope unverified',elapsed_seconds=time.time()-start)
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    for p in sources:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
