"""R31 actual larger pair coverage, retaining frozen 64-query-item proxy."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .evaluate import _load_directory,_stable_seed,BUDGETS,ALC_WEIGHTS
from .metadata_features import encode
from .semantic_features import SemanticFeatures
from .run_metadata_penalty_round import score
from .summarize_candidate import summarize

NAMES=['R25','semantic','smoothed_mean_2','training_selected_smoothed','empirical_mean']


def expanded(directory,fold,sem,meta,engine,root):
    target=root/(directory.split('/')[0]+'.json')
    if target.exists():return json.loads(target.read_text())
    responses,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory)
    groups=responses.groupby('subject_id',sort=False)
    eligible=[sid for sid,g in groups if g['item_id'].nunique()>=80 and sid in subjects]
    chosen=np.sort(np.random.default_rng(_stable_seed(2026,directory,'pair_sample')).choice(len(eligible),100,replace=False));chosen=set(chosen.tolist())
    collected=[];subsets=[];seed_alc={};response_counts=[];before=engine.encoded
    with (root/(directory.split('/')[0]+'_pair_brier.csv')).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['seed','subject_id','in_old_sample','budget',*NAMES])
        for seed in [0,1,2]:
            trajectories=[];used=set()
            for index,sid in enumerate(eligible):
                group=groups.get_group(sid);ids=pd.unique(group['item_id']).copy();rng=np.random.default_rng(_stable_seed(seed,directory,sid));rng.shuffle(ids)
                cut=len(ids)//2;aid=ids[:31];eid=ids[cut:cut+64]
                acquired=group[group['item_id'].isin(aid)].sample(frac=1,random_state=int(rng.integers(2**31)))
                acquired=acquired.drop_duplicates('item_id').set_index('item_id').loc[aid].reset_index();evaluation=group[group['item_id'].isin(eid)]
                aq=acquired['item_id'].tolist();eq=evaluation['item_id'].tolist();used.update(aq);used.update(eq)
                trajectories.append((index,sid,aq,eq,acquired['response'].to_numpy(float),evaluation['response'].to_numpy(float)))
            ids=list(used);vectors={};metadata={}
            for offset in range(0,len(ids),512):
                batch=ids[offset:offset+512];encoded=engine.encode_many([items[i]['content'] for i in batch])
                for i in batch:vectors[i]=encoded[engine.key(items[i]['content'])]['semantic'];metadata[i]=encode(items[i]['item_features'])
            sums={k:np.zeros(6) for k in NAMES};sub={k:np.zeros(6) for k in NAMES};nresponses=0
            for index,sid,aq,eq,a,y in trajectories:
                row={'support':np.stack([vectors[i] for i in aq]),'query':np.stack([vectors[i] for i in eq]),
                     'support_meta':np.stack([metadata[i] for i in aq]),'query_meta':np.stack([metadata[i] for i in eq]),'a':a,'y':y}
                values={'R25':score(row,sem,meta),'semantic':score(row,sem,None)}
                for key,alpha in [('smoothed_mean_2',2),('training_selected_smoothed',[2,2,4][fold]),('empirical_mean',0)]:
                    values[key]=np.array([np.mean((((a[:n].sum()+alpha*.5)/(n+alpha) if n+alpha else .5)-y)**2) for n in BUDGETS])
                for k,v in values.items():sums[k]+=v;sub[k]+=v if index in chosen else 0
                for b,n in enumerate(BUDGETS):writer.writerow([seed,str(sid),int(index in chosen),n,*[values[k][b] for k in NAMES]])
                nresponses+=len(y)
            average={k:v/len(eligible) for k,v in sums.items()};collected.append(average);subsets.append({k:v/100 for k,v in sub.items()})
            seed_alc[str(seed)]={k:float(v@ALC_WEIGHTS) for k,v in average.items()};response_counts.append(nresponses)
            print(directory,'seed',seed,'pairs',len(eligible),'new vectors',engine.encoded-before,flush=True)
    avg={k:np.mean([c[k] for c in collected],axis=0) for k in NAMES};subset={k:np.mean([c[k] for c in subsets],axis=0) for k in NAMES}
    result={'pairs_per_seed':[len(eligible)]*3,'evaluation_responses_per_seed':response_counts,
            'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},
            'alc_per_seed':seed_alc,'original_subset_brier':{k:v.tolist() for k,v in subset.items()},'new_embeddings':engine.encoded-before}
    target.write_text(json.dumps(result,indent=2),encoding='utf-8');return result


def main():
    root=Path('runs/research_loop/R31');root.mkdir(parents=True,exist_ok=True);start=time.time();engine=SemanticFeatures()
    audit=json.loads(Path('runs/research_loop/R30/summary.json').read_text());counts={r['benchmark']:r['eligible_pairs'] for r in audit['by_benchmark']}
    sources=['mlp/run_full_pair_round.py','mlp/run_metadata_penalty_round.py','mlp/metadata_features.py','mlp/semantic_features.py','research_loop/R31_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        selection=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text());old=json.loads(Path(f'runs/research_loop/R25_verify/fold{fold}_evaluation.json').read_text());raw=json.loads(Path(f'runs/research_loop/R27/fold{fold}_evaluation.json').read_text())
        totals={k:np.zeros(6) for k in NAMES};count=0;by={}
        for directory,historic in old['by_benchmark'].items():
            if counts[directory]>100:
                current=expanded(directory,fold,selection['semantic_ridge'],selection['metadata_ridge'],engine,root)
                for k,oldkey in [('R25','separate'),('semantic','semantic'),('smoothed_mean_2','smoothed_mean_2'),('training_selected_smoothed','training_selected_smoothed')]:
                    np.testing.assert_allclose(current['original_subset_brier'][k],historic['brier_mean'][oldkey],atol=1e-7,rtol=0)
                np.testing.assert_allclose(current['original_subset_brier']['empirical_mean'],raw['by_benchmark'][directory]['brier_mean']['empirical_mean'],atol=1e-7,rtol=0)
            else:
                current={'pairs_per_seed':historic['pairs_per_seed'],'brier_mean':{k:historic['brier_mean'][oldkey] for k,oldkey in [('R25','separate'),('semantic','semantic'),('smoothed_mean_2','smoothed_mean_2'),('training_selected_smoothed','training_selected_smoothed')]},'reused_uncapped_from':'R25_verify/R27'}
                current['brier_mean']['empirical_mean']=raw['by_benchmark'][directory]['brier_mean']['empirical_mean']
                current['alc_mean']={k:float(np.asarray(v)@ALC_WEIGHTS) for k,v in current['brier_mean'].items()}
            n=current['pairs_per_seed'][0];assert n==counts[directory];count+=n;by[directory]=current
            for k in NAMES:totals[k]+=n*np.asarray(current['brier_mean'][k])
        result={'selection':selection,'seeds':[0,1,2],'pairs_per_seed':[count]*3,'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    result=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'R25',NAMES[1:]);result.update(elapsed_seconds=time.time()-start,new_embeddings=engine.encoded,scope='actual full eligible-pair coverage; still64evaluationitems; reused benchmarks, not blind')
    (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps(result['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
