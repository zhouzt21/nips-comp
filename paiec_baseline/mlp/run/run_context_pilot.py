"""R42 source-only fixed-rule pilot; no outer score selection or model training."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.context_features import ContextFeatures
from mlp.context_tokens import template,retained,window
from mlp.context_episodes import episodes
from mlp.cohort_semantic import CachedSemantic
from mlp.run.run_metadata_round import prediction
from mlp.measurement_episodes import weighted_brier
from mlp.evaluate import BUDGETS,ALC_WEIGHTS

NAMES=['context_semantic','old_semantic','context_lexical','old_lexical','smoothed_mean_2']


def main():
    start=time.time();root=Path('runs/research_loop/R42');root.mkdir(parents=True,exist_ok=True)
    source=['mlp/context_tokens.py','mlp/context_features.py','mlp/context_episodes.py','mlp/run/run_context_pilot.py','../research_loop/R42_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in source:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    selections=[json.loads(Path(f'runs/research_loop/R25/fold{f}_selection.json').read_text()) for f in range(3)]
    directories=sorted(set.union(*[set(s['train_benchmarks']) for s in selections]));engine=ContextFeatures();old_engine=CachedSemantic();all_results={}
    for directory in directories:
        target=root/(directory.split('/')[0]+'.json')
        if target.exists():all_results[directory]=json.loads(target.read_text());continue
        rows=episodes(directory,engine.tokenizer,old_engine)
        if not rows:continue
        changed=set();changed_queries=np.zeros(6,int);query_total=np.zeros(6,int);active=np.zeros(6,int)
        for row in rows:
            original=[window(t) for t in row['tokens']];variants={}
            for j,n in enumerate(BUDGETS):
                if n<3:continue
                rule=template(row['tokens'][:n]);active[j]+=rule is not None
                selected=[retained(t,rule) for t in row['tokens'][:n]+row['tokens'][31:]]
                source_windows=original[:n]+original[31:]
                for a,b in zip(selected,source_windows):
                    if a!=b:changed.add(a)
                changed_queries[j]+=sum(a!=b for a,b in zip(selected[n:],original[31:]));query_total[j]+=len(original)-31
                variants[n]=selected
            row['original_windows']=original;row['variants']=variants
        print('preflight',directory,'pairs',len(rows),'unique changed windows',len(changed),'changed queries',changed_queries.tolist(),flush=True)
        # Independent forward replay on unchanged windows before using new representations.
        sample=[window(t) for t in rows[0]['tokens'][:3]];replay=engine.encode_tokens(sample)
        error=max(float(np.max(np.abs(replay[k]-rows[0]['old_semantic'][i]))) for i,k in enumerate(sample));assert error<1e-5
        vectors=engine.encode_tokens(changed);configs={}
        for ridge in [4.,1.]:
            values={k:[] for k in NAMES};pair_records=[]
            for row in rows:
                losses={k:np.zeros(6) for k in NAMES}
                for j,n in enumerate(BUDGETS):
                    y=row['labels'][:n];base=(1+y.sum())/(2+n)
                    if n<3:
                        for k in NAMES:losses[k][j]=weighted_brier(base,row['counts'],row['successes'])
                        continue
                    original=row['original_windows'][:n]+row['original_windows'][31:];chosen=row['variants'][n]
                    indices=[*range(n),*range(31,len(row['tokens']))];old=row['old_semantic'][indices];meta=row['metadata'][indices]/4
                    new=np.stack([old[i] if a==b else vectors[a] for i,(a,b) in enumerate(zip(chosen,original))])
                    representations={'old_semantic':old/np.sqrt(ridge),'context_semantic':new/np.sqrt(ridge),
                        'old_lexical':np.stack([engine.lexical(t) for t in original]),'context_lexical':np.stack([engine.lexical(t) for t in chosen])}
                    for name,rep in representations.items():
                        x=np.concatenate([rep,meta],axis=1);p=prediction(x[:n],x[n:],y,1.)
                        losses[name][j]=weighted_brier(p,row['counts'],row['successes'])
                    losses['smoothed_mean_2'][j]=weighted_brier(base,row['counts'],row['successes'])
                for k,v in losses.items():values[k].append(v)
                pair_records.append({'subject_id':row['subject_id'],'brier':{k:v.tolist() for k,v in losses.items()}})
            avg={k:np.mean(v,axis=0) for k,v in values.items()};configs[str(ridge)]={'brier':{k:v.tolist() for k,v in avg.items()},'alc':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},'pairs':pair_records}
        result={'benchmark':directory,'pair_count':len(rows),'source_seed':73,'source_pair_cap':25,'unique_changed_windows':len(changed),
                'changed_queries':changed_queries.tolist(),'query_counts':query_total.tolist(),'active_templates':active.tolist(),
                'forward_replay_max_error':error,'configs':configs}
        target.write_text(json.dumps(result,indent=2),encoding='utf-8');all_results[directory]=result;engine.lexical_cache.clear()
        print('completed source benchmark',directory,configs['4.0']['alc'],flush=True)
    folds=[]
    for fold,s in enumerate(selections):
        rows=[all_results[b] for b in s['train_benchmarks'] if b in all_results];weights=np.array([r['pair_count'] for r in rows]);config=str(float(s['semantic_ridge']))
        brier={k:np.average([r['configs'][config]['brier'][k] for r in rows],axis=0,weights=weights) for k in NAMES}
        alc={k:float(v@ALC_WEIGHTS) for k,v in brier.items()};bm={k:float(np.mean([r['configs'][config]['alc'][k] for r in rows])) for k in NAMES}
        assert weights.sum()==s['training_pairs'];expected=s['training_alc'][s['metadata_grid'].index(16.)]
        assert abs(alc['old_semantic']-expected)<1e-7,(fold,alc['old_semantic'],expected)
        row={'fold':fold,'scope':'source-development only; not OOD','source_benchmarks':s['train_benchmarks'],'pairs':int(weights.sum()),
             'brier':{k:v.tolist() for k,v in brier.items()},'pair_equal_alc':alc,'benchmark_equal_alc':bm}
        (root/f'fold{fold}_source.json').write_text(json.dumps(row,indent=2),encoding='utf-8');folds.append(row)
    admit=all(r['pair_equal_alc']['context_semantic']<r['pair_equal_alc']['old_semantic'] and r['benchmark_equal_alc']['context_semantic']<r['benchmark_equal_alc']['old_semantic'] for r in folds)
    summary={'scope':'Source-only fixed-rule pilot; NO heldout generalization estimate','folds':folds,'admit_outer':admit,
             'new_embeddings_this_run':engine.encoded,'cache_path':str(engine.path),'elapsed_seconds':time.time()-start}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for p in source:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps({'admit_outer':admit,'source_alc':[r['pair_equal_alc'] for r in folds],'elapsed_seconds':summary['elapsed_seconds'],'new_embeddings_this_run':engine.encoded},indent=2),flush=True)


if __name__=='__main__':main()
