"""R26 fixed predictor, online feature-variance acquisition versus random."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.active_episodes import episodes
from mlp.metadata_episodes import episodes as original_episodes
from mlp.semantic_features import SemanticFeatures
from mlp.cohort_semantic import CachedSemantic
from mlp.online_variance_acquisition import select_indices
from mlp.run.run_metadata_penalty_round import score as predictor_score
from mlp.evaluate import BUDGETS,ALC_WEIGHTS
from mlp.reporting.summarize_candidate import summarize

THRESHOLDS=[0.,.05,.1,.2]


def score(row,sem,meta,threshold):
    x=np.concatenate([row['support']/np.sqrt(sem),row['support_meta']/np.sqrt(meta)],axis=1)
    indices=select_indices(x,threshold)
    selected={**row,**{k:row[k][indices] for k in ['a','support','support_meta']}}
    predictions=predictor_score(selected,sem,meta)
    mean=np.array([np.mean(((1+selected['a'][:n].sum())/(2+n)-row['y'])**2) for n in BUDGETS])
    return predictions,mean,indices


def main():
    root=Path('runs/research_loop/R26');root.mkdir(parents=True,exist_ok=True);engine=SemanticFeatures();cached=CachedSemantic();start=time.time();selections=[]
    sources=['mlp/online_variance_acquisition.py','mlp/active_episodes.py','mlp/run/run_acquisition_round.py','../research_loop/R26_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        frozen=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text());sem=frozen['semantic_ridge'];meta=frozen['metadata_ridge'];sums=np.zeros(len(THRESHOLDS));count=0
        for directory in frozen['train_benchmarks']:
            rows=episodes(directory,73,25,engine);original=original_episodes(directory,73,25,cached);assert len(rows)==len(original)
            for row,old in zip(rows,original):
                for key in ['a','support','support_meta']:np.testing.assert_allclose(row[key][:31],old[key],atol=1e-7,rtol=0)
                np.testing.assert_array_equal(row['y'],old['y'])
                sums+=[score(row,sem,meta,t)[0]@ALC_WEIGHTS for t in THRESHOLDS];count+=1
        selection={**frozen,'threshold_grid':THRESHOLDS,'training_acquisition_alc':(sums/count).tolist(),'selected_threshold':THRESHOLDS[int(np.argmin(sums))]}
        selections.append(selection);(root/f'fold{fold}_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('selected',fold,selection['selected_threshold'],flush=True)
    if all(s['selected_threshold']==0 for s in selections):
        result={'status':'source_selection_rejects_active_policy_in_all_folds','selections':selections,
                'outer_evaluation':'not run: selectedpolicy is exactly randomfirst31 for all possible featureinputs',
                'elapsed_seconds':time.time()-start,'new_embeddings':engine.encoded}
        (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    else:
        names=['active','random','active_mean','random_mean','training_selected_smoothed']
        for fold,selection in enumerate(selections):
            old=json.loads(Path(f'runs/research_loop/R25/fold{fold}_evaluation.json').read_text());total={k:np.zeros(6) for k in names};count=0;by={};indices_changed=0
            for directory in selection['heldout_benchmarks']:
                rows=episodes(directory,0,100,engine)
                if not rows:continue
                values={k:[] for k in names[:4]}
                for row in rows:
                    p,m,idx=score(row,selection['semantic_ridge'],selection['metadata_ridge'],selection['selected_threshold'])
                    rp,rm,_=score(row,selection['semantic_ridge'],selection['metadata_ridge'],0)
                    indices_changed+=int(not np.array_equal(idx,np.arange(31)))
                    for k,v in [('active',p),('random',rp),('active_mean',m),('random_mean',rm)]:values[k].append(v)
                avg={k:np.mean(v,axis=0) for k,v in values.items()};n=len(rows);count+=n;historic=old['by_benchmark'][directory]
                np.testing.assert_allclose(avg['random'],historic['brier_mean']['separate'],atol=1e-7,rtol=0)
                np.testing.assert_allclose(avg['random_mean'],historic['brier_mean']['smoothed_mean_2'],atol=1e-7,rtol=0)
                avg['training_selected_smoothed']=np.asarray(historic['brier_mean']['training_selected_smoothed'])
                by[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
                for k,v in avg.items():total[k]+=n*v
                print('outer',fold,directory,flush=True)
            result={'selection':selection,'pairs_per_seed':[count],'changed_trajectories':indices_changed,'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
            (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        result=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'active',names[1:]);result['elapsed_seconds']=time.time()-start;result['new_embeddings']=engine.encoded
        (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(result.get('pair_equal_alc',{'status':result.get('status')}),indent=2),flush=True)


if __name__=='__main__':main()
