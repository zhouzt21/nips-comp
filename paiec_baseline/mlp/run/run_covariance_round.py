"""R29 benchmark-nested selection of sign-invariant source geometry."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.task_covariance import fit_covariance,score
from mlp.source_semantic_episodes import episodes as source_episodes
from mlp.cohort_semantic import CachedSemantic,episodes
from mlp.evaluate import ALC_WEIGHTS,_stable_seed
from mlp.reporting.summarize_candidate import summarize

RHOS=[0.,.25,.75]


def main():
    root=Path('runs/research_loop/R29');root.mkdir(parents=True,exist_ok=True);start=time.time();engine=CachedSemantic();fitted=[]
    for fold in range(3):
        old=json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text());ridge=old['selected_ridge']['semantic']
        train=sorted(old['train_benchmarks']);np.random.default_rng(2901).shuffle(train);halves=[train[::2],train[1::2]]
        data={d:source_episodes(d,engine) for d in train};selection={'fold':fold,'train_benchmarks':train,'heldout_benchmarks':old['heldout_benchmarks'],'inner_halves':halves,'ridge':ridge,'rho_grid':RHOS,'models':{}};models={}
        for name,perm in [('learned',False),('shuffled',True)]:
            sums=np.zeros(3);count=0
            for inner in range(2):
                projection,_=fit_covariance([data[d] for d in halves[inner]],_stable_seed(2902,str(fold),str(inner)) if perm else None)
                for directory in halves[1-inner]:
                    for row in data[directory]:
                        sums+=[score(row,projection,rho,ridge)@ALC_WEIGHTS for rho in RHOS];count+=1
            rho=RHOS[int(np.argmin(sums))];projection,meta=fit_covariance([data[d] for d in train],2903+fold if perm else None)
            models[name]=(projection,rho);selection['models'][name]={'rho':rho,'inner_alc':(sums/count).tolist(),'validation_pairs':count,**meta}
            np.savez_compressed(root/f'fold{fold}_{name}_covariance.npz',projection=projection,rho=rho,ridge=ridge)
            print('selected',fold,name,rho,flush=True)
        fitted.append((selection,models));(root/f'fold{fold}_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    names=['learned','shuffled','semantic','R25','smoothed_mean_2','training_selected_smoothed']
    for fold,(selection,models) in enumerate(fitted):
        old=json.loads(Path(f'runs/research_loop/R25/fold{fold}_evaluation.json').read_text());count=0;totals={k:np.zeros(6) for k in names};by={}
        for directory in selection['heldout_benchmarks']:
            rows=episodes(directory,0,engine)
            if not rows:continue
            avg={k:np.mean([score(row,*models[k],selection['ridge']) for row in rows],axis=0) for k in ['learned','shuffled']}
            historic=old['by_benchmark'][directory];n=len(rows);assert historic['pairs_per_seed'][0]==n
            replay=np.mean([score(row,models['learned'][0],0,selection['ridge']) for row in rows],axis=0)
            np.testing.assert_allclose(replay,historic['brier_mean']['semantic'],atol=1e-7,rtol=0)
            for k,source in [('semantic','semantic'),('R25','separate'),('smoothed_mean_2','smoothed_mean_2'),('training_selected_smoothed','training_selected_smoothed')]:avg[k]=np.asarray(historic['brier_mean'][source])
            count+=n;by[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():totals[k]+=n*v
        result={'selection':selection,'pairs_per_seed':[count],'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('outer',fold,flush=True)
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'learned',names[1:]);d['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/task_covariance.py','mlp/run/run_covariance_round.py','mlp/source_semantic_episodes.py','mlp/local_ridge.py','../research_loop/R29_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
