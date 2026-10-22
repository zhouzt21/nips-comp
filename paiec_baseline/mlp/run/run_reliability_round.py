"""R43 benchmark-nested source selection for query versus episode calibration."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.cohort_semantic import CachedSemantic
from mlp.reliability_training import source_data,combine,train
from mlp.evidence_reliability import FEATURES
from mlp.evaluate import ALC_WEIGHTS


def main():
    start=time.time();root=Path('runs/research_loop/R43');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic()
    source=['mlp/evidence_reliability.py','mlp/reliability_training.py','mlp/run/run_reliability_round.py','mlp/measurement_episodes.py','mlp/local_scale.py','../research_loop/R43_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in source:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    folds=[]
    for fold in range(3):
        target=root/f'fold{fold}_selection.json'
        if target.exists():folds.append(json.loads(target.read_text()));continue
        old=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text());data={}
        for directory in old['train_benchmarks']:
            row=source_data(directory,engine,old['semantic_ridge'])
            if int(row['pairs']):data[directory]=row
        names=sorted(data);np.random.default_rng(4301+fold).shuffle(names);halves=[names[::2],names[1::2]]
        all_data=combine(data,names);assert all_data['pairs']==old['training_pairs']
        baseline=float(sum(int(r['pairs'])*(r['baseline_brier']@ALC_WEIGHTS) for r in data.values())/all_data['pairs'])
        expected=old['training_alc'][old['metadata_grid'].index(16.)];assert abs(baseline-expected)<1e-7
        selection={'fold':fold,'source_benchmarks':old['train_benchmarks'],'heldout_benchmarks':old['heldout_benchmarks'],
                   'inner_halves':halves,'pairs':all_data['pairs'],'rows':len(all_data['z']),'source_baseline_alc':baseline,
                   'semantic_ridge':old['semantic_ridge'],'metadata_ridge':16.,'features':FEATURES,'models':{}}
        for model,episode_only in [('query',False),('episode',True)]:
            curves=[];counts=[];inner_records=[]
            for inner in range(2):
                training=combine(data,halves[inner]);validation=combine(data,halves[1-inner]);assert not set(halves[inner])&set(halves[1-inner])
                folder=root/f'fold{fold}_{model}_inner{inner}'
                history,_=train(training,validation,episode_only,20,folder)
                curves.append([h['validation_alc'] for h in history]);counts.append(validation['pairs'])
                inner_records.append({'train_benchmarks':halves[inner],'validation_benchmarks':halves[1-inner],'train_pairs':training['pairs'],'validation_pairs':validation['pairs'],'history':history})
                print('trained',fold,model,'inner',inner,'best epoch',int(np.argmin(curves[-1])),flush=True)
            curve=np.average(curves,axis=0,weights=counts);assert abs(curve[0]-baseline)<1e-7
            epoch=int(np.argmin(curve));history,artifact=train(all_data,None,episode_only,epoch,root/f'fold{fold}_{model}_refit')
            np.savez_compressed(root/f'fold{fold}_{model}.npz',**artifact)
            selection['models'][model]={'selected_epoch':epoch,'pooled_inner_alc':curve.tolist(),'selected_inner_alc':float(curve[epoch]),'inner_fits':inner_records,
                                         'refit_history':history,'weights':str(root/f'fold{fold}_{model}.npz')}
            print('selected',fold,model,epoch,float(curve[epoch]),flush=True)
        target.write_text(json.dumps(selection,indent=2),encoding='utf-8');folds.append(selection)
    admit=all(r['models']['query']['selected_inner_alc']<min(r['source_baseline_alc'],r['models']['episode']['selected_inner_alc']) for r in folds)
    summary={'scope':'Inner benchmark-grouped model selection on reused sources; no outer performance estimate yet',
             'admit_outer':admit,'folds':folds,'elapsed_seconds':time.time()-start}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for p in source:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps({'admit_outer':admit,'elapsed_seconds':summary['elapsed_seconds'],'folds':[
        {'fold':r['fold'],'R25':r['source_baseline_alc'],**{k:{'epoch':v['selected_epoch'],'inner_alc':v['selected_inner_alc']} for k,v in r['models'].items()}} for r in folds]},indent=2),flush=True)


if __name__=='__main__':main()
