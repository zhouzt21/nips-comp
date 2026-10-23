"""R38 retrospective relation audit; full rates never enter a submitted predictor."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from mlp.subject_affinity import tokens,jaccard
from mlp.model import _SUBJECT_FIELDS
from mlp.visible_item_evidence import plain
from mlp.evaluate import _stable_seed


def main():
    start=time.time();root=Path('runs/research_loop/R38');root.mkdir(parents=True,exist_ok=True)
    methods=['configuration','name','all','permuted_configuration'];records=[];details=[]
    for fold in range(3):
        old=json.loads(Path(f'runs/research_loop/R36/fold{fold}_evaluation.json').read_text())
        for directory,ref in old['by_benchmark'].items():
            folder=Path('D:/paiec_payload/data')/directory
            data=pd.read_parquet(folder/'response.parquet',columns=['subject_id','response']);data=data[data.response.isin([0,1])]
            rates={str(k):float(v) for k,v in data.groupby('subject_id',sort=False).response.mean().items()}
            subjects=pd.read_parquet(folder/'subjects.parquet').set_index('subject_id').to_dict('index')
            subjects={str(k):{f:plain(v.get(f)) for f in _SUBJECT_FIELDS} for k,v in subjects.items()}
            ids=[s for cohort in ref['cohort_members'] for s in cohort];features={s:tokens(subjects[s]) for s in ids}
            keys={s:json.dumps(subjects[s],sort_keys=True) for s in ids};names={s:subjects[s]['normalized_name'].casefold() for s in ids}
            rng=np.random.default_rng(_stable_seed(3801,directory,'configuration_permutation'))
            controls=[dict(zip(ids,rng.permutation(ids))) for _ in range(8)];rows=[]
            for cohort in ref['cohort_members']:
                for target in cohort:
                    peers=[s for s in cohort if keys[s]!=keys[target]]
                    if not peers:continue
                    q=rates[target];qp=np.array([rates[s] for s in peers]);size=len(peers)
                    def assess(weights):
                        w=np.asarray(weights,float);w=w/w.sum()
                        return np.array([np.sum(w*(qp-q)**2),(w@qp-q)**2])
                    scores=np.array([jaccard(features[target],features[s]) for s in peers]);selected=scores==scores.max()
                    matches=np.array([bool(names[target]) and names[target]==names[s] for s in peers]);matches=matches if matches.any() else np.ones(size,bool)
                    values={'configuration':assess(selected),'name':assess(matches),'all':assess(np.ones(size))};shuffled=[]
                    for mapping in controls:
                        sim=np.array([jaccard(features[mapping[target]],features[mapping[s]]) for s in peers]);shuffled.append(assess(sim==sim.max()))
                    values['permuted_configuration']=np.mean(shuffled,axis=0)
                    row={'benchmark':directory,'fold':fold,'subject_id':target,'peers':size,'chosen_peers':int(selected.sum()),
                         'has_extra':any(k.startswith('extra:') for k,v in features[target]),
                         **{k:float(v[0]) for k,v in values.items()},**{k+'_mean_error':float(v[1]) for k,v in values.items()}}
                    rows.append(row);records.append(row)
            if rows:
                frame=pd.DataFrame(rows);keys_metric=[*methods,*[k+'_mean_error' for k in methods]]
                details.append({'benchmark':directory,'fold':fold,'targets':len(rows),**{k:float(frame[k].mean()) for k in keys_metric}})
            print(directory,'eligible peer targets',len(rows),flush=True)
    frame=pd.DataFrame(records);frame.to_csv(root/'target_diagnostics.csv',index=False)
    metric=[*methods,*[k+'_mean_error' for k in methods]]
    overall={k:float(frame[k].mean()) for k in metric};folds={str(f):{k:float(g[k].mean()) for k in metric} for f,g in frame.groupby('fold')}
    rng=np.random.default_rng(3802);counts=np.array([b['targets'] for b in details]);draws=rng.integers(len(details),size=(10000,len(details)))
    comparisons={}
    for method in ['all','name','permuted_configuration']:
        delta=np.array([b['configuration_mean_error']-b[method+'_mean_error'] for b in details])
        boot=(delta[draws]*counts[draws]).sum(1)/counts[draws].sum(1)
        comparisons[method]={'pair_equal_proxy_delta':overall['configuration_mean_error']-overall[method+'_mean_error'],
            'fold_deltas':[folds[str(f)]['configuration_mean_error']-folds[str(f)][method+'_mean_error'] for f in range(3)],
            'cluster_ci95':np.quantile(boot,[.025,.975]).tolist()}
    d={'scope':'retrospective full-rate similarity proxies,not Brier/ALC or a deployed predictor; conditional fixed cohorts',
       'targets':len(frame),'extra_token_coverage':float(frame.has_extra.mean()),'mean_selected_peer_count':float(frame.chosen_peers.mean()),
       'pair_equal_proxy':overall,'benchmark_equal_proxy':{k:float(np.mean([b[k] for b in details])) for k in metric},
       'folds':folds,'by_benchmark':details,'comparisons':comparisons,'elapsed_seconds':time.time()-start}
    d['entry_condition_met']=all(v<0 for v in comparisons['all']['fold_deltas']) and comparisons['permuted_configuration']['pair_equal_proxy_delta']<0
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/audit/audit_subject_affinity.py','mlp/subject_affinity.py','../research_loop/R38_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps({k:d[k] for k in ['pair_equal_proxy','comparisons','entry_condition_met','elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
