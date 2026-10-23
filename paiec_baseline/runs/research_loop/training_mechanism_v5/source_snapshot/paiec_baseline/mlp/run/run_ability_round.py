"""R15b held-out relative ability diagnostic and single-component peer correction."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.relative_ability import aggregate,fit,key
from mlp.run.run_shared_round import records
from mlp.audit.audit_pooling_mechanism import plugin
from mlp.evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from mlp.reporting.summarize_candidate import summarize


def corrected(rows,directory,model,alpha,kappa):
    ability=model['ability'];order=np.arange(len(rows));np.random.default_rng(_stable_seed(193,directory,'cohort')).shuffle(order)
    scores=[]
    for offset in range(0,len(order),4):
        cohort=[rows[i] for i in order[offset:offset+4]]
        for target in cohort:
            ta=ability.get(key(target['subject']));values=[]
            for n in BUDGETS:
                own_s=own_n=peer_s=peer_n=correction=0
                for row in cohort:
                    if row['key']==target['key']:own_s+=sum(row['a'][:n]);own_n+=n
                    else:
                        peer_s+=sum(row['a'][:n]);peer_n+=n
                        pa=ability.get(key(row['subject']))
                        if ta is not None and pa is not None:correction+=n*(ta-pa)
                if peer_n:
                    prior=np.clip((kappa*.5+peer_s+correction)/(kappa+peer_n),0,1)
                    p=(alpha*prior+own_s)/(alpha+own_n)
                else:p=(1+own_s)/(2+own_n)
                q=target['evaluation_mean'];values.append(p*p-2*p*q+q)
            scores.append(values)
    return np.array(scores)


def main():
    root=Path('runs/research_loop/R15');root.mkdir(parents=True,exist_ok=True);start=time.time();models=[]
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text());path=root/f'fold{fold}_ability.json'
        if path.exists():model=json.loads(path.read_text())
        else:
            rows=[]
            for directory in split['train_benchmarks']:rows.extend(aggregate(directory))
            model=fit(rows,split['train_benchmarks']);path.write_text(json.dumps(model,indent=2),encoding='utf-8')
        assert set(model['train_benchmarks'])==set(split['train_benchmarks'])
        assert not set(model['train_benchmarks'])&set(split['development_benchmarks'])
        models.append(model)
    diagnostics=[]
    for fold,model in enumerate(models):
        old=json.loads(Path(f'runs/research_loop/R12/fold{fold}_evaluation.json').read_text());sel=old['selection']
        names=['ability_peer','peer','base'];totals={k:np.zeros(6) for k in names};by_benchmark={};count=0;covered=0
        for directory in sel['heldout_benchmarks']:
            avg={k:[] for k in names};query_means=[];last_rows=[]
            for seed in [0,1,2]:
                rows=records(directory,seed,100)
                if not rows:continue
                last_rows=rows;query_means.append([r['evaluation_mean'] for r in rows])
                avg['ability_peer'].append(corrected(rows,directory,model,sel['plugin_alpha'],sel['plugin_kappa']).mean(0))
                avg['peer'].append(plugin(rows,directory,sel['plugin_alpha'],sel['plugin_kappa']).mean(0))
            if not last_rows:continue
            n=len(last_rows);count+=n
            avg['base']=[old['by_benchmark'][directory]['brier_mean']['smoothed_mean_2']]
            avg={k:np.mean(v,axis=0) for k,v in avg.items()}
            np.testing.assert_allclose(avg['peer'],old['by_benchmark'][directory]['brier_mean']['plugin_selected'],atol=1e-12)
            a=np.array([model['ability'].get(key(r['subject']),np.nan) for r in last_rows]);q=np.mean(query_means,axis=0)
            mask=np.isfinite(a);covered+=int(mask.sum());a,q=a[mask],q[mask];comparisons=[]
            for i in range(len(a)):
                for j in range(i):
                    if a[i]==a[j] or q[i]==q[j]:continue
                    comparisons.append(float((a[i]-a[j])*(q[i]-q[j])>0))
            diagnostics.append({'fold':fold,'benchmark':directory,'pairs':n,'supported_pairs':int(mask.sum()),
                'ranking_comparisons':len(comparisons),'sign_agreement':float(np.mean(comparisons)) if comparisons else None,
                'centered_covariance':float(np.mean((a-a.mean())*(q-q.mean()))) if len(a)>1 else None})
            by_benchmark[directory]={'pairs_per_seed':[n]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():totals[k]+=n*v
        output={'selection':sel,'ability_model':str(root/f'fold{fold}_ability.json'),'supported_pairs':covered,'pairs_per_seed':[count]*3,
                'by_benchmark':by_benchmark,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    s=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'ability_peer',['peer','base']);s['scope']='Source-only relative ability; conditional cohort4, public benchmark holdout, no subject cold-start claim'
    s['ranking_diagnostics']=diagnostics;s['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    sources=['mlp/run/run_ability_round.py','mlp/relative_ability.py','mlp/audit/audit_subject_overlap.py','../research_loop/R15b_protocol.md']
    snapshot=root/'code';snapshot.mkdir(exist_ok=True)
    for p in sources:(snapshot/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps({k:s[k] for k in ['pair_equal_alc','comparisons']},indent=2))


if __name__=='__main__':main()
