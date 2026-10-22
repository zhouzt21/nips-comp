"""R28 symmetric source distribution versus matched-variance Beta control."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .evaluate import BUDGETS,ALC_WEIGHTS
from .empirical_bayes import fit_lookup
from .run_shared_round import records
from .summarize_candidate import summarize


def fit_symmetric(cache, directories):
    qs=[];ws=[]
    for directory in directories:
        e=cache['by_benchmark'][directory]
        if not e['pair_rates']:continue
        rates=np.asarray(e['pair_rates']);counts=np.asarray(e['pair_response_counts'])
        q=(rates*counts+.5)/(counts+1)
        qs.extend(q.tolist());qs.extend((1-q).tolist());ws.extend([.5/len(q)]*(2*len(q)))
    q=np.asarray(qs);w=np.asarray(ws);w/=w.sum();var=float(w@((q-.5)**2));alpha=1/(4*var)-1
    assert 0<var<.25 and alpha>0
    table=np.full((32,32),.5);matched=table.copy()
    for n in range(32):
        for s in range(n+1):
            logw=np.log(w)+s*np.log(q)+(n-s)*np.log1p(-q);weights=np.exp(logw-logw.max())
            table[n,s]=weights@q/weights.sum();matched[n,s]=(s+alpha/2)/(n+alpha)
        np.testing.assert_allclose(table[n,:n+1],1-table[n,:n+1][::-1],atol=1e-12,rtol=0)
    return {'source_revision':cache['source_revision'],'train_benchmarks':directories,'prior_mean':.5,
            'variance':var,'matched_beta_strength':alpha,'symmetric':table.tolist(),'matched_beta':matched.tolist()}


def main():
    root=Path('runs/research_loop/R28');root.mkdir(parents=True,exist_ok=True);start=time.time()
    cache=json.loads(Path('runs/mlp_failure_2026_10_22/stat_prior/pair_rates.json').read_text())
    names=['symmetric','matched_beta','original_E3','smoothed_mean_2','training_selected_smoothed']
    fits=[]
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text())
        assert not set(split['train_benchmarks'])&set(split['development_benchmarks'])
        fit=fit_symmetric(cache,split['train_benchmarks']);fit['original_E3']=fit_lookup(cache,split['train_benchmarks'])['posterior_lookup']
        fit['heldout_benchmarks']=split['development_benchmarks'];fits.append(fit)
        (root/f'fold{fold}_prior.json').write_text(json.dumps(fit,indent=2),encoding='utf-8')
    for fold,fit in enumerate(fits):
        old=json.loads(Path(f'runs/research_loop/R25_verify/fold{fold}_evaluation.json').read_text())
        e3=json.loads(Path(f'runs/mlp_failure_2026_10_22/empirical_bayes/fold{fold}_evaluation.json').read_text())
        totals={k:np.zeros(6) for k in names};by={};count=0
        for directory in fit['heldout_benchmarks']:
            collected={k:[] for k in names[:-1]};per_seed={};counts=[]
            for seed in [0,1,2]:
                rows=records(directory,seed,100)
                if not rows:continue
                values={k:[] for k in collected}
                for row in rows:
                    q=row['evaluation_mean'];cur={k:[] for k in values}
                    for n in BUDGETS:
                        s=sum(row['a'][:n]);ps={k:fit[k][n][s] for k in names[:3]};ps['smoothed_mean_2']=(1+s)/(2+n)
                        for k,p in ps.items():cur[k].append(p*p-2*p*q+q)
                    for k,v in cur.items():values[k].append(v)
                avg={k:np.mean(v,axis=0) for k,v in values.items()}
                if seed==0:
                    np.testing.assert_allclose(avg['original_E3']@ALC_WEIGHTS,e3['by_benchmark'][directory]['alc_mean']['empirical_bayes'],atol=1e-7,rtol=0)
                per_seed[str(seed)]={k:float(v@ALC_WEIGHTS) for k,v in avg.items()}
                for k,v in avg.items():collected[k].append(v)
                counts.append(len(rows))
            if not counts:continue
            avg={k:np.mean(v,axis=0) for k,v in collected.items()};historic=old['by_benchmark'][directory]
            np.testing.assert_allclose(avg['smoothed_mean_2'],historic['brier_mean']['smoothed_mean_2'],atol=1e-7,rtol=0)
            avg['training_selected_smoothed']=np.asarray(historic['brier_mean']['training_selected_smoothed'])
            n=counts[0];assert counts==[n]*3;count+=n
            by[directory]={'pairs_per_seed':counts,'brier_mean':{k:v.tolist() for k,v in avg.items()},
                           'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()},'alc_per_seed':per_seed}
            for k,v in avg.items():totals[k]+=n*v
        result={'seeds':[0,1,2],'pairs_per_seed':[count]*3,'by_benchmark':by,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'symmetric',names[1:]);d['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/run_symmetric_prior_round.py','mlp/empirical_bayes.py','mlp/run_shared_round.py','research_loop/R28_protocol.md']
    snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
