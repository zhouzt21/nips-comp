"""R17 evaluation-label oracle diagnostic, NEVER a submitted predictor."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .diagnose_local_signal import CacheOnly
from .run_semantic_round import episodes
from .local_ridge import kernel_predict
from .evaluate import BUDGETS,ALC_WEIGHTS
from .summarize_candidate import summarize


def optimum(u,v):return np.clip(np.divide(u,v,out=np.zeros_like(np.asarray(u,float)),where=np.asarray(v)>1e-20),0,1)


def main():
    root=Path('runs/research_loop/R17');root.mkdir(parents=True,exist_ok=True);engine=CacheOnly()
    names=['oracle_pair','oracle_benchmark','actual','base']
    for fold in range(3):
        sel=json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text());old=json.loads(Path(f'runs/research_loop/R7_verify/fold{fold}_evaluation.json').read_text())
        total={k:np.zeros(6) for k in names};count=0;by_benchmark={};gamma_stats={}
        for b in sel['heldout_benchmarks']:
            us=[];vs=[];bs=[];actual=[]
            for seed in [0,1,2]:
                data=episodes(b,seed,100,engine)
                for row in data:
                    ru=[];rv=[];rb=[];ra=[]
                    for n in BUDGETS:
                        labels=row['a'][:n];base=(1+labels.sum())/(2+n)
                        p=kernel_predict(row['k_semantic'],row['c_semantic'],labels,sel['selected_ridge']['semantic'])
                        d=p-base;u=float(np.mean(d*(row['y']-base)));v=float(np.mean(d*d));g=float(optimum(u,v))
                        baseline=float(np.mean((base-row['y'])**2))
                        np.testing.assert_allclose(baseline-2*g*u+g*g*v,np.mean((base+g*d-row['y'])**2),atol=1e-12)
                        ru.append(u);rv.append(v);rb.append(baseline);ra.append(float(np.mean((p-row['y'])**2)))
                    us.append(ru);vs.append(rv);bs.append(rb);actual.append(ra)
            if not us:continue
            u,v,base=np.array(us),np.array(vs),np.array(bs);g=optimum(u,v);gb=optimum(u.mean(0),v.mean(0))
            avg={'oracle_pair':np.mean(base-2*g*u+g*g*v,axis=0),'oracle_benchmark':np.mean(base-2*gb*u+gb*gb*v,axis=0),
                 'actual':np.mean(actual,axis=0),'base':base.mean(0)}
            n=len(data);count+=n
            np.testing.assert_allclose(avg['actual'],old['by_benchmark'][b]['brier_mean']['semantic'],atol=1e-7)
            gamma_stats[b]={'benchmark_gamma':gb.tolist(),'pair_gamma_mean':g.mean(0).tolist(),'pairs':n}
            by_benchmark[b]={'pairs_per_seed':[n]*3,'brier_mean':{k:a.tolist() for k,a in avg.items()},'alc_mean':{k:float(a@ALC_WEIGHTS) for k,a in avg.items()}}
            for k,a in avg.items():total[k]+=n*a
        out={'pairs_per_seed':[count]*3,'by_benchmark':by_benchmark,'oracle_gamma':gamma_stats,
             'results':{k:{'brier_mean':(a/count).tolist(),'alc_mean':float(a/count@ALC_WEIGHTS)} for k,a in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    s=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'oracle_pair',names[1:])
    s['scope']='EVALUATION-LABEL ORACLE: empirical bound only for convex shrinkage between frozen R7 and smoothed2; NOT deployable or a candidate'
    s['can_family_meet_absolute_gain_005']=s['pair_equal_alc']['base']-s['pair_equal_alc']['oracle_pair']>=.005
    (root/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    sources=['mlp/oracle_shrinkage_audit.py','research_loop/R17_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps({k:s[k] for k in ['pair_equal_alc','can_family_meet_absolute_gain_005']},indent=2))


if __name__=='__main__':main()
