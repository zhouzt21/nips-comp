"""R35 descriptive oracle/support associations plus source-only objective decomposition."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from mlp.cohort_semantic import CachedSemantic
from mlp.evaluate import BUDGETS, ALC_WEIGHTS
from mlp.measurement_episodes import episodes, weighted_brier
from mlp.run.run_local_scale_round import components, TAUS
from mlp.local_scale import coefficient


def correlation(x,y):
    if x.nunique()<2 or y.nunique()<2: return None
    return float(x.rank().corr(y.rank()))


def describe(frame):
    return {'pair_seed_count':len(frame),'rank_correlation':correlation(frame.scale,frame.oracle_c.clip(0,4)),
            'support_amplify_fraction':float((frame.scale>1.00001).mean()),
            'oracle_amplify_fraction':float((frame.oracle_c>1.00001).mean()),
            'amplification_agreement':float(((frame.scale>1.00001)==(frame.oracle_c>1.00001)).mean()),
            'adaptive_minus_R25':float((frame.adaptive_scale-frame.R25).mean()),
            'oracle_max4_minus_R25':float((frame.oracle_max4-frame.R25).mean())}


def main():
    start=time.time(); root=Path('runs/research_loop/R35'); root.mkdir(parents=True,exist_ok=True)
    records=[]; by={}
    for path in sorted(Path('runs/research_loop/R34').glob('*_pair_brier.csv')):
        benchmark=path.name.removesuffix('_pair_brier.csv')
        actual=pd.read_csv(path,dtype={'subject_id':str})
        oracle=pd.read_csv(Path('runs/research_loop/R33')/(benchmark+'_oracle.csv'),dtype={'subject_id':str})
        oracle=oracle[oracle.scope=='capped'][['subject_id','seed','budget','oracle_c','oracle_max4']]
        merged=actual.merge(oracle,on=['subject_id','seed','budget'],validate='one_to_one')
        assert len(merged)==len(actual)
        merged['benchmark']=benchmark;records.append(merged)
        by[benchmark]={str(n):describe(g) for n,g in merged.groupby('budget') if n>=3}
    data=pd.concat(records,ignore_index=True)
    outer={str(n):describe(g) for n,g in data.groupby('budget') if n>=3}
    source=[]; engine=CachedSemantic()
    for fold in range(3):
        selected=json.loads(Path(f'runs/research_loop/R34/fold{fold}_selection.json').read_text())
        sums=np.zeros((len(TAUS),6)); count=0
        for directory in selected['train_benchmarks']:
            for row in episodes(directory,engine,selected['semantic_ridge'],selected['metadata_ridge'],seeds=(73,),pair_cap=25):
                for j,n in enumerate(BUDGETS):
                    a,b,p,h=components(row,n)
                    for i,tau in enumerate(TAUS):
                        c=coefficient(h,a,tau)
                        sums[i,j]+=weighted_brier(np.clip(b+c*(p-b),0,1),row['counts'],row['successes'])
                count+=1
        scores=sums/count
        np.testing.assert_allclose(scores@ALC_WEIGHTS,selected['adaptive_source_alc'],atol=1e-12,rtol=0)
        source.append({'fold':fold,'source_pairs':count,'taus':TAUS,'brier':scores.tolist(),
                       'preferred_tau_by_budget':[TAUS[i] for i in np.argmin(scores,axis=0)],
                       'selected_tau':selected['tau']})
        print('source decomposition verified',fold,flush=True)
    d={'scope':'retrospective support/oracle association and source-only budget decomposition; no new deployable candidate',
       'outer_by_budget':outer,'outer_by_benchmark':by,'source_decomposition':source,'elapsed_seconds':time.time()-start}
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/audit/diagnose_scale_identifiability.py','mlp/local_scale.py','../research_loop/R35_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps({'outer_by_budget':outer,'source_decomposition':source},indent=2),flush=True)


if __name__=='__main__':main()
