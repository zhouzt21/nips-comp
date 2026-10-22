"""R19 select only on source benchmarks, then test legal peer-item evidence."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .visible_item_evidence import episodes,cohorts
from .peer_item_predictor import means,predict_many
from .evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from .cohort_semantic import CachedSemantic,episodes as semantic_episodes
from .run_component_round import score as component_score
from .summarize_candidate import summarize

TAUS=[None,1.,4.,16.]


def score(rows,directory,tau,permutations=0):
    names=['matched','peer','base']+(['permuted'] if permutations else [])
    output={k:[] for k in names}
    rng=np.random.default_rng(_stable_seed(90210,directory,'item_correspondence'))
    for cohort in cohorts(rows,directory):
        values={k:np.empty((len(cohort),6)) for k in names}
        for j,n in enumerate(BUDGETS):
            context=[{'key':r['key'],'items':r['support_keys'][:n],'labels':r['a'][:n]} for r in cohort]
            rates=means(context)
            for i,r in enumerate(cohort):
                y=np.asarray(r['y']);base,peer=rates[r['key']]
                predictions={'base':base,'peer':peer,'matched':predict_many(r['key'],r['query_keys'],context,tau)}
                for k,p in predictions.items():values[k][i,j]=np.mean((p-y)**2)
            if permutations:
                values['permuted'][:,j]=0
                for repeat in range(permutations):
                    shuffled=[{**r,'items':rng.permutation(r['items']).tolist()} for r in context]
                    for i,r in enumerate(cohort):
                        p=predict_many(r['key'],r['query_keys'],shuffled,tau)
                        values['permuted'][i,j]+=np.mean((p-np.asarray(r['y']))**2)/permutations
        for k in names:output[k].extend(values[k])
    return {k:np.asarray(v) for k,v in output.items()}


def main():
    root=Path('runs/research_loop/R19');root.mkdir(parents=True,exist_ok=True);start=time.time();selections=[]
    sources=['mlp/visible_item_evidence.py','mlp/peer_item_predictor.py','mlp/run_peer_item_round.py','research_loop/R19b_protocol.md']
    snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text());total=np.zeros(len(TAUS));count=0
        assert not set(split['train_benchmarks']) & set(split['development_benchmarks'])
        for directory in split['train_benchmarks']:
            rows=episodes(directory,73,25)
            if not rows:continue
            total+=[np.sum(score(rows,directory,t)['matched']@ALC_WEIGHTS) for t in TAUS];count+=len(rows)
        selection={'fold':fold,'train_benchmarks':split['train_benchmarks'],'heldout_benchmarks':split['development_benchmarks'],
                   'taus':TAUS,'training_alc':(total/count).tolist(),'selected_tau':TAUS[int(np.argmin(total))],'training_pairs':count,'seed':73}
        (root/f'fold{fold}_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8');selections.append(selection)
        print('selected',selection,flush=True)
    engine=CachedSemantic();names=['matched','peer','base','permuted','combined']
    for fold,selection in enumerate(selections):
        totals={k:np.zeros(6) for k in names};count=0;by={}
        local=json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text())
        for directory in selection['heldout_benchmarks']:
            rows=episodes(directory,0)
            if not rows:continue
            out=score(rows,directory,selection['selected_tau'],8)
            old=component_score(semantic_episodes(directory,0,engine),directory,4,local['selected_ridge']['semantic'],4.,8.)
            np.testing.assert_allclose(out['peer'],old['peer'],atol=1e-12,rtol=0)
            out['combined']=old['combined'];n=len(rows);count+=n;avg={k:v.mean(0) for k,v in out.items()}
            by[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():totals[k]+=n*v
        result={'selection':selection,'pairs_per_seed':[count],'by_benchmark':by,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in totals.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    summary=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'matched',names[1:]);summary['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(summary['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
