"""R20 controlled expansion of current item-effect evidence population."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .pooled_item_slope import predict_many
from .peer_item_predictor import means
from .visible_item_evidence import cohorts
from .source_semantic_episodes import episodes as source_episodes
from .cohort_semantic import CachedSemantic,episodes
from .evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from .summarize_candidate import summarize

RIDGES=[None,16.,4.,1.,.25]


def score(rows,directory,pooled_ridge,own_ridge,original_ridge,permutations=0):
    names=['pooled','own_selected','own_original','peer','base']+(['peer_permuted'] if permutations else [])
    output={k:[] for k in names};rng=np.random.default_rng(_stable_seed(4402,directory,'peer_slope'))
    for cohort in cohorts(rows,directory):
        values={k:np.empty((len(cohort),6)) for k in names}
        for j,n in enumerate(BUDGETS):
            context=[{'key':r['key'],'labels':r['a'][:n],'vectors':r['support'][:n]} for r in cohort];rates=means(context)
            for i,r in enumerate(cohort):
                base,peer=rates[r['key']];y=r['y']
                predictions={'base':base,'peer':peer,'pooled':predict_many(r['key'],r['query'],context,pooled_ridge),
                    'own_selected':predict_many(r['key'],r['query'],context,own_ridge,False),
                    'own_original':predict_many(r['key'],r['query'],context,original_ridge,False)}
                for k,p in predictions.items():values[k][i,j]=np.mean((p-y)**2)
                if permutations:
                    values['peer_permuted'][i,j]=0
                    for repeat in range(permutations):
                        shuffled=[{**s,'labels':rng.permutation(s['labels']) if s['key']!=r['key'] else s['labels']} for s in context]
                        p=predict_many(r['key'],r['query'],shuffled,pooled_ridge)
                        values['peer_permuted'][i,j]+=np.mean((p-y)**2)/permutations
        for k,v in values.items():output[k].extend(v)
    return {k:np.asarray(v) for k,v in output.items()}


def main():
    root=Path('runs/research_loop/R20');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic();start=time.time();selections=[]
    sources=['mlp/pooled_item_slope.py','mlp/source_semantic_episodes.py','mlp/run_pooled_slope_round.py','research_loop/R20_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text());sums={k:np.zeros(len(RIDGES)) for k in ['pooled','own_selected']};count=0
        for directory in split['train_benchmarks']:
            rows=source_episodes(directory,engine)
            if not rows:continue
            for j,ridge in enumerate(RIDGES):
                values=score(rows,directory,ridge,ridge,None)
                for k in sums:sums[k][j]+=np.sum(values[k]@ALC_WEIGHTS)
            count+=len(rows)
        selection={'fold':fold,'train_benchmarks':split['train_benchmarks'],'heldout_benchmarks':split['development_benchmarks'],
            'training_pairs':count,'training_alc':{k:(v/count).tolist() for k,v in sums.items()},'ridges':RIDGES,
            'selected_ridge':{k:RIDGES[int(np.argmin(v))] for k,v in sums.items()}}
        selections.append(selection);(root/f'fold{fold}_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('selected',fold,selection['selected_ridge'],flush=True)
    names=['pooled','own_selected','own_original','peer','base','peer_permuted']
    for fold,selection in enumerate(selections):
        total={k:np.zeros(6) for k in names};count=0;by={}
        orig=json.loads(Path(f'runs/research_loop/R7/fold{fold}_selection.json').read_text())
        old=json.loads(Path(f'runs/research_loop/R19/fold{fold}_evaluation.json').read_text())
        for directory in selection['heldout_benchmarks']:
            rows=episodes(directory,0,engine)
            if not rows:continue
            out=score(rows,directory,selection['selected_ridge']['pooled'],selection['selected_ridge']['own_selected'],orig['selected_ridge']['semantic'],8)
            avg={k:v.mean(0) for k,v in out.items()};n=len(rows);count+=n
            for k,old_k in [('own_original','combined'),('peer','peer'),('base','base')]:np.testing.assert_allclose(avg[k],old['by_benchmark'][directory]['brier_mean'][old_k],atol=1e-12,rtol=0)
            by[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():total[k]+=n*v
            print('outer',fold,directory,flush=True)
        result={'selection':selection,'pairs_per_seed':[count],'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'pooled',names[1:]);d['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
