"""R21 fixed degree4 geometry, matched to frozen R20 linear comparisons."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .polynomial_item_kernel import predict_many
from .visible_item_evidence import cohorts
from .source_semantic_episodes import episodes as source_episodes
from .cohort_semantic import CachedSemantic,episodes
from .evaluate import BUDGETS,ALC_WEIGHTS
from .summarize_candidate import summarize

RIDGES=[None,16.,4.,1.,.25]


def score(rows,directory,pooled_ridge,own_ridge):
    output={k:[] for k in ['pooled4','own4']}
    for cohort in cohorts(rows,directory):
        values={k:np.empty((len(cohort),6)) for k in output}
        for j,n in enumerate(BUDGETS):
            context=[{'key':r['key'],'labels':r['a'][:n],'vectors':r['support'][:n]} for r in cohort]
            for i,r in enumerate(cohort):
                for name,ridge,pooled in [('pooled4',pooled_ridge,True),('own4',own_ridge,False)]:
                    p=predict_many(r['key'],r['query'],context,ridge,pooled)
                    values[name][i,j]=np.mean((p-r['y'])**2)
        for k,v in values.items():output[k].extend(v)
    return {k:np.asarray(v) for k,v in output.items()}


def main():
    root=Path('runs/research_loop/R21');root.mkdir(parents=True,exist_ok=True);start=time.time();engine=CachedSemantic();selections=[]
    sources=['mlp/polynomial_item_kernel.py','mlp/run_polynomial_round.py','research_loop/R21_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text());sums={k:np.zeros(len(RIDGES)) for k in ['pooled4','own4']};count=0
        for directory in split['train_benchmarks']:
            rows=source_episodes(directory,engine)
            if not rows:continue
            for j,ridge in enumerate(RIDGES):
                out=score(rows,directory,ridge,ridge)
                for k in sums:sums[k][j]+=np.sum(out[k]@ALC_WEIGHTS)
            count+=len(rows)
        selection={'fold':fold,'train_benchmarks':split['train_benchmarks'],'heldout_benchmarks':split['development_benchmarks'],
            'training_pairs':count,'training_alc':{k:(v/count).tolist() for k,v in sums.items()},'ridges':RIDGES,
            'selected_ridge':{k:RIDGES[int(np.argmin(v))] for k,v in sums.items()}}
        selections.append(selection);(root/f'fold{fold}_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('selected',fold,selection['selected_ridge'],flush=True)
    names=['pooled4','own4','pooled1','own1','own_original','peer','base']
    for fold,selection in enumerate(selections):
        old=json.loads(Path(f'runs/research_loop/R20/fold{fold}_evaluation.json').read_text());total={k:np.zeros(6) for k in names};count=0;by={}
        for directory in selection['heldout_benchmarks']:
            rows=episodes(directory,0,engine)
            if not rows:continue
            out=score(rows,directory,selection['selected_ridge']['pooled4'],selection['selected_ridge']['own4'])
            avg={k:v.mean(0) for k,v in out.items()};n=len(rows);count+=n
            historic=old['by_benchmark'][directory];assert historic['pairs_per_seed'][0]==n
            for k,source in [('pooled1','pooled'),('own1','own_selected'),('own_original','own_original'),('peer','peer'),('base','base')]:avg[k]=np.array(historic['brier_mean'][source])
            by[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():total[k]+=n*v
            print('outer',fold,directory,flush=True)
        result={'selection':selection,'pairs_per_seed':[count],'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'pooled4',names[1:]);d['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
