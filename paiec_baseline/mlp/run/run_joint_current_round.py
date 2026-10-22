"""R23 fixed random-effect variances; source-selected current item covariance."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.joint_current_kernel import predict_many
from mlp.peer_item_predictor import means
from mlp.visible_item_evidence import cohorts
from mlp.source_semantic_episodes import episodes as source_episodes
from mlp.cohort_semantic import CachedSemantic,episodes
from mlp.evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from mlp.reporting.summarize_candidate import summarize

RIDGES=[None,16.,4.,1.,.25]


def score(rows,directory,ridge,permutations=0):
    names=['joint','gaussian_mean','peer','base']+(['peer_permuted'] if permutations else [])
    output={k:[] for k in names};rng=np.random.default_rng(_stable_seed(5102,directory,'joint_peer'))
    for cohort in cohorts(rows,directory):
        values={k:np.empty((len(cohort),6)) for k in names}
        for j,n in enumerate(BUDGETS):
            context=[{'key':r['key'],'labels':r['a'][:n],'vectors':r['support'][:n]} for r in cohort];rates=means(context)
            for i,r in enumerate(cohort):
                base,peer=rates[r['key']]
                predictions={'base':base,'peer':peer,'joint':predict_many(r['key'],r['query'],context,ridge),
                             'gaussian_mean':predict_many(r['key'],r['query'],context,None)}
                for k,p in predictions.items():values[k][i,j]=np.mean((p-r['y'])**2)
                if permutations:
                    values['peer_permuted'][i,j]=0
                    peers=[s['vectors'] for s in context if s['key']!=r['key']]
                    vectors=np.concatenate(peers) if peers else np.empty((0,r['query'].shape[1]))
                    for repeat in range(permutations):
                        shuffled=vectors[rng.permutation(len(vectors))];permuted=[];offset=0
                        for s in context:
                            if s['key']==r['key']:permuted.append(s)
                            else:
                                count=len(s['labels']);permuted.append({**s,'vectors':shuffled[offset:offset+count]});offset+=count
                        p=predict_many(r['key'],r['query'],permuted,ridge)
                        values['peer_permuted'][i,j]+=np.mean((p-r['y'])**2)/permutations
        for k,v in values.items():output[k].extend(v)
    return {k:np.asarray(v) for k,v in output.items()}


def main():
    root=Path('runs/research_loop/R23');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic();start=time.time();selections=[]
    sources=['mlp/joint_current_kernel.py','mlp/run/run_joint_current_round.py','../research_loop/R23_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text());sums=np.zeros(len(RIDGES));count=0
        for directory in split['train_benchmarks']:
            rows=source_episodes(directory,engine)
            if not rows:continue
            sums+=[np.sum(score(rows,directory,r)['joint']@ALC_WEIGHTS) for r in RIDGES];count+=len(rows)
        selection={'fold':fold,'train_benchmarks':split['train_benchmarks'],'heldout_benchmarks':split['development_benchmarks'],
            'training_pairs':count,'training_alc':(sums/count).tolist(),'ridges':RIDGES,'selected_ridge':RIDGES[int(np.argmin(sums))]}
        selections.append(selection);(root/f'fold{fold}_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        print('selected',fold,selection['selected_ridge'],flush=True)
    names=['joint','gaussian_mean','peer','base','peer_permuted','hard_centered']
    for fold,selection in enumerate(selections):
        old=json.loads(Path(f'runs/research_loop/R20/fold{fold}_evaluation.json').read_text());total={k:np.zeros(6) for k in names};count=0;by={}
        for directory in selection['heldout_benchmarks']:
            rows=episodes(directory,0,engine)
            if not rows:continue
            out=score(rows,directory,selection['selected_ridge'],8);avg={k:v.mean(0) for k,v in out.items()};n=len(rows);count+=n
            historic=old['by_benchmark'][directory];assert historic['pairs_per_seed'][0]==n
            for k in ['peer','base']:np.testing.assert_allclose(avg[k],historic['brier_mean'][k],atol=1e-12,rtol=0)
            avg['hard_centered']=np.array(historic['brier_mean']['pooled'])
            by[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():total[k]+=n*v
            print('outer',fold,directory,flush=True)
        result={'selection':selection,'pairs_per_seed':[count],'by_benchmark':by,'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'joint',names[1:]);d['elapsed_seconds']=time.time()-start
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
