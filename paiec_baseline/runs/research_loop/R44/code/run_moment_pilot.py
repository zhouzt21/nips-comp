"""R44 frozen representation/variance factorial, source-only with historical replay."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .moment_covariance import predict
from .cohort_semantic import CachedSemantic
from .metadata_episodes import episodes
from .run_metadata_round import prediction
from .evaluate import BUDGETS,ALC_WEIGHTS

NAMES=['hash_old','hash_fixed','semantic_old','semantic_fixed','semantic_uniform','R25','smoothed_mean_2']


def main():
    start=time.time();root=Path('runs/research_loop/R44');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic()
    source=['mlp/moment_covariance.py','mlp/run_moment_pilot.py','mlp/metadata_episodes.py','mlp/kernel_covariance.py','research_loop/R44_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in source:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    recorded=json.loads(Path('runs/research_loop/R3/fold0_evaluation.json').read_text())
    encoder_hash=hashlib.sha256(json.dumps({'code':recorded['code_sha256'],'revision':recorded['source_revision']},sort_keys=True).encode()).hexdigest();folds=[]
    for fold in range(3):
        target=root/f'fold{fold}_source.json'
        if target.exists():folds.append(json.loads(target.read_text()));continue
        old=json.loads(Path(f'runs/research_loop/R6/fold{fold}_selection.json').read_text());sel=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text());eta=old['selected_eta']
        by={};total={k:np.zeros(6) for k in NAMES};count=0
        for directory in old['train_benchmarks']:
            key=hashlib.sha256(f'{directory}|73|25|64|{encoder_hash}'.encode()).hexdigest();cache=Path('D:/paiec_research_cache/R3')/(key+'.npz')
            assert cache.exists(),('Historical cache absent; refuse recomputation under old key',str(cache))
            with np.load(cache,allow_pickle=False) as pack:lexical=[{k:pack[f'{i}_{k}'].copy() for k in ['k','c','a','y']} for i in range(int(pack['count']))]
            rows=episodes(directory,73,25,engine);assert len(rows)==len(lexical)
            if not rows:continue
            sums={k:np.zeros(6) for k in NAMES}
            for row,hashed in zip(rows,lexical):
                np.testing.assert_array_equal(row['a'],hashed['a']);np.testing.assert_array_equal(row['y'],hashed['y'])
                k=row['support']@row['support'].T;c=row['query']@row['support'].T
                support=np.c_[row['support']/np.sqrt(sel['semantic_ridge']),row['support_meta']/4];query=np.c_[row['query']/np.sqrt(sel['semantic_ridge']),row['query_meta']/4]
                for j,n in enumerate(BUDGETS):
                    y=row['a'][:n]
                    predictions={'hash_old':predict(hashed['k'],hashed['c'],y,eta),'hash_fixed':predict(hashed['k'],hashed['c'],y,eta,True),
                        'semantic_old':predict(k,c,y,eta),'semantic_fixed':predict(k,c,y,eta,True),'semantic_uniform':predict(k,c,y,eta,True,True),
                        'R25':prediction(support[:n],query,y,1.),'smoothed_mean_2':(1+y.sum())/(2+n)}
                    for name,p in predictions.items():sums[name][j]+=float(np.mean((p-row['y'])**2))
            n=len(rows);count+=n
            by[directory]={'pairs':n,'brier':{k:(v/n).tolist() for k,v in sums.items()},'alc':{k:float(v/n@ALC_WEIGHTS) for k,v in sums.items()}}
            for k,v in sums.items():total[k]+=v
        alc={k:float(v/count@ALC_WEIGHTS) for k,v in total.items()}
        assert abs(alc['hash_old']-old['training_alc'][old['eta_grid'].index(eta)])<1e-7
        assert abs(alc['R25']-sel['training_alc'][sel['metadata_grid'].index(16.)])<1e-7
        result={'fold':fold,'scope':'source only, not OOD','source_benchmarks':old['train_benchmarks'],'eta':eta,'pairs':count,
                'pair_equal_brier':{k:(v/count).tolist() for k,v in total.items()},'pair_equal_alc':alc,
                'benchmark_equal_alc':{k:float(np.mean([v['alc'][k] for v in by.values()])) for k in NAMES},'by_benchmark':by}
        target.write_text(json.dumps(result,indent=2),encoding='utf-8');folds.append(result);print(fold,alc,flush=True)
    admit=all(r['pair_equal_alc']['semantic_fixed']<min(r['pair_equal_alc']['R25'],r['pair_equal_alc']['semantic_old']) and r['benchmark_equal_alc']['semantic_fixed']<r['benchmark_equal_alc']['R25'] for r in folds)
    summary={'scope':'Source-only variance-allocation diagnostic; no outer scores','admit_outer':admit,'folds':folds,'elapsed_seconds':time.time()-start}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for p in source:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print('admit_outer',admit,'seconds',summary['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
