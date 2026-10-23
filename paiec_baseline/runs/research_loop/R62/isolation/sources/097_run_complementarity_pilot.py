"""R45 same-window semantic/lexical complementarity, fixed arithmetic averaging."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.metadata_episodes import episodes
from mlp.cohort_semantic import CachedSemantic
from mlp.run.run_metadata_round import prediction
from mlp.evaluate import BUDGETS,ALC_WEIGHTS,_stable_seed
from mlp.model import encode

NAMES=['average','semantic','lexical','permuted_lexical','smoothed_mean_2']


def identity(p,q,y):
    a=float(np.mean(((p+q)*.5-y)**2));b=float(.5*np.mean((p-y)**2)+.5*np.mean((q-y)**2)-.25*np.mean((p-q)**2))
    if abs(a-b)>1e-12:raise AssertionError((a,b))
    return a,float(.25*np.mean((p-q)**2))


class LexicalWindows:
    def __init__(self,engine):
        # Lookup by stored semantic float32 vector. Multiple original rows can have
        # numerically equal vectors; their retained windows must be identical.
        self.windows={};self.features={}
        for vector,window in engine.db.execute('SELECT vector,window FROM features'):
            if vector in self.windows and self.windows[vector]!=window:
                self.windows[vector]=None
            else:self.windows[vector]=window
    def vector(self,v):
        key=np.asarray(v,np.float32).tobytes()
        if key not in self.features:
            text=self.windows.get(key);assert text is not None,'Cannot uniquely recover retained window from cached vector'
            x=(encode([{},{'item_content':text}])-encode([{},{}],include_item=False)).astype(float)
            self.features[key]=x/max(np.linalg.norm(x),1e-12)
        return self.features[key]


def main():
    start=time.time();root=Path('runs/research_loop/R45');root.mkdir(parents=True,exist_ok=True);engine=CachedSemantic()
    source=['mlp/run/run_complementarity_pilot.py','mlp/metadata_episodes.py','mlp/run/run_metadata_round.py','../research_loop/R45_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in source:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    lexical=LexicalWindows(engine);folds=[]
    for fold in range(3):
        target=root/f'fold{fold}_source.json'
        if target.exists():folds.append(json.loads(target.read_text()));continue
        sel=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text());by={};total={k:np.zeros(6) for k in NAMES};dividend=np.zeros(6);count=0
        for directory in sel['train_benchmarks']:
            rows=episodes(directory,73,25,engine)
            if not rows:continue
            sums={k:np.zeros(6) for k in NAMES};div=np.zeros(6);rng=np.random.default_rng(_stable_seed(4501,directory,'lexical_permutation'))
            for row in rows:
                sems=np.c_[row['support']/np.sqrt(sel['semantic_ridge']),row['support_meta']/4];semq=np.c_[row['query']/np.sqrt(sel['semantic_ridge']),row['query_meta']/4]
                lexs=np.c_[np.stack([lexical.vector(x) for x in row['support']]),row['support_meta']/4];lexq=np.c_[np.stack([lexical.vector(x) for x in row['query']]),row['query_meta']/4]
                for j,n in enumerate(BUDGETS):
                    y=row['a'][:n];p=prediction(sems[:n],semq,y,1.);q=prediction(lexs[:n],lexq,y,1.)
                    loss,gain=identity(p,q,row['y']);div[j]+=gain;sums['average'][j]+=loss
                    sums['semantic'][j]+=np.mean((p-row['y'])**2);sums['lexical'][j]+=np.mean((q-row['y'])**2)
                    sums['permuted_lexical'][j]+=np.mean([np.mean(((p+q[rng.permutation(len(q))])*.5-row['y'])**2) for _ in range(4)])
                    sums['smoothed_mean_2'][j]+=np.mean(((1+y.sum())/(2+n)-row['y'])**2)
            n=len(rows);count+=n
            by[directory]={'pairs':n,'brier':{k:(v/n).tolist() for k,v in sums.items()},'alc':{k:float(v/n@ALC_WEIGHTS) for k,v in sums.items()},'disagreement_dividend':(div/n).tolist()}
            for k,v in sums.items():total[k]+=v
            dividend+=div
            lexical.features.clear()
        alc={k:float(v/count@ALC_WEIGHTS) for k,v in total.items()};old=json.loads(Path(f'runs/research_loop/R42/fold{fold}_source.json').read_text())
        assert abs(alc['semantic']-old['pair_equal_alc']['old_semantic'])<1e-7
        assert abs(alc['lexical']-old['pair_equal_alc']['old_lexical'])<1e-7
        result={'fold':fold,'scope':'source only','source_benchmarks':sel['train_benchmarks'],'pairs':count,'pair_equal_alc':alc,
                'pair_equal_brier':{k:(v/count).tolist() for k,v in total.items()},'disagreement_dividend':(dividend/count).tolist(),
                'benchmark_equal_alc':{k:float(np.mean([v['alc'][k] for v in by.values()])) for k in NAMES},'by_benchmark':by}
        target.write_text(json.dumps(result,indent=2),encoding='utf-8');folds.append(result);print(fold,alc,flush=True)
    admit=all(r['pair_equal_alc']['average']<min(r['pair_equal_alc']['semantic'],r['pair_equal_alc']['permuted_lexical']) and r['benchmark_equal_alc']['average']<r['benchmark_equal_alc']['semantic'] for r in folds)
    result={'scope':'Fixed equal average, source-only; no OOD estimate','admit_outer':admit,'folds':folds,'elapsed_seconds':time.time()-start}
    (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for p in source:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print('admit_outer',admit,'seconds',result['elapsed_seconds'],flush=True)


if __name__=='__main__':main()
