"""Independent arithmetic/provenance audit of actual R59/R61 outputs."""
import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO/'paiec_baseline'
sys.path.insert(0,str(BASE))
from mlp.training_mechanism_data import mlp_validation, load_npz, cache_path
from mlp.reproduction_episodes import cache_fingerprint, CACHE_ROOT
from mlp.cohort_semantic import CachedSemantic
from mlp.model import forward_batch
from mlp.training_mechanism_training import PARAMS

R59 = BASE/'runs/research_loop/R59'
R60 = BASE/'runs/research_loop/R60'
R61 = BASE/'runs/research_loop/R61'
WEIGHTS = np.array([.1,.2,.2,.2,.2,.1])
BUDGETS = [0,1,3,7,15,31]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    # Compile only the exact NumPy scoring kernel; no TensorFlow module import.
    original = Path('D:/nips-comp/paiec-dataset/low_budget_prior_sources/uq_benchmark_2019/metrics_lib.py')
    tree = ast.parse(original.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='brier_scores')
    namespace = {'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(original),'exec'),namespace)
    author_score = namespace['brier_scores']
    max_arithmetic = max_decomp = max_author = max_gate_error = 0.
    pairs = groups = 0
    frame = pd.read_csv(R59/'pair_budget_decomposition.csv',dtype={'subject_id':str})
    gate_decisions = pd.DataFrame(read(R60/'decisions.json'))
    selector = gate_decisions.set_index(['fold','benchmark','subject_id'])
    detail = frame.set_index(['fold','benchmark','subject_id','method','budget'])
    gate_analysis = []
    cache_files = set()
    engine = CachedSemantic()
    prior_cache = {str(Path(p).resolve()):h for p,h in read(BASE/'runs/research_loop/training_mechanism_v5/cache_inventory.json').items()}
    for fold in range(3):
        source = read(BASE/f'runs/research_loop/reproduction_v4/source_fold{fold}.json')
        assert set(source['inner_halves'][0]).isdisjoint(source['inner_halves'][1])
        bank = dict(np.load(R59/f'fold{fold}_predictions.npz',allow_pickle=False))
        p_gate = np.load(R61/f'fold{fold}_candidate_predictions.npz',allow_pickle=False)['probabilities']
        records = {(r['benchmark'],r['subject_id']):r for r in read(R61/f'fold{fold}_paired_scores.json')}
        # Independently recompute the sole gate input from original support features and frozen weights.
        support_errors = {}
        epoch = read(BASE/f'runs/research_loop/training_mechanism_v5/R56/fold{fold}_selection.json')['epoch']
        for inner in range(2):
            params = load_npz(BASE/f'runs/research_loop/training_mechanism_v5/R56/fold{fold}_mlp_inner{inner}/epoch{epoch}.npz')
            for benchmark in source['inner_halves'][1-inner]:
                mlp_cache = cache_path('mlp_validation',benchmark)
                semantic_key = hashlib.sha256(f'{benchmark}|73|25|{cache_fingerprint(engine)}'.encode()).hexdigest()
                for path in [mlp_cache,CACHE_ROOT/'source'/(semantic_key+'.npz')]:
                    resolved = str(path.resolve())
                    assert sha(path)==prior_cache[resolved],f'Historical input cache changed: {path}'
                    cache_files.add(resolved)
                for ep in mlp_validation(benchmark,engine):
                    ps = float(forward_batch(ep['support'][:1],tuple(params[k] for k in PARAMS))[0])
                    support_errors[(benchmark,ep['subject_id'])] = (ps-ep['labels'][0])**2
        for index in read(R59/f'fold{fold}_prediction_index.json'):
            pairs += 1
            a,b = index['start'],index['start']+index['size']; c,s = bank['counts'][a:b],bank['successes'][a:b]
            groups += len(c)
            assert np.all(c>0) and np.array_equal(c,c.astype(int))
            assert np.all((s>=0)&(s<=c)) and np.array_equal(s,s.astype(int))
            key = (fold,index['benchmark'],index['subject_id'])
            err = support_errors[key[1:]]
            max_gate_error = max(max_gate_error, abs(err-selector.loc[key,'first_support_sqerr']))
            assert bool(selector.loc[key,'use_R56_at_B1']) == (err < .25)
            expected = bank['R25'][a:b].copy()
            if err < .25:
                expected[:,1] = bank['R56'][a:b,1]
            assert np.array_equal(expected,p_gate[a:b])
            rec = records[key[1:]]
            methods = {m:bank[m][a:b] for m in rec['brier'] if m!='R56_support1_gate'}
            methods['R56_support1_gate'] = p_gate[a:b]
            for method,probs in methods.items():
                assert np.all(np.isfinite(probs)) and np.all((probs>=0)&(probs<=1))
                for j,n in enumerate(BUDGETS):
                    p = probs[:,j]
                    # Explicitly sum success and failure squared errors, avoiding evaluator's expansion.
                    score = float((s*(1-p)**2+(c-s)*p**2).sum()/c.sum())
                    max_arithmetic = max(max_arithmetic, abs(score-rec['brier'][method][j]))
                    if method in ['R25','R54','R56']:
                        pm = np.average(p,weights=c); ym = s.sum()/c.sum()
                        decomp = ym*(1-ym)+(pm-ym)**2+np.average((p-pm)**2,weights=c)-2*(s@p/c.sum()-pm*ym)
                        saved = detail.loc[key+(method,n)]
                        max_decomp = max(max_decomp,abs(score-decomp),abs(score-saved.brier))
                        # Author code omits +1 and class normalization: binary score=(raw+1)/2.
                        pp = np.stack([1-p,p],axis=-1)
                        raw1 = author_score(np.ones(len(p),int),probs=pp)
                        raw0 = author_score(np.zeros(len(p),int),probs=pp)
                        normalized = (s*(raw1+1)/2+(c-s)*(raw0+1)/2).sum()/c.sum()
                        max_author = max(max_author,abs(score-normalized))
            d1 = rec['brier']['R56'][1]-rec['brier']['R25'][1]
            gate_analysis.append({'fold':fold,'benchmark':key[1],'subject_id':key[2],
                'selected':bool(err<.25),'potential_B1_delta_R25':d1,
                'actual_B1_delta_R25':rec['brier']['R56_support1_gate'][1]-rec['brier']['R25'][1],
                'false_selection':bool(err<.25 and d1>0),
                'missed_gain':bool(err>=.25 and d1<0)})
    assert max(max_arithmetic,max_decomp,max_author,max_gate_error)<1e-10
    # Verify every captured dependency/weight and score-stage input, including the original prompt.
    for record,key in [(read(R59/'provenance.json'),'sha256'),(read(R60/'contract.json'),'source_sha256')]:
        for path,expected in record[key].items():
            p=Path(path); p=p if p.is_absolute() else BASE/p
            assert sha(p)==expected,(p,expected)
    outputs = list(R59.glob('*.*'))+list(R60.glob('*.*'))+list(R61.glob('*.*'))
    outputs = [p for p in outputs if p.name not in ['validation.json','artifact_inventory.json']]
    df = pd.DataFrame(gate_analysis)
    df.to_csv(R61/'gate_error_diagnostics.csv',index=False)
    grouped=[]
    for fold,g in df.groupby('fold'):
        selected=g[g.selected]; rejected=g[~g.selected]
        grouped.append({'fold':int(fold),'selected':len(selected),'false_selections':int(selected.false_selection.sum()),
            'false_selection_fraction':float(selected.false_selection.mean()),
            'missed_gains':int(rejected.missed_gain.sum()),'rejected':len(rejected),
            'selected_mean_B1_delta':float(selected.potential_B1_delta_R25.mean()),
            'rejected_mean_potential_B1_delta':float(rejected.potential_B1_delta_R25.mean())})
    audit={'status':'passed','pairs_with_pool_repetition':pairs,'response_groups_with_pool_repetition':groups,
        'used_caches_verified_against_V5_inventory':len(cache_files),
        'direct_success_failure_brier_max_error':max_arithmetic,'decomposition_max_error':max_decomp,
        'author_metric_normalization_max_error':max_author,'gate_support_input_max_error':max_gate_error,
        'author_source_sha256':sha(original),'author_function':'brier_scores',
        'author_transform':'binary_brier=(author_raw+1)/2; not raw score or binned decomposition',
        'candidate_non_B1_predictions':'bitwise identical to R25',
        'candidate_gate':'all flags recomputed from frozen weights and first support only',
        'query_label_use':'scoring only; no fitted gate or threshold search',
        'gate_diagnostics':grouped}
    (R61/'validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (R59/'used_input_cache_inventory.json').write_text(json.dumps({p:sha(p) for p in sorted(cache_files)},indent=2),encoding='utf-8')
    (R61/'artifact_inventory.json').write_text(json.dumps({str(p):sha(p) for p in outputs},indent=2),encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':
    main()
