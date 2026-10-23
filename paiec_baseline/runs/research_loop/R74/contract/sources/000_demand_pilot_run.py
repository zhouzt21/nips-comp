"""Frozen R74 benchmark-grouped teacher/student source pilot. CPU only."""
import argparse
import gc
import time
from collections import Counter
import pandas as pd
from threadpoolctl import threadpool_limits
from historical_positive_common import *
from demand_student_cpu import (fit_ridge, predict_ridge, predict_demands,
    response_features, fit_response, predict_segment)
from mlp.cohort_semantic import CachedSemantic
from mlp.reproduction_episodes import source_episodes, r25_predict
from mlp.run.run_metadata_round import prediction as local_prediction

OUT = ROOT/'R74'
RUBRICS = ['AS', 'MCr', 'QLl', 'KNf']
SEED = 20261023


def rng_for(text):
    return np.random.default_rng(int(hashlib.sha256(f'{SEED}|{text}'.encode()).hexdigest()[:16], 16))


def prepare_annotations():
    items = read(ROOT/'R73/item_mapping.json')
    annotations = [json.loads(line) for line in (ROOT/'R73/final_annotations.jsonl').read_text(encoding='utf8').splitlines()]
    lookup = {(a['item_identity'], a['rubric_id']): a['grade'] for a in annotations}
    assert len(items) == 128 and len(annotations) == 512 and len(lookup) == 512
    grades = np.array([[np.nan if lookup[(r['custom_item_id'], d)] is None else lookup[(r['custom_item_id'], d)]
                        for d in RUBRICS] for r in items], float)
    complete = np.isfinite(grades).all(1)
    assert ((grades[np.isfinite(grades)] >= 0) & (grades[np.isfinite(grades)] <= 5)).all()
    teacher_permuted, student_permuted = grades.copy(), grades.copy()
    for bm in sorted({r['benchmark'] for r in items}):
        group = np.array([i for i, r in enumerate(items) if r['benchmark'] == bm])
        use = group[complete[group]]
        teacher_permuted[use] = grades[rng_for('teacher|' + bm).permutation(use)]
        for j, rubric in enumerate(RUBRICS):
            use = group[np.isfinite(grades[group, j])]
            student_permuted[use, j] = grades[rng_for('student|' + bm + '|' + rubric).permutation(use), j]
    return items, grades, complete, teacher_permuted, student_permuted


def rows_for(episodes, item_index=None, complete=None):
    xs, ys, ws, demands, identities = [], [], [], [], []
    for ep in episodes:
        use = np.arange(len(ep['query']))
        if item_index is not None:
            use = np.array([i for i, key in enumerate(ep['query_item_ids'])
                            if (ep['benchmark'], key) in item_index and complete[item_index[(ep['benchmark'], key)]]], int)
        if not len(use):
            continue
        c, s = ep['counts'][use], ep['successes'][use]
        xs.append(ep['query'][use]); ys.append(s / c); ws.append(c / c.sum())
        for i in use:
            identities.append({'benchmark': ep['benchmark'], 'subject_id': ep['subject_id'],
                               'item_id': ep['query_item_ids'][i], 'count': float(ep['counts'][i]),
                               'successes': float(ep['successes'][i])})
            if item_index is not None:
                demands.append(item_index[(ep['benchmark'], ep['query_item_ids'][i])])
    assert xs, 'No training response rows under the frozen subset'
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), np.asarray(demands, int), identities


def reference_fold(fold, data):
    bank = load_npz(ROOT/f'R59/fold{fold}_predictions.npz')
    indices = read(ROOT/f'R59/fold{fold}_prediction_index.json')
    item_bank = load_npz(ROOT/f'R66/fold{fold}_item_predictions.npz')['p']
    item_idx = {(r['benchmark'], r['subject_id']): r for r in read(ROOT/f'R66/fold{fold}_item_index.json')}
    episodes = {(e['benchmark'], e['subject_id']): e for es in data.values() for e in es}
    result, max_error = {}, 0.
    for ix in indices:
        key = ix['benchmark'], ix['subject_id']; ep = episodes[key]
        assert ix['support_ids'] == ep['support_item_ids']
        assert np.array_equal(ix['support_labels'], ep['labels'])
        qlookup = {k: i for i, k in enumerate(ep['query_item_ids'])}
        vi = np.array([qlookup[q['item_id']] for q in ix['query_groups']], int)
        a, b = ix['start'], ix['start'] + ix['size']
        c, s = bank['counts'][a:b], bank['successes'][a:b]
        assert np.array_equal(np.bincount(vi, weights=c, minlength=len(ep['query'])), ep['counts'])
        assert np.array_equal(np.bincount(vi, weights=s, minlength=len(ep['query'])), ep['successes'])
        ii = item_idx[key]; ia, ib = ii['start'], ii['start'] + ii['size']
        assert np.array_equal(ii['counts'], c) and np.array_equal(ii['successes'], s)
        assert ii['query_ids'] == [q['item_id'] for q in ix['query_groups']]
        r25 = np.column_stack([r25_predict(ep, n, source(fold)['semantic_ridge']) for n in BUDGETS])
        err = float(np.max(np.abs(r25[vi] - bank['R25'][a:b])))
        max_error = max(max_error, err)
        assert err < 1e-7, (key, err)
        result[key] = {'vi': vi, 'counts': c, 'successes': s, 'R25': bank['R25'][a:b],
            'R56': bank['R56'][a:b], 'R66_item': item_bank[ia:ib], 'r25_item': r25,
            'groups': ix['query_groups']}
    return result, max_error


def save_student(folder, fit_names, val_names, items, xitems, grades, permuted):
    folder.mkdir(parents=True, exist_ok=True)
    fit = np.array([r['benchmark'] in fit_names for r in items])
    valid = np.array([r['benchmark'] in val_names for r in items])
    models, controls, rows, audit = [], [], [], []
    for j, rubric in enumerate(RUBRICS):
        fi = np.flatnonzero(fit & np.isfinite(grades[:, j])); vi = np.flatnonzero(valid & np.isfinite(grades[:, j]))
        assert len(fi) >= 2 and len(vi) >= 2
        model = fit_ridge(xitems[fi], grades[fi, j], alpha=.1)
        control = fit_ridge(xitems[fi], permuted[fi, j], alpha=.1)
        models.append(model); controls.append(control)
        np.savez_compressed(folder/f'student_{rubric}.npz', **model)
        np.savez_compressed(folder/f'permuted_student_{rubric}.npz', **control)
        p = predict_ridge(model, xitems[vi], 0., 5.); cp = predict_ridge(control, xitems[vi], 0., 5.)
        constant = float(grades[fi, j].mean())
        for i, pred, ctrl in zip(vi, p, cp):
            rows.append({'item_identity': items[i]['custom_item_id'], 'benchmark': items[i]['benchmark'],
                         'rubric': rubric, 'grade': grades[i, j], 'student': float(pred),
                         'permuted_student': float(ctrl), 'fit_mean': constant})
        audit.append({'rubric': rubric, 'fit_item_ids': [items[i]['custom_item_id'] for i in fi],
                      'heldout_item_ids': [items[i]['custom_item_id'] for i in vi],
                      'fit_grade_mean': constant, 'fit_grade_variance': float(grades[fi,j].var()),
                      'valid_fit_items': len(fi), 'valid_heldout_items': len(vi)})
    write_json(folder/'student_isolation.json', {'fit_benchmarks': fit_names, 'validation_benchmarks': val_names,
                'heads': audit, 'input_fields': '640 existing MiniLM+metadata; no IDs, subject fields or response outcomes',
                'null_policy': 'headwise omitted, never coerced to zero'})
    return models, controls, rows


def score_row(ep, fold, inner, probabilities, c, s):
    return {'fold': fold, 'inner': inner, 'benchmark': ep['benchmark'], 'subject_id': ep['subject_id'], 'seed': 73,
        'brier': {m: [brier(p[:, j], c, s) for j in range(p.shape[1])] for m, p in probabilities.items()},
        'accuracy': {m: [accuracy(p[:, j], c, s) for j in range(p.shape[1])] for m, p in probabilities.items()}}


def audit_summary(rows, weights):
    methods = list(rows[0]['brier']); weights = np.asarray(weights)
    result = {'exposures': len(rows), 'unique_pairs': len({(r['benchmark'], r['subject_id']) for r in rows}),
              'results': {}, 'by_pool': {}, 'by_benchmark': {}, 'comparisons': {}}
    for m in methods:
        curve = np.mean([r['brier'][m] for r in rows], axis=0)
        result['results'][m] = {'brier': curve.tolist(), 'alc': float(curve @ weights),
            'accuracy_at_0_5': np.mean([r['accuracy'][m] for r in rows], axis=0).tolist()}
    for f in range(3):
        group = [r for r in rows if r['fold'] == f]
        result['by_pool'][str(f)] = {m: float(np.mean([r['brier'][m] for r in group], 0) @ weights) for m in methods}
    for bm in sorted({r['benchmark'] for r in rows}):
        group = [r for r in rows if r['benchmark'] == bm]
        result['by_benchmark'][bm] = {'exposures': len(group),
            'scores': {m: float(np.mean([r['brier'][m] for r in group], 0) @ weights) for m in methods}}
    bms = list(result['by_benchmark']); counts = np.array([result['by_benchmark'][b]['exposures'] for b in bms])
    draws = np.random.default_rng(SEED).integers(len(bms), size=(10000, len(bms)))
    for m in methods:
        result['results'][m]['benchmark_equal'] = float(np.mean([result['by_benchmark'][b]['scores'][m] for b in bms]))
        result['comparisons'][m] = {}
        for baseline in methods:
            if baseline == m: continue
            delta = np.array([result['by_benchmark'][b]['scores'][m] - result['by_benchmark'][b]['scores'][baseline] for b in bms])
            weighted = delta * counts
            boot = weighted[draws].sum(1) / counts[draws].sum(1)
            alignment = np.array([b.startswith('alignment_faking/') for b in bms])
            result['comparisons'][m][baseline] = {
                'delta': float(weighted.sum()/counts.sum()), 'benchmark_equal_delta': float(delta.mean()),
                'pool_deltas': [result['by_pool'][str(f)][m]-result['by_pool'][str(f)][baseline] for f in range(3)],
                'cluster_ci95': np.quantile(boot, [.025,.975]).tolist(),
                'without_alignment_delta': float(weighted[~alignment].sum()/counts[~alignment].sum()),
                'alignment_contribution': float(weighted[alignment].sum()/counts.sum()),
                'worst_benchmark': bms[int(delta.argmax())], 'worst_delta': float(delta.max()),
                'improved_benchmarks': int((delta < 0).sum()),
                'regression_ge_005': {b:float(d) for b,d in zip(bms,delta) if d>=.005}}
    return result


def write_prediction_bank(folder, predictions, indices, counts, successes):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'predictions.npz', **{m: np.concatenate(p) for m,p in predictions.items()},
                        counts=np.concatenate(counts), successes=np.concatenate(successes))
    write_json(folder/'index.json', indices)


def main():
    start = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), REPO/'research_loop/demand_student_cpu.py', REPO/'research_loop/demand_pilot_SPEC.md',
        REPO/'research_loop/DUAL_TRACK_SPEC.md', ROOT/'R73/final_annotations.jsonl', ROOT/'R73/item_mapping.json',
        BASE/'mlp/reproduction_episodes.py', BASE/'mlp/run/run_metadata_round.py',
        REPO/'research_loop/literature/reproduction_v4/adele_sources.json',
        Path('D:/nips-comp/paiec-dataset/reproduction_sources/adele/upstream/src/adele/analysis/prediction.py')]
    files += [ROOT/f'reproduction_v4/source_fold{f}.json' for f in range(3)]
    files += [ROOT/f'R59/fold{f}_{kind}' for f in range(3) for kind in ['predictions.npz', 'prediction_index.json']]
    files += [ROOT/f'R66/fold{f}_item_{kind}' for f in range(3) for kind in ['predictions.npz', 'index.json']]
    freeze(OUT/'contract', files, {'threads':2, 'stage':'source_only', 'student_ridge':.1, 'response_ridge':.01,
        'demand_local_divisor':5., 'seed':SEED, 'full_outer_requires_frozen_source_entry':True})
    items, grades, complete, teacher_perm, grade_perm = prepare_annotations()
    item_index = {(r['benchmark'], r['item_id']): i for i,r in enumerate(items)}
    names = sorted({n for f in range(3) for half in source(f)['inner_halves'] for n in half})
    engine = CachedSemantic(); data = {n:source_episodes(n, engine) for n in names}
    item_vectors = {}
    for n, eps in data.items():
        for ep in eps:
            for k,x in zip(ep['query_item_ids'],ep['query']):
                if (n,k) in item_index: item_vectors[n,k]=x
    assert len(item_vectors)==128
    xitems=np.stack([item_vectors[r['benchmark'],r['item_id']] for r in items])
    write_json(OUT/'annotation_scope.json', {'items':128,'complete_items':int(complete.sum()),
        'valid_by_rubric':dict(zip(RUBRICS,np.isfinite(grades).sum(0).tolist())),
        'teacher_permutation_changed_fraction':float(np.mean(np.any(teacher_perm[complete]!=grades[complete],axis=1))),
        'student_permutation_changed_fraction':float(np.mean(grade_perm[np.isfinite(grades)]!=grades[np.isfinite(grades)])),
        'benchmark_complete_counts':dict(Counter(r['benchmark'] for r,c in zip(items,complete) if c)),
        'not_annotated_support': 'Pure teacher B1 and teacher-local six-budget effects are not evaluated.'})
    grade_rows, teacher_rows, deploy_rows, audits, time_rows = [],[],[],[],[]
    for fold in range(3):
        t0=time.perf_counter(); info=source(fold)
        pool={n:data[n] for n in sum(info['inner_halves'],[])}
        reference,replay_error=reference_fold(fold,pool)
        tpred,dpred,tindices,dindices,tc,ts,dc,ds={},{},[],[],[],[],[],[]
        toffset=doffset=0
        for inner in range(2):
            fit_names=info['inner_halves'][inner]; val_names=info['inner_halves'][1-inner]
            assert set(fit_names).isdisjoint(val_names) and set(fit_names+val_names).isdisjoint(info['heldout_benchmarks'])
            folder=OUT/f'pool{fold}_inner{inner}';folder.mkdir(parents=True,exist_ok=True)
            models,controls,grows=save_student(folder,fit_names,val_names,items,xitems,grades,grade_perm)
            grade_rows += [dict(r,fold=fold,inner=inner) for r in grows]
            fit_eps=[e for n in fit_names for e in data[n]];val_eps=[e for n in val_names for e in data[n]]
            x,y,w,gi,fitids=rows_for(fit_eps,item_index,complete)
            teacher_models={}
            for m,features in [('base',response_features(x)),('teacher',response_features(x,grades[gi])),
                ('teacher_permuted',response_features(x,teacher_perm[gi])),('teacher_only',response_features(x,grades[gi],True)),
                ('student_subset',response_features(x,predict_demands(models,x)))]:
                teacher_models[m]=fit_response(features,y,w)
                np.savez_compressed(folder/f'teacher_prior_{m}.npz',**teacher_models[m])
            teacher_constant=float(np.average(y,weights=w))
            write_json(folder/'teacher_response_training_identity.json',fitids)
            x,y,w,_,fullids=rows_for(fit_eps)
            full_constant=float(np.average(y,weights=w)); fullmodels={}
            for m,features in [('base',response_features(x)),('student',response_features(x,predict_demands(models,x))),
                ('permuted_student',response_features(x,predict_demands(controls,x))),
                ('demand_only',response_features(x,predict_demands(models,x),True))]:
                fullmodels[m]=fit_response(features,y,w)
                np.savez_compressed(folder/f'full_prior_{m}.npz',**fullmodels[m])
            write_json(folder/'full_response_training_identity.json',fullids)
            audits.append({'fold':fold,'inner':inner,'fit_benchmarks':fit_names,'heldout_benchmarks':val_names,
                'excluded_outer':info['heldout_benchmarks'],'teacher_response_rows':len(fitids),'full_response_rows':len(fullids),
                'teacher_fit_pairs':len({(r['benchmark'],r['subject_id']) for r in fitids}),
                'r25_replay_max_probability_error':replay_error})
            for ep in val_eps:
                ref=reference[ep['benchmark'],ep['subject_id']];vi=ref['vi'];c,s=ref['counts'],ref['successes']
                sx,qx=ep['support'],ep['query'];sd,qd=predict_demands(models,sx),predict_demands(models,qx)
                sp,qp=predict_demands(controls,sx),predict_demands(controls,qx)
                r25=ref['r25_item']; probs={'R25':ref['R25'],
                    'R56_segment':np.c_[ref['R56'][:,:2],ref['R25'][:,2:]],
                    'R66_item_segment':np.c_[ref['R66_item'][:,:2],ref['R25'][:,2:]],
                    'constant_0.5':np.full_like(ref['R25'],.5)}
                for name,alpha,mean in [('smoothed_mean_2',2.,.5),('training_selected_smoothed',info['selected_alpha'],.5),
                                        ('source_constant_segment',2.,full_constant)]:
                    p=np.column_stack([np.full(len(qx),(alpha*mean+ep['labels'][:n].sum())/(alpha+n)) for n in BUDGETS])
                    if name=='source_constant_segment':p[:,2:]=r25[:,2:]
                    probs[name]=p[vi]
                for m in fullmodels:
                    dsa,dqa=(sp,qp) if m=='permuted_student' else (sd,qd)
                    probs['prior_'+m]=predict_segment(fullmodels[m],sx,qx,dsa,dqa,ep['labels'],r25,
                        'base' if m=='base' else ('demand_only' if m=='demand_only' else 'augmented'))[vi]
                scale=np.r_[np.full(384,np.sqrt(info['semantic_ridge'])),np.full(256,4.)]
                for m,dsa,dqa,only in [('local_student',sd,qd,False),('local_permuted_student',sp,qp,False),
                                      ('local_demand_only',sd,qd,True)]:
                    a=np.c_[np.zeros_like(sx) if only else sx/scale,dsa/5.]
                    q=np.c_[np.zeros_like(qx) if only else qx/scale,dqa/5.]
                    p=r25.copy()
                    for j,n in enumerate(BUDGETS[2:],2):p[:,j]=local_prediction(a[:n],q,ep['labels'][:n],1.)
                    probs[m]=p[vi]
                for m,p in probs.items():
                    assert p.shape==(len(c),6) and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
                    dpred.setdefault(m,[]).append(p)
                deploy_rows.append(score_row(ep,fold,inner,probs,c,s));dc.append(c);ds.append(s)
                dindices.append({'fold':fold,'inner':inner,'benchmark':ep['benchmark'],'subject_id':ep['subject_id'],
                    'seed':73,'start':doffset,'size':len(c),'support_ids':ep['support_item_ids'],
                    'support_labels':ep['labels'].tolist(),'query_groups':ref['groups'],
                    'query_item_ids':ep['query_item_ids'],'variant_item_indices':vi.tolist()});doffset+=len(c)
                use=np.array([i for i,k in enumerate(ep['query_item_ids']) if (ep['benchmark'],k) in item_index
                              and complete[item_index[ep['benchmark'],k]]],int)
                if not len(use):continue
                gidx=np.array([item_index[ep['benchmark'],ep['query_item_ids'][i]] for i in use],int)
                mask=np.isin(vi,use); subset_lookup={v:i for i,v in enumerate(use)}
                svi=np.array([subset_lookup[v] for v in vi[mask]],int)
                features={'base':response_features(qx[use]),'teacher':response_features(qx[use],grades[gidx]),
                    'teacher_permuted':response_features(qx[use],teacher_perm[gidx]),
                    'teacher_only':response_features(qx[use],grades[gidx],True),
                    'student_subset':response_features(qx[use],qd[use])}
                tp={m:predict_ridge(teacher_models[m],feat)[svi,None] for m,feat in features.items()}
                tp.update({'constant_0.5':np.full((mask.sum(),1),.5),'source_constant':np.full((mask.sum(),1),teacher_constant),
                           'R56_B0':ref['R56'][mask,:1],'R66_item_B0':ref['R66_item'][mask,:1]})
                cc,ss=c[mask],s[mask]
                teacher_rows.append(score_row(ep,fold,inner,tp,cc,ss));tc.append(cc);ts.append(ss)
                for m,p in tp.items():tpred.setdefault(m,[]).append(p)
                tindices.append({'fold':fold,'inner':inner,'benchmark':ep['benchmark'],'subject_id':ep['subject_id'],
                    'seed':73,'start':toffset,'size':int(mask.sum()),
                    'query_groups':[g for g,keep in zip(ref['groups'],mask) if keep],
                    'annotation_item_ids':[items[gidx[svi[i]]]['custom_item_id'] for i in range(len(svi))]})
                toffset+=int(mask.sum())
            del x,y,w;gc.collect()
        write_prediction_bank(OUT/f'pool{fold}/teacher',tpred,tindices,tc,ts)
        write_prediction_bank(OUT/f'pool{fold}/deployable',dpred,dindices,dc,ds)
        time_rows.append({'pool':fold,'seconds':time.perf_counter()-t0})
        print(json.dumps({'pool_complete':fold,'seconds':time_rows[-1]['seconds'],'teacher_exposures':len(tindices),
                          'deploy_exposures':len(dindices),'r25_replay_max_error':replay_error}),flush=True)
    write_json(OUT/'teacher_paired_scores.json',teacher_rows);write_json(OUT/'deployable_paired_scores.json',deploy_rows)
    write_json(OUT/'grade_predictions.json',grade_rows);write_json(OUT/'isolation_audit.json',audits)
    teacher_summary=audit_summary(teacher_rows,[1.]);deploy_summary=audit_summary(deploy_rows,ALC_WEIGHTS)
    grade_summary={}
    for rubric in RUBRICS+['ALL']:
        group=[r for r in grade_rows if rubric=='ALL' or r['rubric']==rubric]
        grade_summary[rubric]={'exposures':len(group),'unique_items':len({r['item_identity'] for r in group}),'models':{}}
        truth=np.array([r['grade'] for r in group])
        for m in ['student','permuted_student','fit_mean']:
            p=np.array([r[m] for r in group]);d=p-truth
            grade_summary[rubric]['models'][m]={'mae':float(np.mean(abs(d))),'rmse':float(np.sqrt(np.mean(d*d))),
                'rounded_exact_agreement':float(np.mean(np.rint(p)==truth)),
                'rounded_gt_one_error':float(np.mean(abs(np.rint(p)-truth)>1))}
    teacher_tests={}
    for baseline in ['base','teacher_permuted']:
        c=teacher_summary['comparisons']['teacher'][baseline]
        teacher_tests[baseline]={'two_pools':sum(d<0 for d in c['pool_deltas'])>=2,'pair':c['delta']<0,
            'benchmark':c['benchmark_equal_delta']<0,'without_alignment':c['without_alignment_delta']<0}
    teacher_pass=all(all(v.values()) for v in teacher_tests.values())
    entries={}
    for m,base,perm in [('prior_student','prior_base','prior_permuted_student'),
                        ('local_student','R25','local_permuted_student')]:
        c=deploy_summary['comparisons'][m]['R25']
        tests={'two_pools_R25':sum(d<0 for d in c['pool_deltas'])>=2,'pair_R25':c['delta']<0,
               'benchmark_R25':c['benchmark_equal_delta']<0,'without_alignment_R25':c['without_alignment_delta']<0}
        for b in [base,perm]:
            d=deploy_summary['comparisons'][m][b]
            tests[b+'_pair']=d['delta']<0;tests[b+'_benchmark']=d['benchmark_equal_delta']<0
        entries[m]={'tests':tests,'pass':all(tests.values())}
    eligible=sorted([m for m,v in entries.items() if v['pass']],key=lambda m:(deploy_summary['results'][m]['alc'],m))
    decision={'teacher_tests':teacher_tests,'teacher_entry':teacher_pass,'deployable_entries':entries,
              'full_outer_authorized_by_source_contract':teacher_pass and bool(eligible),
              'candidate':eligible[0] if teacher_pass and eligible else None,
              'action':'notify_root_before_outer' if teacher_pass and eligible else 'stop_at_pilot_keep_R25',
              'no_parameter_retuning':True}
    write_json(OUT/'teacher_summary.json',teacher_summary);write_json(OUT/'deployable_summary.json',deploy_summary)
    write_json(OUT/'grade_summary.json',grade_summary);write_json(OUT/'decision.json',decision)
    elapsed=time.perf_counter()-start
    write_json(OUT/'run_summary.json',{'status':'complete_source_pilot','threads':2,'seconds':elapsed,'pool_timings':time_rows,
        'teacher_exposures':len(teacher_rows),'deployable_exposures':len(deploy_rows),'decision':decision,
        'hardware':'CPU only; host reported by freeze contract','full_outer_evaluated':False})
    pd.DataFrame(grade_rows).to_csv(OUT/'grade_predictions.csv',index=False)
    check_freeze(OUT/'contract')
    print(json.dumps({'seconds':elapsed,'decision':decision,'student_grade':grade_summary['ALL'],
                      'teacher_B0':teacher_summary['results'],'deployable':deploy_summary['results']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
