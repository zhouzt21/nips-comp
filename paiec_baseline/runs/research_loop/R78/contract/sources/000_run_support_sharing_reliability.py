"""R78 legal-source diagnosis, no gate fitting or outer outcome selection."""
import time
from shared_reliability_common import *
from support_sharing_diagnostic import predictions,diagnose,mask_predictions

SCENARIOS=['main','alternative','missing','uneven','permuted']

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R78';root.mkdir(exist_ok=True)
    assert (ROOT/'R77/full/execution.json').exists(),'R77 execution must precede R78'
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/support_sharing_diagnostic.py',REPO/'research_loop/shared_reliability_common.py',
        REPO/'research_loop/shared_slope_model.py',REPO/'research_loop/shared_slope_data.py',REPO/'research_loop/SHARED_RELIABILITY_SPEC.md',
        REPO/'research_loop/SHARED_RELIABILITY_OOF_ADDENDUM.md',REPO/'doc/plan_9.md'],
        {'budgets':[15,31],'mask_seeds':[7801,7802],'scenarios':SCENARIOS,'source_selection':'frozen R75 subjects','gate_not_fitted':True})
    engine=CachedSemantic();folders=outer_folders();total=0;timings=[];audits=[]
    for f in range(3):
        info=source(f);assert set(info['train_benchmarks']).isdisjoint(info['heldout_benchmarks'])
        for oldsource in sorted((ROOT/f'R75/source_fold{f}').iterdir()):
            if not (oldsource/'complete.json').exists():continue
            name=oldsource.name;dest=root/f'source_fold{f}'/name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():timings.append(read(dest/'complete.json'));continue
            begin=time.perf_counter();bm=Benchmark(folders[name],info['semantic_ridge'],engine)
            selection=read(oldsource/'selection.json');selected=set(selection['subjects']);records=[r for r in bm.records if r['subject_id'] in selected]
            maps={c:cohort_map(records,c) for c in [7501,7502]};assert bm.name in info['train_benchmarks']
            write_json(dest/'cohorts.json',{'source_fold':f,'maps':maps,'visible_configurations':[v for v in visible_subjects(bm) if v['subject_id'] in selected],
                'heldout_outer_benchmarks':info['heldout_benchmarks'],'source_benchmark':bm.name})
            rows=[];qindex=[];mindex=[];qbank={};mbank={};qoff=moff=0;leakmax=0.
            for idx in bm.indices:
                sid=idx['subject_id'];seed=idx['seed']
                if sid not in selected:continue
                q=bm.query(idx);counts=np.asarray(idx['counts'],float);success=np.asarray(idx['successes'],float)
                for scenario in SCENARIOS:
                    querypred={k:[] for k in ['own','pooled','partial','mean','meanbase','combined']}
                    for n in [15,31]:
                        own=bm.context(sid,seed,n);peers=maps[7502 if scenario=='alternative' else 7501][sid]
                        pairs=current_pairs(bm,sid,seed,n,peers,scenario if scenario in ['missing','uneven'] else 'full')
                        ctx=[bm.context(s,seed,nn) for s,nn in pairs]
                        if scenario=='permuted':
                            ctx=[{**c,'y':c['y'][np.random.default_rng(stable_seed(f'7803|{bm.name}|{sid}|{seed}|{n}|{k}')).permutation(len(c['y']))]} for k,c in enumerate(ctx)]
                        z,rz,masks,gap=diagnose(own,ctx,bm.name,bm.by[(sid,seed)]['subject_key'],seed)
                        # Adversarial leakage check: changing only heldout target y cannot change the prediction.
                        first=masks[0];fit=np.asarray(first['fit_indices']);hold=np.asarray(first['hold_indices']);changed={**own,'y':own['y'].copy()};changed['y'][hold]=1-changed['y'][hold]
                        again=mask_predictions(changed,ctx,fit,hold)
                        err=max(float(abs(again[k]-first['probabilities'][k]).max()) for k in again);leakmax=max(leakmax,err);assert err==0
                        pp=predictions(own,ctx,q)
                        br={k:brier(v,counts,success) for k,v in pp.items()};ac={k:accuracy(v,counts,success) for k,v in pp.items()}
                        for k,v in pp.items():querypred[k].append(v)
                        rows.append({'selection_fold':f,'fold':f,'benchmark':bm.name,'subject_id':sid,'seed':seed,'identity':idx['identity'],
                             'scenario':scenario,'budget':n,'z':z,'repeat_z':rz,'mean_gap':gap,'query_brier':br,'query_accuracy':ac,
                             'query_delta_partial':br['partial']-br['own'],'query_delta_pooled':br['pooled']-br['own'],
                             'own_labels':n,'peer_labels':sum(nn for _,nn in pairs),'active_peers':len(pairs),
                             'support_holdout_count':2*n,'query_variant_count':len(q)})
                        for mask in masks:
                            size=len(mask['truth']);probs=mask.pop('probabilities');truth=mask.pop('truth')
                            mindex.append({'selection_fold':f,'benchmark':bm.name,'subject_id':sid,'seed':seed,'identity':idx['identity'],
                                'scenario':scenario,'budget':n,'start':moff,'size':size,'support_ids':bm.by[(sid,seed)]['support_ids'][:n],
                                'peer_prefixes':[{'subject_id':s,'n':nn} for s,nn in pairs],**mask});moff+=size
                            for k,v in probs.items():mbank.setdefault(k,[]).append(v)
                            mbank.setdefault('truth',[]).append(truth)
                    qindex.append({'original':idx,'scenario':scenario,'start':qoff,'size':len(q)});qoff+=len(q)
                    for k,v in querypred.items():qbank.setdefault(k,[]).append(np.stack(v,axis=1))
                    qbank.setdefault('counts',[]).append(counts);qbank.setdefault('successes',[]).append(success)
            np.savez_compressed(dest/'query_predictions.npz',**{k:np.concatenate(v) for k,v in qbank.items()})
            np.savez_compressed(dest/'support_predictions.npz',**{k:np.concatenate(v) for k,v in mbank.items()})
            write_json(dest/'query_index.json',qindex);write_json(dest/'support_index.json',mindex);write_json(dest/'diagnostics.json',rows)
            done={'benchmark':bm.name,'selection_fold':f,'diagnostic_records':len(rows),'source_trajectories':len(rows)//10,'support_prediction_rows':moff,
                  'query_variant_rows':qoff,'seconds':time.perf_counter()-begin,'leakage_prediction_max_error':leakmax,'peak_working_set_mib':peak_mb(),'finished':now()}
            write_json(dest/'complete.json',done);total+=len(rows);timings.append(done);print(json.dumps(done),flush=True)
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'finished':now(),'timings':timings,'threads':4})
    check_freeze(root/'contract');check_freeze(ROOT/'R62/isolation')
if __name__=='__main__':main()
