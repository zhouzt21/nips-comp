"""R80 source-only full-support identifiability and masked-query decomposition."""
import time
from partial8_next_common import *

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R80';root.mkdir(exist_ok=True)
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/partial8_next_common.py',REPO/'research_loop/PARTIAL8_NEXT_SPEC.md',
        REPO/'doc/plan_10.md',REPO/'research_loop/shared_slope_model.py',REPO/'research_loop/shared_slope_data.py',REPO/'research_loop/support_sharing_diagnostic.py'],
        {'estimators':['half','loio','k5'],'budgets':[15,31],'cohort':7501,'source':'R78 fixed selections','bootstrap':1000,'refit_bootstrap':32})
    folders=outer_folders();engine=CachedSemantic();timings=[]
    for f in range(3):
        info=source(f)
        for old in sorted((ROOT/f'R78/source_fold{f}').iterdir()):
            if not (old/'complete.json').exists():continue
            dest=root/f'source_fold{f}'/old.name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():timings.append(read(dest/'complete.json'));continue
            begin=time.perf_counter();bm=Benchmark(folders[old.name],info['semantic_ridge'],engine);assert bm.name in info['train_benchmarks'] and bm.name not in info['heldout_benchmarks']
            audit=read(old/'cohorts.json');mapping=audit['maps']['7501'];selected=set(mapping)
            first=min(selected,key=lambda sid:bm.by[(sid,0)]['subject_key']);oldq=dict(np.load(old/'query_predictions.npz'))
            oldidx={(v['original']['subject_id'],v['original']['seed']):v for v in read(old/'query_index.json') if v['scenario']=='main'}
            oldz={(r['subject_id'],r['seed'],r['budget']):r for r in read(old/'diagnostics.json') if r['scenario']=='main'}
            supportbank={};querybank={};index=[];rows=[];bo=[];bootbank={};bootindex=[];so=qo=boff=0;leak=old_error=0.
            write_json(dest/'cohorts.json',audit)
            for idx in bm.indices:
                sid=idx['subject_id'];seed=idx['seed']
                if sid not in selected:continue
                key=bm.by[(sid,seed)]['subject_key'];q=bm.query(idx);counts=np.asarray(idx['counts'],float);succ=np.asarray(idx['successes'],float)
                oi=oldidx[(sid,seed)];a=oi['start'];b=a+oi['size']
                for j,n in enumerate([15,31]):
                    own=bm.context(sid,seed,n);ids=bm.by[(sid,seed)]['support_ids'][:n];pairs=current_pairs(bm,sid,seed,n,mapping[sid]);peers=[bm.context(s,seed,nn) for s,nn in pairs]
                    full=three(own,peers,q)
                    for m in full:old_error=max(old_error,float(abs(full[m]-oldq[m][a:b,j]).max()))
                    full_delta=brier(full['partial'],counts,succ)-brier(full['own'],counts,succ)
                    gap=float(np.sum((own['x'].mean(0)-q.mean(0))**2))
                    for kind in ['half','loio','k5']:
                        reps=1 if kind=='loio' else 2;peritem=np.zeros((reps,n));rd=np.zeros(reps);fitquery=0.;totalweight=0.;fits=[]
                        for rep,h,fit,hold in masks(kind,ids,bm.name,key,seed):
                            train=subset(own,fit);combined=np.vstack([own['x'][hold],q]);pp=three(train,peers,combined);hn=len(hold)
                            if h==0 and rep==0:
                                changed={**own,'y':own['y'].copy()};changed['y'][hold]=1-changed['y'][hold];again=three(subset(changed,fit),peers,combined)
                                leak=max(leak,max(float(abs(pp[m]-again[m]).max()) for m in pp));assert leak==0
                            y=own['y'][hold].copy();lossdiff=(pp['partial'][:hn]-y)**2-(pp['own'][:hn]-y)**2;peritem[rep,hold]=lossdiff
                            qdelta=brier(pp['partial'][hn:],counts,succ)-brier(pp['own'][hn:],counts,succ);weight=hn/(n*reps);fitquery+=weight*qdelta;totalweight+=weight;fits.append(len(fit))
                            index.append({'selection_fold':f,'benchmark':bm.name,'subject_id':sid,'seed':seed,'budget':n,'estimator':kind,'rep':rep,'mask':h,
                                'fit_indices':fit.tolist(),'hold_indices':hold.tolist(),'support_ids':ids,'peer_prefixes':pairs,'support_start':so,'support_size':hn,
                                'query_start':qo,'query_size':len(q),'original':idx,'fit_own_mean':float((1+train['y'].sum())/(2+len(fit)))})
                            for m in pp:supportbank.setdefault(m,[]).append(pp[m][:hn]);querybank.setdefault(m,[]).append(pp[m][hn:])
                            supportbank.setdefault('truth',[]).append(y);querybank.setdefault('counts',[]).append(counts);querybank.setdefault('successes',[]).append(succ);so+=hn;qo+=len(q)
                        assert abs(totalweight-1)<1e-12
                        v=peritem.mean(0);z=float(v.mean());inter=conditional_interval(v,ids,stable_seed(f'8005|{bm.name}|{key}|{seed}|{n}'))
                        if kind=='half':old_error=max(old_error,abs(z-oldz[(sid,seed,n)]['z']['partial']))
                        rows.append({'selection_fold':f,'benchmark':bm.name,'subject_id':sid,'seed':seed,'budget':n,'estimator':kind,
                            'z':z,'repeat_z':peritem.mean(1).tolist(),'query_delta':full_delta,'masked_fit_query_delta':fitquery,'validation_gap':z-fitquery,
                            'fit_size_gap':fitquery-full_delta,'estimation_error':z-full_delta,'feature_mean_gap':gap,'fit_labels_mean':float(np.mean(fits)),
                            'own_labels':n,'peer_labels':sum(nn for _,nn in pairs),'active_peers':len(peers),'interval':inter,
                            'query_brier':{m:brier(p,counts,succ) for m,p in full.items()},'query_accuracy':{m:accuracy(p,counts,succ) for m,p in full.items()}})
                    if sid==first and seed==0:
                        groups=list(dict.fromkeys(ids));rng=np.random.default_rng(stable_seed(f'8006|{bm.name}|{key}|{n}'))
                        for rep in range(32):
                            sample=rng.integers(len(groups),size=len(groups));used={groups[int(k)] for k in sample}
                            fit=np.array([i for k in sample for i,x in enumerate(ids) if x==groups[int(k)]]);hold=np.array([i for i,x in enumerate(ids) if x not in used])
                            if not len(hold):continue
                            pp=three(subset(own,fit),peers,own['x'][hold]);y=own['y'][hold];size=len(hold)
                            bootindex.append({'selection_fold':f,'benchmark':bm.name,'subject_id':sid,'seed':seed,'budget':n,'rep':rep,'fit_indices':fit.tolist(),'hold_indices':hold.tolist(),
                                'support_ids':ids,'start':boff,'size':size});boff+=size
                            for m,p in pp.items():bootbank.setdefault(m,[]).append(p)
                            bootbank.setdefault('truth',[]).append(y)
                            bo.append({'selection_fold':f,'benchmark':bm.name,'subject_id':sid,'budget':n,'rep':rep,'z':float(np.mean((pp['partial']-y)**2-(pp['own']-y)**2)),
                                'query_delta':full_delta,'fit_distinct_items':len(used),'hold_items':len(hold)})
            assert old_error<1e-11
            for filename,bank in [('support_predictions',supportbank),('masked_query_predictions',querybank),('bootstrap_predictions',bootbank)]:
                if bank:np.savez_compressed(dest/(filename+'.npz'),**{m:np.concatenate(v) for m,v in bank.items()})
            write_json(dest/'mask_index.json',index);write_json(dest/'metrics.json',rows);write_json(dest/'bootstrap_index.json',bootindex);write_json(dest/'bootstrap_metrics.json',bo)
            done={'benchmark':bm.name,'fold':f,'rows':len(rows),'source_trajectories':len(rows)//6,'support_prediction_rows':so,'masked_query_variant_rows':qo,'bootstrap_records':len(bo),
                'max_frozen_R78_error':old_error,'heldout_label_leakage_error':leak,'seconds':time.perf_counter()-begin,'peak_working_set_mib':peak_mb(),'finished':now()}
            write_json(dest/'complete.json',done);timings.append(done);print(json.dumps(done),flush=True)
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'timings':timings,'threads':4,'finished':now()});check_freeze(root/'contract')
if __name__=='__main__':main()
