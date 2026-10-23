"""R77: fixed second cohort first, then matched frozen composition stressors."""
import argparse,time
from shared_reliability_common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['alternative','full'],required=True);args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R77'/args.stage;root.mkdir(parents=True,exist_ok=True)
    check_freeze(ROOT/'R76/full/contract');check_freeze(ROOT/'R75/contract')
    if args.stage=='full':assert (ROOT/'R77/alternative/execution.json').exists(),'Complete second cohort first'
    shrink=source_shrinkage()
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/shared_reliability_common.py',REPO/'research_loop/shared_slope_model.py',
        REPO/'research_loop/shared_slope_data.py',REPO/'research_loop/SHARED_RELIABILITY_SPEC.md',REPO/'doc/plan_9.md',ROOT/'R77/source_shrinkage.json'],
        {'stage':args.stage,'cohorts':COHORTS,'budgets':list(map(int,BUDGETS)),'pairs':5242,'scope':'conditional_simulation'})
    engine=CachedSemantic();allrows=[];timings=[]
    for f in range(3):
        for old in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
            dest=root/'outer'/f'fold{f}'/old.name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():allrows+=read(dest/'paired_scores.json');timings.append(read(dest/'complete.json'));continue
            assert not list(dest.glob('predictions_*.npz')),'Preserve unfinished evidence before restart'
            begin=time.perf_counter();bm=Benchmark(old,source(f)['semantic_ridge'],engine)
            maps={c:cohort_map(bm.records,c) for c in ([7502] if args.stage=='alternative' else COHORTS)}
            write_json(dest/'cohorts.json',{'maps':maps,'visible_configuration_keys':visible_subjects(bm),'audit_subject_ids_not_features':True})
            refroot=ROOT/f'R76/full/outer/fold{f}'/old.name;rows=[];costs=[];maxreplay=0.
            for ip in sorted(old.glob('index_*.json')):
                suffix=ip.stem.split('_')[-1];indices=read(ip);base=dict(np.load(refroot/f'predictions_{suffix}.npz'));bank={}
                alt=None
                if args.stage=='full':alt=dict(np.load(ROOT/f'R77/alternative/outer/fold{f}'/old.name/f'predictions_{suffix}.npz'))
                for idx in indices:
                    sid=idx['subject_id'];seed=idx['seed'];a=idx['start'];b=a+idx['size'];pown=base['R25'][a:b];query=bm.query(idx)
                    preds={'R25':pown.copy(),'pool_7502':base['R76_alt3'][a:b].copy(),'partial_7502':pown.copy()}
                    if args.stage=='full':
                        preds['partial_7502']=alt['partial_7502'][a:b].copy()
                        preds.update({'pool_7501':base['R76_peer3'][a:b].copy(),'partial_7501':base['R76_partial_selected'][a:b].copy()})
                        for c in COHORTS[2:]:preds[f'pool_{c}']=pown.copy();preds[f'partial_{c}']=pown.copy()
                        for scenario in ['peer0','peer1','peer2','missing','uneven','ambiguous']:
                            preds[f'pool_{scenario}']=base[f'R76_{scenario}'][a:b].copy();preds[f'partial_{scenario}']=pown.copy()
                        for name in ['smoothed_mean_2','training_selected_smoothed','R72_pooled3','R72_diff3']:preds[name]=base[name][a:b].copy()
                    for j,n in enumerate(BUDGETS):
                        n=int(n);own=bm.context(sid,seed,n)
                        scenarios=([('7502',7502,'full')] if args.stage=='alternative' else
                            [(str(c),c,'full') for c in COHORTS[2:]]+[(s,7501,s) for s in ['peer0','peer1','peer2','missing','uneven','ambiguous']])
                        for label,c,scenario in scenarios:
                            pairs=current_pairs(bm,sid,seed,n,maps[c][sid],scenario)
                            if n>=3 and pairs:
                                ctx=[own,*[bm.context(s,seed,nn) for s,nn in pairs]]
                                preds[f'partial_{label}'][:,j]=predict(ctx,query,partial=True)
                                if c in COHORTS[2:]:preds[f'pool_{label}'][:,j]=predict(ctx,query)
                                elif idx==indices[0]:
                                    expected=preds[f'pool_{label}'][:,j];actual=predict(ctx,query)
                                    maxreplay=max(maxreplay,float(np.max(abs(actual-expected))))
                            costs.append({'fold':f,'benchmark':bm.name,'subject_id':sid,'seed':seed,'scenario':label,'budget':n,'own_labels':n,
                                'peer_labels':sum(nn for _,nn in pairs),'active_peers':len(pairs),'visible_total':n+sum(nn for _,nn in pairs)})
                    if args.stage=='full':
                        for c in COHORTS:preds[f'shrink_{c}']=pown+np.array(shrink[str(f)]['alpha'])*(preds[f'pool_{c}']-pown)
                        for scenario in ['peer0','ambiguous']:assert np.array_equal(preds[f'partial_{scenario}'],pown)
                    for name,p in preds.items():
                        if name!='training_selected_smoothed':assert np.array_equal(p[:,:2],pown[:,:2]),name
                    ep={**idx,'counts':base['counts'][a:b],'successes':base['successes'][a:b]};rows.append(score_episode(ep,preds))
                    for name,p in preds.items():bank.setdefault(name,[]).append(p)
                    if len(rows)%1500==0:print(json.dumps({'stage':args.stage,'benchmark':bm.name,'trajectories':len(rows),'seconds':time.perf_counter()-begin}),flush=True)
                for name in ['counts','successes']:bank[name]=[base[name]]
                np.savez_compressed(dest/f'predictions_{suffix}.npz',**{name:np.concatenate(v) for name,v in bank.items()});write_json(dest/f'index_{suffix}.json',indices)
            assert maxreplay<1e-12,maxreplay
            write_json(dest/'paired_scores.json',rows);pd.DataFrame(costs).to_csv(dest/'label_costs.csv',index=False)
            done={'benchmark':bm.name,'trajectories':len(rows),'seconds':time.perf_counter()-begin,'old_pooled_replay_error':maxreplay,'peak_working_set_mib':peak_mb(),'finished':now()}
            write_json(dest/'complete.json',done);timings.append(done);allrows+=rows
            print(json.dumps({'stage':args.stage,**done}),flush=True)
    summarize(allrows,root,5242)
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'finished':now(),'benchmark_timings':timings,'threads':4})
    check_freeze(root/'contract');check_freeze(ROOT/'R62/isolation')
    print(json.dumps({'stage':args.stage,'completed':len(allrows),'seconds':time.perf_counter()-start}),flush=True)
if __name__=='__main__':main()
