"""R76 full predefined current-peer scenarios; no cohort selected by outcomes."""
import time
from shared_slope_data import *
from shared_slope_model import cohort_map,predict,stable_seed

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R76/full';root.mkdir(parents=True,exist_ok=True)
    check_freeze(ROOT/'R75/contract');selection=read(ROOT/'R75/selection.json')
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/shared_slope_data.py',REPO/'research_loop/shared_slope_model.py',
        REPO/'research_loop/SHARED_SLOPE_CONTRACT.md',ROOT/'R75/selection.json'],{'scope':'conditional_simulation','cohort_seeds':[7501,7502],'pairs':5242,'seeds':[0,1,2]})
    engine=CachedSemantic();allrows=[];timings=[]
    for f in range(3):
        ridge=source(f)['semantic_ridge'];partial=selection[str(f)]['partial_full_enabled']
        for old in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
            dest=root/'outer'/f'fold{f}'/old.name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():allrows+=read(dest/'paired_scores.json');timings.append(read(dest/'complete.json'));continue
            assert not list(dest.glob('predictions_*.npz')),'Inspect and preserve incomplete predictions before restart'
            begin=time.perf_counter();bm=Benchmark(old,ridge,engine);mainmap=cohort_map(bm.records,7501);altmap=cohort_map(bm.records,7502)
            write_json(dest/'cohorts.json',{'main':mainmap,'alternative':altmap,'ambiguity_excluded':[v['subject_id'] for v in bm.records if v['seed']==0 and v['ambiguous_subject']]})
            rows=[];costs=[];maxownerr=0.
            refrows={v['identity']:v for v in read(ROOT/f'R72/full/outer/fold{f}'/old.name/'paired_scores.json')}
            for ip in sorted(old.glob('index_*.json')):
                suf=ip.stem.split('_')[-1];index=read(ip);base=dict(np.load(ROOT/f'R72/full/outer/fold{f}'/old.name/f'predictions_{suf}.npz'));bank={}
                for idx in index:
                    sid=idx['subject_id'];seed=idx['seed'];a=idx['start'];b=a+idx['size'];p=base['R25'][a:b];query=bm.query(idx)
                    names=['R76_peer0','R76_peer1','R76_peer2','R76_peer3','R76_missing','R76_uneven','R76_alt3','R76_ambiguous','R76_partial_selected','R76_partial_perm']
                    preds={m:p.copy() for m in names}
                    for j,n in enumerate(BUDGETS):
                        own=bm.context(sid,seed,n);peers=mainmap[sid];altpeers=altmap[sid]
                        missing=[s for s in peers if stable_seed(f'7505|{bm.name}|{bm.by[(sid,seed)]["subject_key"]}|{bm.by[(s,seed)]["subject_key"]}')%2==0]
                        scenario={
                          'R76_peer0':[], 'R76_peer1':[(s,n) for s in peers[:1]],'R76_peer2':[(s,n) for s in peers[:2]],
                          'R76_peer3':[(s,n) for s in peers[:3]],'R76_missing':[(s,n) for s in missing],
                          'R76_uneven':[(s,n//(2**k)) for k,s in enumerate(peers)],'R76_alt3':[(s,n) for s in altpeers], 'R76_ambiguous':[]}
                        if n and (idx==index[0]):maxownerr=max(maxownerr,float(np.max(np.abs(predict([own],query)-p[:,j]))))
                        for m,pairs in scenario.items():
                            active=[(s,nn) for s,nn in pairs if nn>0]
                            if n and active:preds[m][:,j]=predict([own,*[bm.context(s,seed,nn) for s,nn in active]],query)
                            costs.append({'fold':f,'benchmark':bm.name,'subject_id':sid,'seed':seed,'method':m,'budget':n,'own_labels':n,
                                'peer_labels':sum(nn for _,nn in active),'active_peers':len(active),'visible_total':n+sum(nn for _,nn in active),
                                'largest_dual_matrix_entries':(n+sum(nn for _,nn in active))**2})
                        preds['R76_partial_selected'][:,j]=preds['R76_peer3'][:,j]
                        preds['R76_partial_perm'][:,j]=preds['R76_peer3'][:,j]
                        if partial and n and peers:
                            ctx=[own,*[bm.context(s,seed,n) for s in peers]]
                            preds['R76_partial_selected'][:,j]=predict(ctx,query,partial=True)
                            perm=[own]+[{**c,'y':c['y'][np.random.default_rng(stable_seed(f'7601|{bm.name}|{sid}|{seed}|{n}|{k}')).permutation(n)]} for k,c in enumerate(ctx[1:])]
                            preds['R76_partial_perm'][:,j]=predict(perm,query,partial=True)
                    for m in ['R25','smoothed_mean_2','training_selected_smoothed','R72_same3','R72_pooled3','R72_diff3']:preds[m]=base[m][a:b]
                    assert np.array_equal(preds['R76_peer0'],p) and np.array_equal(preds['R76_ambiguous'],p)
                    assert all(np.array_equal(v[:,0],p[:,0]) for v in preds.values())
                    ep={**idx,'counts':base['counts'][a:b],'successes':base['successes'][a:b]};row=score_episode(ep,preds)
                    ref=refrows[idx['identity']]
                    for k in ['brier','accuracy']:row[k]['R56_segment']=ref[k]['R56_segment']
                    rows.append(row)
                    for m,v in preds.items():bank.setdefault(m,[]).append(v)
                    if len(rows)%900==0:print(json.dumps({'stage':'R76','benchmark':bm.name,'trajectories':len(rows),'seconds':time.perf_counter()-begin}),flush=True)
                for m in ['counts','successes']:bank[m]=[base[m]]
                np.savez_compressed(dest/f'predictions_{suf}.npz',**{m:np.concatenate(v) for m,v in bank.items()});write_json(dest/f'index_{suf}.json',index)
            assert maxownerr<1e-12,maxownerr
            write_json(dest/'paired_scores.json',rows);pd.DataFrame(costs).to_csv(dest/'label_costs.csv',index=False)
            done={'fold':f,'benchmark':bm.name,'trajectories':len(rows),'seconds':time.perf_counter()-begin,'own_replay_error':maxownerr,
                  'peak_working_set_mb':peak_memory_mb(),'partial_enabled':partial,'finished':now()}
            write_json(dest/'complete.json',done);allrows+=rows;timings.append(done)
            print(json.dumps({'stage':'R76',**done}),flush=True)
    summary=summarize(allrows,root,5242)
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'benchmark_timings':timings,'peak_working_set_mb':peak_memory_mb(),'finished':now(),'scope':'conditional_simulation'})
    check_freeze(root/'contract');check_freeze(ROOT/'R75/contract');check_freeze(ROOT/'R62/isolation')
    print(json.dumps({m:v['alc_mean'] for m,v in summary['results'].items()}),flush=True)

if __name__=='__main__':main()
