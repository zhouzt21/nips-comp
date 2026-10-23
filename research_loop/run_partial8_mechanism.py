"""R81 complete matched geometry controls and pre-clip operator audit."""
import time
from partial8_next_common import *

def geometric(own,pool,part,w):
    d=pool-own;e=part-own;valid=abs(d)>1e-12;same=(d*e>0)&valid;flip=(d*e<0)&valid
    return {'pool_cost':float(w@(d*d)),'partial_cost':float(w@(e*e)),
        'same_direction':float(w@same),'direction_flip':float(w@flip),'same_direction_shrink':float(w@(same&(abs(e)<=abs(d)))),
        'same_direction_enlarge':float(w@(same&(abs(e)>abs(d)))),'pool_zero_partial_nonzero':float(w@((~valid)&(abs(e)>1e-12)))}

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R81';root.mkdir(exist_ok=True);assert (ROOT/'R80/execution.json').exists()
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/partial8_next_common.py',REPO/'research_loop/PARTIAL8_NEXT_SPEC.md',REPO/'doc/plan_10.md',
        REPO/'research_loop/shared_slope_model.py',ROOT/'R77/source_shrinkage.json'],
        {'cohorts':COHORTS,'budgets':list(map(int,BUDGETS)),'controls':['R77_source_shrink','point_pooled_sign_partial_magnitude','query_set_exact_cost_pooled'],
         'exact_cost_scope':'offline mechanism diagnostic only; uses all query predictions and response counts, never successes'})
    engine=CachedSemantic();allrows=[];timings=[]
    for f in range(3):
        for old in sorted((ROOT/f'R77/full/outer/fold{f}').iterdir()):
            dest=root/'outer'/f'fold{f}'/old.name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():allrows+=read(dest/'paired_scores.json');timings.append(read(dest/'complete.json'));continue
            begin=time.perf_counter();bm=Benchmark(ROOT/f'R64/full/outer/fold{f}'/old.name,source(f)['semantic_ridge'],engine)
            mapping=read(old/'cohorts.json')['maps'];write_json(dest/'cohorts.json',read(old/'cohorts.json'));rows=[];geo=[];maxerr=0.;maxcosterr=0.
            for ip in sorted(old.glob('index_*.json')):
                suffix=ip.stem.split('_')[-1];indices=read(ip);base=dict(np.load(old/f'predictions_{suffix}.npz'));bank={};rawbank={}
                for idx in indices:
                    a=idx['start'];b=a+idx['size'];sid=idx['subject_id'];seed=idx['seed'];q=bm.query(idx);counts=base['counts'][a:b];success=base['successes'][a:b];w=counts/counts.sum()
                    pp={m:base[m][a:b].copy() for m in ['R25','smoothed_mean_2','training_selected_smoothed',*[f'{kind}_{c}' for c in COHORTS for kind in ['pool','partial','shrink']]]}
                    raw={'own':pp['R25'].copy()}
                    for c in COHORTS:
                        for kind in ['point','cost']:pp[f'{kind}_{c}']=pp['R25'].copy()
                        for kind in ['pool','partial']:raw[f'{kind}_{c}']=pp['R25'].copy()
                    for j,n in enumerate(BUDGETS):
                        n=int(n);own=bm.context(sid,seed,n)
                        if n>=3:raw['own'][:,j]=raw_parts([own],q)[0]
                        maxerr=max(maxerr,float(abs(np.clip(raw['own'][:,j],0,1)-pp['R25'][:,j]).max()))
                        for c in COHORTS:
                            pairs=current_pairs(bm,sid,seed,n,mapping[str(c)][sid]);peers=[bm.context(s,seed,nn) for s,nn in pairs]
                            if n>=3 and peers:
                                raw[f'pool_{c}'][:,j]=raw_parts([own,*peers],q)[0];raw[f'partial_{c}'][:,j]=raw_parts([own,*peers],q,partial=True)[0]
                            else:
                                raw[f'pool_{c}'][:,j]=raw['own'][:,j];raw[f'partial_{c}'][:,j]=raw['own'][:,j]
                            for kind in ['pool','partial']:maxerr=max(maxerr,float(abs(np.clip(raw[f'{kind}_{c}'][:,j],0,1)-pp[f'{kind}_{c}'][:,j]).max()))
                            o=pp['R25'][:,j];p=pp[f'pool_{c}'][:,j];v=pp[f'partial_{c}'][:,j]
                            control=np.clip(o+np.sign(p-o)*abs(v-o),0,1);cc,alpha,target,cost=cost_control(o,p,v,counts)
                            pp[f'point_{c}'][:,j]=control;pp[f'cost_{c}'][:,j]=cc
                            row={'fold':f,'benchmark':bm.name,'subject_id':sid,'seed':seed,'cohort':c,'budget':n,'own_labels':n,'peer_labels':sum(nn for _,nn in pairs),'active_peers':len(peers),
                                 **geometric(o,p,v,w),**{'raw_'+k:z for k,z in geometric(raw['own'][:,j],raw[f'pool_{c}'][:,j],raw[f'partial_{c}'][:,j],w).items()},
                                 'point_cost':float(w@((control-o)**2)),'cost_control_cost':cost,'cost_alpha':float(alpha) if np.isfinite(alpha) else None,'unmatched_cost':max(target-cost,0.),
                                 'point_boundary_mismatch':float(w@(abs(abs(control-o)-abs(v-o))>1e-12)),
                                 'partial_clip_fraction':float(w@((raw[f'partial_{c}'][:,j]<0)|(raw[f'partial_{c}'][:,j]>1))),
                                 'pool_clip_fraction':float(w@((raw[f'pool_{c}'][:,j]<0)|(raw[f'pool_{c}'][:,j]>1)))}
                            residual=v-success/counts
                            for name,ctl in [('point',control),('cost',cc)]:
                                delta=ctl-v;row[name+'_alignment']=float(2*w@(residual*delta));row[name+'_squared_change']=float(w@(delta*delta))
                            geo.append(row)
                            if np.isfinite(alpha):maxcosterr=max(maxcosterr,abs(target-cost))
                    rows.append(score_episode({**idx,'counts':counts,'successes':success},pp))
                    for k,v in pp.items():bank.setdefault(k,[]).append(v)
                    for k,v in raw.items():rawbank.setdefault(k,[]).append(v)
                for k in ['counts','successes']:bank[k]=[base[k]]
                np.savez_compressed(dest/f'predictions_{suffix}.npz',**{k:np.concatenate(v) for k,v in bank.items()})
                np.savez_compressed(dest/f'raw_{suffix}.npz',**{k:np.concatenate(v) for k,v in rawbank.items()});write_json(dest/f'index_{suffix}.json',indices)
            assert maxerr<1e-11 and maxcosterr<1e-11
            write_json(dest/'paired_scores.json',rows);pd.DataFrame(geo).to_csv(dest/'geometry.csv',index=False)
            done={'fold':f,'benchmark':bm.name,'trajectories':len(rows),'max_raw_clipped_replay_error':maxerr,'max_reachable_cost_error':maxcosterr,'seconds':time.perf_counter()-begin,
                'peak_working_set_mib':peak_mb(),'finished':now()};write_json(dest/'complete.json',done);allrows+=rows;timings.append(done);print(json.dumps(done),flush=True)
    summarize(allrows,root,5242);write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'timings':timings,'threads':4,'finished':now()});check_freeze(root/'contract')
if __name__=='__main__':main()
