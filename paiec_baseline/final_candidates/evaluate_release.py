"""Full frozen-input replay and fixed4 evaluation; never writes historical outputs."""
import sys,time
from common import HERE,BASE,REPO,ART
sys.path.insert(0,str(REPO/'research_loop'))
from shared_slope_data import *
from shared_slope_model import predict

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);t=time.perf_counter();out=ART/'evaluation';out.mkdir(parents=True,exist_ok=True)
    freeze(out/'contract',[Path(__file__),HERE/'CONTRACT.md',HERE/'RELEASE_CONFIGURATION_REQUEST.md',REPO/'research_loop/shared_slope_model.py',REPO/'research_loop/shared_slope_data.py'],{'semantic_ridge':[4,4,4],'historical':[4,4,1],'cohort':7501,'budgets':list(map(int,BUDGETS)),'no_selection':True})
    engine=CachedSemantic();allrows=[];evidence=[]
    for f in range(3):
        for old in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
            dest=out/f'fold{f}'/old.name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():
                allrows+=read(dest/'paired_scores.json');evidence.append(read(dest/'complete.json'));continue
            started=time.perf_counter();bm=Benchmark(old,source(f)['semantic_ridge'],engine)
            r76=ROOT/f'R76/full/outer/fold{f}'/old.name;r77=ROOT/f'R77/full/outer/fold{f}'/old.name;r81=ROOT/f'R81/outer/fold{f}'/old.name
            peers=read(r76/'cohorts.json')['main'];assert peers==read(r77/'cohorts.json')['maps']['7501']==read(r81/'cohorts.json')['maps']['7501']
            rows=[];maxerr=0.;nprob=0;refs=[]
            for ip in sorted(old.glob('index_*.json')):
                suf=ip.stem.split('_')[-1];indices=read(ip);banks={};refs.append({'path':str(ip),'sha256':sha(ip)})
                a76=dict(np.load(r76/f'predictions_{suf}.npz'));a77=dict(np.load(r77/f'predictions_{suf}.npz'));a81=dict(np.load(r81/f'predictions_{suf}.npz'))
                for directory in [r76,r77,r81]:
                    p=directory/f'predictions_{suf}.npz';refs.append({'path':str(p),'sha256':sha(p)})
                for idx in indices:
                    a=idx['start'];b=a+idx['size'];sid=idx['subject_id'];seed=idx['seed'];q=bm.query(idx)
                    hist={k:np.empty((len(q),6)) for k in ['R25','Partial8']};fixed={k:np.empty((len(q),6)) for k in hist}
                    for j,n in enumerate(BUDGETS):
                        own=bm.context(sid,seed,int(n));ctx=[own,*[bm.context(s,seed,int(n)) for s in peers[sid]]]
                        hist['R25'][:,j]=predict([own],q)
                        hist['Partial8'][:,j]=predict(ctx,q,partial=True) if n>=3 and len(ctx)>1 else hist['R25'][:,j]
                        if f<2:
                            for k in hist:fixed[k][:,j]=hist[k][:,j]
                        else:
                            # Scaling is an exact change from original semantic ridge 1 to fixed 4.
                            scale=np.r_[np.full(384,.5),np.ones(256)]
                            cc=[{**v,'x':v['x']*scale} for v in ctx];qq=q*scale
                            fixed['R25'][:,j]=predict(cc[:1],qq)
                            fixed['Partial8'][:,j]=predict(cc,qq,partial=True) if n>=3 and len(cc)>1 else fixed['R25'][:,j]
                    for label,ref in [('R76',a76),('R77',a77),('R81',a81)]:
                        for k,key in [('R25','R25'),('Partial8','R76_partial_selected' if label=='R76' else 'partial_7501')]:
                            err=float(abs(hist[k]-ref[key][a:b]).max());maxerr=max(maxerr,err);assert err<1e-10,(old,sid,seed,label,k,err)
                    p={**{'historical_'+k:v for k,v in hist.items()},**{'release_'+k:v for k,v in fixed.items()},'R25':hist['R25'],'smoothed_mean_2':a81['smoothed_mean_2'][a:b]}
                    rows.append(score_episode({**idx,'counts':a81['counts'][a:b],'successes':a81['successes'][a:b]},p));nprob+=hist['R25'].size
                    for k,v in p.items():banks.setdefault(k,[]).append(v)
                banks.update(counts=[a81['counts']],successes=[a81['successes']])
                np.savez_compressed(dest/f'predictions_{suf}.npz',**{k:np.concatenate(v) for k,v in banks.items()})
                write_json(dest/f'index_{suf}.json',indices)
            write_json(dest/'paired_scores.json',rows)
            done={'fold':f,'benchmark':bm.name,'trajectories':len(rows),'probabilities_per_method':nprob,'max_historical_probability_error':maxerr,'seconds':time.perf_counter()-started,'references':refs,'finished':now()}
            write_json(dest/'complete.json',done);evidence.append(done);allrows+=rows
            print(json.dumps({k:v for k,v in done.items() if k!='references'}),flush=True)
    summary=summarize(allrows,out,5242)
    historical=read(ROOT/'R81/summary.json');errors={}
    for name,oldname in [('R25','R25'),('Partial8','partial_7501')]:
        errors[name]=abs(summary['results']['historical_'+name]['alc_mean']-historical['results'][oldname]['alc_mean']);assert errors[name]<1e-10
    from shared_reliability_common import bm_comparison
    comparisons={k:bm_comparison(allrows,'release_'+k,'historical_'+k) for k in ['R25','Partial8']}
    comparisons['release_partial_vs_r25']=bm_comparison(allrows,'release_Partial8','release_R25')
    write_json(out/'release_comparisons.json',comparisons)
    write_json(out/'execution.json',{'finished':now(),'seconds':time.perf_counter()-t,'benchmarks':len(evidence),'trajectories':len(allrows),'unique_pairs':5242,'historical_alc_error':errors,'max_historical_probability_error':max(v['max_historical_probability_error'] for v in evidence),'historical_foldwise_score':{k:summary['results']['historical_'+k] for k in ['R25','Partial8']},'release_fixed4_local_score':{k:summary['results']['release_'+k] for k in ['R25','Partial8']},'hosted_score':None})
    check_freeze(out/'contract');print(json.dumps(read(out/'execution.json')),flush=True)
if __name__=='__main__':main()
