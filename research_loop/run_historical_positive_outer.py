"""R62 and R63 full paired outer evaluation; no training or target selection."""
import argparse
import time
from historical_positive_common import *
from historical_positive_episodes import episodes
from mlp.cohort_semantic import CachedSemantic
from mlp.model import forward_batch
from mlp.training_mechanism_training import PARAMS
from mlp.run.run_metadata_round import prediction
from mlp.evidence_reliability import statistics, forward

def predict_inputs(stage,support,query,labels,variant_indices,models,alpha,mlp_support=None,mlp_query=None):
    """Only input features and legal support outcomes; no query outcomes or target IDs."""
    q=len(variant_indices)
    p25=np.column_stack([prediction(support[:n],query,labels[:n],1.)[variant_indices] for n in BUDGETS])
    result={'R25':p25,
        'smoothed_mean_2':np.broadcast_to([(1+labels[:n].sum())/(n+2) for n in BUDGETS],(q,6)).copy(),
        'training_selected_smoothed':np.broadcast_to([(.5*alpha+labels[:n].sum())/(n+alpha) for n in BUDGETS],(q,6)).copy()}
    diagnostic={}
    if stage=='R62':
        for method in ('R54','R56'):
            art=models[method]['artifact']
            p=forward_batch(np.concatenate([mlp_support,mlp_query]),tuple(art[k] for k in PARAMS)).astype(float)
            ps,pq=p[:31],p[31:]; residual=np.cumsum(labels-ps)
            full=np.column_stack([np.clip(pq+(residual[n-1]/(n+2) if n else 0),0,1) for n in BUDGETS])
            segment=p25.copy(); segment[:,:2]=full[:,:2]
            result[method+'_full']=full; result[method+'_segment']=segment
            diagnostic[method+'_first_support_sqerr']=float((ps[0]-labels[0])**2)
        gate=diagnostic['R56_first_support_sqerr']<.25
        gated=p25.copy()
        if gate: gated[:,1]=result['R56_full'][:,1]
        result['R61_support1_gate']=gated; diagnostic['gate_use_R56']=bool(gate)
        assert all(np.array_equal(result[m][:,2:],p25[:,2:]) for m in ['R54_segment','R56_segment','R61_support1_gate'])
        assert np.array_equal(gated[:,0],p25[:,0])
    else:
        stats={n:statistics(support[:n],query,labels[:n]) for n in BUDGETS[2:]}
        for method,model in models.items():
            out=p25.copy(); gains=[]
            if model['epoch']:
                for j,n in enumerate(BUDGETS[2:],2):
                    z,b,d=stats[n]
                    assert np.allclose(np.clip(b+d,0,1)[variant_indices],p25[:,j],atol=1e-12,rtol=0)
                    g=forward(z,model['artifact']); gains.append(float(g.mean()))
                    out[:,j]=np.clip(b+g*d,0,1)[variant_indices]
            else:
                # Source-selected policy is fixed for this entire fold, not selected per target.
                if model['artifact'] is not None:
                    assert np.allclose(forward(stats[3][0],model['artifact']),1.,atol=1e-12,rtol=0)
            result[method]=out; diagnostic[method+'_mean_g_B3_B31']=gains or [1.]*4
            assert np.array_equal(out[:,:2],p25[:,:2])
    return result,diagnostic

def main(stage):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    threadpool_limits(limits=4)
    start=time.perf_counter(); root=ROOT/stage
    check_freeze(ROOT/'R62/isolation')
    manifests=read(root/'models.json')
    paths=[Path(__file__),REPO/'research_loop/historical_positive_episodes.py',
        REPO/'research_loop/historical_positive_common.py',root/'models.json',
        ROOT/'R62/isolation/contract.json',REPO/'research_loop/HISTORICAL_POSITIVE_FOLLOWUP_SPEC.md']
    paths += [Path(m['weights']) for m in manifests if m['weights']]
    freeze(root/'evaluation_contract',paths,{'stage':stage,'seeds':[0,1,2],'pairs':5242,'epochs':EPOCHS})
    engine=CachedSemantic(); allrows=[]; timings=[]; replay=[]
    for f in range(3):
        info=source(f); old=read(ROOT/f'R31/fold{f}_evaluation.json')
        models={m['method']:{**m,'artifact':load_npz(m['weights']) if m['weights'] else None}
                for m in manifests if m['fold']==f}
        for name in info['heldout_benchmarks']:
            if name not in old['by_benchmark']: continue
            folder=root/'outer'/f'fold{f}'/name.split('/')[0]
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                allrows+=read(folder/'paired_scores.json'); timings.append(read(folder/'complete.json'))
                continue
            assert not list(folder.glob('predictions_*.npz')),f'Inspect incomplete benchmark: {folder}'
            reference=None
            if stage=='R63':
                reference={(r['subject_id'],r['seed']):r for r in read(ROOT/f'R62/outer/fold{f}'/name.split('/')[0]/'paired_scores.json')}
            rows=[]; bank={}; index=[]; offset=0; chunk=0; seconds=time.perf_counter(); predict_seconds=0.
            def flush():
                nonlocal bank,index,offset,chunk
                if not index:return
                np.savez_compressed(folder/f'predictions_{chunk:04d}.npz',**{k:np.concatenate(v) for k,v in bank.items()})
                write_json(folder/f'index_{chunk:04d}.json',index)
                bank={};index=[];offset=0;chunk+=1
            for ep in episodes(name,engine,info['semantic_ridge'],f,include_mlp=stage=='R62'):
                t=time.perf_counter()
                p,diag=predict_inputs(stage,ep['support'],ep['query'],ep['labels'],ep['variant_indices'],
                    models,info['selected_alpha'],ep.get('mlp_support'),ep.get('mlp_query'))
                predict_seconds+=time.perf_counter()-t
                row=score_episode(ep,p,diag)
                if reference is not None:
                    ref=reference[(ep['subject_id'],ep['seed'])]
                    assert row['identity']==ref['identity'],'R62/R63 response trajectory mismatch'
                    for m in BASELINES: assert np.array_equal(row['brier'][m],ref['brier'][m])
                rows.append(row)
                for m,probs in p.items():bank.setdefault(m,[]).append(probs)
                bank.setdefault('counts',[]).append(ep['counts']);bank.setdefault('successes',[]).append(ep['successes'])
                index.append({**ep['metadata'],'identity':ep['identity'],'start':offset,'size':len(ep['counts'])})
                offset+=len(ep['counts'])
                if len(index)>=128:flush()
                if len(rows)%300==0:
                    print(json.dumps({'stage':stage,'fold':f,'benchmark':name,'trajectories':len(rows),
                        'seconds':round(time.perf_counter()-seconds,1)}),flush=True)
            flush()
            assert len(rows)%3==0 and len(rows)//3==old['by_benchmark'][name]['pairs_per_seed'][0]
            summary=aggregate(rows)
            errors={m:float(np.max(np.abs(np.array(summary['results'][m]['brier_mean'])-
                old['by_benchmark'][name]['brier_mean'][m]))) for m in BASELINES}
            assert max(errors.values())<1e-7,(name,errors)
            replay.append({'fold':f,'benchmark':name,'errors':errors})
            write_json(folder/'paired_scores.json',rows);write_json(folder/'summary.json',summary)
            done={'fold':f,'benchmark':name,'trajectories':len(rows),'pairs':len(rows)//3,
                'seconds':time.perf_counter()-seconds,'prediction_seconds':predict_seconds,
                'baseline_replay_max_error':max(errors.values()),'chunks':chunk,'finished_utc':now()}
            write_json(folder/'complete.json',done);timings.append(done);allrows+=rows
            print(json.dumps({'stage':stage,**done}),flush=True)
            assert time.perf_counter()-start<7200,'Scoring time ceiling reached; inspect/resume completed benchmarks'
    summary=summarize(allrows,root,expected_pairs=5242)
    assert len(summary['by_benchmark'])==36
    check_freeze(root/'evaluation_contract');check_freeze(ROOT/'R62/isolation')
    write_json(root/'execution.json',{'seconds_this_invocation':time.perf_counter()-start,
        'benchmark_timings':timings,'baseline_replay':replay,'finished_utc':now()})
    print(json.dumps({'stage':stage,'pairs':5242,'alc':{m:v['alc_mean'] for m,v in summary['results'].items()}}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['R62','R63']); main(p.parse_args().stage)
