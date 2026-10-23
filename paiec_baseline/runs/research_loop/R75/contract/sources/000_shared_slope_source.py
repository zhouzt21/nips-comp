"""R75 source-only finite factor and uncertainty pilot; no outer tuning."""
import time
from shared_slope_data import *
from shared_slope_model import cohort_map,predict,heterogeneity,stable_seed

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R75';root.mkdir(exist_ok=True)
    paths=[Path(__file__),REPO/'research_loop/shared_slope_data.py',REPO/'research_loop/shared_slope_model.py',REPO/'research_loop/SHARED_SLOPE_CONTRACT.md',
           REPO/'research_loop/SHARED_SLOPE_SOURCES.md',root/'literature/manifest.json']
    freeze(root/'contract',paths,{'source_cap':8,'seeds':[0,1,2],'diagnostic_seed':0,'bootstrap':32,'delta_penalty':8,'cohort_seed':7501})
    engine=CachedSemantic();folders=outer_folders();all_diag=[];decisions={};fold_summaries={}
    for f in range(3):
        info=source(f);ridge=info['semantic_ridge'];r20=read(ROOT/f'R20/fold{f}_selection.json')['selected_ridge']['pooled'];rows=[];diagnostics=[]
        # Complete support-only entry diagnosis before any partial-pooling fit/evaluation.
        for name in info['train_benchmarks']:
            short=name.split('/')[0]
            if short not in folders:continue
            dest=root/f'source_fold{f}'/short;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'heterogeneity.json').exists():diagnostics+=read(dest/'heterogeneity.json');continue
            bm=Benchmark(folders[short],ridge,engine)
            unique={v['subject_id']:v for v in bm.records if not v['ambiguous_subject']}
            selected=sorted(unique,key=lambda s:stable_seed('source75|'+unique[s]['subject_key']))[:8]
            if not selected:continue
            records=[v for v in bm.records if v['subject_id'] in selected];cohorts=cohort_map(records,7501)
            write_json(dest/'selection.json',{'source_fold':f,'source_benchmark':name,'subjects':selected,'cohorts':cohorts,'ridge_r25':ridge,'ridge_r36_original':r20,'query_labels_used_for_selection':False})
            diag=[];anchor=next((s for s in selected if len(cohorts[s])==3),None)
            if anchor:
                h=heterogeneity([bm.context(s,0) for s in [anchor,*cohorts[anchor]]],seed=stable_seed(f'7503|{name}'))
                diag=[{'source_fold':f,'benchmark':name,'subjects':[anchor,*cohorts[anchor]],**h}]
            write_json(dest/'heterogeneity.json',diag);diagnostics+=diag
        h=np.array([v['H'] for v in diagnostics]);null=np.array([v['null_H'] for v in diagnostics]);positive=sum(v['positive_heterogeneity'] for v in diagnostics)
        mean_h=float(h.mean());null95=float(np.quantile(null.mean(0),.95));entry=bool(positive>=2 and mean_h>null95)
        write_json(root/f'source_fold{f}'/'entry_before_response_evaluation.json',{'heterogeneity_entry':entry,'mean_H':mean_h,'mean_null95':null95,'positive_benchmarks':positive,'finished':now()})
        for name in info['train_benchmarks']:
            short=name.split('/')[0]
            if short not in folders:continue
            dest=root/f'source_fold{f}'/short;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():
                rows+=read(dest/'paired_scores.json');continue
            bm=Benchmark(folders[short],ridge,engine)
            unique={v['subject_id']:v for v in bm.records if not v['ambiguous_subject']}
            selected=sorted(unique,key=lambda s:stable_seed('source75|'+unique[s]['subject_key']))[:8]
            if not selected:continue
            records=[v for v in bm.records if v['subject_id'] in selected];cohorts=cohort_map(records,7501)
            diag=read(dest/'heterogeneity.json')
            local=[];bank={};indices=[];offset=0
            for idx in bm.indices:
                if idx['subject_id'] not in selected:continue
                query=bm.query(idx);preds={m:[] for m in ['R25','pool_both','own_sem','pool_sem','pool_global_center','pool_both_R36ridge','pool_sem_R36ridge','partial8','partial8_perm']}
                for n in BUDGETS:
                    ctx=[bm.context(s,idx['seed'],n) for s in [idx['subject_id'],*cohorts[idx['subject_id']]]]
                    preds['R25'].append(predict(ctx[:1],query));preds['pool_both'].append(predict(ctx,query))
                    preds['own_sem'].append(predict(ctx[:1],query,semantic=True));preds['pool_sem'].append(predict(ctx,query,semantic=True))
                    preds['pool_global_center'].append(predict(ctx,query,global_center=True))
                    preds['pool_both_R36ridge'].append(predict(ctx,query,ridge=r20/ridge))
                    preds['pool_sem_R36ridge'].append(predict(ctx,query,ridge=r20/ridge,semantic=True))
                    preds['partial8'].append(predict(ctx,query,partial=True) if entry and len(ctx)>1 else preds['pool_both'][-1])
                    perm=[ctx[0]]+[{**c,'y':c['y'][np.random.default_rng(stable_seed(f'7504|{name}|{idx["subject_id"]}|{idx["seed"]}|{n}|{k}')).permutation(n)]} for k,c in enumerate(ctx[1:])]
                    preds['partial8_perm'].append(predict(perm,query,partial=True) if entry and len(ctx)>1 else preds['pool_both'][-1])
                preds={m:np.stack(v,axis=1) for m,v in preds.items()}
                ep={**idx,'fold':f,'counts':np.asarray(idx['counts'],float),'successes':np.asarray(idx['successes'],float)}
                local.append(score_episode(ep,preds))
                for m,p in preds.items():bank.setdefault(m,[]).append(p)
                for m in ['counts','successes']:bank.setdefault(m,[]).append(ep[m])
                indices.append({**idx,'start':offset,'size':len(ep['counts']),'selection_fold':f});offset+=len(ep['counts'])
            np.savez_compressed(dest/'predictions.npz',**{m:np.concatenate(v) for m,v in bank.items()})
            write_json(dest/'index.json',indices);write_json(dest/'paired_scores.json',local);rows+=local
            write_json(dest/'complete.json',{'trajectories':len(local),'finished':now()})
            print(json.dumps({'stage':'R75','source_fold':f,'benchmark':name,'trajectories':len(local),'H':diag[0]['H'] if diag else None,'seconds':time.perf_counter()-start}),flush=True)
        dest=root/f'source_fold{f}';summary=summarize(rows,dest);fold_summaries[f]=summary
        h=np.array([v['H'] for v in diagnostics]);null=np.array([v['null_H'] for v in diagnostics]);positive=sum(v['positive_heterogeneity'] for v in diagnostics)
        mean_h=float(h.mean());null95=float(np.quantile(null.mean(0),.95));entry=bool(positive>=2 and mean_h>null95)
        delta=summary['results']['partial8']['alc_mean']-summary['results']['pool_both']['alc_mean']
        decisions[f]={'diagnosed_benchmarks':len(diagnostics),'positive_heterogeneity_benchmarks':positive,'mean_H':mean_h,'mean_null95':null95,
                      'heterogeneity_entry':entry,'partial_source_delta':delta,'partial_full_enabled':bool(entry and delta<0),
                      'status':'enabled' if entry and delta<0 else 'condition_not_met' if not entry else 'source_gain_failed'}
        all_diag+=diagnostics
    write_json(root/'heterogeneity_all_source.json',all_diag);write_json(root/'selection.json',decisions)
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mb':peak_memory_mb(),'finished':now(),'source_overlap_not_independent':True})
    check_freeze(root/'contract');check_freeze(ROOT/'R62/isolation');print(json.dumps({'decisions':decisions,'seconds':time.perf_counter()-start}),flush=True)

if __name__=='__main__':main()
