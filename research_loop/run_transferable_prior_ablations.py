"""R66 matched 12-fit feature ablations, source-only model selection."""
import gc
import time
import pandas as pd
from threadpoolctl import threadpool_limits
from historical_positive_common import *
from transferable_prior_data import prepare,training,validation,CACHE
from mlp.training_mechanism_training import train,trace_analysis,predict_mlp
from mlp.training_mechanism_data import mlp_training
from mlp.cohort_semantic import CachedSemantic

OUT=ROOT/'R66'

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),REPO/'research_loop/transferable_prior_data.py',
        REPO/'research_loop/TRANSFERABLE_PRIOR_R66_AMENDMENT.md',REPO/'research_loop/TRANSFERABLE_PRIOR_SPEC.md',
        ROOT/'R65/summary.json',BASE/'mlp/training_mechanism_training.py',BASE/'mlp/training_mechanism_data.py',BASE/'mlp/model.py',
        REPO/'research_loop/historical_exact_encoding_cache.py']
    files += [ROOT/f'reproduction_v4/source_fold{f}.json' for f in range(3)]
    files += list((V5/'R56').glob('fold*_selection.json'))
    files.append(OUT/'contract/contract.json')
    freeze(OUT/'contract_v2',files,{'modes':['subject','item'],'fits':12,'epochs':30,'model_seed':2026,'episode_seed':73,
        'repair':'R59 uses query_groups rather than query_ids; completed fits reused unchanged'})
    engine=CachedSemantic();names=sorted({n for f in range(3) for half in source(f)['inner_halves'] for n in half})
    for name in names:prepare(name,engine)
    write_json(OUT/'cache_inventory.json',{str(p):sha(p) for p in CACHE.glob('*/*') if p.is_file()})
    results=[];allrows=[];foldsummaries={}
    for f in range(3):
        halves=source(f)['inner_halves'];assert set(sum(halves,[])).isdisjoint(source(f)['heldout_benchmarks'])
        refs=read(ROOT/f'R59/fold{f}_paired_scores.json');indices=read(ROOT/f'R59/fold{f}_prediction_index.json')
        reflookup={(r['benchmark'],r['subject_id']):r for r in refs};ixlookup={(r['benchmark'],r['subject_id']):r for r in indices}
        bank=load_npz(ROOT/f'R59/fold{f}_predictions.npz')
        trajectories={};epochs={}
        for mode in ['subject','item']:
            histories=[]
            for inner in range(2):
                data=training(halves[inner],mode);val=validation(halves[1-inner],mode)
                oldcfg=read(V5/'R56'/f'fold{f}_mlp_inner{inner}/config.json')
                assert oldcfg['fit_hash']==mlp_training(halves[inner])['data_hash']
                assert oldcfg['validation_identity']==[[e['benchmark'],e['subject_id'],e['seed']] for e in val]
                out=OUT/f'fold{f}_{mode}_inner{inner}'
                write_json(out/'isolation.json',{'fit_benchmarks':halves[inner],'validation_benchmarks':halves[1-inner],
                    'excluded_outer':source(f)['heldout_benchmarks'],'feature_mode':mode,'old_full_config_sha256':sha(V5/'R56'/f'fold{f}_mlp_inner{inner}/config.json')})
                histories.append(train('mlp',data,val,out,epochs=30,risk='mixture'))
                oldh=read(V5/'R56'/f'fold{f}_mlp_inner{inner}/history.json')
                assert [h['steps'] for h in histories[-1]]==[h['steps'] for h in oldh]
                initial=load_npz(out/'epoch0.npz');oldinit=load_npz(V5/'R56'/f'fold{f}_mlp_inner{inner}/epoch0.npz')
                assert all(np.array_equal(initial[k],oldinit[k]) for k in initial)
                del data,val;gc.collect()
            analysis=trace_analysis(histories,'mlp');epoch=analysis['selection']['best30'];epochs[mode]=epoch
            write_json(OUT/f'fold{f}_{mode}_selection.json',analysis)
            preds=[];idx=[];offset=0;selected_train=[]
            for inner in range(2):
                val=validation(halves[1-inner],mode)
                art=load_npz(OUT/f'fold{f}_{mode}_inner{inner}/epoch{epoch}.npz')
                selected_train.append(histories[inner][epoch]['train_metric'])
                fit=mlp_training(halves[inner]);mu=float(np.average(fit['y'],weights=fit['mixture_weight']));del fit
                for ep,prob in zip(val,predict_mlp(val,art)):
                    key=(ep['benchmark'],ep['subject_id']);ref=reflookup[key];ix=ixlookup[key]
                    assert ix['support_ids']==ep['support_ids'] and np.array_equal(ix['support_labels'],ep['labels'])
                    p=np.stack(prob,axis=1);a=ix['start'];b=a+ix['size']
                    assert p.shape==(ix['size'],6)
                    raw=bank['R25'][a:b];c,s=ep['counts'],ep['successes']
                    assert np.array_equal(bank['counts'][a:b],c) and np.array_equal(bank['successes'][a:b],s)
                    assert all(abs(brier(raw[:,j],c,s)-ref['brier']['R25'][j])<1e-12 for j in range(6))
                    r=trajectories.setdefault(key,{'fold':f,'benchmark':key[0],'subject_id':key[1],'seed':73,
                        'brier':{k:v for k,v in ref['brier'].items() if k in BASELINES+['R56']},
                        'accuracy':{k:v for k,v in ref['accuracy'].items() if k in BASELINES+['R56']}})
                    for meth,q in [(mode+'_full',p),(mode+'_segment',np.c_[p[:,:2],raw[:,2:]])]:
                        r['brier'][meth]=[brier(q[:,j],c,s) for j in range(6)]
                        r['accuracy'][meth]=[accuracy(q[:,j],c,s) for j in range(6)]
                    const=np.stack([np.full(len(c),(2*mu+ep['labels'][:n].sum())/(n+2)) for n in BUDGETS],axis=1)
                    for meth,q in [('source_constant_full',const),('source_constant_segment',np.c_[const[:,:2],raw[:,2:]])]:
                        r['brier'][meth]=[brier(q[:,j],c,s) for j in range(6)];r['accuracy'][meth]=[accuracy(q[:,j],c,s) for j in range(6)]
                    r['brier']['R56_segment']=ref['brier']['R56'][:2]+ref['brier']['R25'][2:]
                    r['accuracy']['R56_segment']=ref['accuracy']['R56'][:2]+ref['accuracy']['R25'][2:]
                    preds.append(p);idx.append({'benchmark':key[0],'subject_id':key[1],'inner':inner,'seed':73,'start':offset,'size':len(c),
                        'support_ids':ep['support_ids'],'query_ids':ep['query_ids'],'labels':ep['labels'].tolist(),'counts':c.tolist(),'successes':s.tolist()});offset+=len(c)
            np.savez_compressed(OUT/f'fold{f}_{mode}_predictions.npz',p=np.concatenate(preds));write_json(OUT/f'fold{f}_{mode}_index.json',idx)
            results.append({'fold':f,'mode':mode,'epoch':epoch,'selected_train_pair_brier':selected_train})
        rows=list(trajectories.values());summary=aggregate(rows);foldsummaries[str(f)]=summary
        write_json(OUT/f'fold{f}_paired_scores.json',rows);write_json(OUT/f'fold{f}_summary.json',summary);allrows+=rows
        print(json.dumps({'stage':'R66_fold_done','fold':f,'epochs':epochs,'alc':{k:v['alc_mean'] for k,v in summary['results'].items()}}),flush=True)
    total=aggregate(allrows);decisions={}
    for mode in ['subject','item']:
        meth=mode+'_segment'
        d=[foldsummaries[str(f)]['results'][meth]['alc_mean']-foldsummaries[str(f)]['results']['R25']['alc_mean'] for f in range(3)]
        dc=[foldsummaries[str(f)]['results'][meth]['alc_mean']-foldsummaries[str(f)]['results']['source_constant_segment']['alc_mean'] for f in range(3)]
        other=aggregate([r for r in allrows if not r['benchmark'].startswith('alignment_faking/')])
        tests={'at_least_two_pools_R25':sum(v<0 for v in d)>=2,
            'at_least_two_pools_constant':sum(v<0 for v in dc)>=2,
            'pooled_pair_improves':total['results'][meth]['alc_mean']<total['results']['R25']['alc_mean'],
            'pooled_benchmark_improves':total['benchmark_equal_alc'][meth]<total['benchmark_equal_alc']['R25'],
            'without_alignment_improves':other['results'][meth]['alc_mean']<other['results']['R25']['alc_mean']}
        decisions[mode]={'fold_delta_R25':d,'fold_delta_constant':dc,'tests':tests,'numeric_entry':all(tests.values()),'mechanism_review_required':True}
    write_json(OUT/'summary.json',{'status':'complete','new_fits':12,'folds':foldsummaries,'pooled_exposures':total,
        'selections':results,'entry_diagnostics':decisions,'seconds':time.perf_counter()-start})
    metrics=[]
    for f,summary in foldsummaries.items():
        for m,v in summary['results'].items():
            for j,n in enumerate(BUDGETS):metrics.append({'stage':'R66','scope':f'pool{f}','method':m,'budget':n,'brier':v['brier_mean'][j],
                'accuracy':v['accuracy_at_0_5'][j],'alc':v['alc_mean'],'benchmark_equal_alc':summary['benchmark_equal_alc'][m],
                'delta_R25':v['alc_mean']-summary['results']['R25']['alc_mean']})
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);check_freeze(OUT/'contract_v2')

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
