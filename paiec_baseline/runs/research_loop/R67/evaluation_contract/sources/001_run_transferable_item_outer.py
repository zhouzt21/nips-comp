"""R67: one source-selected item prior, exact R62 outer histories."""
import hashlib
import time
from threadpoolctl import threadpool_limits
from historical_positive_common import *
from historical_exact_encoding_cache import ExactEncodingCache
from transferable_prior_data import training,validation
from mlp.training_mechanism_training import train
from mlp.model import forward_batch,_ITEM_FIELDS,_string
from mlp.evaluate import _load_directory
from mlp.reproduction_episodes import DATA_ROOT
from analyze_historical_positive import paired_comparison

OUT=ROOT/'R67';METHOD='R67_item_segment';PARAMS=('W1','b1','W2','b2','W3','b3')

def predict(query_prior,support_prior,label,r25):
    return np.c_[query_prior,np.clip(query_prior+(label-support_prior)/3,0,1),r25[:,2:]]

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),REPO/'research_loop/TRANSFERABLE_PRIOR_R67_AMENDMENT.md',
           REPO/'research_loop/transferable_prior_data.py',REPO/'research_loop/historical_exact_encoding_cache.py',
           ROOT/'R66/summary.json',ROOT/'R62/summary.json',ROOT/'R62/models.json']
    files += [ROOT/f'reproduction_v4/source_fold{f}.json' for f in range(3)]
    files += list((ROOT/'R66').glob('fold*_item_selection.json'))
    freeze(OUT/'contract',files,{'candidate':METHOD,'source_epochs':[1,2,2],'outer_pairs':5242,'seeds':[0,1,2],'new_fits':3})
    weights={};models=[]
    for f in range(3):
        cfg=source(f);names=sum(cfg['inner_halves'],[]);assert set(names).isdisjoint(cfg['heldout_benchmarks'])
        epoch=read(ROOT/f'R66/fold{f}_item_selection.json')['selection']['best30'];assert epoch==[1,2,2][f]
        data=training(names,'item');monitor=validation(names,'item')[:1]
        folder=OUT/f'refits/fold{f}';train('mlp',data,monitor,folder,epochs=epoch,risk='mixture')
        path=folder/f'epoch{epoch}.npz';weights[f]=tuple(load_npz(path)[k] for k in PARAMS)
        models.append({'fold':f,'fit_benchmarks':names,'epoch':epoch,'model_seed':2026,'weights':str(path),
            'sha256':sha(path),'fit_hash':data['data_hash'],'monitor_role':'fit-only logging, never checkpoint selection'})
        del data,monitor
    write_json(OUT/'models.json',models)
    freeze(OUT/'evaluation_contract',[OUT/'models.json',Path(__file__),*[Path(m['weights']) for m in models]],{'method':METHOD})
    rows=[];saved_support=[];elapsed_bm=[];forward_keys=0
    for path in sorted((ROOT/'R62/outer').glob('fold*/*/paired_scores.json')):
        ref=read(path);name=ref[0]['benchmark'];f=ref[0]['fold'];dest=OUT/'outer'/f'fold{f}'/name.split('/')[0]
        dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():
            rows+=read(dest/'paired_scores.json');elapsed_bm.append(read(dest/'complete.json'));continue
        begin=time.perf_counter();response,_,items=_load_directory(DATA_ROOT,name);items={str(k):v for k,v in items.items()}
        response['_variant']=response.interactors.map(str)
        reps=response.drop_duplicates(['item_id','_variant']);variants={}
        for i,v in reps[['item_id','_variant']].itertuples(index=False,name=None):
            key=(str(i),hashlib.sha256(v.encode()).hexdigest());variants[key]=v
        # Item-only model legitimately shares computation for identical legal input.
        # Prediction cache contains no labels and no subject/benchmark ID feature.
        cache=ExactEncodingCache();prior={};bmrows=[];bm_support=[]
        lookup={(r['subject_id'],r['seed']):r for r in ref}
        for file in sorted(path.parent.glob('predictions_*.npz')):
            ixfile=file.with_name(file.name.replace('predictions_','index_')).with_suffix('.json');indices=read(ixfile)
            needed=set()
            for ix in indices:
                needed.add((ix['support_ids'][0],ix['support_variant_sha256'][0]))
                needed.update((ix['query_ids'][i],h) for i,h in zip(ix['query_variant_item_indices'],ix['query_variant_sha256']))
            missing=sorted(needed-set(prior))
            for pos in range(0,len(missing),512):
                keys=missing[pos:pos+512];X=[]
                for key in keys:
                    item=items[key[0]];inp={'item_content':item['content'],'item_features':item['item_features'],'interactors':variants[key]}
                    x,_=cache.item_cached(tuple(_string(inp.get(k,'')) for k in _ITEM_FIELDS));X.append(x)
                p=forward_batch(np.stack(X),weights[f]).astype(float)
                prior.update(zip(keys,map(float,p)));forward_keys+=len(keys)
            with np.load(file) as archive:
                bank={m:archive[m] for m in BASELINES+['R56_segment','counts','successes']}
            output=[]
            for ix in indices:
                a=ix['start'];b=a+ix['size'];q=np.array([prior[(ix['query_ids'][i],h)] for i,h in zip(ix['query_variant_item_indices'],ix['query_variant_sha256'])])
                ps=prior[(ix['support_ids'][0],ix['support_variant_sha256'][0])]
                p=predict(q,ps,ix['support_labels'][0],bank['R25'][a:b]);output.append(p)
                meta={k:v for k,v in ix.items() if k not in ['identity','start','size']}
                assert hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()==ix['identity']
                c,s=bank['counts'][a:b],bank['successes'][a:b]
                assert np.array_equal(c,ix['counts']) and np.array_equal(s,ix['successes'])
                pred={m:bank[m][a:b] for m in BASELINES+['R56_segment']};pred[METHOD]=p
                ep={**ix,'counts':c,'successes':s};scored=score_episode(ep,pred)
                r=lookup[(ix['subject_id'],ix['seed'])];assert r['identity']==ix['identity']
                for m in BASELINES+['R56_segment']:assert np.max(np.abs(np.array(scored['brier'][m])-r['brier'][m]))<1e-12
                bmrows.append(scored);bm_support.append({'identity':ix['identity'],'prior':ps,'label':ix['support_labels'][0]})
            bank[METHOD]=np.concatenate(output);np.savez_compressed(dest/file.name,**bank)
            write_json(dest/ixfile.name,indices)
        write_json(dest/'paired_scores.json',bmrows);write_json(dest/'support_priors.json',bm_support)
        complete={'benchmark':name,'fold':f,'trajectories':len(bmrows),'distinct_item_variants_forwarded':len(prior),'seconds':time.perf_counter()-begin}
        write_json(dest/'complete.json',complete);rows+=bmrows;elapsed_bm.append(complete)
        print(json.dumps(complete),flush=True)
        assert time.perf_counter()-start<1800,'R67 30min compute limit'
    summary=summarize(rows,OUT,expected_pairs=5242)
    write_json(OUT/'versus_R56.json',paired_comparison(rows,METHOD,'R56_segment'))
    write_json(OUT/'timing.json',{'seconds_this_invocation':time.perf_counter()-start,'benchmarks':elapsed_bm,
        'sum_outer_seconds':sum(v['seconds'] for v in elapsed_bm),'new_fits':3,
        'weights_bytes':sum(Path(m['weights']).stat().st_size for m in models),
        'inference_note':'offline replay with item-only legal-input probability caching; not hosted latency acceptance'})
    write_json(OUT/'status.json',{'status':'evaluation_complete_analysis_pending','finished_utc':now(),'trajectories':len(rows)})
    check_freeze(OUT/'contract');check_freeze(OUT/'evaluation_contract')
    print(json.dumps({'R67_ALC':summary['results'][METHOD]['alc_mean'],'delta':summary['comparisons'][METHOD]['R25']}),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
