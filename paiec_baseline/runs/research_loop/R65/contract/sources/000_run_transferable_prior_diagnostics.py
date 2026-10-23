"""R65: replay frozen probabilities; no new fit or target-conditioned policy."""
import hashlib
import time
import pandas as pd
from historical_positive_common import *
from analyze_historical_positive import decomposition
from mlp.training_mechanism_data import mlp_training
from mlp.evaluate import _load_directory
from mlp.model import _SUBJECT_FIELDS, _string
from mlp.reproduction_episodes import DATA_ROOT
from mlp.cohort_semantic import CachedSemantic

OUT = ROOT/'R65'
FEATURES = ['subject_present','subject_logchars','item_logchars','item_digit_fraction',
            'item_nonascii_fraction','metadata_logchars','semantic_norm',
            'semantic_0','semantic_1','semantic_2','semantic_3']

def descriptor(subject, item, engine, cache):
    key = str(item['content'])
    if key not in cache:
        text = _string(item['content']); v = engine.vector(item['content'])
        cache[key] = [np.log1p(len(text)), sum(c.isdigit() for c in text)/max(1,len(text)),
            sum(ord(c)>127 for c in text)/max(1,len(text)),
            np.log1p(len(_string(item['item_features']))), np.linalg.norm(v), *v[:4]]
    values = [_string(subject.get(k,'')) for k in _SUBJECT_FIELDS]
    return np.array([sum(bool(v) for v in values),np.log1p(sum(map(len,values))),*cache[key]],float)

def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),REPO/'doc/plan_5.md',REPO/'research_loop/TRANSFERABLE_PRIOR_SPEC.md',
           ROOT/'R62/models.json',ROOT/'R62/evaluation_contract/contract.json',DATA_ROOT/'corpus_manifest.json']
    models=read(ROOT/'R62/models.json')
    files += [Path(m['weights']) for m in models]
    files += [ROOT/f'reproduction_v4/source_fold{f}.json' for f in range(3)]
    freeze(OUT/'contract',files,{'stage':'diagnostics_only','features':FEATURES,'new_fits':0})
    engine=CachedSemantic();source_rows={}; source_names={}; source_meta={}; raw_hash={}
    names=sorted(set(b for f in range(3) for b in source(f)['train_benchmarks']))
    for name in names:
        responses, subjects, items=_load_directory(DATA_ROOT,name)
        sampled=responses.sample(n=400,random_state=2026) if len(responses)>400 else responses
        sampled=sampled[sampled.subject_id.isin(subjects)&sampled.item_id.isin(items)]
        cache={};features=[];ident=[]
        for row in sampled.itertuples(index=False):
            features.append(descriptor(subjects[row.subject_id],items[row.item_id],engine,cache))
            ident.append(_string(subjects[row.subject_id].get('normalized_name','')))
        source_rows[name]=np.stack(features);source_names[name]=ident
        for file in ['response.parquet','subjects.parquet','items.parquet']:
            p=DATA_ROOT/name/file;raw_hash[str(p)]=sha(p)
    for f in range(3):
        cfg=source(f); order=cfg['inner_halves'][0]+cfg['inner_halves'][1]
        assert set(order).isdisjoint(cfg['heldout_benchmarks'])
        data=mlp_training(order); X=np.concatenate([source_rows[b] for b in order]);assert len(X)==len(data['y'])
        w=data['mixture_weight'].astype(float);mu=np.average(X,axis=0,weights=w)
        sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=w));sd=np.maximum(sd,1e-6)
        source_meta[f]={'pair_prevalence':float(np.average(data['y'],weights=data['pair_weight'])),
            'mixture_prevalence':float(np.average(data['y'],weights=w)),
            'feature_mean':mu.tolist(),'feature_sd':sd.tolist(),'fit_data_hash':data['data_hash'],
            'rows':len(X),'fit_benchmarks':order,'heldout_benchmarks':cfg['heldout_benchmarks'],
            'subject_names':sorted(set(x for b in order for x in source_names[b]))}
    write_json(OUT/'source_statistics.json',source_meta)
    write_json(OUT/'data_hashes.json',raw_hash)
    details=[];updates=[];features=[];error=0.;chunks=0;identities=set()
    for path in sorted((ROOT/'R62/outer').glob('fold*/*/paired_scores.json')):
        rows=read(path);lookup={(r['subject_id'],r['seed']):r for r in rows};name=rows[0]['benchmark'];f=rows[0]['fold']
        _,subjects,items=_load_directory(DATA_ROOT,name)
        subjects={str(k):v for k,v in subjects.items()};items={str(k):v for k,v in items.items()};cache={}
        cfg=source_meta[f];mu=np.array(cfg['feature_mean']);sd=np.array(cfg['feature_sd'])
        for file in sorted(path.parent.glob('predictions_*.npz')):
            ix=read(file.with_name(file.name.replace('predictions_','index_')).with_suffix('.json'))
            with np.load(file) as bank:
                probs={m:bank[m] for m in ['R25','R54_full','R56_full']}
                for index in ix:
                    meta={k:v for k,v in index.items() if k not in ['identity','start','size']}
                    assert hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()==index['identity']
                    a=index['start'];b=a+index['size'];c=bank['counts'][a:b];s=bank['successes'][a:b]
                    assert np.array_equal(c,index['counts']) and np.array_equal(s,index['successes'])
                    row=lookup[(index['subject_id'],index['seed'])];assert row['identity']==index['identity']
                    identities.add((name,index['subject_id'],index['seed']))
                    label=index['support_labels'][0]
                    pred={m:p[a:b,:2] for m,p in probs.items()}
                    for risk in ['pair','mixture']:
                        q=cfg[risk+'_prevalence'];pred['source_constant_'+risk]=np.tile([q,(2*q+label)/3],(len(c),1))
                    base={k:row[k] for k in ['fold','benchmark','subject_id','seed','identity']}
                    for method,p in pred.items():
                        vals=[]
                        for j in range(2):
                            decomp=decomposition(p[:,j],c,s);score=brier(p[:,j],c,s)
                            if method in probs:
                                error=max(error,abs(score-row['brier'][method][j]));assert error<1e-12
                            vals.append(score)
                            details.append({**base,'method':method,'budget':j,**decomp,'brier':score,
                                            'accuracy':accuracy(p[:,j],c,s)})
                        d=p[:,1]-p[:,0];cross=float(2*(d@(c*p[:,0]-s))/c.sum());energy=float(c@(d*d)/c.sum())
                        assert abs(vals[1]-vals[0]-cross-energy)<1e-12
                        updates.append({**base,'method':method,'B0':vals[0],'B1':vals[1],
                            'update_cross':cross,'update_energy':energy,'update_net':cross+energy,
                            'first_label':label,'first_support_prior':(label+(1-2*label)*np.sqrt(row[method[:3]+'_first_support_sqerr'])) if method[:3] in ['R54','R56'] else (0.5 if method=='R25' else cfg[method.split('_')[-1]+'_prevalence'])})
                    subj=subjects[index['subject_id']]
                    vals=np.stack([descriptor(subj,items[i],engine,cache) for i in index['query_ids']])
                    expanded=vals[np.array(index['query_variant_item_indices'])]
                    averaged=np.average(expanded,axis=0,weights=c)
                    features.append({**base,**dict(zip(FEATURES,averaged)),
                        'source_standardized_distance':float(np.sqrt(np.mean(((averaged-mu)/sd)**2))),
                        'subject_name_seen':int(_string(subj.get('normalized_name','')) in cfg['subject_names']),
                        'R56_mean_p0':float(c@pred['R56_full'][:,0]/c.sum()),
                        'R56_mean_uncertainty':float(c@(pred['R56_full'][:,0]*(1-pred['R56_full'][:,0]))/c.sum()),
                        'first_support_sqerr':row['R56_first_support_sqerr'],
                        'mean_outcome':float(s.sum()/c.sum()),
                        'delta_B0':row['brier']['R56_full'][0]-.25,
                        'delta_B1':row['brier']['R56_full'][1]-row['brier']['R25'][1],
                        'source_pair_prevalence':cfg['pair_prevalence'],'source_mixture_prevalence':cfg['mixture_prevalence']})
            chunks+=1
        print(json.dumps({'stage':'R65','benchmark':name,'trajectories':len(identities),'seconds':time.perf_counter()-start}),flush=True)
    assert len(identities)==15726
    df=pd.DataFrame(details);ud=pd.DataFrame(updates);fd=pd.DataFrame(features)
    df.to_csv(OUT/'decomposition.csv',index=False);ud.to_csv(OUT/'support_update.csv',index=False);fd.to_csv(OUT/'observable_diagnostics.csv',index=False)
    cols=[c for c in df.select_dtypes('number') if c not in ['fold','seed','budget']]
    bm=df.groupby(['fold','benchmark','method','budget'])[cols].mean().reset_index();bm.to_csv(OUT/'benchmark_decomposition.csv',index=False)
    summary=[];us=[]
    for scope,mask in [('all',np.ones(len(df),bool)),('alignment_faking',df.benchmark.str.startswith('alignment_faking/')),('others',~df.benchmark.str.startswith('alignment_faking/'))]:
        for keys,g in df[mask].groupby(['method','budget']):summary.append({'scope':scope,'method':keys[0],'budget':int(keys[1]),'trajectories':len(g),**g[cols].mean().to_dict()})
        m=np.ones(len(ud),bool) if scope=='all' else ud.benchmark.str.startswith('alignment_faking/')==(scope=='alignment_faking')
        for meth,g in ud[m].groupby('method'):us.append({'scope':scope,'method':meth,**g[['B0','B1','update_cross','update_energy','update_net']].mean().to_dict()})
    corr=[]; signals=FEATURES+['source_standardized_distance','subject_name_seen','R56_mean_p0','R56_mean_uncertainty','first_support_sqerr']
    fb=fd.groupby(['fold','benchmark']).mean(numeric_only=True).reset_index()
    for scope,g in [('all',fb),('others',fb[~fb.benchmark.str.startswith('alignment_faking/')])]+[(f'fold{f}',fb[fb.fold==f]) for f in range(3)]:
        for signal in signals:
            for outcome in ['delta_B0','delta_B1']:
                rho=g[signal].corr(g[outcome],method='spearman') if g[signal].nunique()>1 else None
                corr.append({'scope':scope,'signal':signal,'outcome':outcome,'rho':rho,'benchmarks':len(g)})
    pd.DataFrame(summary).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(corr).to_csv(OUT/'benchmark_correlations.csv',index=False)
    fb.to_csv(OUT/'benchmark_observables.csv',index=False)
    write_json(OUT/'summary.json',{'status':'complete','new_fits':0,'trajectories':len(identities),'chunks':chunks,
        'max_rescore_error':error,'metrics':summary,'support_update':us,'seconds':time.perf_counter()-start,
        'aggregation':'response weighted within pair; pair equal; query summaries diagnostic only',
        'identifiability':'observational only; feature-group ablations remain required'})
    check_freeze(OUT/'contract')

if __name__=='__main__':main()
