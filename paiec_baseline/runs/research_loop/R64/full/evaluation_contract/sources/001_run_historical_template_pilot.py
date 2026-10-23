"""R64: one frozen lexical component replacement, with exact residual diagnostics."""
import time
from historical_positive_common import *
from mlp.cohort_semantic import CachedSemantic
from mlp.context_features import ContextFeatures
from mlp.context_tokens import template,retained,window
from mlp.context_episodes import episodes
from mlp.reproduction_episodes import source_episodes
from mlp.run.run_metadata_round import prediction
from mlp.evaluate import _stable_seed

class LexicalOnly:
    """Reuse the original lexical method verbatim; no semantic model initialization."""
    lexical=ContextFeatures.lexical
    def __init__(self):
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained('D:/paiec_pretrained/all-MiniLM-L6-v2',
            local_files_only=True,clean_up_tokenization_spaces=False)
        self.tokenizer.model_max_length=10**9
        self.lexical_cache={}

def predict_inputs(tokens,semantic,metadata,labels,engine,ridge,permutation_seed):
    """Template depends on current support inputs only; no target outcomes."""
    q=len(tokens)-31; semantic=np.c_[semantic/np.sqrt(ridge),metadata/4.]
    raw=np.c_[np.stack([engine.lexical(window(t)) for t in tokens]),metadata/4.]
    output={m:np.zeros((q,6)) for m in ['R25','smoothed_mean_2','template_lexical',
        'raw_lexical','R46_raw_average','template_average']+[f'permuted_template_{s}' for s in range(4)]}
    changed=[]; active=[]; clip_counts=[]; mean_errors=[]
    for j,n in enumerate(BUDGETS):
        rule=template(tokens[:n])
        chosen=[retained(t,rule) for t in tokens[:n]+tokens[31:]]
        indices=[*range(n),*range(31,len(tokens))]
        trim=np.c_[np.stack([engine.lexical(t) for t in chosen]),metadata[indices]/4.]
        p=prediction(semantic[:n],semantic[31:],labels[:n],1.)
        a=prediction(raw[:n],raw[31:],labels[:n],1.)
        b=prediction(trim[:n],trim[n:],labels[:n],1.)
        output['R25'][:,j]=p;output['raw_lexical'][:,j]=a;output['template_lexical'][:,j]=b
        output['R46_raw_average'][:,j]=.5*(p+a);output['template_average'][:,j]=.5*(p+b)
        output['smoothed_mean_2'][:,j]=(1+labels[:n].sum())/(n+2)
        changed.append(sum(t!=window(o) for t,o in zip(chosen[n:],tokens[31:])))
        active.append(rule is not None)
        clips=[];errors=[]
        # Item-index permutation preserves unweighted episode mean of the residual.
        # Clip only to enforce legal diagnostic probabilities; record any resulting change.
        for s in range(4):
            rng=np.random.default_rng(_stable_seed(6401+s,permutation_seed,str(n)))
            delta=(b-p)*.5; perm=delta[rng.permutation(q)]
            assert abs(perm.mean()-delta.mean())<1e-12
            before=p+perm; after=np.clip(before,0,1)
            output[f'permuted_template_{s}'][:,j]=after
            clips.append(int(np.sum(after!=before)));errors.append(float(after.mean()-before.mean()))
        clip_counts.append(clips);mean_errors.append(errors)
    return output,{'active_template':active,'changed_queries':changed,
                   'permutation_clipped_count':clip_counts,'permutation_clip_mean_change':mean_errors}

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4)
    root=ROOT/'R64';root.mkdir(parents=True,exist_ok=True)
    assert (ROOT/'R63/summary.json').exists(),'R63 must finish first'
    check_freeze(ROOT/'R62/isolation')
    paths=[Path(__file__),REPO/'research_loop/historical_positive_common.py',
           REPO/'research_loop/HISTORICAL_POSITIVE_FOLLOWUP_SPEC.md']
    paths += [BASE/f'mlp/{s}.py' for s in ['context_features','context_tokens','context_episodes','model','reproduction_episodes']]
    paths += [ROOT/f'R42/fold{f}_source.json' for f in range(3)]
    freeze(root/'pilot_contract',paths,{'weight':.5,'gate':'2of3_R25_gain_and_pooled_pair_BM_and_raw_average_gain',
        'permutation':'item-uniform residual mean preserved before clipping; report clipping',
        'seed':73,'pair_cap':25,'new_fits':0,'new_embeddings':0})
    engine=LexicalOnly();old_engine=CachedSemantic();allrows=[];decomposition=[];pools=[];start=time.perf_counter()
    for f in range(3):
        info=source(f); rows=[];folder=root/f'pool{f}';folder.mkdir(exist_ok=True)
        for name in sorted(sum(info['inner_halves'],[])):
            dest=folder/name.split('/')[0];dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists():
                rows+=read(dest/'paired_scores.json');decomposition+=read(dest/'decomposition.json');continue
            data=episodes(name,engine.tokenizer,old_engine)
            ids={e['subject_id']:e for e in source_episodes(name,old_engine)}
            records=[];decomp=[];bank={};index=[];offset=0
            reference={r['subject_id']:r for r in read(ROOT/'R42'/f'{name.split("/")[0]}.json')['configs'][str(float(info['semantic_ridge']))]['pairs']}
            for ep in data:
                sid=ep['subject_id'];old=ids[sid]
                for k in ['labels','counts','successes']:assert np.array_equal(ep[k],old[k])
                p,diag=predict_inputs(ep['tokens'],ep['old_semantic'],ep['metadata'],ep['labels'],engine,
                                     info['semantic_ridge'],name+'|'+sid)
                meta={'fold':f,'benchmark':name,'subject_id':sid,'seed':73,
                    'support_ids':old['support_item_ids'],'query_ids':old['query_item_ids'],
                    'labels':ep['labels'].tolist(),'counts':ep['counts'].tolist(),'successes':ep['successes'].tolist()}
                ident=__import__('hashlib').sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
                row=score_episode({**ep,**{k:meta[k] for k in ['fold','benchmark','subject_id','seed']},'identity':ident},p,diag)
                for new,prev in [('R25','old_semantic'),('raw_lexical','old_lexical'),('template_lexical','context_lexical')]:
                    assert np.max(np.abs(np.array(row['brier'][new])-reference[sid]['brier'][prev]))<1e-7
                c,s=ep['counts'],ep['successes']
                for baseline in ['R25','R46_raw_average']:
                    for j,n in enumerate(BUDGETS):
                        a=p[baseline][:,j];d=p['template_average'][:,j]-a
                        cross=float(2*np.sum((c*a-s)*d)/c.sum())
                        energy=float(np.sum(c*d*d)/c.sum())
                        actual=row['brier']['template_average'][j]-row['brier'][baseline][j]
                        assert abs(actual-cross-energy)<1e-12
                        decomp.append({'fold':f,'benchmark':name,'subject_id':sid,'budget':n,'baseline':baseline,
                            'residual_alignment_term':cross,'displacement_cost':energy,'delta_brier':actual})
                records.append(row)
                for method,probs in p.items():bank.setdefault(method,[]).append(probs)
                bank.setdefault('counts',[]).append(c);bank.setdefault('successes',[]).append(s)
                index.append({**meta,'identity':ident,'start':offset,'size':len(c)})
                offset+=len(c)
            np.savez_compressed(dest/'predictions.npz',**{k:np.concatenate(v) for k,v in bank.items()})
            write_json(dest/'index.json',index);write_json(dest/'paired_scores.json',records)
            write_json(dest/'decomposition.json',decomp);write_json(dest/'complete.json',{'pairs':len(records),'utc':now()})
            rows+=records;decomposition+=decomp;engine.lexical_cache.clear()
            print(json.dumps({'stage':'R64','pool':f,'benchmark':name,'pairs':len(records),'seconds':time.perf_counter()-start}),flush=True)
        assert len(rows)==info['training_pairs']
        summary=summarize(rows,folder,expected_pairs=info['training_pairs'])
        assert abs(summary['results']['R25']['alc_mean']-info['baseline_summary']['results']['R25']['alc_mean'])<1e-7
        pools.append(summary);allrows+=rows
    # Pools overlap. This aggregation is a descriptive average of source exposures, not an independent test.
    combined=aggregate(allrows)
    delta=[s['results']['template_average']['alc_mean']-s['results']['R25']['alc_mean'] for s in pools]
    pooled=combined['results']['template_average']['alc_mean']-combined['results']['R25']['alc_mean']
    bm=combined['benchmark_equal_alc']['template_average']-combined['benchmark_equal_alc']['R25']
    raw=combined['results']['template_average']['alc_mean']-combined['results']['R46_raw_average']['alc_mean']
    import pandas as pd
    df=pd.DataFrame(decomposition);df.to_csv(root/'residual_decomposition.csv',index=False)
    align=df[(df.baseline=='R25')].groupby('budget').residual_alignment_term.mean().reindex(BUDGETS).to_numpy()@ALC_WEIGHTS
    passed=bool(sum(d<0 for d in delta)>=2 and pooled<0 and bm<=0 and raw<0 and align<0)
    decision={'source_pool_delta_R25':delta,'pooled_exposure_delta_R25':pooled,'pooled_BM_delta_R25':bm,
              'pooled_delta_raw_average':raw,'residual_alignment_ALC':float(align),
              'enter_outer':passed,'outer_status':'required' if passed else 'condition_not_met',
              'source_exposures':len(allrows),'distinct_pairs':len({(r['benchmark'],r['subject_id']) for r in allrows}),
              'new_fits':0,'new_embeddings':0,'seconds':time.perf_counter()-start,
              'scope':'overlapping, repeatedly used source-development pools; no OOD conclusion',
              'combined':combined}
    write_json(root/'summary.json',decision)
    check_freeze(root/'pilot_contract');check_freeze(ROOT/'R62/isolation')
    print(json.dumps({k:v for k,v in decision.items() if k!='combined'}),flush=True)

if __name__=='__main__':main()
