"""Conditional R64 full outer run, only after the frozen source gate passes."""
import hashlib
import time
import pandas as pd
from historical_positive_common import *
from historical_positive_episodes import episodes
from run_historical_template_pilot import LexicalOnly,predict_inputs
from mlp.cohort_semantic import CachedSemantic
from mlp.semantic_features import SemanticFeatures

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4)
    pilot=read(ROOT/'R64/summary.json')
    assert pilot['enter_outer'],'Frozen R64 source entry not satisfied'
    root=ROOT/'R64/full';root.mkdir(parents=True,exist_ok=True)
    check_freeze(ROOT/'R64/pilot_contract');check_freeze(ROOT/'R62/isolation')
    freeze(root/'evaluation_contract',[Path(__file__),REPO/'research_loop/run_historical_template_pilot.py',
        REPO/'research_loop/historical_positive_episodes.py',REPO/'research_loop/historical_positive_common.py',
        ROOT/'R64/summary.json',ROOT/'R64/pilot_contract/contract.json'],
        {'candidate':'template_average','weight':.5,'seeds':[0,1,2],'pairs':5242,'fit':0})
    engine=LexicalOnly();semantic=CachedSemantic();allrows=[];all_decomp=[];timings=[];start=time.perf_counter()
    for f in range(3):
        info=source(f); old=read(ROOT/f'R46/fold{f}_evaluation.json')
        for name in info['heldout_benchmarks']:
            if name not in old['by_benchmark']:continue
            folder=root/'outer'/f'fold{f}'/name.split('/')[0];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                allrows+=read(folder/'paired_scores.json');all_decomp+=read(folder/'decomposition.json')
                timings.append(read(folder/'complete.json'));continue
            assert not list(folder.glob('predictions_*.npz')),'Inspect incomplete R64 benchmark before resume'
            items=pd.read_parquet(Path('D:/paiec_payload/data')/name/'items.parquet',columns=['item_id','content'])
            items=items.drop_duplicates('item_id');lookup={str(r.item_id):r.content for r in items.itertuples(index=False)}
            assert len(lookup)==len(items)
            token_cache={}
            reference={(r['subject_id'],r['seed']):r for r in read(ROOT/f'R62/outer/fold{f}'/name.split('/')[0]/'paired_scores.json')}
            bank={};index=[];offset=0;chunk=0;rows=[];decomp=[];begin=time.perf_counter()
            def flush():
                nonlocal bank,index,offset,chunk
                if not index:return
                np.savez_compressed(folder/f'predictions_{chunk:04d}.npz',**{k:np.concatenate(v) for k,v in bank.items()})
                write_json(folder/f'index_{chunk:04d}.json',index)
                bank={};index=[];offset=0;chunk+=1
            for ep in episodes(name,semantic,info['semantic_ridge'],f):
                identifiers=ep['support_ids']+ep['query_ids']
                needed=list(dict.fromkeys(i for i in identifiers if i not in token_cache))
                if needed:
                    toks=engine.tokenizer([SemanticFeatures.canonical(lookup[i]) for i in needed],
                        add_special_tokens=False,truncation=False,return_attention_mask=False)['input_ids']
                    token_cache.update({i:tuple(t) for i,t in zip(needed,toks)})
                tokens=[token_cache[i] for i in identifiers]
                features=np.r_[ep['support'],ep['query']]
                p,diag=predict_inputs(tokens,features[:,:384]*np.sqrt(info['semantic_ridge']),features[:,384:]*4.,
                    ep['labels'],engine,info['semantic_ridge'],name+'|'+ep['subject_id']+'|'+str(ep['seed']))
                p={m:v[ep['variant_indices']] for m,v in p.items()}
                alpha=info['selected_alpha'];q=len(ep['counts'])
                p['training_selected_smoothed']=np.broadcast_to([(.5*alpha+ep['labels'][:n].sum())/(alpha+n) for n in BUDGETS],(q,6)).copy()
                row=score_episode(ep,p,diag);ref=reference[(ep['subject_id'],ep['seed'])]
                assert row['identity']==ref['identity']
                for m in BASELINES:assert np.array_equal(row['brier'][m],ref['brier'][m])
                c,s=ep['counts'],ep['successes']
                for baseline in ['R25','R46_raw_average']:
                    for j,n in enumerate(BUDGETS):
                        a=p[baseline][:,j];d=p['template_average'][:,j]-a
                        cross=float(2*np.sum((c*a-s)*d)/c.sum());energy=float(np.sum(c*d*d)/c.sum())
                        actual=row['brier']['template_average'][j]-row['brier'][baseline][j]
                        assert abs(actual-cross-energy)<1e-12
                        decomp.append({'fold':f,'benchmark':name,'subject_id':ep['subject_id'],'seed':ep['seed'],
                            'budget':n,'baseline':baseline,'residual_alignment_term':cross,'displacement_cost':energy,'delta_brier':actual})
                rows.append(row)
                for m,v in p.items():bank.setdefault(m,[]).append(v)
                bank.setdefault('counts',[]).append(c);bank.setdefault('successes',[]).append(s)
                index.append({**ep['metadata'],'identity':ep['identity'],'start':offset,'size':q});offset+=q
                if len(index)>=128:flush()
                if len(rows)%600==0:print(json.dumps({'stage':'R64full','benchmark':name,'trajectories':len(rows),'seconds':time.perf_counter()-begin}),flush=True)
            flush();summary=aggregate(rows)
            assert len(rows)//3==old['by_benchmark'][name]['pairs_per_seed'][0]
            errors={}
            for m,prev in [('R25','semantic'),('R46_raw_average','average'),('raw_lexical','lexical')]:
                # Historical R46 stores R25 under semantic; no new tuning of that failed reference.
                v=old['by_benchmark'][name]['brier_mean']
                if prev not in v and m=='R25':prev='R25'
                errors[m]=float(np.max(np.abs(np.array(summary['results'][m]['brier_mean'])-v[prev])))
                assert errors[m]<1e-7,(name,m,errors[m])
            write_json(folder/'paired_scores.json',rows);write_json(folder/'decomposition.json',decomp)
            write_json(folder/'summary.json',summary)
            done={'fold':f,'benchmark':name,'pairs':len(rows)//3,'seconds':time.perf_counter()-begin,
                  'baseline_replay_errors':errors,'finished_utc':now()}
            write_json(folder/'complete.json',done);timings.append(done);allrows+=rows;all_decomp+=decomp
            engine.lexical_cache.clear()
            print(json.dumps({'stage':'R64full',**done}),flush=True)
            assert time.perf_counter()-start<7200
    summary=summarize(allrows,root,expected_pairs=5242)
    pd.DataFrame(all_decomp).to_csv(root/'residual_decomposition.csv',index=False)
    check_freeze(root/'evaluation_contract');check_freeze(ROOT/'R64/pilot_contract');check_freeze(ROOT/'R62/isolation')
    write_json(root/'execution.json',{'seconds_this_invocation':time.perf_counter()-start,'benchmark_timings':timings,'finished_utc':now()})
    print(json.dumps({'stage':'R64full','alc':{m:v['alc_mean'] for m,v in summary['results'].items()}}),flush=True)

if __name__=='__main__':main()
