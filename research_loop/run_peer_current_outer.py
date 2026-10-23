"""R72 full conditional-simulation prediction; selection frozen before scores."""
import time,hashlib
import pandas as pd
from historical_positive_common import *
from peer_current_predictor import predict_current, r36_family
from mlp.cohort_semantic import CachedSemantic
from mlp.metadata_features import encode as meta_encode
from mlp.evaluate import _stable_seed
from mlp.conditional_components import components

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R72/full';root.mkdir(parents=True,exist_ok=True)
    check_freeze(ROOT/'R72/coverage_contract');check_freeze(ROOT/'R62/isolation')
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/peer_current_predictor.py',REPO/'research_loop/R72_OPERATOR_CONTRACT.md',
        ROOT/'R72/coverage_summary.json',REPO/'research_loop/PAIEC_RUNTIME_EVIDENCE_CONTRACT_CLIENT_ADDENDUM.json',
        *[ROOT/f'R20/fold{f}_selection.json' for f in range(3)]],{'scope':'conditional_simulation','pairs':5242,'seeds':[0,1,2],'permutations':2,'tau':4.})
    engine=CachedSemantic();allrows=[];timings=[]
    for f in range(3):
        info=source(f);r=info['semantic_ridge'];r20=read(ROOT/f'R20/fold{f}_selection.json')['selected_ridge']['pooled']
        for old in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
            name=old.name+'/formatted_tables';folder=root/'outer'/f'fold{f}'/old.name;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():allrows+=read(folder/'paired_scores.json');timings.append(read(folder/'complete.json'));continue
            assert not list(folder.glob('predictions_*.npz')),'Inspect incomplete R72 benchmark before restart'
            begin=time.perf_counter();records=read(ROOT/'R72/contexts'/old.name/'index.json')
            ctxmap={(v['subject_id'],v['seed']):v for v in records}
            # Only support outcomes are retained in the peer context map.
            for ip in sorted(old.glob('index_*.json')):
                for v in read(ip):ctxmap[(v['subject_id'],v['seed'])]['labels']=np.array(v['support_labels'],float)
            it=pd.read_parquet(Path('D:/paiec_payload/data')/name/'items.parquet',columns=['item_id','content','item_features']).drop_duplicates('item_id')
            items={str(v.item_id):(v.content,v.item_features) for v in it.itertuples(index=False)};vectors={}
            def vector(i):
                if i not in vectors:
                    content,meta=items[i];vectors[i]=np.r_[engine.vector(content)/np.sqrt(r),meta_encode(meta)/4.]
                return vectors[i]
            refs={v['identity']:v for v in read(ROOT/f'R62/outer/fold{f}'/old.name/'paired_scores.json')};rows=[];max_r36=0.;checked=set();nopeer_count=0
            for ip in sorted(old.glob('index_*.json')):
                suf=ip.stem.split('_')[-1];index=read(ip);arr=dict(np.load(old/f'predictions_{suf}.npz'));bank={}
                for idx in index:
                    a=idx['start'];b=a+idx['size'];p=arr['R25'][a:b];rowctx=ctxmap[(idx['subject_id'],idx['seed'])]
                    cohort=[rowctx]+[ctxmap[(sid,idx['seed'])] for sid in rowctx['peer_ids']]
                    assert all(v['subject_key']!=rowctx['subject_key'] for v in cohort[1:])
                    assert len({v['subject_key'] for v in cohort})==len(cohort)
                    for v in cohort:assert set(v['support_ids']).isdisjoint(v['query_ids'])
                    xx=[np.stack([vector(i) for i in v['support_ids']]) for v in cohort]
                    query=np.stack([vector(idx['query_ids'][i]) for i in idx['query_variant_item_indices']])
                    contexts=[[{'x':x[:n],'y':v['labels'][:n],'keys':v['support_keys'][:n]} for x,v in zip(xx,cohort)] for n in BUDGETS]
                    permutations=[]
                    for k in range(2):
                        orders=[]
                        for n in BUDGETS:
                            orders.append([np.random.default_rng(_stable_seed(7201+k,name,f'{idx["subject_id"]}|{v["subject_id"]}|{idx["seed"]}|{n}')).permutation(n) for v in cohort[1:]])
                        permutations.append(orders)
                    preds={'R25':p,**predict_current(p,query,contexts,rowctx['query_keys'],r,r20,permutations)}
                    for m in ['smoothed_mean_2','training_selected_smoothed']:preds[m]=arr[m][a:b]
                    assert all(np.array_equal(v[:,0],p[:,0]) for v in preds.values())
                    if len(cohort)==1:
                        nopeer_count+=1
                        assert all(np.array_equal(v,p) for m,v in preds.items() if m not in ['smoothed_mean_2','training_selected_smoothed'])
                    elif idx['seed'] not in checked:
                        checked.add(idx['seed']);ctx=contexts[-1]
                        oldctx=[{'key':v['subject_key'],'labels':c['y'],'vectors':c['x'][:,:384]*np.sqrt(r)} for c,v in zip(ctx,cohort)]
                        original=components(oldctx,[query[:,:384]*np.sqrt(r)]*len(oldctx),r20,r20,r20,np.random.default_rng(72),permutations=0)[0]
                        for new,base in [('R36_own_pooled3','base'),('R20_peer_pooled3','peer')]:
                            max_r36=max(max_r36,float(np.max(np.abs(preds[new][:,-1]-np.clip(original[base]+original['pooled'],0,1)))))
                        assert max_r36<1e-12,max_r36
                    ep={**idx,'counts':arr['counts'][a:b],'successes':arr['successes'][a:b]};row=score_episode(ep,preds)
                    ref=refs[idx['identity']];assert row['brier']['R25']==ref['brier']['R25']
                    for k in ['brier','accuracy']:row[k]['R56_segment']=ref[k]['R56_segment']
                    rows.append(row)
                    for k,v in preds.items():bank.setdefault(k,[]).append(v)
                    if len(rows)%600==0:print(json.dumps({'stage':'R72full','benchmark':name,'trajectories':len(rows),'seconds':time.perf_counter()-begin}),flush=True)
                for k in ['counts','successes']:bank[k]=[arr[k]]
                np.savez_compressed(folder/f'predictions_{suf}.npz',**{k:np.concatenate(v) for k,v in bank.items()})
                write_json(folder/f'index_{suf}.json',index)
            write_json(folder/'paired_scores.json',rows)
            done={'fold':f,'benchmark':name,'trajectories':len(rows),'nopeer_trajectories':nopeer_count,'R36_formula_replay_error':max_r36,'seconds':time.perf_counter()-begin,'finished':now()}
            write_json(folder/'complete.json',done);timings.append(done);allrows+=rows
            print(json.dumps({'stage':'R72full',**done}),flush=True)
            assert time.perf_counter()-start<5400,'R72 90-minute cap'
    summary=summarize(allrows,root,5242);write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'benchmark_timings':timings,'finished':now(),'scope':'conditional_simulation'})
    check_freeze(root/'contract');check_freeze(ROOT/'R72/coverage_contract');check_freeze(ROOT/'R62/isolation')
    print(json.dumps({m:v['alc_mean'] for m,v in summary['results'].items()}),flush=True)

if __name__=='__main__':main()
