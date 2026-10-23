"""R71 full frozen reliability experiment with honest support permutation controls."""
import csv, time
import pandas as pd
from lexical_peer_common import *
from historical_positive_episodes import episodes
from run_historical_template_pilot import LexicalOnly
from mlp.cohort_semantic import CachedSemantic
from mlp.semantic_features import SemanticFeatures

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R71/full';root.mkdir(parents=True,exist_ok=True)
    check_freeze(ROOT/'R70/calibration_contract');check_freeze(ROOT/'R62/isolation')
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/lexical_peer_common.py',ROOT/'R70/calibration_contract/contract.json',
        REPO/'research_loop/historical_positive_episodes.py'],{'pairs':5242,'seeds':[0,1,2],'support_permutations':4,'energy_controls':4})
    engine=LexicalOnly();semantic=CachedSemantic();allrows=[];timings=[]
    for f in range(3):
        info=source(f);r=info['semantic_ridge'];coeff=read(ROOT/f'R70/calibration/fold{f}/coefficients.json')
        for old in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
            name=old.name+'/formatted_tables';folder=root/'outer'/f'fold{f}'/old.name;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():allrows+=read(folder/'paired_scores.json');timings.append(read(folder/'complete.json'));continue
            assert not list(folder.glob('predictions_*.npz')),'Inspect incomplete R71 benchmark before restart'
            begin=time.perf_counter();items=pd.read_parquet(Path('D:/paiec_payload/data')/name/'items.parquet',columns=['item_id','content']).drop_duplicates('item_id')
            lookup={str(v.item_id):v.content for v in items.itertuples(index=False)};tok={}
            refs={v['identity']:v for v in read(ROOT/f'R62/outer/fold{f}'/old.name/'paired_scores.json')}
            oldrefs={v['identity']:v for v in read(old/'paired_scores.json')}
            rows=[];bank={};index=[];offset=0;chunk=0;max_energy=0.;replay=0.
            def flush():
                nonlocal bank,index,offset,chunk
                if not index:return
                np.savez_compressed(folder/f'predictions_{chunk:04d}.npz',**{k:np.concatenate(v) for k,v in bank.items()})
                write_json(folder/f'index_{chunk:04d}.json',index);bank={};index=[];offset=0;chunk+=1
            with (folder/'decomposition.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=None
                for ep in episodes(name,semantic,r,f):
                    ids=ep['support_ids']+ep['query_ids'];needed=list(dict.fromkeys(i for i in ids if i not in tok))
                    if needed:
                        tokenized=engine.tokenizer([SemanticFeatures.canonical(lookup[i]) for i in needed],add_special_tokens=False,truncation=False,return_attention_mask=False)['input_ids']
                        tok.update({i:tuple(t) for i,t in zip(needed,tokenized)})
                    features=np.r_[ep['support'],ep['query']]
                    rng_id=name+'|'+ep['subject_id']+'|'+str(ep['seed'])
                    base=predict_bank([tok[i] for i in ids],features[:,:384]*np.sqrt(r),features[:,384:]*4.,ep['labels'],engine,[r],rng_id)
                    base={k:v[ep['variant_indices']] for k,v in base.items()}
                    preds,error=full_predictions(base,r,coeff,ep['counts'],rng_id);max_energy=max(max_energy,error)
                    for m,alpha in [('smoothed_mean_2',2.),('training_selected_smoothed',info['selected_alpha'])]:
                        preds[m]=np.broadcast_to([(.5*alpha+sum(ep['labels'][:n]))/(alpha+n) for n in BUDGETS],preds['R25'].shape).copy()
                    row=score_episode(ep,preds,{'random_energy_error':error});ref=refs[ep['identity']];previous=oldrefs[ep['identity']]
                    for m,orig in [('R25','R25'),('R64','template_average')]:replay=max(replay,float(np.max(np.abs(np.array(row['brier'][m])-previous['brier'][orig]))))
                    assert replay<1e-12,(name,replay)
                    for k in ['brier','accuracy']:row[k]['R56_segment']=ref[k]['R56_segment']
                    rows.append(row)
                    diagnostic={k:v for k,v in preds.items() if not k.startswith('perm')}
                    dc=decompose(ep,diagnostic)
                    if writer is None:writer=csv.DictWriter(stream,fieldnames=list(dc[0]));writer.writeheader()
                    writer.writerows(dc)
                    for k,v in preds.items():bank.setdefault(k,[]).append(v)
                    for k in ['counts','successes']:bank.setdefault(k,[]).append(ep[k])
                    bank.setdefault('neighbor_agreement',[]).append(base['z_'+rk(r)])
                    q=len(ep['counts']);index.append({**ep['metadata'],'identity':ep['identity'],'start':offset,'size':q});offset+=q
                    if len(index)>=128:flush()
                    if len(rows)%600==0:print(json.dumps({'stage':'R71full','benchmark':name,'trajectories':len(rows),'seconds':time.perf_counter()-begin}),flush=True)
                flush()
            write_json(folder/'paired_scores.json',rows)
            done={'benchmark':name,'fold':f,'trajectories':len(rows),'seconds':time.perf_counter()-begin,'R64_replay_error':replay,'energy_match_error':max_energy,'finished':now()}
            write_json(folder/'complete.json',done);timings.append(done);allrows+=rows;engine.lexical_cache.clear()
            print(json.dumps({'stage':'R71full',**done}),flush=True)
            assert time.perf_counter()-start<5400,'R71 predeclared time cap reached'
    summary=summarize(allrows,root,5242)
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'benchmark_timings':timings,'finished':now()})
    check_freeze(root/'contract');check_freeze(ROOT/'R70/calibration_contract');check_freeze(ROOT/'R62/isolation')
    print(json.dumps({m:v['alc_mean'] for m,v in summary['results'].items() if not m.startswith('perm')}),flush=True)

if __name__=='__main__':main()
