"""R70 frozen source coefficients applied to exact R64 outer probabilities."""
import time
import pandas as pd
from lexical_peer_common import *

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R70/full';root.mkdir(exist_ok=True)
    check_freeze(ROOT/'R70/calibration_contract');check_freeze(ROOT/'R62/isolation')
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/lexical_peer_common.py',ROOT/'R70/calibration_contract/contract.json'],{'pairs':5242,'seeds':[0,1,2],'candidates':['independent','monotone','global']})
    rows=[];decomp=[]
    for f in range(3):
        coeff=read(ROOT/f'R70/calibration/fold{f}/coefficients.json')
        for old in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
            folder=root/'outer'/f'fold{f}'/old.name;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():
                rows+=read(folder/'paired_scores.json');decomp+=read(folder/'decomposition.json');continue
            refs={r['identity']:r for r in read(ROOT/f'R62/outer/fold{f}'/old.name/'paired_scores.json')}
            local=[];dc=[]
            for indexpath in sorted(old.glob('index_*.json')):
                suf=indexpath.stem.split('_')[-1];index=read(indexpath);arr=dict(np.load(old/f'predictions_{suf}.npz'));bank={}
                for idx in index:
                    a=idx['start'];b=a+idx['size'];p=arr['R25'][a:b];d=arr['template_average'][a:b]-p
                    preds={k:arr[k][a:b] for k in BASELINES};preds['R64']=p+d
                    for key in ['independent','monotone','global']:preds['R70_'+key]=p+d*np.array(coeff[key])
                    ep={**idx,'counts':arr['counts'][a:b],'successes':arr['successes'][a:b]}
                    row=score_episode(ep,preds);ref=refs[idx['identity']]
                    assert row['brier']['R25']==ref['brier']['R25']
                    for k in ['brier','accuracy']:row[k]['R56_segment']=ref[k]['R56_segment']
                    local.append(row);dc+=decompose(ep,preds)
                    for k,v in preds.items():bank.setdefault(k,[]).append(v)
                for k in ['counts','successes']:bank[k]=[arr[k]]
                np.savez_compressed(folder/f'predictions_{suf}.npz',**{k:np.concatenate(v) for k,v in bank.items()})
                write_json(folder/f'index_{suf}.json',index)
            write_json(folder/'paired_scores.json',local);write_json(folder/'decomposition.json',dc);write_json(folder/'complete.json',{'trajectories':len(local),'utc':now()})
            rows+=local;decomp+=dc;print(json.dumps({'stage':'R70full','fold':f,'benchmark':old.name,'trajectories':len(local),'seconds':time.perf_counter()-start}),flush=True)
    summary=summarize(rows,root,5242);df=pd.DataFrame(decomp);df.to_csv(root/'decomposition.csv',index=False)
    # Explicitly post-hoc diagnostic, never used by the candidate.
    raw=df[df.method=='R64'].groupby('budget')[['A','cost']].mean().reindex(BUDGETS)
    raw['posthoc_oracle_alpha']=np.clip(-raw.A/np.maximum(raw.cost,1e-30),0,1);raw.to_csv(root/'outer_oracle_diagnostic.csv')
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'finished':now()})
    check_freeze(root/'contract');check_freeze(ROOT/'R70/calibration_contract');check_freeze(ROOT/'R62/isolation')
    print(json.dumps({m:v['alc_mean'] for m,v in summary['results'].items()}),flush=True)

if __name__=='__main__':main()
