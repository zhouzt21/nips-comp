"""Deterministic identity, saved-score, fit-contract and preservation audit."""
import hashlib
import pandas as pd
from historical_positive_common import *
from mlp.evaluate import _load_directory
from mlp.reproduction_episodes import DATA_ROOT

def main():
    audit={};cache={};n=0;error=0.
    for f in range(3):
        refidx={(r['benchmark'],r['subject_id']):r for r in read(ROOT/f'R59/fold{f}_prediction_index.json')}
        scores={(r['benchmark'],r['subject_id']):r for r in read(ROOT/f'R66/fold{f}_paired_scores.json')}
        for mode in ['subject','item']:
            bank=load_npz(ROOT/f'R66/fold{f}_{mode}_predictions.npz')['p']
            for ix in read(ROOT/f'R66/fold{f}_{mode}_index.json'):
                name=ix['benchmark'];key=(name,ix['subject_id']);old=refidx[key]
                if name not in cache:
                    response,_,_=_load_directory(DATA_ROOT,name)
                    cache[name]={str(sid):g for sid,g in response.groupby('subject_id',sort=False)}
                g=cache[name][ix['subject_id']];q=g[g.item_id.astype(str).isin(ix['query_ids'])].copy();q['_variant']=q.interactors.map(str)
                agg=q.groupby(['item_id','_variant'],sort=False,dropna=False).response.agg(['size','sum'])
                groups=[{'item_id':str(i),'interactor_variant_sha256':hashlib.sha256(str(v).encode()).hexdigest()} for i,v in agg.index]
                assert groups==old['query_groups'],key
                assert set(ix['query_ids'])=={r['item_id'] for r in groups}
                assert np.array_equal(agg['size'],ix['counts']) and np.array_equal(agg['sum'],ix['successes'])
                assert ix['support_ids']==old['support_ids'] and np.array_equal(ix['labels'],old['support_labels'])
                assert set(ix['support_ids']).isdisjoint(ix['query_ids'])
                p=bank[ix['start']:ix['start']+ix['size']];c=np.array(ix['counts']);s=np.array(ix['successes'])
                for j in range(6):
                    err=abs(brier(p[:,j],c,s)-scores[key]['brier'][mode+'_full'][j]);error=max(error,err)
                    assert err<1e-12
                    assert abs(accuracy(p[:,j],c,s)-scores[key]['accuracy'][mode+'_full'][j])<1e-12
                n+=1
    audit['R66']={'aligned_trajectories_across_two_ablations':n,'query_item_variant_order':'exact reconstruction equals R59 query_groups',
                  'rescore_max_error':error,'fits':len(list((ROOT/'R66').glob('fold*_inner*/complete.json')))}
    assert n==1648 and audit['R66']['fits']==12
    old=pd.read_csv(ROOT/'R65/decomposition.csv');new=pd.read_csv(ROOT/'R65/descriptor_v2/decomposition.csv')
    assert old.equals(new)
    audit['R65']={'descriptor_repair':'complete','Brier_decomposition_unchanged':True,
        'current_descriptors':'R65/descriptor_v2','historical_descriptor_files':'R65 root retained, superseded for feature claims'}
    print(json.dumps(audit),flush=True)
    write_json(ROOT/'R66/identity_and_rescore_audit.json',audit)
    if not (ROOT/'R67/summary.json').exists():return
    error=0.;n=0;allrows=[]
    for path in sorted((ROOT/'R67/outer').glob('fold*/*/paired_scores.json')):
        rows=read(path);allrows+=rows;lookup={(r['subject_id'],r['seed']):r for r in rows}
        supports={r['identity']:r for r in read(path.parent/'support_priors.json')}
        for file in path.parent.glob('predictions_*.npz'):
            bank=load_npz(file)
            for ix in read(file.with_name(file.name.replace('predictions_','index_')).with_suffix('.json')):
                meta={k:v for k,v in ix.items() if k not in ['identity','start','size']}
                assert hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()==ix['identity']
                a=ix['start'];b=a+ix['size'];c=bank['counts'][a:b];s=bank['successes'][a:b]
                assert np.array_equal(c,ix['counts']) and np.array_equal(s,ix['successes'])
                row=lookup[(ix['subject_id'],ix['seed'])];assert row['identity']==ix['identity']
                for method in row['brier']:
                    p=bank[method][a:b]
                    for j in range(6):
                        err=abs(brier(p[:,j],c,s)-row['brier'][method][j]);error=max(error,err);assert err<1e-12
                        assert abs(accuracy(p[:,j],c,s)-row['accuracy'][method][j])<1e-12
                p=bank['R67_item_segment'][a:b];support=supports[ix['identity']]
                assert np.array_equal(p[:,1],np.clip(p[:,0]+(support['label']-support['prior'])/3,0,1))
                assert np.array_equal(p[:,2:],bank['R25'][a:b,2:]);n+=1
    assert n==15726 and len({(r['benchmark'],r['subject_id']) for r in allrows})==5242
    summary=read(ROOT/'R67/summary.json');computed=aggregate(allrows)
    for m in computed['results']:
        assert abs(computed['results'][m]['alc_mean']-summary['results'][m]['alc_mean'])<1e-12
    audit['R67']={'trajectories':n,'unique_pairs':5242,'rescore_max_error':error,'B1_allowed_support_only':True,'B3_B31_equals_R25':True}
    for folder in [ROOT/'R65/descriptor_v2/contract_v2',ROOT/'R66/contract_v2',ROOT/'R67/contract',ROOT/'R67/evaluation_contract']:check_freeze(folder)
    # Prior delivery identities, including immutable old MLP code and submission fallback.
    old_contract=read(ROOT/'R62/isolation/contract.json')
    for p,h in old_contract['sha256'].items():assert sha(p)==h,p
    audit['preservation']={'R62_frozen_inputs_unchanged':len(old_contract['sha256']),'old_fallback_unchanged':True}
    write_json(ROOT/'R67/validation.json',audit)
    print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
