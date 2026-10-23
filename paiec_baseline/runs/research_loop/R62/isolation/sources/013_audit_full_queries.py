"""R32 full-query measurement contrast; all label-dependent diagnostics stay outside prediction."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from mlp.evaluate import _load_directory,_stable_seed,BUDGETS,ALC_WEIGHTS
from mlp.semantic_features import SemanticFeatures
from mlp.metadata_features import encode
from mlp.run.run_metadata_round import prediction

NAMES=['R25','semantic','smoothed_mean_2','training_selected_smoothed']


def select(census):
    data=pd.read_csv(census,dtype={'subject_id':str});data['size_bin']=np.where(data.evaluation_pool_items<=128,0,np.where(data.evaluation_pool_items<=512,1,2));data['repeat_bin']=(data.repeated_item_groups>0).astype(int)
    out=[]
    for (directory,size,repeat),group in data.groupby(['benchmark','size_bin','repeat_bin'],sort=True):
        n=min(len(group),32 if directory=='alignment_faking/formatted_tables' else 2)
        rng=np.random.default_rng(_stable_seed(3201,directory,f'{size}:{repeat}'));chosen=rng.choice(len(group),n,replace=False)
        for row in group.iloc[np.sort(chosen)].to_dict('records'):
            out.append({**row,'stratum':f'{size}:{repeat}','population_count':len(group),'selected_count':n,'design_weight':len(group)/n})
    return out


def brier(p,m,s):
    return (p*p*m-2*p*s+s).sum(axis=-1)/m.sum()


def evaluate_pair(group,items,vectors,metas,sid,directory,seed,sem,meta,fold):
    ids=pd.unique(group['item_id']).copy();rng=np.random.default_rng(_stable_seed(seed,directory,sid));rng.shuffle(ids);cut=len(ids)//2;aid=ids[:31];eid=ids[cut:]
    acquired=group[group['item_id'].isin(aid)].sample(frac=1,random_state=int(rng.integers(2**31)))
    acquired=acquired.drop_duplicates('item_id').set_index('item_id').loc[aid].reset_index();labels=acquired['response'].to_numpy(float)
    evaluated=group[group['item_id'].isin(eid)].groupby('item_id',sort=False)['response'].agg(['size','sum']).loc[eid]
    m=evaluated['size'].to_numpy(float);s=evaluated['sum'].to_numpy(float)
    x=np.stack([vectors[i] for i in aid])/np.sqrt(sem);q=np.stack([vectors[i] for i in eid])/np.sqrt(sem)
    xm=np.concatenate([x,np.stack([metas[i] for i in aid])/np.sqrt(meta)],axis=1);qm=np.concatenate([q,np.stack([metas[i] for i in eid])/np.sqrt(meta)],axis=1)
    preds={k:[] for k in NAMES}
    for n in BUDGETS:
        preds['R25'].append(prediction(xm[:n],qm,labels[:n],1))
        preds['semantic'].append(prediction(x[:n],q,labels[:n],1))
        for k,alpha in [('smoothed_mean_2',2),('training_selected_smoothed',[2,2,4][fold])]:preds[k].append(np.full(len(eid),(labels[:n].sum()+alpha*.5)/(n+alpha)))
    preds={k:np.asarray(v) for k,v in preds.items()};cap=min(64,len(eid));full={k:brier(v,m,s) for k,v in preds.items()};capped={k:brier(v[:,:cap],m[:cap],s[:cap]) for k,v in preds.items()}
    qbar=s/m;noise=float(np.sum(s*(1-qbar))/m.sum());identity_error=0
    for k,p in preds.items():identity_error=max(identity_error,float(np.max(abs(full[k]-((p-qbar)**2*m).sum(1)/m.sum()-noise))))
    assert identity_error<1e-12
    random=np.random.default_rng(_stable_seed(3202,directory,f'{sid}:{seed}'));draws=[]
    for rep in range(16):
        keep=random.choice(len(eid),cap,replace=False);vals={k:brier(v[:,keep],m[keep],s[keep]) for k,v in preds.items()}
        draws.append(float((vals['R25']-vals['smoothed_mean_2'])@ALC_WEIGHTS))
    return {'seed':seed,'full_items':len(eid),'full_responses':int(m.sum()),'capped_responses':int(m[:cap].sum()),
            'full_brier':{k:v.tolist() for k,v in full.items()},'capped_brier':{k:v.tolist() for k,v in capped.items()},
            'conditional_64_subset_delta_alc':draws,'empirical_within_item_variance':noise,'identity_max_error':identity_error,
            'full_delta_alc':float((full['R25']-full['smoothed_mean_2'])@ALC_WEIGHTS),'capped_delta_alc':float((capped['R25']-capped['smoothed_mean_2'])@ALC_WEIGHTS)}


def main():
    root=Path('runs/research_loop/R32');start=time.time();chosen=select(root/'pair_census.csv');selection=root/'selection.json'
    if selection.exists():assert json.loads(selection.read_text())==chosen
    else:selection.write_text(json.dumps(chosen,indent=2),encoding='utf-8')
    engine=SemanticFeatures();by={}
    for row in chosen:by.setdefault(row['benchmark'],[]).append(row)
    for directory,selected in by.items():
        target=root/(directory.split('/')[0]+'_full_query.json')
        if target.exists():continue
        response,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory);groups=response.groupby('subject_id',sort=False);sidmap={str(s):s for s in groups.groups};ids=set()
        for row in selected:ids.update(groups.get_group(sidmap[row['subject_id']])['item_id'].tolist())
        ids=list(ids);keys={engine.key(items[i]['content']) for i in ids};missing=sum(engine.db.execute('SELECT 1 FROM features WHERE key=?',(key,)).fetchone() is None for key in keys)
        print(directory,'selected pairs',len(selected),'unique items',len(ids),'missing vectors',missing,flush=True)
        vectors={};metas={}
        for offset in range(0,len(ids),512):
            batch=ids[offset:offset+512];encoded=engine.encode_many([items[i]['content'] for i in batch])
            for i in batch:vectors[i]=encoded[engine.key(items[i]['content'])]['semantic'];metas[i]=encode(items[i]['item_features'])
            if offset and offset%4096==0:print(directory,'encoded/read',offset,'/',len(ids),flush=True)
        fold=int(selected[0]['fold']);parameters=json.loads(Path(f'runs/research_loop/R25/fold{fold}_selection.json').read_text());results=[]
        for row in selected:
            sid=sidmap[row['subject_id']];g=groups.get_group(sid)
            runs=[evaluate_pair(g,items,vectors,metas,sid,directory,seed,parameters['semantic_ridge'],parameters['metadata_ridge'],fold) for seed in [0,1,2]]
            result={**row,'seeds':runs};results.append(result)
        target.write_text(json.dumps(results,indent=2),encoding='utf-8');print('completed',directory,flush=True)
    records=[r for directory in by for r in json.loads((root/(directory.split('/')[0]+'_full_query.json')).read_text())]
    full={k:np.zeros(6) for k in NAMES};capped={k:np.zeros(6) for k in NAMES};count=sum(r['design_weight'] for r in records);strata={};bench={}
    for r in records:
        f={k:np.mean([s['full_brier'][k] for s in r['seeds']],axis=0) for k in NAMES};c={k:np.mean([s['capped_brier'][k] for s in r['seeds']],axis=0) for k in NAMES};shift=float(((f['R25']-f['smoothed_mean_2'])-(c['R25']-c['smoothed_mean_2']))@ALC_WEIGHTS)
        strata.setdefault((r['benchmark'],r['stratum']),[]).append((shift,r));w=r['design_weight']
        for k in NAMES:full[k]+=w*f[k];capped[k]+=w*c[k]
        b=bench.setdefault(r['benchmark'],{'population_pairs':0,'selected_pairs':0,'weighted_cap_shift':0});b['population_pairs']+=w;b['selected_pairs']+=1;b['weighted_cap_shift']+=w*shift
    assert round(count)==5242
    variance=0
    for values in strata.values():
        n=len(values);pop=values[0][1]['population_count']
        if n<pop:variance+=pop*pop*(1-n/pop)*np.var([v for v,r in values],ddof=1)/n
    shift=float(((full['R25']-full['smoothed_mean_2'])-(capped['R25']-capped['smoothed_mean_2']))@ALC_WEIGHTS/count);se=np.sqrt(variance)/count
    for b in bench.values():b['estimated_delta_change']=b.pop('weighted_cap_shift')/b['population_pairs']
    d={'scope':'stratified representative full-item contrast; estimated population scores are NOT actual full5242-pair scores','selected_pairs':len(records),'represented_pairs':count,
       'estimated_full_brier':{k:(v/count).tolist() for k,v in full.items()},'estimated_capped_brier':{k:(v/count).tolist() for k,v in capped.items()},
       'estimated_full_alc':{k:float(v/count@ALC_WEIGHTS) for k,v in full.items()},'estimated_capped_alc':{k:float(v/count@ALC_WEIGHTS) for k,v in capped.items()},
       'cap_removal_effect_on_candidate_minus_baseline':shift,'stratified_design_standard_error':se,'approx_design_ci95':[shift-1.96*se,shift+1.96*se],
       'by_benchmark':bench,'elapsed_seconds':time.time()-start,'new_embeddings':engine.encoded,'identity_max_error':max(s['identity_max_error'] for r in records for s in r['seeds'])}
    (root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    sources=['mlp/audit/audit_evaluation_items.py','mlp/audit/audit_full_queries.py','mlp/run/run_metadata_round.py','mlp/metadata_features.py','../research_loop/R32_protocol.md','../research_loop/R32b_protocol.md'];snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in d.items() if k not in ['by_benchmark','estimated_full_brier','estimated_capped_brier']},indent=2),flush=True)


if __name__=='__main__':main()
