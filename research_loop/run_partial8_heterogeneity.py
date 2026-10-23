"""R81 C: isolated offset on heldout labels and real-geometry synthetic worlds."""
import time
from partial8_next_common import *

def expected_brier(p,t,w):return float(w@(p*p-2*p*t+t))

def cosine(a,b):
    a=a-a.mean();b=b-b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-20 else None

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R81/heterogeneity';root.mkdir(parents=True,exist_ok=True)
    assert (ROOT/'R81/execution.json').exists()
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/partial8_next_common.py',REPO/'research_loop/PARTIAL8_NEXT_SPEC.md',ROOT/'R77/cohort_heterogeneity.json'],
        {'group':'R77 first four-subject main7501 group','budgets':[15,31],'item_seed':0,'replicates':32,'worlds':['common','heterogeneous'],
         'slope_anchor_sd':.15,'subject_offset_anchor_sd':.15,'paired_uniform_random_numbers_across_worlds':True,'status':'mechanism_only_no_candidate_selection'})
    groups=[g for g in read(ROOT/'R77/cohort_heterogeneity.json') if g['cohort']==7501];engine=CachedSemantic();timings=[]
    for g in groups:
        name=g['benchmark'].split('/')[0];dest=root/name;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():timings.append(read(dest/'complete.json'));continue
        begin=time.perf_counter();f=g['fold'];bm=Benchmark(ROOT/f'R64/full/outer/fold{f}'/name,source(f)['semantic_ridge'],engine);sids=g['subjects']
        realbank={};realindex=[];realrows=[];realoff=0;sim_bank={};sim_index=[];sim_rows=[];simoff=0;worlds=[];geometry={}
        idxmap={v['subject_id']:v for v in bm.indices if v['seed']==0};write_json(dest/'group.json',g)
        for n in [15,31]:
            contexts=[bm.context(s,0,n) for s in sids];anchor=np.concatenate([c['x'] for c in contexts]);ax=anchor-anchor.mean(0)
            for target,sid in enumerate(sids):
                own=contexts[target];other=[c for i,c in enumerate(contexts) if i!=target];key=bm.by[(sid,0)]['subject_key'];ids=bm.by[(sid,0)]['support_ids'][:n]
                for scenario in ['actual','permuted']:
                    peers=other
                    if scenario=='permuted':
                        peers=[{**c,'y':c['y'][np.random.default_rng(stable_seed(f'8103|{bm.name}|{key}|{n}|{i}')).permutation(n)]} for i,c in enumerate(other)]
                    offsets={};risk=[]
                    for rep,h,fit,hold in masks('half',ids,bm.name,key,0):
                        train=subset(own,fit);q=np.vstack([own['x'][hold],anchor]);raw,off=raw_parts([train,*peers],q,partial=True);hn=len(hold)
                        full=np.clip(raw[:hn],0,1);shared=np.clip(raw[:hn]-off[:hn],0,1);y=own['y'][hold]
                        offsets[(rep,h)]=off[hn:];delta=(full-y)**2-(shared-y)**2;risk.extend(delta.tolist())
                        realindex.append({'benchmark':bm.name,'subject_id':sid,'budget':n,'scenario':scenario,'rep':rep,'half':h,'fit_indices':fit.tolist(),'hold_indices':hold.tolist(),
                            'support_ids':ids,'start':realoff,'size':hn});realoff+=hn
                        for k,v in [('partial',full),('shared_component',shared),('truth',y),('raw_offset',off[:hn])]:realbank.setdefault(k,[]).append(v)
                    cs=[cosine(offsets[(r,0)],offsets[(r,1)]) for r in range(2)]
                    realrows.append({'fold':f,'benchmark':bm.name,'subject_id':sid,'budget':n,'scenario':scenario,'offset_validation_delta':float(np.mean(risk)),
                        'offset_repeat_cosine':float(np.mean([v for v in cs if v is not None])) if any(v is not None for v in cs) else None})
                    geometry[f'offsets_{n}_{target}_{scenario}']=np.stack([offsets[(r,h)] for r in range(2) for h in range(2)])
            x,y,_,_=centered_arrays(contexts);beta=x.T@np.linalg.solve(x@x.T+np.eye(len(y)),y);sd=float(np.std(ax@beta))
            rng=np.random.default_rng(stable_seed(f'8104|{bm.name}|{n}'))
            if sd<1e-12:beta=rng.normal(size=x.shape[1]);sd=float(np.std(ax@beta))
            beta=beta*(.15/max(sd,1e-12));offsets=rng.normal(size=(4,x.shape[1]));offsets-=offsets.mean(0)
            offsets*=.15/max(float(np.std(ax@offsets.T)),1e-12)
            for target in range(4):geometry[f'x_{n}_{target}']=contexts[target]['x'];geometry[f'q_{n}_{target}']=bm.query(idxmap[sids[target]])
            for world in ['common','heterogeneous']:
                offsets_true=np.zeros_like(offsets) if world=='common' else offsets;trainps=[];queryps=[];cliprates=[]
                for t,c in enumerate(contexts):
                    base=(1+c['y'].sum())/(2+n);bvec=beta+offsets_true[t];q=geometry[f'q_{n}_{t}'];center=c['x'].mean(0)
                    tr=base+(c['x']-center)@bvec;qr=base+(q-center)@bvec;trainps.append(np.clip(tr,.01,.99));queryps.append(np.clip(qr,.01,.99))
                    cliprates.append(float(np.mean((np.r_[tr,qr]<.01)|(np.r_[tr,qr]>.99))))
                worlds.append({'budget':n,'world':world,'common_beta':beta.tolist(),'offsets':offsets_true.tolist(),'own_bases':[(1+c['y'].sum())/(2+n) for c in contexts],
                    'probability_clip_rates':cliprates,'support_probabilities':[v.tolist() for v in trainps]})
                for rep in range(32):
                    rng=np.random.default_rng(stable_seed(f'8105|{bm.name}|{n}|{rep}'));labels=[(rng.random(n)<p).astype(float) for p in trainps]
                    cc=[{**c,'y':labels[t]} for t,c in enumerate(contexts)]
                    for t,sid in enumerate(sids):
                        original=idxmap[sid];q=geometry[f'q_{n}_{t}'];counts=np.asarray(original['counts'],float);w=counts/counts.sum();truth=queryps[t]
                        p=three(cc[t],[c for i,c in enumerate(cc) if i!=t],q);p['cost']=cost_control(p['own'],p['pooled'],p['partial'],counts)[0]
                        p['point']=np.clip(p['own']+np.sign(p['pooled']-p['own'])*abs(p['partial']-p['own']),0,1)
                        row={'fold':f,'benchmark':bm.name,'subject_id':sid,'budget':n,'world':world,'rep':rep,**{'brier_'+k:expected_brier(v,truth,w) for k,v in p.items()}}
                        sim_rows.append(row);sim_index.append({'original':original,'budget':n,'world':world,'rep':rep,'start':simoff,'size':len(q),'support_labels':[v.tolist() for v in labels]});simoff+=len(q)
                        for k,v in p.items():sim_bank.setdefault(k,[]).append(v)
                        sim_bank.setdefault('truth_probability',[]).append(truth);sim_bank.setdefault('counts',[]).append(counts)
        for filename,bank in [('real_predictions',realbank),('synthetic_predictions',sim_bank),('geometry',geometry)]:
            np.savez_compressed(dest/(filename+'.npz'),**({k:np.concatenate(v) for k,v in bank.items()} if filename!='geometry' else bank))
        write_json(dest/'real_index.json',realindex);write_json(dest/'real_metrics.json',realrows);write_json(dest/'synthetic_index.json',sim_index);write_json(dest/'synthetic_metrics.json',sim_rows);write_json(dest/'worlds.json',worlds)
        done={'benchmark':bm.name,'real_rows':len(realrows),'synthetic_rows':len(sim_rows),'seconds':time.perf_counter()-begin,'peak_working_set_mib':peak_mb(),'finished':now()};write_json(dest/'complete.json',done);timings.append(done);print(json.dumps(done),flush=True)
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'timings':timings,'threads':4,'finished':now()});check_freeze(root/'contract')
if __name__=='__main__':main()
