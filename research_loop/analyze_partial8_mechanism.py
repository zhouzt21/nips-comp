"""R81 independent probability replay, five-cohort comparisons and geometry."""
import time
from partial8_next_common import *
from shared_slope_analyze import replay

def cluster_mean(values):
    a=np.asarray(values,float);draw=np.random.default_rng(20261023).integers(len(a),size=(10000,len(a)))
    return {'mean':float(a.mean()),'ci95':np.quantile(a[draw].mean(1),[.025,.975]).tolist(),'benchmarks':len(a)}

def analyze_full(root):
    rows=[];checks=[];frames=[];rawerr=0.;geoerr=0.
    for folder in sorted((root/'outer').glob('fold*/*')):
        rr,c=replay(folder);rows+=rr;checks.append(c);df=pd.read_csv(folder/'geometry.csv');direction_rows=[]
        for ip in sorted(folder.glob('index_*.json')):
            suffix=ip.stem.split('_')[-1];p=dict(np.load(folder/f'predictions_{suffix}.npz'));raw=dict(np.load(folder/f'raw_{suffix}.npz'))
            for name,v in raw.items():rawerr=max(rawerr,float(abs(np.clip(v,0,1)-p['R25' if name=='own' else name]).max()))
            # Enforce the preregistered numerical zero on BOTH vectors. Keep runner CSV immutable.
            for idx in read(ip):
                a=idx['start'];b=a+idx['size'];w=p['counts'][a:b];w=w/w.sum()
                for co in COHORTS:
                    out={}
                    for prefix,o,po,pa in [('',p['R25'][a:b],p[f'pool_{co}'][a:b],p[f'partial_{co}'][a:b]),
                         ('raw_',raw['own'][a:b],raw[f'pool_{co}'][a:b],raw[f'partial_{co}'][a:b])]:
                        d=po-o;e=pa-o;vd=abs(d)>1e-12;ve=abs(e)>1e-12;same=(d*e>0)&vd&ve;flip=(d*e<0)&vd&ve
                        for name,mask in [('same_direction',same),('direction_flip',flip),('same_direction_shrink',same&(abs(e)<=abs(d))),
                            ('same_direction_enlarge',same&(abs(e)>abs(d))),('pool_zero_partial_nonzero',(~vd)&ve)]:out[prefix+name]=w@mask
                    for j,n in enumerate(BUDGETS):direction_rows.append({'subject_id':idx['subject_id'],'seed':idx['seed'],'cohort':co,'budget':int(n),**{k:float(v[j]) for k,v in out.items()}})
        corrected=pd.DataFrame(direction_rows);join=['subject_id','seed','cohort','budget'];fields=[k for k in corrected if k not in join]
        df=df.drop(columns=fields).merge(corrected,on=join,validate='one_to_one');frames.append(df)
        indexed=df.set_index(join).sort_index()
        # Saved risk identity, aggregated by response then pair: ctl-part = alignment + squared change.
        for r in rr:
            for cohort in COHORTS:
                g=indexed.loc[(r['subject_id'],r['seed'],cohort)]
                for kind in ['cost','point']:
                    expected=np.array(r['brier'][f'{kind}_{cohort}'])-r['brier'][f'partial_{cohort}'];actual=(g[kind+'_alignment']+g[kind+'_squared_change']).to_numpy()
                    geoerr=max(geoerr,float(abs(expected-actual).max()))
        print(json.dumps({'replayed':str(folder),'trajectories':len(rr),'max_raw_clip_error':rawerr,'max_risk_identity_error':geoerr}),flush=True)
    assert len(rows)==15726 and rawerr<1e-11 and geoerr<1e-11
    comp={str(c):{b:bm_comparison(rows,f'partial_{c}',f'{b}_{c}') for b in ['pool','shrink','point','cost']} for c in COHORTS}
    for r in rows:
        for k in ['partial','pool','shrink','point','cost']:r['brier'][k+'_avg']=np.mean([r['brier'][f'{k}_{c}'] for c in COHORTS],axis=0).tolist()
    avg={b:bm_comparison(rows,'partial_avg',b+'_avg') for b in ['pool','shrink','point','cost']}
    baseline={str(c):{b:bm_comparison(rows,f'partial_{c}',b) for b in ['R25','smoothed_mean_2','training_selected_smoothed']} for c in COHORTS}
    write_json(root/'comparisons.json',{'cohorts':comp,'cohort_average':avg,'baseline_comparisons':baseline,'average_definition':'within-pair/seed average risks, no ensemble and no best-cohort selection'})
    geo=pd.concat(frames,ignore_index=True);geo.to_csv(root/'geometry.csv',index=False)
    geo.groupby(['cohort','budget']).mean(numeric_only=True).to_csv(root/'geometry_by_cohort_budget.csv')
    geo.groupby(['benchmark','cohort','budget']).mean(numeric_only=True).to_csv(root/'geometry_by_benchmark.csv')
    valid=geo[geo.budget>=3];finite=valid.cost_alpha.notna()
    write_json(root/'geometry_summary.json',{'cost_unreachable_requests':int((~finite).sum()),'cost_total_requests':len(valid),
        'unmatched_cost_mean':float(valid.unmatched_cost.mean()),'max_reachable_cost_error':float(abs(valid.loc[finite,'cost_control_cost']-valid.loc[finite,'partial_cost']).max()),
        'diagnostic_only':True,'source_shrinkage_reused':True,
        'direction_tolerance_note':'Canonical analysis applies abs>1e-12 to both pooled and partial displacement. Immutable runner CSV originally thresholded only pooled in same/flip masks; probabilities and risk metrics unaffected.'})
    write_json(root/'replay_validation.json',{'checks':checks,'trajectories':len(rows),'pairs':5242,'benchmarks':36,'max_brier_error':max(c['max_brier_error'] for c in checks),
        'max_accuracy_error':max(c['max_accuracy_error'] for c in checks),'max_raw_clip_error':rawerr,'max_risk_decomposition_error':geoerr,'finished':now()})
    return avg

def analyze_hetero(root):
    real=[];sim=[];maxreal=maxsim=maxtruth=maxcos=maxpoint=0.;clip=[];label_checks=0
    for folder in sorted((root/'heterogeneity').iterdir()):
        if not (folder/'complete.json').exists():continue
        r=read(folder/'real_metrics.json');s=read(folder/'synthetic_metrics.json');a=dict(np.load(folder/'real_predictions.npz'));p=dict(np.load(folder/'synthetic_predictions.npz'))
        geometry=dict(np.load(folder/'geometry.npz'));group=read(folder/'group.json');worlds=read(folder/'worlds.json');truths={};support_ps={}
        for world in worlds:
            n=world['budget'];slopes=np.asarray(world['offsets'])+np.asarray(world['common_beta']);support_ps[(n,world['world'])]=world['support_probabilities']
            for t,sid in enumerate(group['subjects']):
                x=geometry[f'x_{n}_{t}'];q=geometry[f'q_{n}_{t}'];base=world['own_bases'][t]
                tp=np.clip(base+(x-x.mean(0))@slopes[t],.01,.99);qp=np.clip(base+(q-x.mean(0))@slopes[t],.01,.99)
                maxtruth=max(maxtruth,float(abs(tp-world['support_probabilities'][t]).max()));truths[(sid,n,world['world'])]=qp
        risks={}
        for idx in read(folder/'real_index.json'):
            k=(idx['subject_id'],idx['budget'],idx['scenario']);start=idx['start'];end=start+idx['size'];y=a['truth'][start:end]
            risks.setdefault(k,[]).extend(((a['partial'][start:end]-y)**2-(a['shared_component'][start:end]-y)**2).tolist())
            assert set(idx['support_ids'][i] for i in idx['fit_indices']).isdisjoint(idx['support_ids'][i] for i in idx['hold_indices'])
        for row in r:
            maxreal=max(maxreal,abs(float(np.mean(risks[(row['subject_id'],row['budget'],row['scenario'])]))-row['offset_validation_delta']))
            t=group['subjects'].index(row['subject_id']);off=geometry[f"offsets_{row['budget']}_{t}_{row['scenario']}"];cs=[]
            for i in [0,2]:
                left=off[i]-off[i].mean();right=off[i+1]-off[i+1].mean();den=np.linalg.norm(left)*np.linalg.norm(right)
                if den>1e-20:cs.append(left@right/den)
            if cs:maxcos=max(maxcos,abs(float(np.mean(cs))-row['offset_repeat_cosine']))
            else:assert row['offset_repeat_cosine'] is None
        lookup={(v['subject_id'],v['budget'],v['world'],v['rep']):v for v in s}
        for idx in read(folder/'synthetic_index.json'):
            st=idx['start'];end=st+idx['size'];w=p['counts'][st:end];w=w/w.sum();t=p['truth_probability'][st:end];v=lookup[(idx['original']['subject_id'],idx['budget'],idx['world'],idx['rep'])]
            n=idx['budget'];rng=np.random.default_rng(stable_seed(f"8105|{group['benchmark']}|{n}|{idx['rep']}"))
            labels=[(rng.random(n)<np.asarray(tp)).astype(float) for tp in support_ps[(n,idx['world'])]]
            assert np.array_equal(labels,idx['support_labels']);label_checks+=1
            maxtruth=max(maxtruth,float(abs(t-truths[(idx['original']['subject_id'],n,idx['world'])]).max()))
            expected_point=np.clip(p['own'][st:end]+np.sign(p['pooled'][st:end]-p['own'][st:end])*abs(p['partial'][st:end]-p['own'][st:end]),0,1)
            maxpoint=max(maxpoint,float(abs(expected_point-p['point'][st:end]).max()))
            for m in ['own','pooled','partial','cost','point']:
                pp=p[m][st:end];br=float(w@(pp*pp-2*pp*t+t));maxsim=max(maxsim,abs(br-v['brier_'+m]))
        for v in read(folder/'worlds.json'):clip.append({'benchmark':r[0]['benchmark'],'budget':v['budget'],'world':v['world'],'clip_fraction':float(np.mean(v['probability_clip_rates']))})
        real+=r;sim+=s
    assert max(maxreal,maxsim,maxtruth,maxcos,maxpoint)<1e-11
    rr=pd.DataFrame(real);ss=pd.DataFrame(sim);rr.to_csv(root/'heterogeneity/real_metrics.csv',index=False);ss.to_csv(root/'heterogeneity/synthetic_metrics.csv',index=False)
    result={};realresult={}
    for world,d in ss.groupby('world'):
        result[world]={}
        for n,g in d.groupby('budget'):
            result[world][str(n)]={}
            for other in ['pooled','cost','point','own']:
                delta=(g.brier_partial-g['brier_'+other]);vals=delta.groupby(g.benchmark).mean();result[world][str(n)][other]=cluster_mean(vals)
    for scene,d in rr.groupby('scenario'):
        realresult[scene]={}
        for n,g in d.groupby('budget'):
            z=g.groupby('benchmark').mean(numeric_only=True)
            realresult[scene][str(n)]={'offset_risk':cluster_mean(z.offset_validation_delta),'repeat_cosine':cluster_mean(z.offset_repeat_cosine.dropna())}
    write_json(root/'heterogeneity/summary.json',{'real':realresult,'synthetic':result,'probability_clipping':clip,'synthetic_not_PAIEC_score':True})
    write_json(root/'heterogeneity/replay_validation.json',{'max_real_brier_error':maxreal,'max_expected_brier_error':maxsim,
        'max_generated_probability_error':maxtruth,'max_repeat_cosine_error':maxcos,'max_point_control_error':maxpoint,'paired_uniform_label_regenerations':label_checks,
        'real_metric_rows':len(rr),'synthetic_metric_rows':len(ss),'benchmarks':rr.benchmark.nunique(),'finished':now()})
    return {'real':realresult,'synthetic':result}

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['full','heterogeneity'],required=True);args=p.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R81'
    result=analyze_full(root) if args.stage=='full' else analyze_hetero(root)
    write_json(root/f'analysis_{args.stage}_execution.json',{'seconds':time.perf_counter()-start,'peak_working_set_mib':peak_mb(),'finished':now()});check_freeze(root/'contract')
    print(json.dumps(plain({'stage':args.stage,'result':result})),flush=True)
if __name__=='__main__':main()
