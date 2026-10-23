"""Bounded additional checks on real source geometry, kept outside frozen runners."""
from shared_slope_data import *
from shared_slope_model import predict,cohort_map
from peer_current_predictor import r36_family,joint

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);engine=CachedSemantic();folders=outer_folders();records=[]
    for f in range(3):
        selection=next(p for p in sorted((ROOT/f'R75/source_fold{f}').glob('*/selection.json')) if any(read(p)['cohorts'].values()));meta=read(selection);name=meta['source_benchmark'];bm=Benchmark(folders[name.split('/')[0]],meta['ridge_r25'],engine)
        sid=next(s for s in meta['subjects'] if meta['cohorts'][s]);idx=next(v for v in bm.indices if v['subject_id']==sid and v['seed']==0)
        query=bm.query(idx);errors=[]
        for n in [1,3,7,15,31]:
            ctx=[bm.context(s,0,n) for s in [sid,*meta['cohorts'][sid]]]
            r36=r36_family(ctx,query,np.sqrt(meta['ridge_r25']),meta['ridge_r36_original'])[0]
            scaled=predict(ctx,query,semantic=True,ridge=meta['ridge_r36_original']/meta['ridge_r25'])
            err=float(np.max(np.abs(r36-scaled)));joint_err=float(np.max(np.abs(joint(ctx,query)-predict(ctx,query))));assert max(err,joint_err)<1e-12
            errors.append({'budget':n,'original_R36_vs_scaled_sem_error':err,'original_R72_vs_new_pooled_error':joint_err})
        records.append({'source_fold':f,'benchmark':name,'ridge_r25':meta['ridge_r25'],'ridge_r36_original':meta['ridge_r36_original'],'checks':errors})
    write_json(ROOT/'R76/smoke/real_source_equivalence.json',{'status':'passed','records':records,'finished':now()});print(json.dumps(records))

if __name__=='__main__':main()
