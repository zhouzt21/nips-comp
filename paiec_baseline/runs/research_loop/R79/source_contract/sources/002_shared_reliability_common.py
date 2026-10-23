"""New plan_9 accounting helpers; frozen predictors imported unchanged."""
import ctypes
from ctypes import wintypes
from shared_slope_data import *
from shared_slope_model import cohort_map,predict,stable_seed

COHORTS=[7501,7502,7701,7702,7703]

def peak_mb():
    class Info(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                  *[(k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]]
    get=ctypes.windll.kernel32.GetCurrentProcess;get.restype=wintypes.HANDLE
    call=ctypes.windll.psapi.GetProcessMemoryInfo
    call.argtypes=[wintypes.HANDLE,ctypes.POINTER(Info),wintypes.DWORD];call.restype=wintypes.BOOL
    info=Info();info.cb=ctypes.sizeof(info)
    assert call(get(),ctypes.byref(info),info.cb),'Windows memory measurement failed'
    assert info.PeakWorkingSetSize>0
    return info.PeakWorkingSetSize/1024**2

def visible_subjects(bm,seed=0):
    return [{'subject_id':v['subject_id'],'subject_key':v['subject_key'],'ambiguous_subject':v['ambiguous_subject']}
            for v in bm.records if v['seed']==seed]

def current_pairs(bm,sid,seed,n,peers,scenario='full'):
    if scenario=='missing':
        peers=[s for s in peers if stable_seed(f'7505|{bm.name}|{bm.by[(sid,seed)]["subject_key"]}|{bm.by[(s,seed)]["subject_key"]}')%2==0]
    if scenario.startswith('peer'):peers=peers[:int(scenario[-1])]
    if scenario=='ambiguous':peers=[]
    pairs=[(s,n//(2**k) if scenario=='uneven' else n) for k,s in enumerate(peers)]
    return [(s,nn) for s,nn in pairs if nn>0]

def source_shrinkage():
    path=ROOT/'R77/source_shrinkage.json'
    if path.exists():return read(path)
    result={}
    for f in range(3):
        cross=np.zeros(6);cost=np.zeros(6);count=0;files=[]
        for folder in sorted((ROOT/f'R75/source_fold{f}').iterdir()):
            if not (folder/'predictions.npz').exists():continue
            p=dict(np.load(folder/'predictions.npz'));files.append({'path':str(folder/'predictions.npz'),'sha256':sha(folder/'predictions.npz')})
            for idx in read(folder/'index.json'):
                a=idx['start'];b=a+idx['size'];w=p['counts'][a:b];w=w/w.sum()
                d=p['pool_both'][a:b]-p['R25'][a:b];v=p['partial8'][a:b]-p['R25'][a:b]
                cross+=(w[:,None]*d*v).sum(0);cost+=(w[:,None]*d*d).sum(0);count+=1
        alpha=np.divide(cross,cost,out=np.ones(6),where=cost>1e-30);alpha=np.clip(alpha,0,1)
        result[str(f)]={'alpha':alpha.tolist(),'cross':cross.tolist(),'cost':cost.tolist(),'source_trajectories':count,'files':files,'query_outcomes_used':False}
    write_json(path,result);return result

def bm_comparison(rows,method,baseline,weights=ALC_WEIGHTS):
    names=sorted({r['benchmark'] for r in rows});counts=[];sums=[];by={}
    for b in names:
        vals=[float((np.array(r['brier'][method])-r['brier'][baseline])@weights) for r in rows if r['benchmark']==b]
        counts.append(len(vals));sums.append(sum(vals));by[b]=float(np.mean(vals))
    counts=np.asarray(counts);sums=np.asarray(sums);draw=np.random.default_rng(20261023).integers(len(names),size=(10000,len(names)))
    boot=sums[draw].sum(1)/counts[draw].sum(1);mask=np.array(['alignment_faking/' in b for b in names])
    return {'delta':float(sums.sum()/counts.sum()),'benchmark_equal_delta':float(np.mean(sums/counts)),
            'cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'fold_delta':[float(np.mean([(np.array(r['brier'][method])-r['brier'][baseline])@weights for r in rows if r['fold']==f])) for f in range(3)],
            'per_budget_delta':np.mean([np.array(r['brier'][method])-r['brier'][baseline] for r in rows],axis=0).tolist(),
            'alignment_contribution':float(sums[mask].sum()/counts.sum()),'others_contribution':float(sums[~mask].sum()/counts.sum()),
            'without_alignment_delta':float(sums[~mask].sum()/counts[~mask].sum()),'by_benchmark':by,
            'worst_benchmark':max(by,key=by.get),'worst_delta':max(by.values()),'regression_ge_005':{b:v for b,v in by.items() if v>=.005}}
