"""Stateless current-label operators for R75/R76. No outcome-dependent selection."""
import hashlib
import numpy as np
from peer_current_predictor import joint

def stable_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')

def cohort_map(records, seed):
    unique={v['subject_id']:v for v in records}
    eligible=sorted([s for s,v in unique.items() if not v['ambiguous_subject']],key=lambda s:unique[s]['subject_key'])
    rng=np.random.default_rng(stable_seed(f'{seed}|{records[0]["benchmark"]}'))
    eligible=[eligible[i] for i in rng.permutation(len(eligible))]
    result={s:[] for s in unique}
    for start in range(0,len(eligible),4):
        group=eligible[start:start+4]
        for sid in group:result[sid]=[p for p in group if p!=sid]
    return result

def transform(context, query, semantic=False):
    if not semantic:return context,query
    return [{**c,'x':c['x'][:,:384]} for c in context],query[:,:384]

def centered_arrays(context, global_center=False):
    if global_center:
        x=np.concatenate([c['x'] for c in context]);y=np.concatenate([c['y'] for c in context])
        center=x.mean(0)
        return x-center,y-y.mean(),center,np.concatenate([np.full(len(c['y']),i) for i,c in enumerate(context)])
    xs=[];ys=[];ids=[]
    for i,c in enumerate(context):
        if not len(c['y']):continue
        xs.append(c['x']-c['x'].mean(0));ys.append(c['y']-c['y'].mean());ids.append(np.full(len(c['y']),i))
    return np.concatenate(xs),np.concatenate(ys),context[0]['x'].mean(0),np.concatenate(ids)

def predict(context,query,ridge=1.,semantic=False,global_center=False,partial=False):
    context,query=transform(context,query,semantic)
    own=context[0]['y'];base=(1+own.sum())/(2+len(own))
    if not len(own):return np.full(len(query),base)
    x,y,center,ids=centered_arrays(context,global_center)
    kernel=x@x.T
    if partial:kernel*=1+(ids[:,None]==ids[None,:])/8.
    coef=np.linalg.solve(kernel+ridge*np.eye(len(y)),y)
    cross=(query-center)@x.T
    if partial:cross*=1+(ids==0)/8.
    return np.clip(base+cross@coef,0,1)

def heterogeneity(context, reps=32, seed=7503):
    """Cross-half repeatability compared with fitted common-slope Bernoulli noise."""
    anchor=np.concatenate([c['x'] for c in context]);anchor=anchor-anchor.mean(0)
    # Fixed design projection, reused for observed and simulated labels.
    operators=[]
    for c in context:
        halves=[]
        for parity in [0,1]:
            x=c['x'][parity::2];x=x-x.mean(0)
            halves.append(anchor@x.T@np.linalg.inv(x@x.T+np.eye(len(x))))
        operators.append(halves)
    def stat(labels):
        z=[]
        for ops,y in zip(operators,labels):
            z.append([op@(y[k::2]-y[k::2].mean()) for k,op in enumerate(ops)])
        z=np.array(z)
        h=np.mean([np.mean((z[i,0]-z[j,0])*(z[i,1]-z[j,1])) for i in range(len(z)) for j in range(i)])
        repeat=np.mean(z[:,0]*z[:,1])
        cosine=[float(z[i,0]@z[i,1]/max(np.linalg.norm(z[i,0])*np.linalg.norm(z[i,1]),1e-30)) for i in range(len(z))]
        return float(h),float(repeat),cosine
    labels=[c['y'] for c in context];observed,repeat,cos=stat(labels)
    x,y,_,_=centered_arrays(context);beta=x.T@np.linalg.solve(x@x.T+np.eye(len(y)),y)
    ps=[np.clip(c['y'].mean()+(c['x']-c['x'].mean(0))@beta,.01,.99) for c in context]
    rng=np.random.default_rng(seed)
    null=[stat([rng.binomial(1,p).astype(float) for p in ps])[0] for _ in range(reps)]
    threshold=float(np.quantile(null,.95))
    return {'H':observed,'repeatability':repeat,'half_cosines':cos,'null_H':null,'null95':threshold,
            'p_upper':float((1+sum(v>=observed for v in null))/(reps+1)),
            'positive_heterogeneity':bool(observed>threshold and repeat>0),'n_subjects':len(context),'support_each':31}
