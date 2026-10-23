"""New plan10 helpers; old scientific source files stay immutable."""
from shared_reliability_common import *
from shared_slope_model import centered_arrays
from support_sharing_diagnostic import split_masks

def subset(c,idx):
    return {'x':c['x'][idx],'y':c['y'][idx],'keys':[c['keys'][int(i)] for i in idx]}

def three(own,peers,q):
    a=predict([own],q)
    return {'own':a,'pooled':predict([own,*peers],q) if peers else a.copy(),
            'partial':predict([own,*peers],q,partial=True) if peers else a.copy()}

def masks(kind,ids,bm,key,seed):
    n=len(ids);groups=list(dict.fromkeys(ids));ng=len(groups)
    if kind=='half' and ng==n:
        yield from split_masks(n,bm,key,seed);return
    if kind=='loio':
        for i,g in enumerate(groups):
            hold=np.flatnonzero(np.asarray(ids)==g);fit=np.flatnonzero(np.asarray(ids)!=g)
            yield 0,i,fit,hold
        return
    k=2 if kind=='half' else min(5,ng)
    for rep in range(2):
        s=(7801 if kind=='half' else 8001)+rep
        order=np.random.default_rng(stable_seed(f'{s}|{bm}|{key}|{seed}|{n}')).permutation(ng)
        for j in range(k):
            keep={groups[int(i)] for i in order[j::k]};hold=np.array([i for i,x in enumerate(ids) if x in keep]);fit=np.array([i for i,x in enumerate(ids) if x not in keep])
            assert set(ids[i] for i in fit).isdisjoint(ids[i] for i in hold)
            yield rep,j,fit,hold

def conditional_interval(values,ids,seed):
    groups=list(dict.fromkeys(ids));v=np.array([np.mean(values[np.asarray(ids)==g]) for g in groups])
    rng=np.random.default_rng(seed);draw=rng.integers(len(v),size=(1000,len(v)))
    b=v[draw].mean(1)
    return {'lo':float(np.quantile(b,.025)),'hi':float(np.quantile(b,.975)),'bootstrap_positive_fraction':float((b>0).mean()),
        'scope':'conditional fixed-prediction item bootstrap, not guaranteed CV confidence coverage'}

def raw_parts(context,q,partial=False):
    own=context[0]['y'];base=(1+own.sum())/(2+len(own))
    if not len(own):return np.full(len(q),base),np.zeros(len(q))
    x,y,center,ids=centered_arrays(context);kernel=x@x.T
    if partial:kernel*=1+(ids[:,None]==ids[None,:])/8.
    coef=np.linalg.solve(kernel+np.eye(len(y)),y);cross=(q-center)@x.T
    shared=base+cross@coef
    offset=(cross[:,ids==0]@coef[ids==0])/8 if partial else np.zeros(len(q))
    return shared+offset,offset

def cost_control(own,pool,part,counts):
    w=counts/counts.sum();d=pool-own;target=float(w@((part-own)**2));limit=np.where(d>0,1.,np.where(d<0,0.,own));maxcost=float(w@((limit-own)**2))
    if target<=1e-30:return own.copy(),0.,target,0.
    if maxcost<target-1e-14:return limit,np.inf,target,maxcost
    lo=0.;hi=1.
    def candidate(a):return np.clip(own+a*d,0,1)
    while float(w@((candidate(hi)-own)**2))<target and hi<2**40:hi*=2
    for _ in range(60):
        mid=(lo+hi)/2
        if float(w@((candidate(mid)-own)**2))<target:lo=mid
        else:hi=mid
    out=candidate((lo+hi)/2)
    return out,(lo+hi)/2,target,float(w@((out-own)**2))

def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.bool_):return bool(v)
    return v
