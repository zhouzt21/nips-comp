"""Pure current-evidence peer operators. No query outcomes, dataset IDs, or state."""
import numpy as np
from collections import Counter

def centered_system(context,ridge=1.,semantic_only=False):
    xs=[];ys=[];centers=[]
    for row in context:
        x=row['x'][:,:384] if semantic_only else row['x'];y=row['y']
        center=x.mean(0) if len(y) else np.zeros(x.shape[1]);centers.append(center)
        xs.append(x-center);ys.append(y-y.mean() if len(y) else y)
    xx=np.concatenate(xs);yy=np.concatenate(ys)
    if len(yy) and ridge is not None:
        transform=xx.T@np.linalg.inv(xx@xx.T+ridge*np.eye(len(yy)))
    else:transform=np.zeros((xx.shape[1],len(yy)))
    return transform,yy,centers

def joint(context,query,ridge=1.):
    transform,y,centers=centered_system(context,ridge)
    base=(1+sum(context[0]['y']))/(2+len(context[0]['y']))
    return np.clip(base+(query-centers[0])@(transform@y),0,1)

def same_input_residual(context,query_keys,p,tau=4.):
    effects={}
    for row in context[1:]:
        y=row['y'];keys=row['keys'];counts=Counter(keys)
        if len(y)<2:continue
        residual=y-y.mean()
        for key,value in zip(keys,residual):
            # Reject a repeated/ambiguous within-peer visible key; one outcome per peer per query key.
            if counts[key]!=1:continue
            total,n=effects.get(key,(0.,0));effects[key]=(total+float(value),n+1)
    delta=np.array([effects.get(k,(0.,0))[0]/(tau+effects.get(k,(0.,0))[1]) for k in query_keys])
    return np.clip(p+delta,0,1)

def different_input_joint(context,query,query_keys,full_prediction,ridge=1.):
    """Exclude peer evidence matching THIS query, then re-center/re-solve. Own stays intact."""
    out=full_prediction.copy();allkeys=set(k for row in context[1:] for k in row['keys']);cache={}
    for i,key in enumerate(query_keys):
        if key not in allkeys:continue
        if key not in cache:
            changed=[context[0]]
            for row in context[1:]:
                keep=np.array([v!=key for v in row['keys']],bool)
                changed.append({'x':row['x'][keep],'y':row['y'][keep],'keys':[v for v in row['keys'] if v!=key]})
            # Same full visible query key implies identical modeled input.
            cache[key]=float(joint(changed,query[i:i+1],ridge)[0])
        out[i]=cache[key]
    return out

def r36_family(context,query,semantic_scale,pooled_ridge):
    # Restore original unscaled MiniLM as in R36; no metadata in this historical control.
    sem=[{'x':r['x'][:,:384]*semantic_scale,'y':r['y']} for r in context]
    q=query[:,:384]*semantic_scale
    transform,y,centers=centered_system(sem,pooled_ridge)
    slope=(q-centers[0])@(transform@y)
    own_y=sem[0]['y'];n=len(own_y);s=float(own_y.sum());base=(1+s)/(2+n)
    pn=sum(len(r['y']) for r in sem[1:]);ps=sum(float(r['y'].sum()) for r in sem[1:])
    peer=(4*(8*.5+ps)/(8+pn)+s)/(4+n) if pn else base
    return np.clip(base+slope,0,1),np.clip(peer+slope,0,1)

def predict_current(p,query,context_by_budget,query_keys,r25_ridge,r20_ridge,permutations):
    """permutations is supplied as outcome-independent prefix index arrays."""
    names=['R72_nopeer','R72_same1','R72_same3','R72_pooled1','R72_pooled3','R72_diff3',
           'R72_morelabels_mean3','R36_own_pooled3','R20_peer_pooled3']
    names += [f'perm{k}_{m}' for k in range(2) for m in ['R72_same3','R72_pooled3','R72_diff3','R36_own_pooled3']]
    out={m:p.copy() for m in names}
    for j,context in enumerate(context_by_budget):
        n=len(context[0]['y']);peers=context[1:]
        if not n or not peers:continue
        for k in [1,3]:
            ctx=[context[0],*peers[:k]]
            out[f'R72_same{k}'][:,j]=same_input_residual(ctx,query_keys,p[:,j])
            out[f'R72_pooled{k}'][:,j]=joint(ctx,query)
        out['R72_diff3'][:,j]=different_input_joint(context,query,query_keys,out['R72_pooled3'][:,j])
        own=context[0]['y'];all_y=np.concatenate([r['y'] for r in context]);delta=(1+all_y.sum())/(2+len(all_y))-(1+own.sum())/(2+n)
        out['R72_morelabels_mean3'][:,j]=np.clip(p[:,j]+delta,0,1)
        v,w=r36_family(context,query,np.sqrt(r25_ridge),r20_ridge)
        out['R36_own_pooled3'][:,j]=v;out['R20_peer_pooled3'][:,j]=w
        for k in range(2):
            perm=[context[0]]+[{'x':r['x'],'keys':r['keys'],'y':r['y'][order]} for r,order in zip(peers,permutations[k][j])]
            out[f'perm{k}_R72_same3'][:,j]=same_input_residual(perm,query_keys,p[:,j])
            full=joint(perm,query);out[f'perm{k}_R72_pooled3'][:,j]=full
            out[f'perm{k}_R72_diff3'][:,j]=different_input_joint(perm,query,query_keys,full)
            out[f'perm{k}_R36_own_pooled3'][:,j]=r36_family(perm,query,np.sqrt(r25_ridge),r20_ridge)[0]
    return out
