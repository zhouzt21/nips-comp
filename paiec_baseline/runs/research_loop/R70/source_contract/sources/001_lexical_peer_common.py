"""R70/R71: strict source calibration and observable lexical reliability."""
from historical_positive_common import *
from scipy.optimize import isotonic_regression
from mlp.run.run_metadata_round import prediction
from mlp.context_tokens import template, retained
from mlp.evaluate import _stable_seed

RIDGES=[None,16.,4.,1.,.25]
def rk(r):return 'none' if r is None else str(float(r))

def views(tokens, semantic, metadata, engine, ridge):
    sem=np.c_[semantic/np.sqrt(ridge or 1.),metadata/4.]
    for j,n in enumerate(BUDGETS):
        rule=template(tokens[:n]);chosen=[retained(t,rule) for t in tokens[:n]+tokens[31:]]
        idx=[*range(n),*range(31,len(tokens))]
        lex=np.c_[np.stack([engine.lexical(t) for t in chosen]),metadata[idx]/4.]
        z=np.zeros(len(tokens)-31,dtype=np.uint8)
        if n>=3:
            def near(a,b):
                a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
                b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
                return np.argmax(b@a.T,axis=1)
            z=(near(sem[:n],sem[31:])==near(lex[:n],lex[n:])).astype(np.uint8)
        yield j,n,sem[:n],sem[31:],lex[:n],lex[n:],z

def predict_bank(tokens,semantic,metadata,labels,engine,ridges,identity,permutations=True):
    out={};q=len(tokens)-31
    # Feature construction is deterministic and uses no query labels.
    for r in ridges:
        key=rk(r);p=np.empty((q,6));lex=np.empty_like(p);z=np.zeros((q,6),np.uint8)
        ps=[np.empty_like(p) for _ in range(4)];ls=[np.empty_like(p) for _ in range(4)]
        for j,n,sx,qx,sl,ql,zz in views(tokens,semantic,metadata,engine,r):
            p[:,j]=prediction(sx,qx,labels[:n],None if r is None else 1.)
            lex[:,j]=prediction(sl,ql,labels[:n],1.);z[:,j]=zz
            if permutations:
                for k in range(4):
                    rng=np.random.default_rng(_stable_seed(7101+k,identity,str(n)))
                    y=labels[:n][rng.permutation(n)]
                    ps[k][:,j]=prediction(sx,qx,y,None if r is None else 1.)
                    ls[k][:,j]=prediction(sl,ql,y,1.)
        out['p_'+key]=p;out['z_'+key]=z;out['lex']=lex
        if permutations:
            for k in range(4):out[f'perm{k}_p_'+key]=ps[k];out[f'perm{k}_lex']=ls[k]
    return out

def stats(ep,r):
    p=ep['bank']['p_'+rk(r)];d=(ep['bank']['lex']-p)*.5
    c=ep['counts'][:,None];s=ep['successes'][:,None]
    return ((c*p-s)*d)/c.sum(),c*d*d/c.sum(),ep['bank']['z_'+rk(r)]

def select_ridge(episodes):
    losses=[]
    for r in RIDGES:
        losses.append(np.mean([sum(w*brier(ep['bank']['p_'+rk(r)][:,j],ep['counts'],ep['successes']) for j,w in enumerate(ALC_WEIGHTS)) for ep in episodes]))
    return RIDGES[int(np.argmin(losses))],losses

def calibrate(records):
    aa=[];cc=[];ag=np.zeros((6,2));cg=np.zeros((6,2));support=[[set() for _ in range(2)] for j in range(6)]
    ng=np.zeros((6,2),int)
    for ep,r in records:
        a,c,z=stats(ep,r);aa.append(a.sum(0));cc.append(c.sum(0))
        for j in range(2,6):
            for g in range(2):
                sel=z[:,j]==g;ag[j,g]+=a[sel,j].sum();cg[j,g]+=c[sel,j].sum()
                if c[sel,j].sum()>1e-16:ng[j,g]+=1;support[j][g].add(ep['benchmark'])
    a=np.mean(aa,0);c=np.mean(cc,0)
    independent=np.clip(np.divide(-a,c,out=np.zeros(6),where=c>1e-16),0,1);independent[:2]=0
    monotone=np.zeros(6)
    if np.any(c[2:]>1e-16):
        # Zero-weight empty coordinates receive negligible weight; C>0 in observed data.
        w=np.maximum(c[2:]*ALC_WEIGHTS[2:],1e-30)
        target=np.divide(-a[2:],c[2:],out=np.zeros(4),where=c[2:]>1e-16)
        monotone[2:]=np.clip(isotonic_regression(target,weights=w,increasing=False).x,0,1)
    global_a=float(np.clip(-float(a@ALC_WEIGHTS)/max(float(c@ALC_WEIGHTS),1e-30),0,1))
    grouped=np.zeros((6,2))
    for j in range(2,6):
        for g in range(2):
            if ng[j,g]>=20 and len(support[j][g])>=3 and cg[j,g]>1e-16:grouped[j,g]=np.clip(-ag[j,g]/cg[j,g],0,1)
    return {'independent':independent.tolist(),'monotone':monotone.tolist(),'global':[0.,0.]+[global_a]*4,
        'grouped':grouped.tolist(),'A':a.tolist(),'C':c.tolist(),'group_A_sum':ag.tolist(),'group_C_sum':cg.tolist(),
        'group_pairs':ng.tolist(),'group_BMs':[[len(v) for v in row] for row in support],
        'training_records':len(records),'training_benchmarks':sorted({ep['benchmark'] for ep,r in records})}

def apply(p,lex,z,coeff):
    d=(lex-p)*.5
    out={'R25':p,'R64':p+d}
    for key in ['independent','monotone','global']:out['R70_'+key]=p+d*np.array(coeff[key])
    alpha=np.array(coeff['grouped'])[np.arange(6)[None,:],z]
    out['R71_grouped']=p+d*alpha
    return out,alpha

def random_energy_control(p,lex,alpha,counts,identity,seed):
    d=(lex-p)*.5;result=p.copy();error=0.
    for j,n in enumerate(BUDGETS):
        energy=counts*d[:,j]**2;target=float(np.sum(energy*alpha[:,j]**2));remaining=target
        g=np.zeros(len(p));rng=np.random.default_rng(_stable_seed(seed,identity,str(n)))
        for i in rng.permutation(len(p)):
            if energy[i]>0:
                used=min(remaining,energy[i]);g[i]=np.sqrt(max(0,used/energy[i]));remaining-=used
        result[:,j]=p[:,j]+g*d[:,j]
        error=max(error,abs(float(np.sum(counts*(result[:,j]-p[:,j])**2))-target))
    return result,error

def full_predictions(bank,r,coeff,counts,identity):
    key=rk(r);p=bank['p_'+key];lex=bank['lex'];z=bank['z_'+key]
    out,alpha=apply(p,lex,z,coeff);error=0.
    for k in range(4):
        pp,_=apply(bank[f'perm{k}_p_'+key],bank[f'perm{k}_lex'],z,coeff)
        for name,v in pp.items():out[f'perm{k}_'+name]=v
        out[f'random{k}_R71'],e=random_energy_control(p,lex,alpha,counts,identity,7111+k);error=max(error,e)
    return out,error

def decompose(ep,preds):
    p=preds['R25'];c=ep['counts'][:,None];s=ep['successes'][:,None];out=[]
    for name,v in preds.items():
        if name=='R25':continue
        d=v-p;a=((c*p-s)*d).sum(0)/c.sum();cost=(c*d*d).sum(0)/c.sum()
        for j,n in enumerate(BUDGETS):
            delta=brier(v[:,j],ep['counts'],ep['successes'])-brier(p[:,j],ep['counts'],ep['successes'])
            assert abs(delta-2*a[j]-cost[j])<1e-12
            out.append({'fold':ep['fold'],'benchmark':ep['benchmark'],'subject_id':ep['subject_id'],'seed':ep['seed'],
                'method':name,'budget':n,'A':float(a[j]),'alignment':float(2*a[j]),'cost':float(cost[j]),'delta_brier':delta})
    return out
