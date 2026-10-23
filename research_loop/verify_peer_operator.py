"""Small deterministic numerical checks directly supporting the R72 experiment."""
from historical_positive_common import *
from peer_current_predictor import *
from mlp.conditional_components import components
from mlp.run.run_metadata_round import prediction

rng=np.random.default_rng(7200);r=4.;x=rng.normal(size=(31,640))*.03;q=rng.normal(size=(9,640))*.03;y=rng.integers(2,size=31).astype(float)
rows=[{'x':x,'y':y,'keys':[f'own{i}' for i in range(31)]}]
for k in range(3):rows.append({'x':rng.normal(size=(31,640))*.03,'y':rng.integers(2,size=31).astype(float),'keys':[f'peer{k}_{i}' for i in range(31)]})
keys=['peer0_0','peer1_1','peer2_2',*['absent']*6]
base=np.column_stack([prediction(x[:n],q,y[:n],1.) for n in BUDGETS]);ctx=[[{'x':v['x'][:n],'y':v['y'][:n],'keys':v['keys'][:n]} for v in rows] for n in BUDGETS]
perms=[[[rng.permutation(n) for v in rows[1:]] for n in BUDGETS] for k in range(2)]
out=predict_current(base,q,ctx,keys,r,4.,perms);one={}
for i in range(len(q)):
    preds=predict_current(base[i:i+1],q[i:i+1],ctx,keys[i:i+1],r,4.,perms)
    for m,p in preds.items():one.setdefault(m,[]).append(p)
batch_error=max(float(np.max(np.abs(v-np.concatenate(one[m])))) for m,v in out.items());assert batch_error<1e-12
own=predict_current(base,q,[[v[0]] for v in ctx],keys,r,4.,[[[] for n in BUDGETS] for k in range(2)])
assert all(np.array_equal(v,base) for v in own.values());assert all(np.array_equal(v[:,0],base[:,0]) for v in out.values())
refctx=[{'key':str(i),'labels':v['y'],'vectors':v['x'][:,:384]*np.sqrt(r)} for i,v in enumerate(rows)]
ref=components(refctx,[q[:,:384]*np.sqrt(r)]*4,4.,4.,4.,rng,permutations=0)[0]
replay=max(float(np.max(np.abs(out[m][:,-1]-np.clip(ref[k]+ref['pooled'],0,1)))) for m,k in [('R36_own_pooled3','base'),('R20_peer_pooled3','peer')]);assert replay<1e-12
result={'batch_vs_single_query_error':batch_error,'R36_formula_error':replay,'own_only_exact':True,'B0_exact':True,'finite_bounds':all(np.all((v>=0)&(v<=1)) for v in out.values()),'seed':7200,'finished':now()}
write_json(ROOT/'R72/operator_checks.json',result);print(json.dumps(result))
