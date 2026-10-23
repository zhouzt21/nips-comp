"""Synthetic-only acceptance: adapter input boundaries and quadratic equivalence."""
import copy,time
from historical_positive_common import *
from shared_slope_adapter import predict_raw,SUBJECT_FIELDS
from shared_slope_model import predict,centered_arrays

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);root=ROOT/'R76/smoke';root.mkdir(parents=True,exist_ok=True);start=time.perf_counter();cases={}
    rng=np.random.default_rng(7606);ctx=[{'x':rng.normal(size=(n,7)),'y':rng.integers(0,2,n).astype(float)} for n in [9,7,5]];q=rng.normal(size=(11,7))
    x,y,center,ids=centered_arrays(ctx);T=len(ctx);d=x.shape[1]
    design=np.zeros((len(y),d*(T+1)));design[:,:d]=x
    for s in range(T):design[ids==s,d*(s+1):d*(s+2)]=x[ids==s]
    penalty=np.diag(np.r_[np.ones(d),np.full(d*T,8.)]);coef=np.linalg.solve(design.T@design+penalty,design.T@y)
    primal=np.clip((1+ctx[0]['y'].sum())/(2+len(ctx[0]['y']))+(q-center)@(coef[:d]+coef[d:2*d]),0,1)
    dual=predict(ctx,q,partial=True);err=float(np.max(np.abs(primal-dual)));assert err<1e-12
    W=np.stack([coef[:d]+coef[d*(s+1):d*(s+2)] for s in range(T)],axis=1);R=np.eye(T)-np.ones((T,T))/T
    a=8/(1+8*T);g=8-a;pen=coef@penalty@coef;graph=g*np.linalg.norm(W@R)**2+a*np.linalg.norm(W)**2
    assert abs(pen-graph)<1e-12;cases['partial_primal_dual_graph_equivalence']={'max_error':err,'penalty_error':abs(pen-graph),'MALSAR_half_SSE_rho_graph':g/2,'MALSAR_half_SSE_rho_L2':a/2}
    def subject(i):return {k:('name'+str(i) if k=='normalized_name' else 'provider' if k=='provider' else '') for k in SUBJECT_FIELDS}
    subjects=[subject(i) for i in range(4)]
    def item(i,group='opaque-current-token'):return {'item_content':str(i),'item_features':'','interactors':'','benchmark_id':group}
    def encode(it):
        z=float(it['item_content']);return np.array([z,z*z/10,1.])
    own_input=[subjects[0],item(1.5)];labels=[[[s,item(i)],int((i+j)%3!=0)] for j,s in enumerate(subjects) for i in range(4)]
    def call(ls,**kw):return predict_raw(own_input,ls,encoder=encode,**kw)
    for peers in [0,1,2,3]:
        ls=labels[:4*(peers+1)];v=call(ls,visible_subjects=subjects[:peers+1]);assert v['peers']==peers;cases[f'peers_{peers}']=v
    no_roster=call(labels);assert no_roster['mode']=='own_only';cases['flat_labels_no_uniqueness_evidence']=no_roster
    duplicate=call(labels,visible_subjects=subjects+[subjects[1]]);assert duplicate['mode']=='own_only';cases['duplicate_full_configuration']=duplicate
    conflict=call(labels+[ [labels[4][0],1-labels[4][1]] ],visible_subjects=subjects);assert conflict['mode']=='own_only';cases['conflicting_label_for_visible_key']=conflict
    repeated=call(labels+[labels[4]],visible_subjects=subjects);assert repeated['mode']=='own_only';cases['duplicate_visible_input']=repeated
    b0=call(labels[4:],visible_subjects=subjects);assert b0['prediction']==.5 and b0['mode']=='own_only';cases['B0_visible_peers']=b0
    mixed=labels+[[[subjects[1],item(99,'another-group')],0]];assert call(mixed,visible_subjects=subjects)==call(labels,visible_subjects=subjects);cases['cross_group_filtered']=True
    # Internal IDs are not used as identity evidence or modeled fields.
    with_ids=copy.deepcopy(labels)
    for i,entry in enumerate(with_ids):entry[0][0]['subject_id']='forbidden-'+str(i);entry[0][1]['item_id']='forbidden-'+str(i)
    assert call(with_ids,visible_subjects=subjects)==call(labels,visible_subjects=subjects);cases['hidden_ids_ignored']=True
    before=call(labels,visible_subjects=subjects);call([],visible_subjects=subjects);assert call(labels,visible_subjects=subjects)==before;cases['no_cross_call_label_state']=True
    reverse=call(list(reversed(labels)),visible_subjects=list(reversed(subjects)));assert abs(reverse['prediction']-before['prediction'])<1e-12;cases['input_order_invariant']=True
    bad=copy.deepcopy(labels);bad[0][1]=.3
    try:call(bad,visible_subjects=subjects)
    except ValueError:cases['nonbinary_rejected']=True
    else:raise AssertionError('nonbinary label accepted')
    custom=call(labels,own_predictor=lambda inp,ls:.123);assert custom['prediction']==.123;cases['custom_no_peer_fallback']=True
    write_json(root/'results.json',{'status':'passed_synthetic_only','cases':cases,'count':len(cases),'seconds':time.perf_counter()-start,'finished':now(),
        'limitations':['Not Hosted acceptance','Roster must be explicitly current-visible','Undetectable hidden collisions cannot be resolved from flat visible dictionaries']})
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/shared_slope_adapter.py',REPO/'research_loop/shared_slope_model.py'],{'synthetic':True})
    print(json.dumps({'cases':len(cases),'status':'passed_synthetic_only','dual_error':err}))

if __name__=='__main__':main()
