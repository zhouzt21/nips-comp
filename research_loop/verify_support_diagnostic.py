"""Deterministic boundaries required by plan_9, not a statistical pilot."""
from shared_reliability_common import *
from support_sharing_diagnostic import predictions,diagnose,mask_predictions,split_masks
def main():
    rng=np.random.default_rng(7800);checks=[]
    for n in [15,31]:
        own={'x':rng.normal(size=(n,8)),'y':rng.integers(0,2,n).astype(float),'keys':[str(i) for i in range(n)]}
        peers=[{'x':rng.normal(size=(n,8)),'y':rng.integers(0,2,n).astype(float),'keys':[str(i) for i in range(n)]} for _ in range(3)]
        z,rz,masks,gap=diagnose(own,peers,'synthetic','visible subject',0)
        err=0.
        for mask in masks:
            fit=np.asarray(mask['fit_indices']);hold=np.asarray(mask['hold_indices']);changed={**own,'y':own['y'].copy()};changed['y'][hold]=1-changed['y'][hold]
            p=mask_predictions(changed,peers,fit,hold)
            err=max(err,max(float(abs(p[k]-mask['probabilities'][k]).max()) for k in p))
        assert err==0
        pp=predictions(own,[],rng.normal(size=(5,8)));assert np.array_equal(pp['own'],pp['partial']) and np.array_equal(pp['own'],pp['pooled'])
        assert all(len([m for m in masks if i in m['hold_indices']])==2 for i in range(n))
        checks.append({'budget':n,'masks':4,'each_support_heldout_exactly_twice':True,'heldout_truth_mutation_prediction_error':err,'no_peer_fallback_exact':True,'z_finite':all(np.isfinite(v) for v in z.values())})
    dest=ROOT/'shared_reliability_v9';dest.mkdir(exist_ok=True)
    write_json(dest/'support_diagnostic_checks.json',{'status':'passed_synthetic_only','checks':checks,'source_hashes':{str(p):sha(p) for p in [Path(__file__),REPO/'research_loop/support_sharing_diagnostic.py']},'finished':now()})
    print(json.dumps(checks))
if __name__=='__main__':main()
