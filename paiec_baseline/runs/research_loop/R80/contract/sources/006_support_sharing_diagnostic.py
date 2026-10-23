"""R78 stateless support masking; heldout labels used only after predictions."""
from shared_reliability_common import predict,stable_seed,np
METRICS=['partial','pooled','mean','combined']

def predictions(own,peers,query):
    b=(1+own['y'].sum())/(len(own['y'])+2)
    ownp=predict([own],query)
    pool=predict([own,*peers],query) if peers else ownp.copy()
    part=predict([own,*peers],query,partial=True) if peers else ownp.copy()
    pb=float(np.mean([(1+c['y'].sum())/(len(c['y'])+2) for c in peers])) if peers else float(b)
    return {'own':ownp,'pooled':pool,'partial':part,'mean':np.full(len(query),pb),
            'meanbase':np.full(len(query),b),'combined':np.clip(part+pb-b,0,1)}

def split_masks(n,benchmark,subject_key,item_seed):
    for rep in range(2):
        rng=np.random.default_rng(stable_seed(f'{7801+rep}|{benchmark}|{subject_key}|{item_seed}|{n}'))
        perm=rng.permutation(n)
        for half in range(2):
            hold=perm[half::2];fit=perm[1-half::2]
            assert set(fit).isdisjoint(hold) and sorted(np.r_[fit,hold].tolist())==list(range(n))
            yield rep,half,fit,hold

def mask_predictions(own,peers,fit,hold):
    train={'x':own['x'][fit],'y':own['y'][fit],'keys':[own['keys'][int(k)] for k in fit]}
    return predictions(train,peers,own['x'][hold])

def diagnose(own,peers,benchmark,subject_key,item_seed):
    masks=[];reps=[{k:[] for k in METRICS} for _ in range(2)];repeat_y=[[],[]];mean_gap=[]
    for rep,half,fit,hold in split_masks(len(own['y']),benchmark,subject_key,item_seed):
        pp=mask_predictions(own,peers,fit,hold)
        truth=own['y'][hold].copy()  # first read of heldout y after all fits/predictions
        ownrisk=(pp['own']-truth)**2
        for k in METRICS:
            base=pp['meanbase'] if k=='mean' else pp['own']
            reps[rep][k].extend(((pp[k]-truth)**2-(base-truth)**2).tolist())
        mean_gap.append(float(abs(pp['mean'][0]-pp['meanbase'][0])))
        masks.append({'rep':rep,'half':half,'fit_indices':fit.tolist(),'hold_indices':hold.tolist(),'probabilities':pp,'truth':truth,
                      'fit_own_mean':float(pp['meanbase'][0]),'peer_mean':float(pp['mean'][0])})
    z={k:float(np.mean([np.mean(reps[r][k]) for r in range(2)])) for k in METRICS}
    return z,{k:[float(np.mean(reps[r][k])) for r in range(2)] for k in METRICS},masks,float(np.mean(mean_gap))
