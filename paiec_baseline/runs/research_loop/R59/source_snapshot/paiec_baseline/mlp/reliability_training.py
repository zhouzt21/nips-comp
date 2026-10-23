"""R43 source episodic data and small CPU MLP, with portable NumPy export."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .evidence_reliability import statistics,forward,QUERY_START
from .measurement_episodes import episodes,weighted_brier
from .evaluate import BUDGETS,ALC_WEIGHTS


def source_data(directory,engine,ridge):
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path('mlp/evidence_reliability.py').read_bytes()+Path('mlp/measurement_episodes.py').read_bytes()+engine.fingerprint.encode()).hexdigest()
    key=hashlib.sha256((fingerprint+directory+str(ridge)).encode()).hexdigest();folder=Path('D:/paiec_research_cache/evidence_reliability');folder.mkdir(parents=True,exist_ok=True);path=folder/(key+'.npz')
    if path.exists():
        with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
    blocks={k:[] for k in ['z','base','delta','fraction','weight']};baseline=np.zeros(6);pairs=0
    for row in episodes(directory,engine,ridge,16.,seeds=(73,),pair_cap=25):
        for j,n in enumerate(BUDGETS):
            y=row['labels'][:n];b=(1+y.sum())/(2+n)
            if n<3:p=b
            else:
                z,b,d=statistics(row['support'][:n],row['query'],y);p=np.clip(b+d,0,1)
                blocks['z'].append(z);blocks['base'].append(np.full(len(d),b));blocks['delta'].append(d)
                blocks['fraction'].append(row['successes']/row['counts']);blocks['weight'].append(ALC_WEIGHTS[j]*row['counts']/row['counts'].sum())
            baseline[j]+=weighted_brier(p,row['counts'],row['successes'])
        pairs+=1
    result={k:np.concatenate(v).astype(np.float32) if v else np.empty((0,16) if k=='z' else 0,np.float32) for k,v in blocks.items()}
    result.update(pairs=np.array(pairs),baseline_brier=baseline/max(pairs,1),constant=np.array(float(baseline[:2]@ALC_WEIGHTS[:2])))
    np.savez_compressed(path,**result);return result


def combine(data,names):
    rows=[data[n] for n in names]
    return {**{k:np.concatenate([r[k] for r in rows]) for k in ['z','base','delta','fraction','weight']},
            'pairs':sum(int(r['pairs']) for r in rows),'constant':sum(float(r['constant']) for r in rows)}


def score(data,artifact):
    g=forward(data['z'],artifact);p=np.clip(data['base'].astype(float)+g*data['delta'],0,1)
    loss=p*p-2*p*data['fraction']+data['fraction']
    return float((np.sum(data['weight'].astype(float)*loss)+data['constant'])/data['pairs'])


def train(train_data,val_data,episode_only,epochs,out):
    import torch
    torch.set_num_threads(4);torch.manual_seed(4307);rng=np.random.default_rng(4307);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    z=train_data['z'].astype(float).copy()
    if episode_only:z[:,QUERY_START:]=0
    mean=np.average(z,axis=0,weights=train_data['weight']);scale=np.sqrt(np.average((z-mean)**2,axis=0,weights=train_data['weight']));scale=np.where(scale>1e-8,scale,1.)
    normalized=np.clip((z-mean)/scale,-5,5).astype(np.float32);x=torch.from_numpy(normalized)
    base=torch.from_numpy(train_data['base']);delta=torch.from_numpy(train_data['delta']);fraction=torch.from_numpy(train_data['fraction'])
    weight=torch.from_numpy(train_data['weight']*len(z)/train_data['weight'].sum())
    model=torch.nn.Sequential(torch.nn.Linear(16,16),torch.nn.ReLU(),torch.nn.Linear(16,8),torch.nn.ReLU(),torch.nn.Linear(8,1))
    torch.nn.init.zeros_(model[-1].weight);torch.nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);history=[]
    def artifact():
        a={'mean':mean,'scale':scale,'episode_only':np.array(episode_only)}
        for i,j in enumerate([0,2,4],1):a[f'W{i}']=model[j].weight.detach().numpy().T.copy();a[f'b{i}']=model[j].bias.detach().numpy().copy()
        return a
    for epoch in range(epochs+1):
        if epoch:
            order=rng.permutation(len(z))
            for start in range(0,len(z),1024):
                idx=torch.from_numpy(order[start:start+1024]);g=4*torch.sigmoid(model(x[idx]).reshape(-1)+float(np.log(1/3)))
                p=torch.clamp(base[idx]+g*delta[idx],0,1);loss=(weight[idx]*(p*p-2*p*fraction[idx]+fraction[idx]+.01*(g-1)**2)).mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
        a=artifact();np.savez_compressed(out/f'epoch{epoch}.npz',**a)
        with torch.no_grad():expected=(4*torch.sigmoid(model(x[:32]).reshape(-1)+float(np.log(1/3)))).numpy()
        error=float(np.max(np.abs(forward(train_data['z'][:32],a)-expected)));assert error<1e-5
        entry={'epoch':epoch,'train_alc':score(train_data,a),'export_error':error}
        if val_data is not None:entry['validation_alc']=score(val_data,a)
        history.append(entry)
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return history,a
