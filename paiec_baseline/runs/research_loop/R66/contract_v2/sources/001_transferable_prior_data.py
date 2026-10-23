"""From-scratch feature ablations; exactly reuse historical rows and episodes."""
from types import FunctionType
import pandas as pd
from historical_positive_common import *
from historical_exact_encoding_cache import ExactEncodingCache
from mlp.model import _SUBJECT_FIELDS,_ITEM_FIELDS,_string,encode
from mlp.train import _load_examples
from mlp.evaluate import _vectors
from mlp.training_mechanism_data import mlp_training,mlp_validation,raw_episodes,row_weights,digest
from mlp.reproduction_episodes import DATA_ROOT

CACHE=Path('D:/paiec_research_cache/transferable_prior_v1')

def clone(fn, **changes):
    f=FunctionType(fn.__code__,{**fn.__globals__,**changes},fn.__name__,fn.__defaults__,fn.__closure__)
    f.__kwdefaults__=fn.__kwdefaults__;return f

class Collector:
    def __init__(self):self.cache=ExactEncodingCache();self.values={'subject':[],'item':[]}
    def __call__(self, inp, mode='all'):
        assert mode=='all'
        subject,item=inp
        xs,_=self.cache.subject_cached(tuple(_string(subject.get(k,'')) for k in _SUBJECT_FIELDS))
        xi,_=self.cache.item_cached(tuple(_string(item.get(k,'')) for k in _ITEM_FIELDS))
        self.values['subject'].append(xs.copy());self.values['item'].append(xi.copy())
        return self.cache(inp)

def prepare(name,engine):
    folder=CACHE/name.split('/')[0];folder.mkdir(parents=True,exist_ok=True)
    trainfile=folder/'training.npz';valfile=folder/'validation.npz'
    if trainfile.exists() and valfile.exists():return
    collector=Collector()
    x,y,g=clone(_load_examples,encode_mode=collector)(DATA_ROOT,[name],400,2026,'all','response')
    old=mlp_training([name]);assert np.array_equal(x,old['x']) and np.array_equal(y,old['y']) and np.array_equal(g,old['groups'])
    np.savez_compressed(trainfile,y=y,groups=g,**{m:np.stack(v) for m,v in collector.values.items()})
    # _vectors locally caches repeated rows, so collect per-call feature arrays separately.
    def vectors(subject,items,rows,feature_mode='all'):
        out=[];ab={'subject':[],'item':[]}
        for i,interactors in rows[['item_id','interactors']].itertuples(index=False,name=None):
            item=items[i];inp=[subject,{'item_content':item['content'],'item_features':item['item_features'],'interactors':interactors}]
            out.append(collector(inp));ab['subject'].append(collector.values['subject'][-1]);ab['item'].append(collector.values['item'][-1])
        captured.append({m:np.stack(v) for m,v in ab.items()})
        return np.stack(out)
    captured=[];oldval=mlp_validation(name,engine)
    newval=list(clone(raw_episodes,_vectors=vectors)(name,engine,1.,include_mlp=True))
    assert len(newval)==len(oldval) and len(captured)==2*len(newval)
    meta=[];arrays={}
    for i,(ep,ref) in enumerate(zip(newval,oldval)):
        for key in ['support','query','labels','counts','successes']:assert np.array_equal(ep[key],ref[key]),(name,i,key)
        assert ep['subject_id']==ref['subject_id'] and ep['support_ids']==ref['support_ids'] and ep['query_ids']==ref['query_ids']
        meta.append({k:v for k,v in ep.items() if k not in ['support','query','labels','counts','successes']})
        for key in ['labels','counts','successes']:arrays[f'{i}_{key}']=ep[key]
        for mode in ['subject','item']:
            arrays[f'{i}_{mode}_support']=captured[2*i][mode];arrays[f'{i}_{mode}_query']=captured[2*i+1][mode]
    np.savez_compressed(valfile,metadata=json.dumps(meta),**arrays)
    write_json(folder/'audit.json',{'benchmark':name,'training_rows':len(y),'validation_pairs':len(meta),
        'full_encoder_and_sampler_replay':'array_equal','training_sha256':sha(trainfile),'validation_sha256':sha(valfile),
        'data_revision':'54074453da7b9911be0f48e4c560be547bc84d21'})
    print(json.dumps({'prepared':name,'rows':len(y),'pairs':len(meta)}),flush=True)

def training(names,mode):
    rows=[load_npz(CACHE/n.split('/')[0]/'training.npz') for n in names]
    data={'x':np.concatenate([r[mode] for r in rows]),'y':np.concatenate([r['y'] for r in rows]),'groups':np.concatenate([r['groups'] for r in rows])}
    data['pair_weight'],data['mixture_weight'],data['weight_audit']=row_weights(data['groups'])
    data['data_hash']=digest(*(data[k].tobytes() for k in ['x','y','groups']))
    return data

def validation(names,mode):
    result=[]
    for name in names:
        bank=load_npz(CACHE/name.split('/')[0]/'validation.npz')
        for i,m in enumerate(json.loads(str(bank['metadata']))):
            result.append({**m,**{k:bank[f'{i}_{k}'] for k in ['labels','counts','successes']},
                'support':bank[f'{i}_{mode}_support'],'query':bank[f'{i}_{mode}_query']})
    return result
