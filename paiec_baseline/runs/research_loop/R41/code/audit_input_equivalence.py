"""R41 finite-evaluation equivalence loss. No trained/deployable oracle is produced."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .evaluate import _load_directory,_stable_seed
from .model import _string
from .semantic_features import SemanticFeatures,MODEL_REVISION
from .metadata_features import encode

STAGES=['raw','canonical','token_window','R25_information']


def class_loss(keys,counts,successes):
    grouped={}
    for key,m,s in zip(keys,counts,successes):
        old=grouped.setdefault(key,[0.,0.]);old[0]+=m;old[1]+=s
    total=sum(counts)
    return sum(s-s*s/m for m,s in grouped.values())/total,len(grouped)


def digest(*parts):
    h=hashlib.sha256()
    for p in parts:
        b=p.encode() if isinstance(p,str) else p
        h.update(len(b).to_bytes(8,'little'));h.update(b)
    return h.digest()


def main():
    from transformers import AutoTokenizer
    start=time.time();root=Path('runs/research_loop/R41');root.mkdir(parents=True,exist_ok=True)
    source=['mlp/audit_input_equivalence.py','mlp/semantic_features.py','mlp/structured_text.py','mlp/metadata_features.py','research_loop/R41_protocol.md']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source}
    if (root/'provenance.json').exists():assert json.loads((root/'provenance.json').read_text())==hashes
    (root/'provenance.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');snap=root/'code';snap.mkdir(exist_ok=True)
    for p in source:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    tokenizer=AutoTokenizer.from_pretrained('D:/paiec_pretrained/all-MiniLM-L6-v2',local_files_only=True,clean_up_tokenization_spaces=False)
    result=[]
    for fold in range(3):
        reference=json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text())
        for directory,old in reference['by_benchmark'].items():
            target=root/(directory.split('/')[0]+'.json')
            if target.exists():result.append(json.loads(target.read_text()));continue
            response,subjects,items=_load_directory(Path('D:/paiec_payload/data'),directory)
            response['interactors']=response['interactors'].astype(str)
            pending=[];needed=set();pairs=0
            for sid,g in response.groupby('subject_id',sort=False):
                if sid not in subjects or g.item_id.nunique()<80:continue
                pairs+=1
                grouped=g.groupby('item_id',sort=False).agg(count=('response','size'),positive=('response','sum'),interactors=('interactors','first'),n_interactors=('interactors','nunique'))
                assert grouped.n_interactors.max()==1
                for seed in [0,1,2]:
                    ids=pd.unique(g.item_id).copy();np.random.default_rng(_stable_seed(seed,directory,sid)).shuffle(ids)
                    chosen=ids[len(ids)//2:len(ids)//2+64];needed.update(chosen);table=grouped.loc[chosen]
                    pending.append((str(sid),seed,chosen,table['count'].to_numpy(float),table.positive.to_numpy(float),[_string(x) for x in table.interactors]))
            assert pairs==old['pairs_per_seed'][0]
            canonical={i:SemanticFeatures.canonical(items[i]['content']) for i in needed}
            unique=list(dict.fromkeys(canonical.values()));token_keys={}
            for offset in range(0,len(unique),256):
                batch=unique[offset:offset+256];tokens=tokenizer(batch,add_special_tokens=False,truncation=False,return_attention_mask=False)['input_ids']
                for text,ids in zip(batch,tokens):
                    if len(ids)>126:ids=ids[:96]+ids[-30:]
                    token_keys[text]=digest(np.asarray(ids,dtype='<i4').tobytes())
            raw={i:_string(items[i]['content']) for i in needed};metadata={i:_string(items[i]['item_features']) for i in needed}
            last={i:digest(token_keys[canonical[i]],encode(items[i]['item_features']).astype('<f8').tobytes()) for i in needed}
            sums=np.zeros(4);counts=np.zeros(4);merged=np.zeros(4);max_violation=0.
            with (root/(directory.split('/')[0]+'_pairs.csv')).open('w',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['subject_id','seed','query_items','response_count',*[s+'_oracle_loss' for s in STAGES],*[s+'_unique' for s in STAGES]])
                for sid,seed,chosen,m,s,interactors in pending:
                    keys=[[],[],[],[]]
                    for i,interactor in zip(chosen,interactors):
                        keys[0].append(digest(raw[i],metadata[i],interactor));keys[1].append(digest(canonical[i],metadata[i],interactor))
                        keys[2].append(digest(token_keys[canonical[i]],metadata[i],interactor));keys[3].append(last[i])
                    scored=[class_loss(k,m,s) for k in keys];losses=np.array([v[0] for v in scored]);unique_n=np.array([v[1] for v in scored])
                    violation=max(0.,float(-np.min(np.diff(losses))));max_violation=max(max_violation,violation);assert violation<1e-12
                    sums+=losses;counts+=unique_n;merged+=len(chosen)-unique_n
                    writer.writerow([sid,seed,len(chosen),m.sum(),*losses,*unique_n])
            row={'benchmark':directory,'fold':fold,'pairs':pairs,'episodes':len(pending),'unique_items_tokenized':len(needed),
                 'oracle_loss':dict(zip(STAGES,(sums/len(pending)).tolist())),'unique_classes_per_episode':dict(zip(STAGES,(counts/len(pending)).tolist())),
                 'merged_items_per_episode':dict(zip(STAGES,(merged/len(pending)).tolist())),'maximum_monotonicity_violation':max_violation}
            row['collision_cost']=row['oracle_loss']['R25_information']-row['oracle_loss']['raw']
            target.write_text(json.dumps(row,indent=2),encoding='utf-8');result.append(row);print(directory,'collision cost',row['collision_cost'],flush=True)
    weights=np.array([r['pairs'] for r in result]);pair={s:float(np.average([r['oracle_loss'][s] for r in result],weights=weights)) for s in STAGES}
    summary={'scope':'Evaluation-label-fitted empirical approximation costs; NOT predictor scores, ALC, noise-floor estimates or blind-test evidence',
             'pairs':int(weights.sum()),'seeds':[0,1,2],'evaluation_item_cap':64,'model_revision':MODEL_REVISION,'pair_equal_oracle_loss':pair,
             'pair_equal_collision_cost':pair['R25_information']-pair['raw'],
             'benchmark_equal_collision_cost':float(np.mean([r['collision_cost'] for r in result])),
             'by_benchmark':result,'elapsed_seconds':time.time()-start}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for p in source:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==hashes[p]
    print(json.dumps({k:v for k,v in summary.items() if k!='by_benchmark'},indent=2),flush=True)


if __name__=='__main__':main()
