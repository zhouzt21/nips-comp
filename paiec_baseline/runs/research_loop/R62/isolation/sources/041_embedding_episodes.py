"""Cache public episode input vectors with support and query labels separated."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluate import _load_directory, _stable_seed
from .model import encode_mode


def episodes(directory, seed, max_pairs=100):
    data=Path("D:/paiec_payload/data")
    revision=json.loads((data/"corpus_manifest.json").read_text())["revision"]
    code=Path(__file__).read_bytes()+Path("mlp/model.py").read_bytes()
    key=hashlib.sha256(code+f"{revision}|{directory}|{seed}|{max_pairs}|64".encode()).hexdigest()
    path=Path("D:/paiec_research_cache/input_episodes")/(key+".npz")
    if path.exists():
        with np.load(path,allow_pickle=False) as pack:
            return [{k:pack[f"{i}_{k}"] for k in ["x","a","y"]} for i in range(int(pack["count"]))]
    responses,subjects,items=_load_directory(data,directory)
    groups=[(s,g) for s,g in responses.groupby("subject_id",sort=False)
            if g["item_id"].nunique()>=80 and s in subjects]
    if len(groups)>max_pairs:
        rng=np.random.default_rng(_stable_seed(2026,directory,"pair_sample"))
        groups=[groups[i] for i in np.sort(rng.choice(len(groups),max_pairs,replace=False))]
    @lru_cache(maxsize=4096)
    def vector(subject_id,item_id,interactors):
        item=items[item_id]
        return encode_mode([subjects[subject_id],{"item_content":item["content"],"item_features":item["item_features"],"interactors":interactors}])
    out=[]
    for sid,group in groups:
        ids=pd.unique(group["item_id"]).copy()
        rng=np.random.default_rng(_stable_seed(seed,directory,sid)); rng.shuffle(ids)
        cut=len(ids)//2; acquired_ids=ids[:31]; evaluation_ids=ids[cut:cut+64]
        acquired=group[group["item_id"].isin(acquired_ids)].sample(frac=1,random_state=int(rng.integers(2**31)))
        acquired=acquired.drop_duplicates("item_id").set_index("item_id").loc[acquired_ids].reset_index()
        evaluation=group[group["item_id"].isin(evaluation_ids)]
        rows=pd.concat([acquired,evaluation],ignore_index=True)
        x=np.stack([vector(sid,i,str(t)) for i,t in rows[["item_id","interactors"]].itertuples(index=False,name=None)])
        out.append({"x":x,"a":acquired["response"].to_numpy(np.float32),"y":evaluation["response"].to_numpy(np.float32)})
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,count=len(out),**{f"{i}_{k}":v for i,row in enumerate(out) for k,v in row.items()})
    return out
