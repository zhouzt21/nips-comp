"""R18a deterministic input-only tokenization/equivalence audit."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from mlp.semantic_features import SemanticFeatures
from mlp.evaluate import _stable_seed


def main():
    from transformers import AutoTokenizer
    root=Path('runs/research_loop/R18');root.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained('D:/paiec_pretrained/all-MiniLM-L6-v2',local_files_only=True)
    directories=sorted({b for f in range(3) for b in json.loads(Path(f'runs/mlp_cv/fold{f}_split.json').read_text())['development_benchmarks']})
    output=[]
    for directory in directories:
        path=Path('D:/paiec_payload/data')/directory
        df=pd.read_parquet(path/'items.parquet',columns=['item_id','content'])
        if len(df)>100:
            rng=np.random.default_rng(_stable_seed(123,directory,'input_audit'));df=df.iloc[np.sort(rng.choice(len(df),100,replace=False))]
        raw=[str(v) if v is not None else '' for v in df['content']];canonical=[SemanticFeatures.canonical(v) for v in raw]
        tokens=tokenizer(canonical,add_special_tokens=False,truncation=False,return_attention_mask=False)['input_ids']
        window=[tuple(v if len(v)<=126 else v[:96]+v[-30:]) for v in tokens]
        lengths=np.array([len(v) for v in tokens]);meta=pd.read_parquet(path/'benchmarks.parquet').iloc[0]
        modality=meta['modality'].tolist() if hasattr(meta['modality'],'tolist') else meta['modality']
        output.append({'benchmark':directory,'items':len(raw),'modality':modality,'raw_unique':len(set(raw)),
            'canonical_unique':len(set(canonical)),'window_unique':len(set(window)),
            'over126':int((lengths>126).sum()),'over254':int((lengths>254).sum()),
            'token_quantiles':np.quantile(lengths,[0,.25,.5,.75,.95,1]).tolist(),
            'image_reference_syntax':sum(bool(re.search(r'image|\.png|\.jpg|\.jpeg',v,re.I)) for v in raw),
            'empty_canonical':sum(not v.strip() for v in canonical)})
    summary={'scope':'Input-only sample, not performance or information-theoretic bound','seed':123,'max_items_per_benchmark':100,
        'benchmarks':output,'totals':{k:sum(r[k] for r in output) for k in ['items','raw_unique','canonical_unique','window_unique','over126','over254','image_reference_syntax','empty_canonical']}}
    (root/'input_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['totals'],indent=2))


if __name__=='__main__':main()
