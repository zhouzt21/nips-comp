"""Field-aware current-item metadata; no IDs, provenance paths or outcome fields."""
import hashlib
import json
import numpy as np

DIM=256
FIELDS=frozenset('category difficulty task domain mmlu_subject context_condition question_polarity group lang shot data_modality task_type prompt_condition prompt_variant question_type component natural_language programming_language stage subcategory visual_input'.split())


def fields(value):
    if not isinstance(value,str):return {}
    text=value.strip()
    try:parsed=json.loads(text)
    except (ValueError,TypeError):parsed=None
    if not isinstance(parsed,dict):parsed=dict(part.split('=',1) for part in text.split(';') if '=' in part)
    out={}
    for key,value in parsed.items():
        key=key.strip().lower()
        if key not in FIELDS or isinstance(value,(dict,list)) or value is None:continue
        value=str(value).strip().casefold()
        if not value or len(value)>64 or '://' in value:continue
        out[key]=value
    return out


def encode(value):
    out=np.zeros(DIM,float)
    for key,value in fields(value).items():
        digest=hashlib.blake2b((key+'='+value).encode(),digest_size=8).digest();index=int.from_bytes(digest[:4],'little')%DIM
        out[index]+=1 if digest[4]&1 else -1
    norm=np.linalg.norm(out)
    return out/norm if norm else out
