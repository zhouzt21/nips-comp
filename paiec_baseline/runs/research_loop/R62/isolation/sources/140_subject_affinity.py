"""Outcome-free configuration relation; no stored benchmark/subject lookup."""
import json
import re

STANDARD=('normalized_name','provider','harness','harness_version','reasoning_effort')
EXCLUDE=re.compile(r'(?:^|_)(?:id|uuid|path|file|filename|url|source|benchmark|dataset|run|timestamp|date|label|outcome|grade|score|accuracy|success|result)(?:_|$)|identifier')


def tokens(subject):
    out=set()
    for key in STANDARD:
        value=subject.get(key,'')
        if value is not None:
            value=str(value).strip().casefold()
            if value and value not in {'nan','none','<na>'}:out.add((key,value))
    text=subject.get('subject_features_extra','')
    if not isinstance(text,str):return frozenset(out)
    try:extra=json.loads(text)
    except (ValueError,TypeError):extra=None
    if not isinstance(extra,dict):extra=dict(part.split('=',1) for part in text.split(';') if '=' in part)
    for key,value in extra.items():
        key=str(key).strip().casefold()
        if EXCLUDE.search(key) or isinstance(value,(dict,list)) or value is None:continue
        value=str(value).strip().casefold()
        if value and len(value)<=128 and '://' not in value and '\\' not in value:out.add(('extra:'+key,value))
    return frozenset(out)


def jaccard(a,b):
    union=a|b
    return len(a&b)/len(union) if union else 0.
