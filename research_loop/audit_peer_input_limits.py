"""Audit distinct visible inputs, including interactor variants, in frozen cohorts."""
from historical_positive_common import *
records=[]
for folder in sorted((ROOT/'R72/contexts').iterdir()):
    ix=read(folder/'index.json');meta=read(folder/'complete.json');lookup={(v['subject_id'],v['seed']):v for v in ix}
    groups=meta['cohorts']+[[sid] for sid in sorted({v['subject_id'] for v in ix if not v['peer_ids']})]
    seen=set()
    for seed in [0,1,2]:
        for group in groups:
            ident=(tuple(group),seed)
            if ident in seen:continue
            seen.add(ident);inputs=set();raw=0
            for sid in group:
                row=lookup[(sid,seed)];inputs.update((row['subject_key'],k) for k in row['support_keys']+row['query_keys']);raw+=31+len(row['query_keys'])
            records.append({'benchmark':meta['benchmark'],'seed':seed,'cohort_members':group,'distinct_visible_inputs':len(inputs),'support_plus_query_variants':raw})
assert all(v['distinct_visible_inputs']<=1000 for v in records)
write_json(ROOT/'R72/input_limit_audit.json',{'groups':len(records),'max_distinct_visible_inputs':max(v['distinct_visible_inputs'] for v in records),
    'max_raw_support_plus_query_variants':max(v['support_plus_query_variants'] for v in records),'over1000':0,
    'max_record':max(records,key=lambda v:v['distinct_visible_inputs']),'finished':now()})
print(json.dumps(read(ROOT/'R72/input_limit_audit.json')))
