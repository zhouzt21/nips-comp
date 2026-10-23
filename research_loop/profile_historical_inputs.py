"""Input-only estimate of exact feature-cache reuse; never inspect outcomes."""
import json
import time
from collections import Counter
from historical_positive_common import *
import pandas as pd
from mlp.model import _SUBJECT_FIELDS,_string

def main():
    start=time.perf_counter()
    folder=Path('D:/paiec_payload/data/alignment_faking/formatted_tables')
    frame=pd.read_parquet(folder/'subjects.parquet',columns=['subject_id',*_SUBJECT_FIELDS])
    frame=frame.drop_duplicates('subject_id')
    keys=[tuple(_string(row[k]) for k in _SUBJECT_FIELDS) for row in frame.to_dict('records')]
    counts=Counter(keys)
    record={'benchmark':'alignment_faking','subject_ids':len(frame),'unique_encoded_subject_inputs':len(counts),
            'largest_exact_duplicates':sorted(counts.values(),reverse=True)[:10],
            'seconds':time.perf_counter()-start,'purpose':'input-only exact-cache cost diagnosis; no labels or scores'}
    write_json(ROOT/'R62/input_cache_profile.json',record)
    print(json.dumps(record))

if __name__=='__main__':main()
