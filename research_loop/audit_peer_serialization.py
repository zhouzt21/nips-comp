"""Check reconstructed interactor hashes are unambiguous at the visible-string level."""
from collections import defaultdict
import hashlib
import pandas as pd
from historical_positive_common import *
from audit_peer_coverage import plain

rows=[]
for f in range(3):
    for folder in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
        name=folder.name+'/formatted_tables';values=pd.read_parquet(Path('D:/paiec_payload/data')/name/'response.parquet',columns=['interactors']).interactors.drop_duplicates()
        mapping=defaultdict(set)
        for v in values:mapping[hashlib.sha256(str(v).encode()).hexdigest()].add(plain(v))
        collisions=sum(len(v)>1 for v in mapping.values());assert collisions==0,(name,collisions)
        rows.append({'benchmark':name,'unique_raw_interactor_values':len(values),'canonical_hash_collisions':collisions})
write_json(ROOT/'R72/serialization_audit.json',{'benchmarks':rows,'ambiguous_interactor_hashes':0,'finished':now()})
print(json.dumps({'benchmarks':len(rows),'ambiguous_interactor_hashes':0}))
