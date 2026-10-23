from historical_positive_common import *
import requests
root=ROOT/'R69';head=read(root/'retrieval.json')['official_commit'];records=[]
for name in ['tools/streaming_ingestion.py','tools/smoke_test.py','tools/prepare_data.py']:
    url=f'https://raw.githubusercontent.com/aims-foundations/paiec_baseline/{head}/{name}'
    r=requests.get(url,timeout=30);r.raise_for_status();p=root/'official_baseline'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(r.content)
    records.append({'url':url,'path':str(p),'sha256':sha(p),'bytes':len(r.content)})
write_json(root/'runner_retrieval.json',records);print(json.dumps(records))
