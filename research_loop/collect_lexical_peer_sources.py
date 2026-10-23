"""Bounded original-source retrieval; never reads credentials or private APIs."""
from historical_positive_common import *
import requests, time

out=ROOT/'R69';out.mkdir(exist_ok=True)
session=requests.Session();session.headers['User-Agent']='PAIEC-public-research-audit/1.0'
records=[]
def fetch(url,name):
    start=time.perf_counter()
    try:
        r=session.get(url,timeout=30);r.raise_for_status()
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
        records.append({'url':url,'file':str(p),'sha256':sha(p),'status':r.status_code,'bytes':len(r.content),'seconds':time.perf_counter()-start})
        return r
    except Exception as e:
        records.append({'url':url,'error':type(e).__name__+': '+str(e)[:350],'seconds':time.perf_counter()-start})
        return None

r=fetch('https://api.github.com/repos/aims-foundations/paiec_baseline/commits/main','official_head.json')
head=r.json()['sha'] if r is not None else 'cc88a959c1e5e7462f31a9215baba066ce6d6e2b'
fetch(f'https://api.github.com/repos/aims-foundations/paiec_baseline/git/trees/{head}?recursive=1','official_tree.json')
for f in ['README.md','empirical_mean/model.py']:
    fetch(f'https://raw.githubusercontent.com/aims-foundations/paiec_baseline/{head}/{f}','official_baseline/'+f)
fetch('https://www.codabench.org/api/competitions/17828/','codabench_public_api.json')
fetch('https://www.jstatsoft.org/article/view/v102c01/4254','pava_paper.pdf')
fetch('https://raw.githubusercontent.com/scipy/scipy/v1.13.1/scipy/optimize/_isotonic.py','scipy_isotonic.py')
fetch('https://raw.githubusercontent.com/scipy/scipy/v1.13.1/scipy/optimize/_pava/pava.cpp','scipy_pava.cpp')
write_json(out/'retrieval.json',{'retrieved_utc':now(),'official_commit':head,'records':records})
raw=(REPO/'research_loop/literature/lexical_peer_official_interface.txt').read_text(encoding='utf-8-sig')
obj=json.loads(raw[raw.index('{'):]);(out/'official_interface.md').write_text(obj['content'],encoding='utf-8')
print(json.dumps({'head':head,'records':records}),flush=True)
