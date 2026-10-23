"""Public original-source acquisition for plan_7; no credentials or private API."""
from pathlib import Path
import requests,json,hashlib,datetime,time
from concurrent.futures import ThreadPoolExecutor

ROOT=Path('D:/nips-comp/paiec-official/starting-kit')
OUT=ROOT/'retrieval';OUT.mkdir(parents=True,exist_ok=True)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fetch(url,path):
    start=time.perf_counter();record={'url':url,'path':str(path),'at':stamp()}
    try:
        r=requests.get(url,timeout=25,headers={'User-Agent':'PAIEC-public-runtime-audit/1.0'})
        record.update(status=r.status_code,final_url=r.url,content_type=r.headers.get('Content-Type'))
        r.raise_for_status();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.content)
        record.update(bytes=len(r.content),sha256=hashlib.sha256(r.content).hexdigest())
    except Exception as e:record['error']=f'{type(e).__name__}: {str(e)[:500]}'
    record['seconds']=time.perf_counter()-start
    return record

records=[]
for url,name in [('https://api.github.com/repos/aims-foundations/paiec_baseline/commits/main','github_head.json'),
                 ('https://api.github.com/repos/aims-foundations/paiec_baseline/issues/3','issue3.json'),
                 ('https://api.github.com/repos/aims-foundations/paiec_baseline/issues/3/comments','issue3_comments.json'),
                 ('https://www.codabench.org/competitions/17828/','codabench_page.html')]:
    rec=fetch(url,OUT/name);records.append(rec);print(json.dumps(rec),flush=True)
head=json.loads((OUT/'github_head.json').read_text(encoding='utf-8'))['sha']
treefile=OUT/'github_tree.json';records.append(fetch(f'https://api.github.com/repos/aims-foundations/paiec_baseline/git/trees/{head}?recursive=1',treefile))
tree=json.loads(treefile.read_text(encoding='utf-8'))['tree'];snapshot=ROOT/'github'/head
names=[v['path'] for v in tree if v['type']=='blob' and Path(v['path']).suffix.lower() in {'.py','.md','.json','.yaml','.yml','.txt','.toml','.sh'} and Path(v['path']).name not in {'.env'}]
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(lambda name:fetch(f'https://raw.githubusercontent.com/aims-foundations/paiec_baseline/{head}/{name}',snapshot/name),names))
records+=results
(OUT/'public_source_manifest.json').write_text(json.dumps({'retrieved':stamp(),'github_commit':head,'github_snapshot':str(snapshot),'records':records},indent=2),encoding='utf-8')
print(json.dumps({'commit':head,'files':len(names),'errors':[v for v in records if 'error'in v]}),flush=True)
