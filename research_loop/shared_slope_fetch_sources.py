"""Archive primary sources; AnySearch discovery is kept separately from evidence."""
import hashlib,json,subprocess,sys
from pathlib import Path
import requests

root=Path(__file__).resolve().parents[1]/'paiec_baseline/runs/research_loop/R75/literature'
root.mkdir(parents=True,exist_ok=True)
cli=Path('C:/Users/Administrator/.codex/skills/anysearch/scripts/anysearch_cli.py')
assert hashlib.sha256(cli.read_bytes()).hexdigest()=='f029d4841d856085e27ebd226d36714bfc45898ee1f60f210d04195fca868a7d'
for name,args in [('academic_discovery',['get_sub_domains','--domain','academic']),('code_discovery',['get_sub_domains','--domain','code']),
                  ('paper_search',['search','Regularized multi task learning','--domain','academic','--sub_domain','academic.search','--max_results','3']),
                  ('repo_extract',['extract','https://github.com/jiayuzhou/MALSAR'])]:
    out=subprocess.run([sys.executable,str(cli),*args],capture_output=True,text=True,encoding='utf-8',timeout=90)
    # Only known public-result text is retained; no credentials/config are read or copied.
    txt=out.stdout
    if 'api_key' in txt or 'auto_registered' in txt:raise RuntimeError('Service credential response must not be persisted')
    (root/f'{name}.txt').write_text(txt,encoding='utf-8')
session=requests.Session();manifest=[]
def save(url,name):
    r=session.get(url,timeout=60);r.raise_for_status();p=root/name;p.write_bytes(r.content)
    if name.endswith('.pdf'):assert r.content.startswith(b'%PDF')
    manifest.append({'url':url,'file':str(p),'sha256':hashlib.sha256(r.content).hexdigest(),'bytes':len(r.content)})
    return r
repo='jiayuzhou/MALSAR';commit=session.get(f'https://api.github.com/repos/{repo}/commits/master',timeout=60).json()['sha']
tree=session.get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1',timeout=60).json()['tree']
paths=[v['path'] for v in tree if v['type']=='blob' and ('mean' in v['path'].lower() or 'mtl.m' in v['path'].lower() or 'l2' in v['path'].lower() or v['path'].startswith('manual'))]
(root/'repository_tree_candidates.json').write_text(json.dumps({'commit':commit,'paths':paths},indent=2),encoding='utf-8')
save(f'https://raw.githubusercontent.com/{repo}/{commit}/README.md','MALSAR_README.md')
for path in paths:
    if 'mean' in path.lower() or 'l2' in path.lower() or path.lower().endswith('manual.pdf') or path.endswith('Least_SRMTL.m'):
        save(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}',path.replace('/','__'))
errors=[]
for url in ['http://www0.cs.ucl.ac.uk/staff/m.pontil/reading/mt-kdd.pdf','https://www.cs.ucl.ac.uk/staff/M.Pontil/reading/mt-kdd.pdf']:
    try:
        save(url,'Evgeniou_Pontil_KDD2004.pdf');break
    except Exception as e:errors.append({'url':url,'error':type(e).__name__})
(root/'retrieval_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
(root/'manifest.json').write_text(json.dumps({'repo':repo,'commit':commit,'sources':manifest},indent=2),encoding='utf-8')
print(json.dumps({'commit':commit,'saved':len(manifest),'candidate_paths':paths},indent=2))
