"""Read public official sources only; do not resume paused Codabench access."""
import sys,subprocess,concurrent.futures
import requests
from common import *
def main():
    out=HERE/'official';out.mkdir(exist_ok=True)
    cli=Path('C:/Users/Administrator/.codex/skills/anysearch/scripts/anysearch_cli.py')
    assert sha(cli)=='f029d4841d856085e27ebd226d36714bfc45898ee1f60f210d04195fca868a7d'
    p=subprocess.run([sys.executable,str(cli),'extract','https://github.com/aims-foundations/paiec_baseline'],capture_output=True,text=True,encoding='utf-8',timeout=90)
    text=p.stdout
    if 'api_key' in text or 'auto_registered' in text:text='Credential-bearing service output discarded.'
    (out/'anysearch_official_extract.txt').write_text(text,encoding='utf-8')
    repo='aims-foundations/paiec_baseline';records=[]
    def get(url,path):
        r=requests.get(url,timeout=40);r.raise_for_status();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.content)
        return {'url':url,'path':str(path),'bytes':len(r.content),'sha256':sha(path)}
    try:
        records.append(get(f'https://api.github.com/repos/{repo}/commits/main',out/'head.json'));head=read(out/'head.json')['sha']
        records.append(get(f'https://api.github.com/repos/{repo}/git/trees/{head}?recursive=1',out/'tree.json'))
        paths=[v['path'] for v in read(out/'tree.json')['tree'] if v['type']=='blob' and Path(v['path']).suffix in {'.py','.md','.json','.txt','.toml','.yaml','.yml','.sh'} and Path(v['path']).name!='.env']
        snap=out/'github'/head
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(lambda name:get(f'https://raw.githubusercontent.com/{repo}/{head}/{name}',snap/name),paths):records.append(result)
        status='refreshed_public_head'
    except Exception as exc:
        status='refresh_failed_using_previous_snapshot';head='cc88a959c1e5e7462f31a9215baba066ce6d6e2b';snap=Path('D:/nips-comp/paiec-official/starting-kit/github')/head
        records.append({'error':type(exc).__name__})
    result={'retrieved':now(),'status':status,'commit':head,'snapshot':str(snap),'anysearch_exit':p.returncode,'records':records,'official_zip':'not_obtained_user_paused','hosted_score':None}
    write(HERE/'official_manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
