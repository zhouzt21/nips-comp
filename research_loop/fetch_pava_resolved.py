from historical_positive_common import *
import requests
root=ROOT/'R69';records=[]
def get(url,name,pdf=False):
    try:
        r=requests.get(url,timeout=40);r.raise_for_status()
        if pdf:assert r.content.startswith(b'%PDF'),r.headers.get('Content-Type')
        p=root/name;p.write_bytes(r.content);records.append({'url':url,'path':str(p),'sha256':sha(p),'bytes':len(r.content)})
        return r
    except Exception as e:records.append({'url':url,'error':type(e).__name__+': '+str(e)[:250]})
r=get('https://scholarlypublications.universiteitleiden.nl/download/1887/3515640/1?col=81522199280002711','pava_original.pdf',True)
if r is not None:
    try:
        import fitz
        doc=fitz.open(stream=r.content,filetype='pdf');(root/'pava_original.txt').write_text('\n'.join(f'PAGE {i+1}\n'+p.get_text() for i,p in enumerate(doc)),encoding='utf-8')
    except ImportError:
        import subprocess
        subprocess.run(['D:/texlive/2025/bin/windows/pdftotext.exe','-layout',str(root/'pava_original.pdf'),str(root/'pava_original.txt')],check=True)
r=get('https://api.github.com/repos/cran/monotone/git/trees/master?recursive=1','author_tree.json')
if r is not None:
    tree=r.json();head=tree['sha']
    for v in tree['tree']:
        path=v['path']
        if path.startswith('src/') and path.endswith(('.c','.cpp','.h')):get(f'https://raw.githubusercontent.com/cran/monotone/{head}/{path}','author_'+Path(path).name)
write_json(root/'pava_resolved_retrieval.json',records);print(json.dumps(records))
