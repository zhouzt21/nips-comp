"""Read-only source helpers for the separate plan_7 audit."""
from pathlib import Path
import hashlib,json,datetime
REPO=Path('D:/nips-comp/nips-comp')
ROOT=Path('D:/nips-comp/paiec-official/starting-kit')
RETRIEVAL=ROOT/'retrieval'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def github():return Path(read(RETRIEVAL/'public_source_manifest.json')['github_snapshot'])
def numbered(p):return '\n'.join(f'{i:04d}: {line}' for i,line in enumerate(Path(p).read_text(encoding='utf-8-sig').splitlines(),1))+'\n'
