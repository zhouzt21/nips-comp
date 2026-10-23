"""Read-only historical preservation snapshots; never repairs old artifacts."""
import re,sys
from common import *
def main():
    mode=sys.argv[1];path=ART/'historical_preservation_before.json'
    if mode=='before':
        if path.exists():raise RuntimeError('Snapshot already exists')
        root=BASE/'runs/research_loop';files=set()
        for p in root.iterdir():
            match=re.match(r'^R(\d+)(?:_|$)',p.name)
            if match and 1<=int(match.group(1))<=81:
                if p.is_file():files.add(p)
                else:files.update(v for v in p.rglob('*') if v.is_file() and '__pycache__' not in v.parts)
        files.update(p for p in (REPO/'research_loop').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
        files.update(Path('D:/nips-comp/paiec-dataset/paiec_git_backups').glob('*/workspace/paiec_baseline/dist/smoothed_mean_strength2.zip'))
        files.update((BASE/'dist').glob('smoothed_mean_strength2.zip'))
        entries=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
        write(path,{'created':now(),'entries':entries,'files':len(entries),'bytes':sum(v['bytes'] for v in entries)})
        print(json.dumps({'files':len(entries),'bytes':sum(v['bytes'] for v in entries)}),flush=True)
    else:
        entries=read(path)['entries'];errors=[v['path'] for v in entries if not Path(v['path']).exists() or sha(v['path'])!=v['sha256']]
        result={'finished':now(),'files_checked':len(entries),'unchanged':not errors,'changed':errors}
        write(HERE/'historical_preservation.json',result);print(json.dumps(result));assert not errors
if __name__=='__main__':main()
