"""Safely preserve and inspect a user-downloaded official Kit, without executing it."""
import argparse,zipfile,stat,shutil,re
from pathlib import PurePosixPath
from starting_kit_audit_common import *

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',type=Path,required=True)
    p.add_argument('--source-page',default='https://www.codabench.org/competitions/17828/')
    p.add_argument('--download-note',default='User supplied file; authenticated/latest provenance requires visible page evidence.')
    args=p.parse_args();src=args.zip.resolve()
    assert src.is_file() and zipfile.is_zipfile(src),'No valid ZIP at supplied path'
    digest=sha(src);out=ROOT/'originals'/digest[:16]/src.name;out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():assert sha(out)==digest
    elif out.resolve()!=src:shutil.copy2(src,out)
    target=(ROOT/'extracted'/digest[:16]).resolve();target.mkdir(parents=True,exist_ok=True)
    inventory=[];seen=set();total=0
    with zipfile.ZipFile(out) as z:
        members=z.infolist();assert len(members)<=10000,'Inspect unexpectedly large archive manually'
        for info in members:
            name=info.filename.replace('\\','/');path=PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and ':' not in name,('Unsafe archive path',name)
            assert not stat.S_ISLNK(info.external_attr>>16),('Symlink member',name)
            for part in path.parts:
                assert part and part.rstrip(' .')==part and not re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',part),('Unsafe Windows member',name)
            canonical=str(path).casefold();assert canonical not in seen,('Duplicate member',name);seen.add(canonical)
            dest=(target/str(path)).resolve();assert dest==target or target in dest.parents
            total+=info.file_size;assert total<=500*1024**2,'Inspect expansion above 500 MiB manually'
        assert z.testzip() is None,'ZIP CRC check failed'
        for info in members:
            dest=target/info.filename.replace('\\','/')
            if info.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
            data=z.read(info)
            if dest.exists():assert hashlib.sha256(dest.read_bytes()).digest()==hashlib.sha256(data).digest()
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            inventory.append({'member':info.filename,'bytes':len(data),'sha256':sha(dest),'path':str(dest)})
    record={'status':'local_zip_integrity_verified','file':str(out),'bytes':out.stat().st_size,'sha256':digest,'source_page':args.source_page,
            'received':now(),'download_note':args.download_note,'authenticated_download':None,'latest_visible_version':None,
            'extracted':str(target),'crc_pass':True,'safe_paths_pass':True,'expanded_bytes':total,'inventory':inventory}
    write(ROOT/'zip_inspection'/f'{digest}.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='inventory'},ensure_ascii=False))
if __name__=='__main__':main()
