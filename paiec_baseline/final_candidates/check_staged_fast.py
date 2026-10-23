"""Apply the existing public-source rules using one git cat-file batch process."""
import sys,subprocess,json,importlib.util
from pathlib import PurePosixPath
from common import *
def main():
    spec=importlib.util.spec_from_file_location('source_policy',BASE/'tools/check_public_source.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    entries=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO).split(b'\0');errors=[];count=0
    p=subprocess.Popen(['git','cat-file','--batch'],cwd=REPO,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    for row in entries:
        if not row:continue
        meta,name=row.split(b'\t',1);mode,oid,stage=meta.decode().split();path=PurePosixPath(name.decode());count+=1
        research=path.parts[:1]==('runs',) or path.parts[:2]==('paiec_baseline','runs');allowed=research and path.suffix.lower() in policy.RESEARCH_SUFFIXES
        if (mode not in {'100644','100755'} or stage!='0' or any(x in policy.PRIVATE_DIRECTORIES and not (x=='runs' and allowed) for x in path.parts) or path.name in policy.PRIVATE_FILENAMES or (path.name.startswith('.env') and path.name!='.env.example') or (path.suffix.lower() in policy.ARTIFACT_SUFFIXES and not (allowed and path.suffix.lower()=='.log'))):
            errors.append({'path':str(path),'reason':'prohibited artifact'});continue
        p.stdin.write(oid.encode()+b'\n');p.stdin.flush();head=p.stdout.readline().split();length=int(head[2]);data=p.stdout.read(length);assert p.stdout.read(1)==b'\n'
        if policy.CREDENTIAL.search(data):errors.append({'path':str(path),'reason':'credential pattern; value withheld'})
        if path.suffix=='.json':
            try:v=json.loads(data)
            except (ValueError,UnicodeDecodeError):
                local=path.parts[1:] if path.parts[:1]==('paiec_baseline',) else path.parts
                if not (local[:3]==('research_loop','literature','raw') and data.removeprefix(b'\xef\xbb\xbf').startswith(b'## Query ')):errors.append({'path':str(path),'reason':'invalid JSON'})
            else:
                if isinstance(v,dict) and any(v.get('api_keys',{}).values()):errors.append({'path':str(path),'reason':'nonempty credential config; value withheld'})
    p.stdin.close();p.wait();result={'finished':now(),'files_checked':count,'policy_sha256':sha(BASE/'tools/check_public_source.py'),'passed':not errors,'errors':errors}
    write(HERE/'staged_source_check.json',result);print(json.dumps(result));assert not errors
if __name__=='__main__':main()
