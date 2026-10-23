"""Selective publication of source and compact evidence, never datasets or weights."""
import subprocess,sys,re
from common import *
def main():
    raw=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=REPO).decode().split('\0')
    included=[];excluded=[]
    suffixes={'.py','.md','.json','.jsonl','.txt','.log','.csv','.png','.svg','.ps1'}
    bulk_names={'paired_scores.json','decomposition.json','coverage.json','mask_index.json','routes.json'}
    credential=re.compile(rb'(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
    for name in raw:
        if not name:continue
        p=REPO/name;reason=None
        if p.name.startswith('.env'):reason='private_configuration'
        elif p.suffix.lower() not in suffixes:reason='non_source_or_original_download'
        elif p.stat().st_size>2*1024**2:reason='large_local_evidence'
        elif 'runs' in p.parts and (p.name in bulk_names or p.name.startswith(('index_','index.','mask_'))):reason='per_example_local_evidence'
        elif p.name in {'anysearch_raw.json'}:reason='raw_retrieval'
        if reason is None:
            data=p.read_bytes()
            if credential.search(data):reason='credential_pattern_value_withheld'
            elif p.suffix=='.json':
                try:json.loads(data)
                except (ValueError,UnicodeDecodeError):reason='historical_non_json_transcript'
        if reason:excluded.append({'path':name,'bytes':p.stat().st_size,'reason':reason})
        else:included.append(name)
    # Preserve all pre-existing tracked edits requested by the user's latest-repository push.
    tracked=subprocess.check_output(['git','diff','--name-only','-z'],cwd=REPO).decode().split('\0')
    included+=list(filter(None,tracked))
    report={'created':now(),'included_count':len(included),'excluded_count':len(excluded),'excluded_bytes':sum(v['bytes'] for v in excluded),'excluded':excluded,'policy':'Only compact source/config/summary evidence; local raw records are preserved and indexed. No ignored file is force-added.'}
    write(HERE/'publication_scope.json',report)
    if '--stage' in sys.argv:
        paths=ART/'git_pathspec.bin';paths.parent.mkdir(exist_ok=True);paths.write_bytes('\0'.join(included+['paiec_baseline/final_candidates/publication_scope.json']).encode()+b'\0')
        logs=HERE/'logs';logs.mkdir(exist_ok=True)
        with (logs/'git_add.txt').open('w',encoding='utf-8') as log:
            subprocess.run(['git','add','--pathspec-from-file='+str(paths),'--pathspec-file-nul'],cwd=REPO,stdout=log,stderr=log,check=True)
        report['total_staged_changed_files']=len([v for v in subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=REPO).split(b'\0') if v])
        report['included_count_scope']='new or modified paths in this invocation; total_staged_changed_files is cumulative'
        write(HERE/'publication_scope.json',report)
        subprocess.run(['git','add','--','paiec_baseline/final_candidates/publication_scope.json'],cwd=REPO,check=True)
    print(json.dumps({k:v for k,v in report.items() if k!='excluded'}),flush=True)
if __name__=='__main__':main()
