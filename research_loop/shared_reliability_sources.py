"""Narrow AnySearch refresh; original R75 and DomainBed sources remain authority."""
import subprocess,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
CLI=Path('C:/Users/Administrator/.codex/skills/anysearch/scripts/anysearch_cli.py')
OUT=Path(__file__).parent/'literature/shared_reliability'
def scrub(x):
    if isinstance(x,dict):return {k:scrub(v) for k,v in x.items() if k.lower() not in ['api_key','token','authorization','secret']}
    if isinstance(x,list):return [scrub(v) for v in x]
    return x
def main():
    assert hashlib.sha256(CLI.read_bytes()).hexdigest()=='f029d4841d856085e27ebd226d36714bfc45898ee1f60f210d04195fca868a7d'
    OUT.mkdir(parents=True,exist_ok=True)
    calls=[('academic_capabilities',['get_sub_domains','--domain','academic']),
           ('mtl_search',['search','Regularized Multi Task Learning Evgeniou Pontil 2004','--domain','academic','--sub_domain','academic.search','--max_results','3']),
           ('domainbed_search',['search','In Search of Lost Domain Generalization Gulrajani Lopez Paz','--domain','academic','--sub_domain','academic.search','--max_results','3']),
           ('domainbed_original',['extract','https://arxiv.org/abs/2007.01434']),
           ('author_selection_code',['extract','https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/model_selection.py'])]
    for name,args in calls:
        p=subprocess.run([sys.executable,str(CLI),*args],capture_output=True,text=True,encoding='utf-8',timeout=90)
        try:output=scrub(json.loads(p.stdout))
        except ValueError:output=p.stdout if not any(v in p.stdout.lower() for v in ['api_key','auto_registered']) else '[credential-shaped raw response withheld]'
        (OUT/(name+'.json')).write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'args':args,'returncode':p.returncode,'output':output},ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'name':name,'returncode':p.returncode,'characters':len(p.stdout)}),flush=True)
if __name__=='__main__':main()
