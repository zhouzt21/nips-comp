"""AnySearch discovery, retained separately from original full-text evidence."""
import subprocess,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent/'literature/dual_track_demand'
CLI=Path('C:/Users/Administrator/.codex/skills/anysearch/scripts/anysearch_cli.py')
def scrub(v):
    if isinstance(v,dict): return {k:scrub(x) for k,x in v.items() if k.lower() not in ['api_key','token','authorization','secret']}
    if isinstance(v,list): return [scrub(x) for x in v]
    return v
def main():
    assert hashlib.sha256(CLI.read_bytes()).hexdigest()=='f029d4841d856085e27ebd226d36714bfc45898ee1f60f210d04195fca868a7d'
    ROOT.mkdir(parents=True,exist_ok=True)
    calls=[('academic_capabilities',['get_sub_domains','--domain','academic']),('code_capabilities',['get_sub_domains','--domain','code']),('paper_search',['search','General scales unlock AI evaluation with explanatory and predictive power','--domain','academic','--sub_domain','academic.search','--max_results','5']),('code_search',['search','build_annotation_prompt','--domain','code','--sub_domain','code.snippet','--sub_domain_params',json.dumps({'repo':'Kinds-of-Intelligence-CFI/ADeLe-AIEvaluation'}),'--max_results','5']),('paper_extract',['extract','https://www.nature.com/articles/s41586-026-10303-2'])]
    for name,args in calls:
        p=subprocess.run([sys.executable,str(CLI),*args],capture_output=True,text=True,encoding='utf-8',timeout=100)
        try: output=scrub(json.loads(p.stdout))
        except ValueError:
            output=p.stdout if not any(k in p.stdout.lower() for k in ['api_key','auto_registered']) else '[raw output withheld credential-shaped field]'
        (ROOT/(name+'.json')).write_text(json.dumps({'retrieved_utc':datetime.now(timezone.utc).isoformat(),'args':args,'returncode':p.returncode,'output':output},ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'name':name,'returncode':p.returncode,'output_characters':len(p.stdout)}),flush=True)
if __name__=='__main__':main()
