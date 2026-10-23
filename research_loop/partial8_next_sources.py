"""Narrow primary-source retrieval through the user-requested AnySearch CLI."""
import subprocess,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from shared_reliability_sources import scrub
CLI=Path('C:/Users/Administrator/.codex/skills/anysearch/scripts/anysearch_cli.py')
OUT=Path(__file__).parent/'literature/partial8_next'
def main():
    assert hashlib.sha256(CLI.read_bytes()).hexdigest()=='f029d4841d856085e27ebd226d36714bfc45898ee1f60f210d04195fca868a7d'
    OUT.mkdir(parents=True,exist_ok=True)
    calls=[('academic_capabilities',['get_sub_domains','--domain','academic']),
        ('cv_variance_search',['search','Bengio Grandvalet No Unbiased Estimator of the Variance of K-Fold Cross-Validation 2004','--domain','academic','--sub_domain','academic.search','--max_results','3']),
        ('cv_original',['extract','https://www.jmlr.org/papers/v5/grandvalet04a.html']),
        ('grouped_cv_docs',['extract','https://scikit-learn.org/stable/modules/cross_validation.html']),
        ('grouped_cv_code',['extract','https://github.com/scikit-learn/scikit-learn/blob/1.5.2/sklearn/model_selection/_split.py']),
        ('mtl_code',['extract','https://github.com/jiayuzhou/MALSAR/blob/fb9751594983df020ddc4f7e4a40520ee7c37989/MALSAR/functions/SRMTL/Least_SRMTL.m'])]
    for name,args in calls:
        p=subprocess.run([sys.executable,str(CLI),*args],capture_output=True,text=True,encoding='utf-8',timeout=90)
        try:v=scrub(json.loads(p.stdout))
        except ValueError:v=p.stdout if not any(k in p.stdout.lower() for k in ['api_key','auto_registered']) else '[credential-shaped response withheld]'
        (OUT/(name+'.json')).write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'args':args,'returncode':p.returncode,'output':v},ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'name':name,'returncode':p.returncode,'characters':len(p.stdout)}),flush=True)
if __name__=='__main__':main()
