"""Launch actual ZIPs in a clean independent environment; preserve all command logs."""
import os,sys,subprocess,time,shutil,zipfile
from common import *
WORK=Path('D:/nips-comp/paiec-final-acceptance')
PYTHON=WORK/'win312/Scripts/python.exe'
def env(root):
    keep=['SystemRoot','WINDIR','PATH','TEMP','TMP','USERPROFILE','APPDATA','LOCALAPPDATA','COMSPEC','PATHEXT']
    e={k:os.environ[k] for k in keep if k in os.environ};e.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HOME=str(root/'empty_hf'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    return e
def run(cmd,name,cwd,timeout=7200):
    logs=HERE/'logs';logs.mkdir(exist_ok=True);t=time.perf_counter()
    with (logs/(name+'.stdout.txt')).open('w',encoding='utf-8') as out,(logs/(name+'.stderr.txt')).open('w',encoding='utf-8') as err:
        try:p=subprocess.run(cmd,cwd=cwd,env=env(cwd),stdout=out,stderr=err,timeout=timeout);code=p.returncode;error=None
        except subprocess.TimeoutExpired:code=None;error='timeout'
    rec={'command':list(map(str,cmd)),'cwd':str(cwd),'exit_code':code,'error':error,'seconds':time.perf_counter()-t,'stdout':str(logs/(name+'.stdout.txt')),'stderr':str(logs/(name+'.stderr.txt'))}
    write(ART/'commands'/(name+'.json'),rec);print(json.dumps({'name':name,'exit_code':code,'seconds':rec['seconds']}),flush=True);return rec
def main():
    manifests=read(HERE/'build_manifest.json');WORK.mkdir(exist_ok=True)
    fixtures=WORK/'fixtures';shutil.copytree(HERE/'fixtures',fixtures,dirs_exist_ok=True)
    worker=WORK/'accept_worker.py';shutil.copyfile(HERE/'accept_worker.py',worker);rows=[]
    official=Path(read(HERE/'official_manifest.json')['snapshot'])
    for kind,v in manifests.items():
        root=WORK/kind;root.mkdir(exist_ok=True);package=root/'submission';package.mkdir(exist_ok=True)
        assert sha(v['zip'])==v['sha256']
        with zipfile.ZipFile(v['zip']) as z:z.extractall(package)
        # Check both current local and fixed latest upstream copies; neither is Hosted.
        for label,checker in [('repo',BASE/'check_submission_zip.py'),('upstream',official/'check_submission_zip.py')]:
            rows.append(run([str(PYTHON),'-I','-B',str(checker),v['zip']],kind+'_'+label+'_zipcheck',root,120))
        for mode in ['full','fresh1','fresh2']:
            result=root/(mode+'.json');rows.append(run([str(PYTHON),'-I','-B',str(worker),str(package),str(fixtures),str(result),'full' if mode=='full' else 'fresh',kind],kind+'_'+mode,root))
            if result.exists():shutil.copyfile(result,ART/(kind+'_'+mode+'.json'))
        if (root/'fresh1.json').exists() and (root/'fresh2.json').exists():assert read(root/'fresh1.json')['predictions']==read(root/'fresh2.json')['predictions']
    # Baseline-specific upstream smoke tool is a tool health check, never candidate acceptance.
    rows.append(run([str(PYTHON),'-I','-B',str(official/'tools/smoke_test.py'),'--mock'],'upstream_ble_mock',WORK,180))
    rows.append(run([str(PYTHON),'-I','-m','pip','check'],'clean_pip_check',WORK,60))
    write(HERE/'acceptance_commands.json',{'finished':now(),'commands':rows,'hosted_score':None,'real_hosted_calls':0,'official_submissions':0})
if __name__=='__main__':main()
