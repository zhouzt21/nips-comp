"""Version audit of actually obtained public files; never calls them a Starting Kit."""
import ast,difflib
from starting_kit_audit_common import *

snap=github();out=ROOT/'audit';out.mkdir(exist_ok=True)
symbols=['PIPELINE_RUNTIME_VERSION','PAIEC_STREAM_ROLE','configure','acquisition_finished','evaluation_finished','prediction_workers','competition_contract']
keys=['README.md','tools/streaming_ingestion.py','check_submission_zip.py','tools/smoke_test.py','tools/run_smoke_test.py','tools/check_submission_zip.py','empirical_mean/model.py']
comparisons=[]
for name in keys:
    remote=snap/name;local=REPO/'paiec_baseline'/name
    row={'file':name,'github_exists':remote.exists(),'local_exists':local.exists(),'kit_exists':None,'kit_comparison':'not_obtained'}
    for label,path in [('github',remote),('local',local)]:
        if path.exists():row[label+'_sha256']=sha(path)
    if remote.exists():
        text=remote.read_text(encoding='utf-8');(out/(name.replace('/','__')+'.numbered.txt')).write_text(numbered(remote),encoding='utf-8')
        row['symbols']={s:[i for i,line in enumerate(text.splitlines(),1) if s in line] for s in symbols}
        if remote.suffix=='.py':
            row['functions']=[{'name':n.name,'line':n.lineno,'arguments':ast.unparse(n.args)} for n in ast.walk(ast.parse(text)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        if local.exists():
            row['identical_bytes']=row['github_sha256']==row['local_sha256']
            diff=''.join(difflib.unified_diff(local.read_text(encoding='utf-8').splitlines(True),text.splitlines(True),fromfile='local/'+name,tofile='github/'+name))
            (out/(name.replace('/','__')+'.diff')).write_text(diff,encoding='utf-8')
    comparisons.append(row)
issue=read(RETRIEVAL/'issue3.json');head=read(RETRIEVAL/'github_head.json')
write(out/'source_comparison.json',{'date':now(),'github_commit':head['sha'],'github_commit_date':head['commit']['committer']['date'],
    'issue3':{'state':issue['state'],'comments':issue['comments'],'author_association':issue['author_association'],'created_at':issue['created_at'],'updated_at':issue['updated_at'],
              'evidence_type':'community report; not a maintainer confirmation or downloaded Kit'},'files':comparisons})
access={'date':now(),'starting_kit_downloaded':False,'latest_version_verified':False,'account_access_verified':False,
        'authenticated_browser_session_available_to_agent':False,'methods':[
        {'method':'AnySearch extract','url':'https://www.codabench.org/competitions/17828/','result':'extract_failed'},
        {'method':'requests with certificate verification','result':'TLS SSLEOFError before HTTP response'},
        {'method':'Microsoft Edge via existing Playwright CLI 0.1.21','result':'net::ERR_CONNECTION_CLOSED; chrome-error page',
         'screenshot':str(ROOT/'output/playwright/codabench-connection-closed.png')}],
        'blocker_class':'network transport plus no connected authenticated session; account permission is unknown',
        'api_download_attempted':False,'api_reason':'No official documented Starting Kit download endpoint acquired; did not guess URLs or use submission token.',
        'manual_download_pending':True}
write(out/'access_attempts.json',access)
write(REPO/'research_loop/PAIEC_STARTING_KIT_VERSION_MANIFEST.json',{'status':'awaiting_official_zip','audit_date':now(),
      'official_page':'https://www.codabench.org/competitions/17828/','kit':{'downloaded':False,'file':None,'bytes':None,'sha256':None,'runtime_version':None,'authenticated':None,'latest':None,'candidates_observed':[]},
      'public_github':{'commit':head['sha'],'snapshot':str(snap),'files_downloaded':40,'retrieval':str(RETRIEVAL/'public_source_manifest.json')},
      'comparison':str(out/'source_comparison.json'),'access_attempts':str(out/'access_attempts.json'),'tests':str(ROOT/'tests/summary.json')})
print(json.dumps({'files_compared':len(comparisons),'github_commit':head['sha'],'kit':'not obtained','different_local_files':[x['file'] for x in comparisons if x.get('identical_bytes') is False]}))
