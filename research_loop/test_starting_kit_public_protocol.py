"""Synthetic public-client checks only; no Hosted calls or competition outcomes."""
import sys,os,subprocess,importlib.util,contextlib,time,copy,math,zipfile
from starting_kit_audit_common import *
sys.dont_write_bytecode=True
SNAP=github();OUT=ROOT/'tests';OUT.mkdir(exist_ok=True)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
runtime=load(SNAP/'tools/streaming_ingestion.py','public_streaming')
mean=load(SNAP/'empirical_mean/model.py','public_mean')
results=[]

SUB={'normalized_name':'synthetic-own','provider':'synthetic','release_date':'','access_date':'','harness':'synthetic','harness_version':'1','reasoning_effort':'','subject_features_extra':''}
def subject(i):return dict(SUB,normalized_name=f'synthetic-peer-{i}')
def item(text='synthetic-query',meta='v=1',bench='synthetic-B'):
    return {'item_content':text,'item_features':meta,'interactors':'','benchmark_id':bench}
def label(s,i,y):return [[s,i],y]
OWN=[SUB,item()]
fixtures=[
    ('B0_empty',0,[],[OWN]),
    ('B1_own',1,[label(SUB,item('own-support'),1)],[OWN]),
    ('one_peer',3,[label(SUB,item('own-support'),1),label(subject(1),item(),0)],[OWN,[subject(1),item()]]),
    ('three_peers',3,[label(SUB,item('own-support'),1)]+[label(subject(i),item(),i%2) for i in range(1,4)],[OWN]),
    ('shared_benchmark_different_subjects',3,[label(subject(1),item(),0),label(subject(2),item(),1)],[OWN,[subject(2),item()]]),
    ('identical_visible_subjects',3,[label(dict(SUB),item('support1'),1),label(dict(SUB),item('support2'),0)],[OWN]),
    ('same_text_different_metadata',3,[label(subject(1),item(meta='v=2'),1)],[OWN]),
    ('no_matching_peer_input',3,[label(subject(1),item('different-query'),0)],[OWN]),
    ('repeated_responses',3,[label(SUB,item('support'),1),label(SUB,item('support'),0)],[OWN,copy.deepcopy(OWN),[SUB,item(meta='v=2')]]),
    ('mixed_benchmarks',3,[label(SUB,item('support'),1),label(subject(1),item(bench='synthetic-C'),0)],[OWN]),
]
def probe(inp,labeled=None):
    # Finite diagnostic value, never a hidden-data output channel: all labels below are authored here.
    return float((1+sum(v[1] for v in (labeled or [])))/(2+len(labeled or [])))
def valid(x):
    v=float(x)
    if not math.isfinite(v) or not 0<=v<=1:raise ValueError('invalid probability')
    return v
def grouping(inputs,n,deadline):
    unique=[];groups=[];keys={}
    for row in inputs:
        key=json.dumps(row,sort_keys=True)
        if key not in keys:keys[key]=len(unique);unique.append(row);groups.append([])
        groups[keys[key]].append(sum(len(g) for g in groups))
    return unique,groups
def execute_case(name,event,expected_error=None,broken_stream=False):
    folder=OUT/'protocol'/name;folder.mkdir(parents=True,exist_ok=True)
    config=folder/'config.json';write(config,{'protocol':'streaming_alc_v1'})
    frozen=copy.deepcopy(event);observed=[];requests=[];logs=[]
    def exchange(payload):
        requests.append(copy.deepcopy(payload))
        if len(requests)==1:return copy.deepcopy(event)
        return {'protocol':'streaming_alc_v1','type':'finished','predictions_csv':'synthetic_id,probability\n'}
    def stream(model_path,inputs,labeled,**kw):
        if broken_stream:return
        for i,row in enumerate(inputs):
            observed.append({'input':copy.deepcopy(row),'labeled':copy.deepcopy(labeled),'kwargs':kw})
            yield i,probe(row,labeled)
    error=None
    try:
        stats=runtime.run_streaming(config_path=config,predict=probe,acquisition_function=None,model_path=__file__,output_dir=folder/'outputs',
            concurrency=4,deadline=time.monotonic()+10,per_call_timeout=1,acquisition_timeout=1,
            call_window=lambda *_:contextlib.nullcontext(),validate_score=valid,prediction_stream=stream,group_inputs=grouping,log=logs.append,exchange=exchange)
    except Exception as exc:
        stats=None;error=f'{type(exc).__name__}: {exc}'
    if expected_error:assert error and expected_error in error,(name,error)
    else:
        assert error is None,(name,error)
        assert all(v['labeled']==event['labeled'] for v in observed)
        assert event==frozen
        assert all(math.isfinite(v) and 0<=v<=1 for v in requests[-1].get('probabilities',[]))
    result={'case':name,'passed':True,'error':error,'stats':stats,'observed':observed,'exchange_payloads':requests,'logs':logs,
            'scope':'local public run_streaming with injected synthetic exchange, grouping and prediction_stream; not real worker/coordinator'}
    write(folder/'result.json',result);results.append({'case':name,'passed':True,'expected_error':expected_error,'result':str(folder/'result.json')})

for name,n,evidence,targets in fixtures:
    event={'protocol':'streaming_alc_v1','type':'evaluate','event_id':'synthetic-'+name,'budget':n,'labeled':evidence,'targets':[{'input':t} for t in targets]}
    write(OUT/'fixtures'/f'{name}.json',event);execute_case(name,event)

# These old-client behavior checks expose the boundary; no newer Kit behavior is inferred.
for kind in ['configure','acquisition_finished','evaluation_finished']:
    event={'protocol':'streaming_alc_v1','type':kind,'event_id':'synthetic-'+kind,'labeled':[]}
    execute_case('old_client_rejects_'+kind,event,'Unknown streaming coordinator event')
execute_case('wrong_protocol',{'protocol':'synthetic_v2','type':'evaluate'},'Unexpected streaming coordinator protocol')
execute_case('incomplete_predictions',read(OUT/'fixtures/B1_own.json'),'incomplete predictions',broken_stream=True)
bad=read(OUT/'fixtures/B1_own.json');bad['budget']=0
execute_case('B0_nonempty_not_rejected_by_client',bad)

# Direct pure-function behavior for identical visible configurations and metadata.
assert mean.predict(OWN,fixtures[5][2])==.5
assert mean.predict(OWN,[label(SUB,item('a'),1),label(SUB,item('a'),0),label(SUB,item('b'),1)])==2/3
assert mean.predict(OWN,[label(subject(1),item(),1)])==.5

def child(cmd,name,cwd=SNAP):
    start=time.perf_counter();safe_names=['SystemRoot','WINDIR','PATH','TEMP','TMP','USERPROFILE','APPDATA','LOCALAPPDATA','COMSPEC','PATHEXT']
    env={k:os.environ[k] for k in safe_names if k in os.environ};env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONIOENCODING']='utf-8'
    try:
        p=subprocess.run(cmd,cwd=cwd,env=env,text=True,encoding='utf-8',errors='replace',capture_output=True,timeout=90)
        rec={'name':name,'command':cmd,'cwd':str(cwd),'returncode':p.returncode,'seconds':time.perf_counter()-start}
        (OUT/(name+'.stdout.txt')).write_text(p.stdout,encoding='utf-8');(OUT/(name+'.stderr.txt')).write_text(p.stderr,encoding='utf-8')
    except subprocess.TimeoutExpired:rec={'name':name,'command':cmd,'returncode':None,'error':'90-second timeout','seconds':time.perf_counter()-start}
    write(OUT/(name+'.json'),rec);return rec

# A disposable synthetic diagnostic package, never a real submission.
pack=OUT/'synthetic_probe';pack.mkdir(exist_ok=True)
(pack/'model.py').write_text('def predict(input, labeled=None):\n    return 0.5\n',encoding='utf-8')
archive=OUT/'synthetic_probe_DO_NOT_SUBMIT.zip'
with zipfile.ZipFile(archive,'w') as z:z.write(pack/'model.py','model.py')
native=[]
native.append(child([sys.executable,str(SNAP/'check_submission_zip.py'),'--help'],'checker_help'))
native.append(child([sys.executable,str(SNAP/'check_submission_zip.py'),str(archive)],'native_github_zip_check'))
native.append(child([sys.executable,str(SNAP/'tools/smoke_test.py'),'--help'],'github_smoke_help'))
native.append(child([sys.executable,str(SNAP/'tools/smoke_test.py'),'--mock'],'native_github_ble_mock'))

# Two fresh OS processes on identical synthetic input; this is a harness test, not Hosted restart evidence.
worker=OUT/'fresh_process_probe.py'
worker.write_text('import json,sys,importlib.util\nfrom pathlib import Path\ns=importlib.util.spec_from_file_location("m",sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\ne=json.loads(Path(sys.argv[2]).read_text());print(json.dumps([m.predict(t["input"],e["labeled"]) for t in e["targets"]]))\n',encoding='utf-8')
for i in [1,2]:native.append(child([sys.executable,str(worker),str(SNAP/'empirical_mean/model.py'),str(OUT/'fixtures/three_peers.json')],f'fresh_process_{i}',OUT))
assert (OUT/'fresh_process_1.stdout.txt').read_text()==(OUT/'fresh_process_2.stdout.txt').read_text()

# Frozen R72 numeric operator on synthetic vectors. No text encoder or real data.
sys.path.insert(0,str(REPO/'research_loop'))
from peer_current_predictor import predict_current,joint
import numpy as np
rng=np.random.default_rng(7300);x=rng.normal(size=(31,640))*.02;y=rng.integers(2,size=31).astype(float);q=rng.normal(size=(3,640))*.02
budget=[0,1,3,7,15,31];rows=[{'x':x,'y':y,'keys':[f'own{i}' for i in range(31)]}]
for j in range(3):rows.append({'x':rng.normal(size=(31,640))*.02,'y':rng.integers(2,size=31).astype(float),'keys':[f'peer{j}_{i}' for i in range(31)]})
base=np.column_stack([joint([{'x':x[:n],'y':y[:n]}],q) for n in budget]);errors={}
for k in [0,1,3]:
    ctx=[[{'x':v['x'][:n],'y':v['y'][:n],'keys':v['keys'][:n]} for v in rows[:k+1]] for n in budget]
    perms=[[[rng.permutation(n) for _ in range(k)] for n in budget] for _ in range(2)]
    pred=predict_current(base,q,ctx,['peer0_0','peer1_1','absent'],4.,4.,perms)
    assert all(np.isfinite(v).all() and ((0<=v)&(v<=1)).all() for v in pred.values())
    assert all(np.array_equal(v[:,0],base[:,0]) for v in pred.values())
    if k==0:assert all(np.array_equal(v,base) for v in pred.values())
    errors[str(k)]={'finite':True,'B0_exact':True,'nopeer_exact':k==0,'methods':len(pred)}
write(OUT/'r72_synthetic_operator.json',{'cases':errors,'scope':'numeric core with authored features; not a predict(input,labeled) adapter or Hosted test','source_sha256':sha(REPO/'research_loop/peer_current_predictor.py')})
summary={'finished':now(),'public_protocol_cases':results,'native_commands':native,'fresh_process_repeat_identical':True,'R72_synthetic':errors,
         'kit_native_tests':'not_run_kit_not_obtained','real_hosted_events':0,'trained_models':0,'official_submissions':0}
write(OUT/'summary.json',summary)
print(json.dumps({'protocol_cases':len(results),'native':native,'R72_synthetic_cases':len(errors),'real_hosted_events':0}),flush=True)
