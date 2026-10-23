"""Run inside an extracted ZIP and clean venv; never imports research code."""
import os,sys,json,time,math,copy,importlib.util,ctypes,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def peak_mb():
    if os.name!='nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    from ctypes import wintypes as w
    class Info(ctypes.Structure):
        _fields_=[('cb',w.DWORD),('PageFaultCount',w.DWORD),*[(k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]]
    i=Info();i.cb=ctypes.sizeof(i);get=ctypes.windll.kernel32.GetCurrentProcess;get.restype=w.HANDLE
    fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[w.HANDLE,ctypes.POINTER(Info),w.DWORD];assert fn(get(),ctypes.byref(i),i.cb)
    return i.PeakWorkingSetSize/1024**2

def main():
    folder=Path(sys.argv[1]).resolve();fixtures=Path(sys.argv[2]).resolve();out=Path(sys.argv[3]).resolve();mode=sys.argv[4];kind=sys.argv[5]
    started=time.perf_counter();access=[];blocked=[]
    forbidden=['d:/paiec_pretrained','d:/paiec_research_cache','d:/paiec_payload','d:/paiec_research_deps','d:/nips-comp/paiec-dataset','d:/nips-comp/nips-comp']
    def audit(event,args):
        if event in ['socket.connect','socket.getaddrinfo']:
            blocked.append(event);raise RuntimeError('Inference network disabled')
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            p=os.fsdecode(args[0]).replace('\\','/').lower()
            if any(p.startswith(x) for x in forbidden):
                blocked.append(p);raise RuntimeError('External research file access blocked')
            if 'weights' in p:access.append(p)
    sys.addaudithook(audit);sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('submission',folder/'model.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    import numpy as np
    from candidate.adapter import canonical,ITEM_FIELDS
    from candidate.core import predict as core
    checkrows=[];timings=[];predictions=[]
    def call(inp,labeled,roster=None):
        begin=time.perf_counter();before=copy.deepcopy([inp,labeled,roster])
        result=m.predict(inp,labeled) if roster is None else m.predict_with_roster(inp,labeled,roster)['prediction']
        timings.append(time.perf_counter()-begin);assert isinstance(result,float) and math.isfinite(result) and 0<=result<=1
        assert [inp,labeled,roster]==before,'input mutation';predictions.append(result);return result
    paths=sorted(fixtures.glob('*.json'))
    if mode=='fresh':paths=paths[:1]
    for path in paths:
        v=json.loads(path.read_text(encoding='utf-8'));subjects=v['subjects'];maxerr=0.;operator_gap=0.
        ns=[0,1,3,7,15,31] if mode!='fresh' else [0,31]
        for n in ns:
            j=v['budgets'].index(n);labels=[r for c in v['contexts'] for r in c[:n]]
            qs=v['queries'] if mode!='fresh' else v['queries'][:1]
            for qi,q in enumerate(qs):
                ref=v['expected'][j][qi];p=call([subjects[0],q],labels);maxerr=max(maxerr,abs(p-ref['r25']))
                if kind=='partial8_shared':
                    p=call([subjects[0],q],labels,subjects);maxerr=max(maxerr,abs(p-ref['partial_adapter']));operator_gap=max(operator_gap,abs(p-ref['partial_operator']))
        f=np.load(path.with_name(path.stem+'_features.npz'));items=[json.loads(k) for k in f['keys']]
        if mode!='fresh':
            encoded=m._runtime.features.encode_items(items);encoded[:,:384]/=2.;encoded[:,384:]/=4.
            feature_error=float(abs(encoded-f['vectors']).max());assert feature_error<2e-6,(path,feature_error)
        else:feature_error=None
        assert maxerr<2e-6,(path,maxerr)
        checkrows.append({'benchmark':v['benchmark'],'fold':v['fold'],'queries':len(qs),'max_probability_error':maxerr,'max_scaled_feature_error':feature_error,'partial_operator_to_adapter_gap':operator_gap})
        print(json.dumps(checkrows[-1]),flush=True)
    synthetic=[]
    if mode!='fresh':
        sub={'normalized_name':'synthetic-own','provider':'test','release_date':'','access_date':'','harness':'local','harness_version':'1','reasoning_effort':'','subject_features_extra':''}
        subs=[sub,*[dict(sub,normalized_name='synthetic-peer-'+str(k)) for k in range(3)]]
        def item(i,group='new-group'):
            return {'item_content':f'Question {i}: infer the answer to a reasoning task number {i*i}.','item_features':'category=reasoning;difficulty=medium','interactors':'','benchmark_id':group}
        contexts=[[[s,item(100*k+i)],int((i+k)%3!=0)] for k,s in enumerate(subs) for i in range(31)]
        cs=[contexts[k*31:(k+1)*31] for k in range(4)];q=item(989);inp=[sub,q]
        enc=m._runtime.features.encode_items([q]+[v[0][1] for v in contexts]);enc[:,:384]/=2.;enc[:,384:]/=4.;query=enc[:1]
        cc=[{'x':enc[1+k*31:1+(k+1)*31],'y':np.array([r[1] for r in cs[k]],float)} for k in range(4)]
        for n in [0,1,3,7,15,31]:
            for k in [0,1,2,3]:
                labels=[r for c in cs[:k+1] for r in c[:n]];ctx=[{'x':v['x'][:n],'y':v['y'][:n]} for v in cc[:k+1]]
                own=float(core(ctx[:1],query)[0]);expected=float(core(ctx,query,partial=True)[0]) if k and n else own
                a=call(inp,labels);b=call(inp,labels,subs[:k+1]);target=expected if kind=='partial8_shared' else own
                assert abs(a-own)<2e-6 and abs(b-target)<2e-6
                synthetic.append({'case':f'B{n}_peer{k}','error':max(abs(a-own),abs(b-target))})
        labels=[r for c in cs for r in c[:7]];base=call(inp,cs[0][:7]);base_labels=copy.deepcopy(labels)
        # Duplicate roster means indistinguishable Subject identities; exact R25 fallback.
        p=call(inp,labels,[*subs,dict(subs[1])]);assert p==call(inp,labels)
        synthetic.append({'case':'ambiguous_roster','exact_fallback':True})
        # Duplicate modeled supports, including conflicting repeated responses, retain own multiplicity.
        repeated=copy.deepcopy(cs[0][:7]);repeated+=[copy.deepcopy(repeated[0]),[copy.deepcopy(repeated[0][0]),1-repeated[0][1]]]
        together=repeated+cs[1][:7];own=call(inp,repeated);p=call(inp,together,subs[:2]);assert p==own
        r=m.predict_with_roster(inp,together,subs[:2]);assert r['own_labels']==9 and r['mode']=='own_only'
        synthetic.append({'case':'duplicate_conflicting_responses','own_labels':9,'exact_fallback':True})
        # Query repeated responses use identical modeled input -> identical probabilities.
        assert call(inp,labels)==call(copy.deepcopy(inp),labels)
        synthetic.append({'case':'duplicate_query_response','identical_probability':True})
        # Ambiguous incomplete Subject fields; no hidden identifier can repair them.
        incomplete=copy.deepcopy(inp);incomplete[0].pop('harness_version');incomplete_labels=copy.deepcopy(labels)
        for row in incomplete_labels:
            if row[0][0]['normalized_name']==sub['normalized_name']:row[0][0].pop('harness_version',None)
        p=call(incomplete,incomplete_labels,subs);assert p==call(incomplete,incomplete_labels)
        synthetic.append({'case':'incomplete_subject','exact_fallback':True})
        # Hidden IDs are ignored, and other anonymous Benchmark labels never enter this context.
        hidden=copy.deepcopy(inp);hidden[0]['subject_id']='ignored';hidden[1]['item_id']='ignored';assert call(hidden,labels)==call(inp,labels)
        other=[[[subs[1],item(202,'different-group')],1]];assert call(inp,cs[0][:7]+other)==base
        assert call(inp,cs[1][:7],subs)==.5
        synthetic.extend([{'case':'hidden_ids_ignored','passed':True},{'case':'other_benchmark_ignored','passed':True},{'case':'peer_only_without_own','probability':.5}])
        # No label state can survive a call; model weights and tokenizer only persist.
        call(inp,labels,subs);assert call(inp,[])==.5;assert labels==base_labels
        synthetic.append({'case':'B31_to_B0_no_label_state','passed':True})
        with ThreadPoolExecutor(max_workers=4) as pool:
            values=list(pool.map(lambda _:m.predict(inp,labels),range(4)))
        assert len(set(values))==1;synthetic.append({'case':'concurrent_calls','identical':True})
        for invalid in [2,-1,'1']:
            try:m.predict(inp,[[[sub,item(23)],invalid]])
            except (ValueError,TypeError):pass
            else:raise AssertionError('invalid binary label accepted')
        synthetic.append({'case':'invalid_labels_rejected','passed':True})
    import numpy,torch,transformers,tokenizers,huggingface_hub,safetensors
    deps={x.__name__:{'version':x.__version__,'path':x.__file__} for x in [numpy,torch,transformers,tokenizers,huggingface_hub,safetensors]}
    assert all(str(Path(x['path']).resolve()).lower().startswith(sys.prefix.lower()) for x in deps.values())
    resource={'seconds':time.perf_counter()-started,'cold_load_seconds':m._runtime.features.cold_load_seconds,'max_request_seconds':max(timings),'median_request_seconds':float(np.median(timings)),'p95_request_seconds':float(np.quantile(timings,.95)),'peak_rss_mib':peak_mb(),'calls':len(timings)}
    assert resource['cold_load_seconds']<60 and resource['max_request_seconds']<60 and resource['peak_rss_mib']<2048,resource
    assert not blocked,blocked
    result={'status':'PASS','kind':kind,'mode':mode,'fixtures':checkrows,'synthetic_cases':synthetic,'resources':resource,'dependencies':deps,'prefix':sys.prefix,'python':sys.version,'network_attempts':blocked,'model_file_reads':len(access),'predictions':predictions,'sys_path':sys.path}
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'status':'PASS','kind':kind,'mode':mode,'resources':resource}),flush=True)
if __name__=='__main__':main()
