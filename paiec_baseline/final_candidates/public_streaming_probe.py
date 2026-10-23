"""Execute latest official public client with actual candidate predict and synthetic exchange."""
import sys,json,time,contextlib,importlib.util,copy
from pathlib import Path
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    package=Path(sys.argv[1]);runtime_path=Path(sys.argv[2]);fixture=Path(sys.argv[3]);out=Path(sys.argv[4]);out.mkdir(exist_ok=True)
    sys.path.insert(0,str(package));m=load(package/'model.py','submission');rt=load(runtime_path,'official_streaming');v=json.loads(fixture.read_text(encoding='utf-8'));inp=[v['subjects'][0],v['queries'][0]];results=[]
    def grouping(inputs,n,deadline):
        unique=[];groups=[];by={}
        for i,row in enumerate(inputs):
            key=json.dumps(row,sort_keys=True)
            if key not in by:by[key]=len(unique);unique.append(row);groups.append([])
            groups[by[key]].append(i)
        return unique,groups
    def valid(x):
        p=float(x);assert 0<=p<=1;return p
    config=out/'config.json';config.write_text(json.dumps({'protocol':'streaming_alc_v1'}))
    for j,n in enumerate(v['budgets']):
        labels=[r for c in v['contexts'] for r in c[:n]];targets=[inp,copy.deepcopy(inp),[v['subjects'][0],v['queries'][1]]]
        event={'protocol':'streaming_alc_v1','type':'evaluate','event_id':'local-'+str(n),'budget':n,'labeled':labels,'targets':[{'input':t} for t in targets]};exchanges=[];calls=[]
        def exchange(payload):
            exchanges.append(copy.deepcopy(payload))
            return copy.deepcopy(event) if len(exchanges)==1 else {'protocol':'streaming_alc_v1','type':'finished','predictions_csv':'synthetic_id,probability\n'}
        def stream(model_path,inputs,labeled,**kwargs):
            assert labeled==labels
            for i,query in enumerate(inputs):calls.append(query);yield i,m.predict(query,labeled)
        stats=rt.run_streaming(config_path=config,predict=m.predict,acquisition_function=None,model_path=str(package/'model.py'),output_dir=out/('B'+str(n)),concurrency=2,deadline=time.monotonic()+60,per_call_timeout=30,acquisition_timeout=30,call_window=lambda *_:contextlib.nullcontext(),validate_score=valid,prediction_stream=stream,group_inputs=grouping,log=lambda _:None,exchange=exchange)
        assert len(calls)==2,'duplicate inputs not grouped';probs=exchanges[-1]['probabilities'];assert probs[0]==probs[1]
        assert max(abs(probs[k]-v['expected'][j][0 if k<2 else 1]['r25']) for k in range(3))<2e-6
        results.append({'budget':n,'predictions':probs,'unique_actual_model_calls':len(calls),'stats':stats})
    (out/'result.json').write_text(json.dumps({'status':'PASS','cases':results,'scope':'official public run_streaming + real ZIP predict, injected local exchange/grouping; not worker isolation or Hosted','hosted_score':None},indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','cases':len(results)}),flush=True)
if __name__=='__main__':main()
