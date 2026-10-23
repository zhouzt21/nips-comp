"""R68 source/resource audit only: no API inference or training."""
import collections
import runpy
import subprocess
import requests
from historical_positive_common import *

OUT=ROOT/'R68'; UP=Path('D:/nips-comp/paiec-dataset/reproduction_sources/adele/upstream')
PREP=Path('D:/nips-comp/paiec-dataset/reproduction_v4/adele_pilot')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=read(PREP/'manifest.json');items=read(PREP/'items.json')
    records=[json.loads(x) for x in (PREP/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records)==512 and len(items)==128
    assert len({r['custom_id'] for r in records})==512
    assert sha(PREP/'requests.jsonl')=='573243e3a4cd89a7dbbb7813abf33d418702e684318c044c652b7b221a6d1584'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()
    assert commit==manifest['upstream_commit']
    script=UP/'src/adele/annotation/prompts.py';author=runpy.run_path(str(script))
    names={'AS':'Attention and Scan','MCr':'Identifying Relevant Information','QLl':'Logical Reasoning','KNf':'Knowledge of Formal Sciences'}
    counts=collections.Counter();rubrics={}
    for dim in names:
        p=UP/f'src/adele/rubrics/data_v1/{dim}.txt';rubrics[dim]=p.read_text(encoding='utf-8').strip()
        assert subprocess.check_output(['git','diff','HEAD','--',str(p.relative_to(UP))],cwd=UP,text=True)==''
    assert subprocess.check_output(['git','diff','HEAD','--',str(script.relative_to(UP))],cwd=UP,text=True)==''
    for r in records:
        dim=r['custom_id'].rsplit('__',1)[1];counts[dim]+=1
        assert len(r['body']['messages'])==1 and r['body']['messages'][0]['role']=='user'
        prompt=r['body']['messages'][0]['content']
        task=prompt.split('TASK INSTANCE: ',1)[1].split('\n\nINSTRUCTION: ',1)[0]
        assert author['build_annotation_prompt'](names[dim],rubrics[dim],task)==prompt
        rebuilt=author['build_batch_request'](r['custom_id'].rsplit('__',1)[0],dim,prompt,manifest['request_model_placeholder'])
        assert rebuilt==r
    adele={'status':'waiting_for_annotation_resources','requests':len(records),'distinct_items':len(items),'dimensions':dict(counts),
        'author_prompt_reconstruction':'all 512 exact strings and request bodies equal','upstream_commit':commit,
        'input_characters':sum(len(r['body']['messages'][0]['content']) for r in records),
        'new_api_calls':0,'paid_usage':0,'new_models_downloaded':0,
        'model_snapshot':None,'authorization':'prior user instruction: ADeLe waits for resources',
        'cpu_alternative':'Only MiniLM is available in project model locations; no validated local generative rubric annotator. Heuristic length/type proxies are legal diagnostics, not original demand labels or a faithful ADeLe substitute.',
        'hidden_inference':'single input task annotation would require the same frozen judge/rubrics and measured latency/resources; current empty model snapshot cannot run',
        'hashes':{str(p):sha(p) for p in [PREP/'manifest.json',PREP/'items.json',PREP/'requests.jsonl',script]}}
    write_json(OUT/'adele_audit.json',adele)
    # Read only immutable official baseline files; public downloads, no credentials.
    rev='cc88a959c1e5e7462f31a9215baba066ce6d6e2b';assets=[]
    for relative in ['README.md','empirical_mean/model.py']:
        url=f'https://raw.githubusercontent.com/aims-foundations/paiec_baseline/{rev}/{relative}'
        r=requests.get(url,timeout=30);r.raise_for_status()
        p=OUT/'official_baseline'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content)
        assets.append({'url':url,'path':str(p),'sha256':sha(p)})
    official=REPO/'research_loop/literature/transferable_prior_official_interface.txt'
    raw=official.read_text(encoding='utf-8-sig');doc=json.loads(raw)['content']
    (OUT/'official_interface.md').write_text(doc,encoding='utf-8')
    assert 'across sampled subject' in doc
    shared={'status':'blocked_by_runtime_contract','public_interface_checked_utc':now(),'official_commit':rev,'assets':assets,
        'confirmed':{'input':'one subject dict and one item dict; plain strings',
            'labeled':'[[subject,item],label] across sampled subject-benchmark pairs, allowed budget only; B0 empty',
            'benchmark_matching':'stable anonymous benchmark_id supplied in item',
            'subject_matching':'visible subject attributes available; no subject_id in prediction input',
            'item_matching':'content, item_features, interactors available; no stable item_id in prediction interface'},
        'unresolved':['exact peer visibility/cohort composition at each scoring call','cross-pair process/state isolation and call order',
            'whether identical content/metadata denotes the same task in all records; no stable item id'],
        'starter_access':'AnySearch extract of public Codabench 17828 failed; public baseline repository is not the hosted runner',
        'decision':'No new cohort simulation or hidden-state predictor. Multi-pair evidence is documented, but prior cohort4 results remain conditional and are not hosted gains.',
        'historical_evidence':{'R20_pooled':0.202834515,'R20_own_selected':0.203917943,
            'R36_own_mean_pooled_direction':0.195253399,'R36_strong_own':0.195325060,
            'interpretation':'R20 item correspondence adds information conditional on shared labels; R36 gain vs strong own is only 0.000071661 with fold regression and CI crossing zero.'},
        'official_interface_sha256':sha(OUT/'official_interface.md')}
    write_json(OUT/'shared_runtime_audit.json',shared)
    write_json(OUT/'status.json',{'stage':'R68','audit':'complete','adele':'waiting_for_annotation_resources','shared':'blocked_by_runtime_contract',
        'new_fits':0,'new_effectiveness_scores':False,'finished_utc':now()})
    print(json.dumps({'R68':'audit_complete','author_prompts_exact':512,'shared':shared['status']}))

if __name__=='__main__':main()
