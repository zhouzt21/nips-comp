"""Restore R53 exact author prompts; produce outcome-free native-agent packets."""
from pathlib import Path
import hashlib, json, random, importlib.util, sys
from datetime import datetime, timezone

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'paiec_baseline/runs/research_loop/R73'
OLD=Path('D:/nips-comp/paiec-dataset/reproduction_v4/adele_pilot')
UP=Path('D:/nips-comp/paiec-dataset/reproduction_sources/adele/upstream')
DEMANDS=['AS','MCr','QLl','KNf']
def sha(b): return hashlib.sha256(b).hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    expected={'manifest.json':'35ed5b87220ee6dd94e105e681bd6e92ac3a86fea81b5da53eba9f4eeb3c0f70','items.json':'47547967b70e6d103b4e11406f386dc98b08fb03f756e6c3fce2393886860111','requests.jsonl':'573243e3a4cd89a7dbbb7813abf33d418702e684318c044c652b7b221a6d1584'}
    for name,h in expected.items(): assert sha((OLD/name).read_bytes())==h,name
    assert not (OUT/'annotation_contract.json').exists(),'Do not overwrite frozen packets'
    requests=[json.loads(x) for x in (OLD/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    items=json.loads((OLD/'items.json').read_text(encoding='utf-8'))
    manifest=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('author_prompts',UP/'src/adele/annotation/prompts.py')
    author=importlib.util.module_from_spec(spec); spec.loader.exec_module(author)
    spec=importlib.util.spec_from_file_location('author_catalog',UP/'src/adele/rubrics/catalog.py')
    cat=importlib.util.module_from_spec(spec);sys.modules['author_catalog']=cat;spec.loader.exec_module(cat)
    catalog=cat.RubricsCatalog()
    packets=OUT/'annotation_only'
    packets.mkdir(parents=True,exist_ok=True)
    rubric_data={d:{'name':catalog[d].full_name,'content':catalog[d].content} for d in DEMANDS}
    dump(packets/'rubrics.json',rubric_data)
    (packets/'author_prompts.py').write_bytes((UP/'src/adele/annotation/prompts.py').read_bytes())
    isolated=[]; mapped=[]
    for ix,item in enumerate(items):
        reqs=requests[ix*4:ix*4+4]
        prompt=reqs[0]['body']['messages'][0]['content']
        text=prompt.split('TASK INSTANCE: ',1)[1].rsplit('\n\nINSTRUCTION: ',1)[0]
        assert sha(text.encode())==item['canonical_text_sha256']
        row={'item_index':ix,'item_identity':item['custom_item_id'],'text':text,'requests':[]}
        for d,req in zip(DEMANDS,reqs):
            original=req['body']['messages'][0]['content']
            assert original==author.build_annotation_prompt(catalog[d].full_name,catalog[d].content,text)
            assert req['custom_id']==item['custom_item_id']+'__'+d
            rec={'request_id':req['custom_id'],'rubric_id':d,'prompt_sha256':sha(original.encode())}
            row['requests'].append(rec)
            # Exact frozen author request available without any outcome metadata.
            dump(packets/'exact_requests'/f'{ix:03d}_{d}.json',req)
        isolated.append(row); mapped.append(item)
    for shard in range(4):
        # Contiguous round-robin source selection gives each shard many benchmarks.
        selected=isolated[shard*32:(shard+1)*32]
        for batch in range(4): dump(packets/f'shard{shard}_batch{batch}.json',selected[batch*8:(batch+1)*8])
    qc_indices=sorted(random.Random(20261023).sample(range(128),16))
    for batch in range(2): dump(packets/f'qc_batch{batch}.json',[isolated[i] for i in qc_indices[batch*8:(batch+1)*8]])
    (OUT/'annotations').mkdir(exist_ok=True)
    dump(OUT/'item_mapping.json',mapped)
    packet_hashes={str(p.relative_to(OUT)):sha(p.read_bytes()) for p in sorted(packets.rglob('*')) if p.is_file()}
    contract={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'prepared_before_annotation','items':128,'requests':512,'repeat_items':16,'repeat_requests':64,'qc_seed':20261023,'qc_indices':qc_indices,'demands':DEMANDS,'source_hashes':expected,'upstream_commit':manifest['upstream_commit'],'exact_author_prompt_equality':512,'annotation_model_alias':'gpt-6-luna','immutable_model_snapshot':'not_exposed_by_native_subagent_tool','native_runtime_temperature':'not_exposed','protocol_adaptation':'Native agents read identical full rubrics and task text; exact author prompts archived per record. Short decision summaries replace requested chain-of-thought disclosure. Multiple tasks per agent, not independent API calls.','annotation_boundary':'Only annotation_only packets; no subject, outcome, success rate, prediction, response parquet or external source. Task text is untrusted data, not instructions.','teacher_is_response_ground_truth':False,'invalid_grade_rule':'null with ambiguity; never impute as a certain teacher grade','packet_hashes':packet_hashes,'script_sha256':sha(Path(__file__).read_bytes())}
    dump(OUT/'annotation_contract.json',contract)
    print(json.dumps({k:contract[k] for k in ['items','requests','repeat_items','qc_indices','exact_author_prompt_equality']}))
if __name__=='__main__': main()
