"""Outcome-blind integration and quality audit of native Luna annotations."""
import hashlib,json,time,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
REPO=Path(__file__).resolve().parents[1];OUT=REPO/'paiec_baseline/runs/research_loop/R73'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def main():
    start=time.perf_counter();contract=read(OUT/'annotation_contract.json')
    for p,h in contract['packet_hashes'].items(): assert sha(OUT/p)==h,p
    mapping=read(OUT/'item_mapping.json');byid={x['custom_item_id']:x for x in mapping}
    expected={};texts={}
    for p in sorted((OUT/'annotation_only').glob('shard*_batch*.json')):
        for item in read(p):
            texts[item['item_identity']]=item['text']
            for req in item['requests']:expected[req['request_id']]={**req,'item_identity':item['item_identity'],'item_index':item['item_index']}
    sources=[]
    for name in ['shard0','shard1','shard2','shard3','qc']:
        path=OUT/'annotations'/f'{name}.jsonl'
        alt=OUT/'annotation_only/annotations'/f'{name}.jsonl'
        if alt.exists():
            if not path.exists():shutil.copyfile(alt,path)
            assert sha(path)==sha(alt)
            sources.append({'agent':name,'original_path':str(alt),'normalized_copy':str(path),'sha256':sha(path),'copy_equals_original':sha(path)==sha(alt)})
        else:sources.append({'agent':name,'original_path':str(path),'sha256':sha(path)})
    primary=sum([load(OUT/'annotations'/f'shard{i}.jsonl') for i in range(4)],[])
    qc=load(OUT/'annotations/qc.jsonl')
    assert len(primary)==512 and len(qc)==64
    required={'item_identity','item_index','request_id','rubric_id','grade','rationale','ambiguity','model_version','model_snapshot','prompt_sha256','raw_response'}
    for seq in [primary,qc]:
        assert len({r['request_id'] for r in seq})==len(seq)
        for row in seq:
            assert required<=row.keys(),row
            ref=expected[row['request_id']]
            for k in ['item_identity','item_index','rubric_id','prompt_sha256']: assert row[k]==ref[k],(k,row['request_id'])
            assert row['model_version']=='gpt-6-luna' and row['model_snapshot']=='not_exposed'
            assert row['grade'] is None or type(row['grade'])==int and 0<=row['grade']<=5
            if row['grade'] is None:assert row['ambiguity'],row
            assert row['raw_response'] and row['rationale']
    assert sorted({r['item_index'] for r in qc})==contract['qc_indices']
    primary.sort(key=lambda r:(r['item_index'],contract['demands'].index(r['rubric_id'])))
    # No outcome-based arbitration or grade modification. Null remains missing.
    (OUT/'final_annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in primary),encoding='utf-8')
    lookup={r['request_id']:r for r in primary};pairs=[]
    for r in qc:
        a=lookup[r['request_id']];both=a['grade'] is not None and r['grade'] is not None
        pairs.append({'request_id':r['request_id'],'item_index':r['item_index'],'rubric_id':r['rubric_id'],'primary_grade':a['grade'],'repeat_grade':r['grade'],'both_valid':both,'absolute_difference':abs(a['grade']-r['grade']) if both else None,'null_status_agreement':(a['grade'] is None)==(r['grade'] is None),'primary_rationale':a['rationale'],'repeat_rationale':r['rationale']})
    def agreement(rows):
        valid=[r for r in rows if r['both_valid']];diff=[r['absolute_difference'] for r in valid]
        return {'repeat_records':len(rows),'both_valid':len(valid),'exact_agreement':float(np.mean(np.array(diff)==0)) if diff else None,'over_one_level_fraction':float(np.mean(np.array(diff)>1)) if diff else None,'mean_absolute_difference':float(np.mean(diff)) if diff else None,'null_status_agreement':float(np.mean([r['null_status_agreement'] for r in rows]))}
    def dist(rows):
        grades=[r['grade'] for r in rows if r['grade'] is not None]
        return {'total':len(rows),'valid':len(grades),'null':len(rows)-len(grades),'ambiguity_noted':sum(bool(r['ambiguity']) for r in rows),'counts':dict(sorted(Counter(map(str,grades)).items())),'mean':float(np.mean(grades)) if grades else None}
    perbm={}
    for bm in sorted({x['benchmark'] for x in mapping}):
        perbm[bm]={d:dist([r for r in primary if byid[r['item_identity']]['benchmark']==bm and r['rubric_id']==d]) for d in contract['demands']}
    length=[]
    for item in mapping:
        rows=[r for r in primary if r['item_identity']==item['custom_item_id']]
        length.append({'item_identity':item['custom_item_id'],'benchmark':item['benchmark'],'canonical_characters':item['canonical_characters'],'at_20000_character_cap':item['canonical_characters']>=20000,'null_dimensions':sum(r['grade'] is None for r in rows),'ambiguous_dimensions':sum(bool(r['ambiguity']) for r in rows),'complete_four_grades':all(r['grade'] is not None for r in rows)})
    def lengths(rows):return {'items':len(rows),'complete_four_grades':sum(r['complete_four_grades'] for r in rows),'null_dimensions':sum(r['null_dimensions'] for r in rows)}
    # Measure the existing tokenizer's window, never download another model.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('D:/paiec_pretrained/all-MiniLM-L6-v2',local_files_only=True)
    for row in length:
        row['content_tokens_without_truncation']=len(tokenizer.encode(texts[row['item_identity']],truncation=False,add_special_tokens=False))
        row['exceeds_project_126_content_token_window']=row['content_tokens_without_truncation']>126
        row['project_token_rule']='first96 + last30 content tokens; add CLS/SEP => maximum128'
    dump(OUT/'repeat_comparisons.json',pairs);dump(OUT/'item_length_audit.json',length)
    summary={'status':'complete','primary_records':512,'repeat_records':64,'source_records':sources,'final_sha256':sha(OUT/'final_annotations.jsonl'),'invalid_parse_records':0,'missing_expected_records':0,'primary_distribution':dist(primary),'per_rubric':{d:dist([r for r in primary if r['rubric_id']==d]) for d in contract['demands']},'agreement':agreement(pairs),'per_rubric_agreement':{d:agreement([r for r in pairs if r['rubric_id']==d]) for d in contract['demands']},'large_disagreements':[r for r in pairs if r['both_valid'] and r['absolute_difference']>1],'null_disagreements':[r for r in pairs if not r['null_status_agreement']],'benchmark_distributions':perbm,'length_groups':{'at_cap':lengths([r for r in length if r['at_20000_character_cap']]),'below_cap':lengths([r for r in length if not r['at_20000_character_cap']]),'above_minilm_window':lengths([r for r in length if r['exceeds_project_126_content_token_window']]),'within_minilm_window':lengths([r for r in length if not r['exceeds_project_126_content_token_window']])},'project_encoder_rule':'content first96+last30 if >126; +CLS/SEP total<=128; source mlp/semantic_features.py encode_many','complete_four_grade_items':sum(r['complete_four_grades'] for r in length),'teacher_annotation_model':'gpt-6-luna','fixed_snapshot_available':False,'arbitration':'none; preserve primary grades and independent repeat; source pilot excludes missing teacher dimensions instead of inventing labels','validity_boundary':'Same-model inter-rater agreement, not human truth or calibrated rubric validity. Per-BM grade differences do not establish bias without reference labels.','seconds':time.perf_counter()-start}
    dump(OUT/'quality_summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['primary_distribution','agreement','complete_four_grade_items','length_groups']},ensure_ascii=False))
if __name__=='__main__':main()
