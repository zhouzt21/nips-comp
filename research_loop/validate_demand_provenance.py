"""Author parser replay and retained packet/paper provenance; no response labels."""
from pathlib import Path
import importlib.util,hashlib,json,sys
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'paiec_baseline/runs/research_loop/R73'
UP=Path('D:/nips-comp/paiec-dataset/reproduction_sources/adele/upstream')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=UP/'src/adele/annotation/parsing.py';spec=importlib.util.spec_from_file_location('adele_author_parsing',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    groups={}
    for name in ['final_annotations','annotations/qc']:
        rows=[json.loads(s) for s in (ROOT/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        valid=0;missing=0;errors=[];false_numeric=[]
        for row in rows:
            grade,ok=module.extract_demand_level(row['raw_response'])
            if row['grade'] is None:
                missing+=1
                if ok:false_numeric.append({'request_id':row['request_id'],'author_parsed':grade})
            else:
                valid+=1
                if not ok or grade!=row['grade']:errors.append({'request_id':row['request_id'],'author_parsed':str(grade),'expected':row['grade'],'ok':bool(ok)})
        groups[name]={'records':len(rows),'numeric_grades':valid,'deliberate_null':missing,'numeric_parse_disagreements':errors,'null_parsed_as_numeric':false_numeric}
        assert not errors,errors
    source_manifest=json.loads((REPO/'research_loop/literature/reproduction_v4/adele_sources.json').read_text(encoding='utf-8'))
    checked=[]
    for a in source_manifest['artifacts']:
        path=Path(a['path'])
        assert sha(path)==a['sha256'],path
        checked.append({'path':str(path),'sha256':a['sha256'],'role':a['role']})
    ann=json.loads((ROOT/'annotation_contract.json').read_text(encoding='utf-8'))
    for rel,h in ann['packet_hashes'].items():assert sha(ROOT/rel)==h,rel
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'groups':groups,'source_files_verified':checked,'packet_files_verified':len(ann['packet_hashes']),'annotation_labels_changed':False,'extra_external_paid_api_requests':0,'native_agents_requested':5,'native_model_alias':'gpt-6-luna','native_token_usage':None,'native_billing':None,'native_snapshot':None,'annotation_context_isolation':'five fork_turns=none agent invocations with disjoint allowlisted packets; instruction boundary, not filesystem sandbox','compiler_python':sys.version,'source_script_sha256':sha(Path(__file__))}
    (ROOT/'provenance_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'groups':groups,'source_files_verified':len(checked),'packet_files_verified':len(ann['packet_hashes'])},ensure_ascii=False))
if __name__=='__main__':main()
