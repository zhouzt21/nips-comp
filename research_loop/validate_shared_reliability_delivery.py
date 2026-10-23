"""Plan_9 preservation, consistency and delivery checks (no rerun of models)."""
import ast,re
from shared_reliability_common import *

def entries(path):
    value=read(path)
    return value if isinstance(value,list) else value.get('entries',value.get('files'))

def main():
    dest=ROOT/'shared_reliability_v9';dest.mkdir(exist_ok=True);report=REPO/'reports';src=REPO/'research_loop'
    allowed={str((src/n).resolve()) for n in ['HIGH_LEVEL_STATE.md','HISTORICAL_POSITIVE_SIGNAL_LEDGER.md']}
    checks={};cached={}
    def digest(path):
        path=str(Path(path).resolve())
        if path not in cached:cached[path]=sha(path)
        return cached[path]
    for stage in ['R72','R76','dual_track_v8']:
        unchanged=0;changed=[]
        for row in entries(ROOT/stage/'delivery_manifest.json'):
            p=str(Path(row['path']).resolve());actual=digest(p)
            if p in allowed:changed.append({'path':p,'old':row['sha256'],'current':actual,'authorized_global_update':True})
            else:assert actual==row['sha256'],p;unchanged+=1
        checks[stage]={'unchanged_scientific_and_other_files':unchanged,'intentional_updates':changed}
    contracts=[ROOT/'R62/isolation',ROOT/'R75/contract',ROOT/'R76/full/contract',ROOT/'R77/alternative/contract',ROOT/'R77/full/contract',ROOT/'R77/heterogeneity_contract',ROOT/'R78/contract']
    if (ROOT/'R79/source_contract').exists():contracts.append(ROOT/'R79/source_contract')
    for p in contracts:check_freeze(p)
    checks['frozen_contracts']=[str(p) for p in contracts]
    for stage in ['alternative','full']:
        r=read(ROOT/f'R77/{stage}/replay_validation.json');assert (r['pairs'],r['trajectories'],r['benchmarks'],r['chunks'])==(5242,15726,36,149)
        assert r['max_brier_error']<1e-12 and r['max_accuracy_error']<1e-12
    r=read(ROOT/'R78/replay_validation.json');assert max(r[k] for k in ['max_query_brier_error','max_query_accuracy_error','max_support_z_error'])<1e-12
    d=read(ROOT/'R78/decision.json')
    if not d['R79_diagnostic_entry']:assert not (ROOT/'R79').exists(),'Untriggered R79 must not be presented as executed'
    if (ROOT/'R79').exists():
        r79=read(ROOT/'R79/decision.json');assert r79['R79_source_executed']
        assert read(ROOT/'R79/source_replay_validation.json')['max_brier_accuracy_error']<1e-12
        assert read(ROOT/'R79/source_isolation_validation.json')['status']=='passed'
        if not r79['R79_outer_authorized']:
            assert not r79['R79_outer_executed'] and not (ROOT/'R79/outer').exists() and not (ROOT/'R79/full').exists()
        checks['R79_decision']=r79
    checks['R78_decision']=d;checks['R77_decision']=read(ROOT/'R77/decision.json')
    assert read(dest/'support_diagnostic_checks.json')['status']=='passed_synthetic_only'
    checks['fallback']=[]
    expected='e55ee0b267f3b345685ef072cfa511c3c017d9082256573793961ea20feadcb9'
    for name in ['layout_20261022_120427','nips-comp_20261022_113534']:
        p=Path('D:/nips-comp/paiec-dataset/paiec_git_backups')/name/'workspace/paiec_baseline/dist/smoothed_mean_strength2.zip';assert digest(p)==expected
        checks['fallback'].append({'path':str(p),'sha256':expected})
    required=['shared_cohort_robustness_findings.md','support_sharing_reliability_findings.md','shared_reliability_tables.md','shared_reliability_reproduction.md']
    if (ROOT/'R79').exists():required.append('shared_reliability_policy_findings.md')
    for name in required:
        p=report/name;assert p.exists()
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            assert (p.parent/target.split('#')[0]).exists(),(str(p),target)
    assert (src/'SHARED_RELIABILITY_DECISION.md').exists()
    for name in ['SHARED_RELIABILITY_DECISION.md','SHARED_RELIABILITY_SPEC.md','SHARED_RELIABILITY_OOF_ADDENDUM.md']:
        p=src/name
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            assert (p.parent/target.split('#')[0]).exists(),(str(p),target)
    source_manifest=read(src/'literature/shared_reliability/original_source_hashes.json')
    for row in source_manifest['entries']:assert digest(row['path'])==row['sha256']
    checks['original_source_hashes']=source_manifest
    checks['figure_inspection']={'files':['cohort_increment.png','benchmark_cohort_matrix.png','support_query_association.png'],
        'method':'main agent opened all three exported images; axis overlap and colorbar clipping corrected; no change to data'}
    newcode=[]
    for pattern in ['*shared_reliability*.py','*shared_cohort*.py','*support_sharing*.py','diagnose_cohort_heterogeneity.py','verify_support_diagnostic.py']:
        newcode+=list(src.glob(pattern))
    for p in set(newcode):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    checks['python_static_parse_files']=len(set(newcode));checks['report_links']='passed'
    checks['review']='primary-agent deterministic source/identity/probability/leakage verification; no claim of independent external review'
    checks['status']='passed';checks['finished']=now();write_json(dest/'delivery_validation.json',checks)
    paths=set(newcode)
    paths.update(src.glob('SHARED_RELIABILITY*'));paths.add(REPO/'doc/plan_9.md');paths.update(src/n for n in ['HIGH_LEVEL_STATE.md','HISTORICAL_POSITIVE_SIGNAL_LEDGER.md'])
    for folder in [ROOT/'R77',ROOT/'R78',dest,src/'literature/shared_reliability',report/'shared_reliability_figures']:
        paths.update(p for p in folder.rglob('*') if p.is_file())
    if (ROOT/'R79').exists():paths.update(p for p in (ROOT/'R79').rglob('*') if p.is_file())
    paths.update(report/n for n in required)
    paths.update([Path('D:/todo/tasks/graph/nips-comp.shared-reliability-2026-10-23.md'),Path('D:/todo/reports/work-traces/nips-comp/2026-10-23-shared-reliability.md')])
    result=[]
    for p in sorted(paths,key=str):
        if p==dest/'delivery_manifest.json':continue
        assert p.suffix!='.pyc' and '__pycache__' not in p.parts and not p.name.startswith('.env')
        result.append({'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)})
    write_json(dest/'delivery_manifest.json',{'finished':now(),'files':len(result),'bytes':sum(v['bytes'] for v in result),'entries':result,'scope':'R77/R78 and only actually triggered R79; conditional simulation'})
    print(json.dumps({'status':'passed','files':len(result),'bytes':sum(v['bytes'] for v in result),'R79_source_triggered':d['R79_diagnostic_entry'],
        'R79_full_executed':checks.get('R79_decision',{}).get('R79_outer_executed',False),'preservation':{k:v.get('unchanged_scientific_and_other_files') for k,v in checks.items() if k in ['R72','R76','dual_track_v8']}}),flush=True)
if __name__=='__main__':main()
