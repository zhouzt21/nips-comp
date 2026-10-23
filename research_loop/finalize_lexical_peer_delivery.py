"""Final artifact inventory and old-asset isolation; no submission or git mutation."""
import time,re
from historical_positive_common import *

def main():
    start=time.perf_counter()
    contracts=[ROOT/'R62/isolation',ROOT/'R70/source_contract',ROOT/'R70/calibration_contract',ROOT/'R70/full/contract',ROOT/'R71/full/contract',ROOT/'R72/coverage_contract',ROOT/'R72/full/contract']
    for path in contracts:check_freeze(path)
    stages={s:read(ROOT/s/'full/replay_validation.json') for s in ['R70','R71','R72']}
    for s,v in stages.items():
        assert v['pairs']==5242 and v['trajectories']==15726 and v['benchmarks']==36
        assert v['max_brier_error']<1e-12 and v['max_accuracy_error']<1e-12
        assert (ROOT/s/'full/execution.json').exists()
    fallback={str(p/'dist/smoothed_mean_strength2.zip'):sha(p/'dist/smoothed_mean_strength2.zip') for p in ARCHIVES}
    assert all(v=='e55ee0b267f3b345685ef072cfa511c3c017d9082256573793961ea20feadcb9' for v in fallback.values())
    for s in ['R70','R71','R72']:
        status_path=ROOT/s/'status.json'
        finished=read(status_path).get('finished',now()) if status_path.exists() else now()
        write_json(status_path,{'status':'complete','full_evaluation':True,'replay_checked':True,'finished':finished,'scope':'conditional_simulation' if s=='R72' else 'public_exploratory_outer'})
    files=[]
    for s in ['R69','R70','R71','R72']:
        for p in (ROOT/s).rglob('*'):
            if p.is_file() and p.name not in ['delivery_manifest.json','delivery_validation.json']:
                files.append(p)
    for pattern in ['lexical_peer*']:
        for p in (REPO/'reports').glob(pattern):
            if p.is_file():files.append(p)
            elif p.is_dir():files.extend(q for q in p.rglob('*') if q.is_file())
    for p in (REPO/'research_loop').glob('*'):
        if p.is_file() and any(v in p.name.lower() for v in ['lexical_peer','lexical_source','lexical_calibration','lexical_budget','lexical_reliability','peer_current','peer_coverage','peer_operator','peer_input','peer_serialization','peer_postrun','runtime_evidence_contract','r72_operator']):files.append(p)
    files += [REPO/'doc/plan_6.md',REPO/'research_loop/HIGH_LEVEL_STATE.md',REPO/'research_loop/HISTORICAL_POSITIVE_SIGNAL_LEDGER.md',REPO/'research_loop/literature/lexical_peer_evidence.md']
    files += [REPO/'research_loop'/name for name in ['fetch_pava_original.py','fetch_pava_resolved.py','fetch_official_runner.py','consolidate_runtime_contract.py','finalize_runtime_evidence.py']]
    files += list((REPO/'research_loop/literature').glob('lexical_peer*'))
    links_checked=0;linked_paths=[]
    generated_paths={(ROOT/'R72/delivery_manifest.json').resolve(),(ROOT/'R72/delivery_validation.json').resolve()}
    for p in [REPO/'reports'/name for name in ['lexical_peer_evidence_findings.md','lexical_peer_evidence_tables.md','lexical_peer_evidence_reproduction.md']]:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            resolved=(p.parent/target.split('#')[0]).resolve()
            assert resolved.exists() or resolved in generated_paths,(str(p),target)
            linked_paths.append(resolved)
            links_checked+=1
    report=(REPO/'reports/lexical_peer_evidence_findings.md').read_text(encoding='utf-8')
    for stage,method in [('R70','R70_independent'),('R71','R71_grouped'),('R72','R72_same3'),('R72','R72_pooled3'),('R72','R72_diff3')]:
        value=read(ROOT/stage/'full/summary.json')['results'][method]['alc_mean']
        assert f'{value:.9f}' in report,(stage,method,value)
    files=sorted(set(files));manifest=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'valid_results':'invalid_control_seed_20261023' not in str(p)} for p in files]
    write_json(ROOT/'R72/delivery_manifest.json',manifest)
    summary={'status':'complete','finished':now(),'stages':stages,'contracts_checked':[str(p) for p in contracts],
        'old_frozen_assets_unchanged':len(read(ROOT/'R62/isolation/contract.json')['sha256']),
        'fallback_sha256':fallback,'files':len(manifest),'bytes':sum(v['bytes'] for v in manifest),
        'invalid_control_run_retained_and_excluded':True,'new_neural_network_fits':0,'new_embeddings':0,'official_submission':False,
        'R72_scope':'conditional_simulation; actual hosted coordinator not verified','seconds':time.perf_counter()-start,
        'report_local_links_checked':links_checked,'headline_ALC_values_checked':True,
        'manifest_sha256':sha(ROOT/'R72/delivery_manifest.json')}
    write_json(ROOT/'R72/delivery_validation.json',summary)
    assert all(p.exists() for p in linked_paths)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
