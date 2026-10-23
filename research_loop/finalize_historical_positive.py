"""Record final evidence inventory after all authorized stages and reports exist."""
import importlib.metadata
import shutil
from historical_positive_common import *

def main():
    r62=read(ROOT/'R62/summary.json');r63=read(ROOT/'R63/summary.json');pilot=read(ROOT/'R64/summary.json')
    stages=[('R62',ROOT/'R62'),('R63',ROOT/'R63')]
    if pilot['enter_outer']:
        assert (ROOT/'R64/full/summary.json').exists()
        stages.append(('R64',ROOT/'R64/full'))
    contracts=[ROOT/'R62/isolation',ROOT/'R62/evaluation_contract',ROOT/'R62/exact_cache_contract',
               ROOT/'R63/evaluation_contract',ROOT/'R64/pilot_contract']
    if pilot['enter_outer']:contracts.append(ROOT/'R64/full/evaluation_contract')
    for p in contracts:check_freeze(p)
    evidence={}
    for stage,path in stages:
        score=read(path/'summary.json');audit=read(path/'analysis_audit.json')
        assert score['unique_pairs']==5242 and score['trajectories']==15726 and len(score['by_benchmark'])==36
        assert audit['rescore_max_error']==0
        assert abs(score['results']['R25']['alc_mean']-.19512392597599454)<1e-12
        assert abs(score['results']['smoothed_mean_2']['alc_mean']-.19868736276798543)<1e-12
        evidence[stage]={'alc':{m:v['alc_mean'] for m,v in score['results'].items()},
            'rescore_max_error':audit['rescore_max_error'],'trajectories':score['trajectories'],
            'unique_pairs':score['unique_pairs'],'benchmarks':len(score['by_benchmark']),
            'execution':read(path/'execution.json')}
    overlap=read(ROOT/'R62/cache_overlap_audit.json')
    assert overlap['max_absolute_error']==0 and overlap['complete_overlap_chunks']==7
    required=['historical_positive_followup_findings.md','historical_positive_tables.md',
        'historical_positive_metrics.json','historical_positive_metrics.csv','historical_positive_benchmark_metrics.csv',
        'historical_positive_reproduction.md','historical_positive_structural_review.md']
    for name in required:assert (REPO/'reports'/name).exists(),name
    report=(REPO/'reports/historical_positive_followup_findings.md').read_text(encoding='utf-8')
    for stage,m in [('R62','R56_segment'),('R63','R55_resampled_query')]:
        assert f"{evidence[stage]['alc'][m]:.9f}" in report
    assert '正在执行' not in report and '最终数值待补' not in report
    for old,new in [(ROOT/'R63_refit.log',ROOT/'R63/refit.log'),(ROOT/'R64_pilot.log',ROOT/'R64/pilot.log')]:
        if old.exists():
            if not new.exists():shutil.copyfile(old,new)
            assert sha(old)==sha(new)
    manifest={str((a/'dist/smoothed_mean_strength2.zip').resolve()):sha(a/'dist/smoothed_mean_strength2.zip') for a in ARCHIVES}
    assert all(h=='e55ee0b267f3b345685ef072cfa511c3c017d9082256573793961ea20feadcb9' for h in manifest.values())
    packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','torch','transformers','matplotlib','pyarrow']}
    validation={'finished_utc':now(),'stages':evidence,'R64_source_entered_outer':pilot['enter_outer'],
        'new_fits':10,'reused_R43_weights':3,'new_embeddings':0,'model_training_seeds':[2026,4307],
        'item_split_seeds':[0,1,2], 'bootstrap_seed':20261022,'bootstrap_draws':10000,
        'environment':{'python':sys.version,'packages':packages,'cpu':'AMD Ryzen 7 4800H, 8 cores/16 threads',
                       'physical_memory_bytes':16505966592,'torch_and_BLAS_threads':4},
        'fallback_sha256':manifest,'cache_overlap':overlap,
        'verification':'real-data isolation/replay and saved-probability rescore, one scoped read-only structural reviewer; no new full test-suite run',
        'research_success':'not_achieved','reference':'R25','submission':'not_submitted','ADeLe':'waiting_for_resources'}
    # Every evaluated deployable candidate failed at least one original numerical gate.
    assert all(not d['simple_numeric_gate_pass'] for stage,path in stages for d in read(path/'analysis_audit.json')['decisions'].values())
    write_json(ROOT/'R64/delivery_validation.json',validation)
    files=[]
    for name in ['R62','R63','R64']:
        files += [p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_manifest.json']
    files += [p for p in (REPO/'research_loop').glob('*historical*.py')]
    files += [REPO/'research_loop'/n for n in ['HISTORICAL_POSITIVE_FOLLOWUP_SPEC.md','HISTORICAL_POSITIVE_FOLLOWUP_STATE.md','HISTORICAL_POSITIVE_SIGNAL_LEDGER.md']]
    files += [REPO/'reports'/n for n in required]
    files += list((REPO/'reports/historical_positive_figures').glob('*'))
    files += [REPO/'research_loop/literature/historical_positive_followup.md']
    entries=[{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(files))]
    write_json(ROOT/'R64/delivery_manifest.json',{'files':entries,'count':len(entries),'total_bytes':sum(e['bytes'] for e in entries),'created_utc':now()})
    print(json.dumps({'delivery_files':len(entries),'bytes':sum(e['bytes'] for e in entries),'research_success':'not_achieved',
                      'stage_count':len(stages),'new_fits':10,'fallback_unchanged':True}))

if __name__=='__main__':main()
