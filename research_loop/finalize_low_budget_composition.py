"""Record actual stage closure, source snapshots, and deliverable inventory."""
import hashlib
import json
import re
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd

REPO=Path(__file__).resolve().parents[1]
RUN=REPO/'paiec_baseline/runs/research_loop'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report=REPO/'reports/low_budget_prior_composition_findings.md'
    text=report.read_text(encoding='utf-8')
    checked=[]
    for url in re.findall(r'\]\(([^)]+)\)',text):
        if not url.startswith(('https://','http://')):
            assert (report.parent/url).exists(),url
            checked.append(url)
    metrics=pd.read_csv(REPO/'reports/low_budget_prior_composition_metrics.csv')
    assert len(metrics)==180 and not metrics.duplicated(['fold','method','budget']).any()
    outcome=read(RUN/'R61/summary.json')
    assert not outcome['all_source_numeric_pass'] and outcome['outer_status']=='condition_not_met'
    assert read(RUN/'R61/validation.json')['status']=='passed'
    provenance=read(RUN/'R59/provenance.json')
    assert sha(REPO/'doc/plan_3.md')==provenance['sha256']['..\\doc\\plan_3.md']
    complete=read(RUN/'R59/complete.json')
    complete['status']='complete'
    complete['operator_decision']='R60 single support-label proper-loss selector; R61 source gate failed'
    write(RUN/'R59/complete.json',complete)
    write(RUN/'R60/summary.json',{'status':'complete','candidates':1,'candidate':'R56_support1_gate',
        'new_fits':0,'threshold_search':False,'decision':'frozen before R61 scoring'})
    sources=[REPO/'research_loop'/name for name in ['run_low_budget_composition_pilot.py','audit_low_budget_composition.py',
        'report_low_budget_composition.py','finalize_low_budget_composition.py','low_budget_prior_composition_validation.md']]
    sources += list((REPO/'research_loop/literature').glob('*low_budget*'))
    target=RUN/'R61/supporting_source_snapshot'; target.mkdir(exist_ok=True)
    for p in sources:
        (target/p.name).write_bytes(p.read_bytes())
    files=[]
    for stage in ['R59','R60','R61']:
        files += [p for p in (RUN/stage).rglob('*') if p.is_file() and p.name!='delivery_inventory.json']
    files += [p for p in (REPO/'reports').glob('low_budget*') if p.is_file()]
    files += list((REPO/'reports/low_budget_prior_composition_figures').glob('*'))
    files += sources+[REPO/'research_loop/LOW_BUDGET_PRIOR_COMPOSITION_SPEC.md',
        REPO/'research_loop/LOW_BUDGET_PRIOR_OPERATOR_DECISION.md',REPO/'doc/plan_3.md']
    extra=Path('D:/nips-comp/paiec-dataset/low_budget_prior_sources')
    files += [p for p in extra.rglob('*') if p.is_file()]
    write(RUN/'R61/delivery_inventory.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
        'status':'complete_source_task_outer_condition_not_met','metric_rows':len(metrics),'links_checked':len(checked),
        'sha256':{str(p):sha(p) for p in sorted(set(files))}})
    print(json.dumps({'status':'complete','metric_rows':len(metrics),'links_checked':len(checked),'hashed_files':len(set(files))}))


if __name__=='__main__':
    main()
