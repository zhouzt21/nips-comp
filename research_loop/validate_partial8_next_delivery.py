"""Plan10 artifact, immutable-history and numerical-replay validation."""
import ast,re,subprocess
from partial8_next_common import *

def main():
    dest=ROOT/'partial8_next_v10';dest.mkdir(exist_ok=True);src=REPO/'research_loop';report=REPO/'reports';checks={};cache={}
    def digest(p):
        p=str(Path(p).resolve())
        if p not in cache:cache[p]=sha(p)
        return cache[p]
    allowed={str((src/n).resolve()) for n in ['HIGH_LEVEL_STATE.md','HISTORICAL_POSITIVE_SIGNAL_LEDGER.md']}
    for stage in ['R72','R76','dual_track_v8','shared_reliability_v9']:
        old=read(ROOT/stage/'delivery_manifest.json');entries=old if isinstance(old,list) else old['entries'];same=0;changed=[]
        for row in entries:
            p=str(Path(row['path']).resolve());actual=digest(p)
            if p in allowed:
                changed.append({'path':p,'old':row['sha256'],'current':actual,'authorized_global_update':True})
            else:assert actual==row['sha256'],p;same+=1
        checks[stage]={'unchanged_files':same,'intentional_updates':changed}
    contracts=[ROOT/p for p in ['R62/isolation','R75/contract','R76/full/contract','R77/alternative/contract','R77/full/contract',
        'R77/heterogeneity_contract','R78/contract','R79/source_contract','R80/contract','R81/contract','R81/heterogeneity/contract']]
    for p in contracts:check_freeze(p)
    checks['contracts']=[str(p) for p in contracts]
    decision=read(ROOT/'R80/decision.json');assert not decision['R82_authorized'] and not decision['R83_authorized']
    assert not (ROOT/'R82').exists() and not (ROOT/'R83').exists()
    checks['conditional_decision']=decision
    r80=read(ROOT/'R80/replay_validation.json');assert r80['max_error']<1e-12
    assert r80['source_trajectories']==1146 and r80['dedup_trajectories']==573
    r81=read(ROOT/'R81/replay_validation.json');assert (r81['pairs'],r81['trajectories'],r81['benchmarks'])==(5242,15726,36)
    assert sum(x['chunks'] for x in r81['checks'])==149
    full_summary=read(ROOT/'R81/summary.json');assert full_summary['unique_pairs']==5242 and full_summary['trajectories']==15726
    assert all(r81[k]<1e-11 for k in ['max_brier_error','max_accuracy_error','max_raw_clip_error','max_risk_decomposition_error'])
    hc=read(ROOT/'R81/heterogeneity/replay_validation.json');assert hc['benchmarks']==25 and hc['real_metric_rows']==400 and hc['synthetic_metric_rows']==12800
    assert max(hc[k] for k in ['max_real_brier_error','max_expected_brier_error','max_generated_probability_error','max_repeat_cosine_error','max_point_control_error'])<1e-11
    assert hc['paired_uniform_label_regenerations']==12800
    checks['numerical_replays']={'R80':{k:v for k,v in r80.items() if k!='checks'},'R81':{k:v for k,v in r81.items() if k!='checks'},'R81_C':hc}
    checks['execution']={}
    for name in ['R80/execution.json','R80/analysis_execution.json','R81/execution.json','R81/heterogeneity/execution.json',
                 'R81/analysis_full_execution.json','R81/analysis_heterogeneity_execution.json']:
        v=read(ROOT/name);assert v['seconds']>0 and v['peak_working_set_mib']>0;checks['execution'][name]={k:v[k] for k in ['seconds','peak_working_set_mib','finished']}
    checks['fallback']=[]
    expected='e55ee0b267f3b345685ef072cfa511c3c017d9082256573793961ea20feadcb9'
    for name in ['layout_20261022_120427','nips-comp_20261022_113534']:
        p=Path('D:/nips-comp/paiec-dataset/paiec_git_backups')/name/'workspace/paiec_baseline/dist/smoothed_mean_strength2.zip'
        assert digest(p)==expected;checks['fallback'].append({'path':str(p),'sha256':expected})
    original=src/'literature/partial8_next';manifest=read(original/'original_manifest.json')
    for row in manifest:
        p=original/row['file'];assert digest(p)==row['sha256'] and p.stat().st_size==row['bytes']
        if p.suffix=='.pdf':assert p.read_bytes().startswith(b'%PDF')
    checks['original_sources']=manifest
    required=[report/n for n in ['partial8_identifiability_findings.md','partial8_mechanism_findings.md','partial8_decision_tables.md','partial8_next_reproduction.md']]
    required += [src/'PARTIAL8_NEXT_SPEC.md',src/'PARTIAL8_NEXT_DECISION.md']
    for p in required:
        assert p.exists()
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            assert (p.parent/target.split('#')[0]).exists(),(p,target)
    prompt=REPO/'doc/plan_10.md';assert digest(prompt)=='6685dadac0c93971673bdcb6873691199aa4dcca8993a36fc136c487af87df79'
    code=set(src.glob('*partial8*.py'))
    for p in code:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    checks['static_parse_files']=len(code);checks['report_links']='passed'
    checks['figure_inspection']={'files':['risk_sign_and_error.png','cost_direction_controls.png'],
        'method':'primary agent opened both exported PNGs and inspected axes, labels and confidence intervals'}
    checks['operational_notes']=[
        'Original PDF extraction initially lacked pypdf/fitz; reused installed TeX Live pdftotext, no environment changes.',
        'Immutable R81 runner geometry CSV used the numerical-zero threshold only on pooled displacement. Canonical analysis recomputed direction masks with both displacements >1e-12; probabilities/risks unchanged.',
        'Markdown table header literal pipes escaped in report generator; no metric changes.']
    checks['review']='primary-agent deterministic identity, saved-probability, group-isolation, source-hash and immutable-history checks; not independent review'
    todo_files=['tasks/graph/nips-comp.partial8-identifiability-2026-10-23.md','reports/work-traces/nips-comp/2026-10-23-partial8-identifiability.md']
    todo_commit=subprocess.check_output(['git','-C','D:/todo','rev-parse','HEAD'],text=True).strip()
    todo_status=subprocess.check_output(['git','-C','D:/todo','status','--porcelain','--',*todo_files],text=True).strip()
    assert not todo_status,'Final task/trace must be committed before final delivery snapshot'
    checks['todo_sync']={'status':'done','local_commit':todo_commit,'files':todo_files,'pushed':False,
        'reason':'17 pre-existing unpublished commits and unrelated worktree changes; only the two current files were staged and committed'}
    checks['status']='passed';checks['finished']=now();write_json(dest/'delivery_validation.json',checks)
    paths=set(code)|set(required)|{prompt,src/'HIGH_LEVEL_STATE.md',src/'HISTORICAL_POSITIVE_SIGNAL_LEDGER.md'}
    for folder in [ROOT/'R80',ROOT/'R81',dest,original,report/'partial8_next_figures']:
        paths.update(p for p in folder.rglob('*') if p.is_file())
    paths.update([Path('D:/todo/tasks/graph/nips-comp.partial8-identifiability-2026-10-23.md'),
        Path('D:/todo/reports/work-traces/nips-comp/2026-10-23-partial8-identifiability.md')])
    records=[]
    for p in sorted(paths,key=str):
        if p==dest/'delivery_manifest.json':continue
        assert p.suffix!='.pyc' and '__pycache__' not in p.parts and not p.name.startswith('.env')
        records.append({'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)})
    write_json(dest/'delivery_manifest.json',{'finished':now(),'files':len(records),'bytes':sum(r['bytes'] for r in records),'entries':records,
        'scope':'R80 source identifiability, R81 full mechanism and real/synthetic offset diagnostics; R82/R83 not triggered'})
    print(json.dumps({'status':'passed','files':len(records),'bytes':sum(r['bytes'] for r in records),'preservation':{k:checks[k]['unchanged_files'] for k in ['R72','R76','dual_track_v8','shared_reliability_v9']}}),flush=True)

if __name__=='__main__':main()
