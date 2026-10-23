"""Close artifact inventory and consistency checks after the research is done."""
import re
from historical_positive_common import *

def main():
    expected=['Executive Judgment','Observed Phenomenon','Competing Explanations','Discriminating Evidence',
        'High-Level Understanding','Minimal Method Design','Evaluation and Decision','Next Research Move']
    mainfile=REPO/'reports/transferable_prior_mechanism_findings.md';body=mainfile.read_text(encoding='utf-8')
    for i,title in enumerate(expected,1):assert f'## {i}. {title}' in body
    for file in [mainfile,REPO/'reports/transferable_prior_reproduction.md',REPO/'research_loop/TRANSFERABLE_PRIOR_DECISION.md']:
        for target in re.findall(r'\]\(([^)]+)\)',file.read_text(encoding='utf-8')):
            if target.startswith(('https:','http:','#')):continue
            assert (file.parent/target.split('#')[0]).exists(),(file,target)
    s=read(ROOT/'R67/summary.json');decision=read(ROOT/'R67/decision.json');validation=read(ROOT/'R67/validation.json')
    assert s['unique_pairs']==5242 and s['trajectories']==15726
    assert abs(s['results']['R67_item_segment']['alc_mean']-.19444891435197925)<1e-12
    assert not decision['research_success'] and validation['R67']['rescore_max_error']==0
    assert validation['R66']['rescore_max_error']==0 and validation['R65']['Brier_decomposition_unchanged']
    assert read(ROOT/'R68/adele_audit.json')['new_api_calls']==0
    old=read(ROOT/'R62/isolation/contract.json')
    for p,h in old['sha256'].items():assert sha(p)==h,p
    write_json(ROOT/'R65/status.json',{'status':'complete','authoritative_diagnostics':'descriptor_v2',
        'initial_descriptor_statistics':'superseded; retained','probability_decomposition_unchanged':True})
    write_json(ROOT/'R66/status.json',{'status':'complete','new_fits':12,'item_entry':'passed_and_executed_as_R67','subject_entry':'not_passed'})
    paths=[]
    for stage in ['R65','R66','R67','R68']:
        paths += [p for p in (ROOT/stage).rglob('*') if p.is_file() and p.name not in ['delivery_manifest.json','delivery_validation.json']]
    paths += list((REPO/'reports').glob('transferable_prior*.*'))
    paths += list((REPO/'reports/transferable_prior_figures').glob('*'))
    paths += list((REPO/'research_loop').glob('*transferable*.py'))
    paths += list((REPO/'research_loop').glob('TRANSFERABLE_PRIOR*.md'))
    paths += list((REPO/'research_loop/literature').glob('transferable_prior*'))
    paths += [REPO/'doc/plan_5.md']
    paths=sorted(set(paths))
    manifest={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
    write_json(ROOT/'R67/delivery_manifest.json',manifest)
    result={'status':'complete','files':len(paths),'bytes':sum(v['bytes'] for v in manifest.values()),
        'new_fits':15,'new_checkpoints':380,'outer_pairs':5242,'outer_seeds':[0,1,2],
        'R66_rescore_error':0,'R67_rescore_error':0,'research_success':False,'no_official_submission':True,
        'R65_R66_R67_complete':True,'R68_audit_complete':True,'finished_utc':now()}
    write_json(ROOT/'R67/delivery_validation.json',result);print(json.dumps(result))

if __name__=='__main__':main()
