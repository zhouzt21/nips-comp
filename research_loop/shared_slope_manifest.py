"""Final Track B artifact hashes and checks; no global state files are modified."""
from historical_positive_common import *

def main():
    historical=[ROOT/'R62/isolation',ROOT/'R72/coverage_contract',ROOT/'R72/full/contract']
    current=[ROOT/'R75/contract',ROOT/'R76/full/contract',ROOT/'R76/smoke/contract',ROOT/'R76/cpu_adapter_smoke/contract',ROOT/'R76/cpu_adapter_cost/contract']
    checks=[]
    for path in [*historical,*current]:check_freeze(path);checks.append(str(path))
    fallbacks=[]
    for base in ARCHIVES:
        p=base/'dist/smoothed_mean_strength2.zip';h=sha(p);assert h=='e55ee0b267f3b345685ef072cfa511c3c017d9082256573793961ea20feadcb9';fallbacks.append({'path':str(p),'sha256':h})
    files=[]
    for root in [ROOT/'R75',ROOT/'R76']:
        files += [p for p in root.rglob('*') if p.is_file() and p.name!='delivery_manifest.json']
    files += list((REPO/'research_loop').glob('shared_slope_*.*'))
    files += [REPO/'research_loop/SHARED_SLOPE_CONTRACT.md',REPO/'research_loop/SHARED_SLOPE_SOURCES.md',REPO/'research_loop/dual_track_cpu_features.py']
    files += list((REPO/'reports').glob('shared_slope_*.md'))
    files += list((REPO/'reports/shared_slope_figures').glob('*'))
    files=sorted(set(files));entries=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    result={'finished':now(),'scope':'Track B conditional_simulation','files':len(entries),'bytes':sum(v['bytes'] for v in entries),'entries':entries,
            'contracts_checked':checks,'protected_R62_assets':len(read(ROOT/'R62/isolation/contract.json')['sha256']),'fallbacks':fallbacks,
            'decision':'research_candidate_retained_full_gate_not_passed_due_severe_regression','hosted_confirmed':False,'formal_submission':False,
            'probability_replay':{k:v for k,v in read(ROOT/'R76/full/replay_validation.json').items() if k!='checks'}}
    write_json(ROOT/'R76/delivery_manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['entries','probability_replay']}))

if __name__=='__main__':main()
