"""Freeze the unchanged R62 scorer without starting its uncached slow path."""
from historical_positive_common import *

if __name__=='__main__':
    check_freeze(ROOT/'R62/isolation')
    root=ROOT/'R62';manifests=read(root/'models.json')
    paths=[REPO/'research_loop/run_historical_positive_outer.py',REPO/'research_loop/historical_positive_episodes.py',
        REPO/'research_loop/historical_positive_common.py',root/'models.json',
        ROOT/'R62/isolation/contract.json',REPO/'research_loop/HISTORICAL_POSITIVE_FOLLOWUP_SPEC.md']
    paths += [Path(m['weights']) for m in manifests if m['weights']]
    freeze(root/'evaluation_contract',paths,{'stage':'R62','seeds':[0,1,2],'pairs':5242,'epochs':EPOCHS})
    print('Frozen R62 evaluation contract; safe to run cached wrapper')
