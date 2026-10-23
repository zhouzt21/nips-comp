"""Audit historical fold isolation, then perform only preselected full-source fits."""
import argparse
import shutil
import time
from historical_positive_common import *
from mlp.cohort_semantic import CachedSemantic
from mlp.training_mechanism_data import (mlp_training, mlp_validation, reliability_fixed,
    concatenate, resampled_banks, bank_load)
from mlp.training_mechanism_training import train

AUDIT = ROOT/'R62/isolation'

def audit():
    paths = [Path(__file__), REPO/'research_loop/historical_positive_common.py',
             REPO/'research_loop/HISTORICAL_POSITIVE_FOLLOWUP_SPEC.md', REPO/'doc/plan_4.md',
             Path('D:/paiec_payload/data/corpus_manifest.json')]
    paths += sorted((BASE/'mlp').rglob('*.py'))
    records=[]
    for f in range(3):
        info=source(f); names=info['inner_halves'][0]+info['inner_halves'][1]
        assert len(names)==len(set(names))
        assert set(names)<=set(info['train_benchmarks'])
        assert set(info['train_benchmarks']).isdisjoint(info['heldout_benchmarks'])
        paths += [ROOT/f'reproduction_v4/source_fold{f}.json', ROOT/f'R25/fold{f}_selection.json',
                  ROOT/f'R43/fold{f}_selection.json', V5/f'R54/fold{f}_mlp_analysis.json',
                  V5/f'R56/fold{f}_selection.json', V5/f'R55/fold{f}_selection.json']
        assert read(paths[-3])['selection']['best30']==EPOCHS['R54'][f]
        assert read(paths[-2])['epoch']==EPOCHS['R56'][f]
        s55=read(paths[-1])
        assert s55['epochs']['query']==EPOCHS['R55_resampled_query'][f]
        assert s55['epochs']['episode']==EPOCHS['R55_resampled_episode'][f]
        s43=read(ROOT/f'R43/fold{f}_selection.json')
        assert s43['inner_halves']==info['inner_halves']
        assert s43['heldout_benchmarks']==info['heldout_benchmarks']
        assert s43['models']['query']['selected_epoch']==EPOCHS['R43_query_best20'][f]
        for fit in s43['models']['query']['inner_fits']:
            assert set(fit['train_benchmarks']).isdisjoint(fit['validation_benchmarks'])
            assert set(fit['train_benchmarks']+fit['validation_benchmarks'])==set(names)
        archived=[p/f'runs/research_loop/R43/fold{f}_query.npz' for p in ARCHIVES]
        assert sha(archived[0])==sha(archived[1])
        for p in ARCHIVES:
            assert sha(p/f'runs/research_loop/R43/fold{f}_selection.json')==sha(ROOT/f'R43/fold{f}_selection.json')
        paths+=archived
        records.append({'fold':f, 'fit_benchmarks':names, 'checkpoint_selection_benchmarks':names,
            'calibration_benchmarks':[], 'normalization_benchmarks_R55':names,
            'heldout_benchmarks':info['heldout_benchmarks'], 'intersection':[],
            'selected_epochs':{m:e[f] for m,e in EPOCHS.items()},
            'R43_archive_weights':str(archived[0]),'R43_sha256':sha(archived[0]),
            'MLP_refit_order':'inner_half0 followed by inner_half1',
            'R55_refit_order':'inner_half0 followed by inner_half1',
            'refit_monitor':'fit-only source episodes; never used for epoch selection'})
    fallback=[p/'dist/smoothed_mean_strength2.zip' for p in ARCHIVES]
    assert sha(fallback[0])==sha(fallback[1])
    paths+=fallback
    freeze(AUDIT,paths,{'records':records,'epochs':EPOCHS})
    write_json(AUDIT/'audit.json',{'passed':True,'records':records,'checked_utc':now(),
        'scope':'foldwise Benchmark separation; repeated public-data adaptive selection remains',
        'fallback_sha256':sha(fallback[0])})
    print(json.dumps({'audit':'passed','folds':3,'epochs':EPOCHS}),flush=True)

def refit62():
    check_freeze(AUDIT)
    engine=CachedSemantic(); start=time.perf_counter(); manifests=[]
    for f in range(3):
        info=source(f); names=sum(info['inner_halves'],[])
        data=mlp_training(names)
        assert set(g.split('\0',1)[0] for g in data['groups'])<=set(names)
        monitor=[ep for name in names for ep in mlp_validation(name,engine)]
        for method,risk in [('R54','pair'),('R56','mixture')]:
            out=ROOT/f'R62/refits/fold{f}_{method}'
            epoch=EPOCHS[method][f]
            train('mlp',data,monitor,out,epochs=epoch,risk=risk)
            path=out/f'epoch{epoch}.npz'
            manifests.append({'fold':f,'method':method,'epoch':epoch,'weights':str(path),
                'sha256':sha(path),'fit_benchmarks':names, 'fit_rows':len(data['x']),
                'training_seed':2026,'fit_complete':read(out/'complete.json'),
                'label':'foldwise refit of historical candidate'})
            print(json.dumps({'refit':method,'fold':f,'epoch':epoch,'seconds':time.perf_counter()-start}),flush=True)
    write_json(ROOT/'R62/models.json',manifests)
    check_freeze(AUDIT)

def refit63():
    check_freeze(AUDIT)
    assert (ROOT/'R62/summary.json').exists(),'R62 must finish first'
    engine=CachedSemantic(); start=time.perf_counter(); manifests=[]
    for f,record in enumerate(read(AUDIT/'audit.json')['records']):
        info=source(f); names=sum(info['inner_halves'],[])
        target=ROOT/f'R63/reused/fold{f}_R43_query.npz'
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copyfile(record['R43_archive_weights'],target)
        assert sha(target)==record['R43_sha256']
        manifests.append({'fold':f,'method':'R43_query_best20','epoch':EPOCHS['R43_query_best20'][f],
            'weights':str(target),'sha256':sha(target),'label':'reused original full-source refit',
            'fit_benchmarks':names,'training_seed':4307})
        fixed=concatenate([reliability_fixed(name,engine,info['semantic_ridge']) for name in names])
        banks={name:resampled_banks(name,engine,info['semantic_ridge']) for name in names}
        def epoch_data(epoch):
            return concatenate([bank_load(banks[name][5500+epoch]) for name in names])
        for family in ['query','episode']:
            method=f'R55_resampled_{family}'; epoch=EPOCHS[method][f]
            entry={'fold':f,'method':method,'epoch':epoch,'fit_benchmarks':names,
                   'training_seed':4307,'label':'foldwise refit of historical candidate'}
            if epoch:
                out=ROOT/f'R63/refits/fold{f}_{method}'
                train(family,fixed,fixed,out,epochs=epoch,episode_only=family=='episode',epoch_data=epoch_data)
                path=out/f'epoch{epoch}.npz'
                entry.update(weights=str(path),sha256=sha(path),fit_complete=read(out/'complete.json'),
                    episode_bank_sha256={str(banks[n][5500+e]):sha(banks[n][5500+e])
                        for n in names for e in range(1,epoch+1)})
            else:
                entry.update(weights=None,sha256=None,policy='fixed source-selected fold-level R25 fallback')
            manifests.append(entry)
            print(json.dumps({'refit':method,'fold':f,'epoch':epoch,'seconds':time.perf_counter()-start}),flush=True)
    write_json(ROOT/'R63/models.json',manifests)
    check_freeze(AUDIT)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['audit','r62','r63'])
    args=parser.parse_args()
    {'audit':audit,'r62':refit62,'r63':refit63}[args.stage]()
