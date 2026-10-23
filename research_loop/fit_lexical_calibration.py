"""Benchmark-grouped base selection and source-only R70/R71 coefficient fitting."""
import time
from lexical_peer_common import *

def load_source():
    out={}
    for folder in sorted((ROOT/'R70/source_bank').iterdir()):
        if not folder.is_dir():continue
        arrays=dict(np.load(folder/'predictions.npz',allow_pickle=False));rows=[]
        for idx in read(folder/'index.json'):
            a=idx['start'];b=a+idx['size'];bank={k:v[a:b] for k,v in arrays.items()}
            rows.append({**idx,'counts':bank['counts'],'successes':bank['successes'],'bank':bank})
        if rows:out[rows[0]['benchmark']]=rows
    return out

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();data=load_source();results=[]
    check_freeze(ROOT/'R70/source_contract')
    for f in range(3):
        info=source(f);halves=info['inner_halves'];oof=[];cv=[];audit=[]
        folder=ROOT/f'R70/calibration/fold{f}';folder.mkdir(parents=True,exist_ok=True)
        for h in range(2):
            fit_names=halves[1-h];val_names=halves[h]
            assert set(fit_names).isdisjoint(val_names)
            assert set(fit_names+val_names).isdisjoint(info['heldout_benchmarks'])
            train=[e for b in fit_names for e in data[b]];val=[e for b in val_names for e in data[b]]
            ridge,loss=select_ridge(train);coef=calibrate([(ep,ridge) for ep in train])
            audit.append({'heldout_half':h,'fit_names':fit_names,'validation_names':val_names,
                'selected_ridge':ridge,'ridge_training_alc':loss,'calibration_fit':'same inner fit half only',
                'validation_labels_not_used':True,'validation_pairs':len(val),'fit_coefficients':coef})
            for ep in val:
                oof.append((ep,ridge));p,err=full_predictions(ep['bank'],ridge,coef,ep['counts'],ep['identity'])
                cv.append(score_episode({**ep,'fold':f},p,{'inner_half':h,'selected_ridge':ridge,'random_energy_error':err}))
        final=calibrate(oof);final.update(fold=f,base_ridge_on_outer=info['semantic_ridge'],outer_benchmarks=info['heldout_benchmarks'],oof_base_selection=audit)
        write_json(folder/'coefficients.json',final);write_json(folder/'source_cv_scores.json',cv)
        summary=summarize(cv,folder,info['training_pairs'])
        oof_rows=[]
        for ep,r in oof:
            a,c,z=stats(ep,r)
            for j,n in enumerate(BUDGETS):
                oof_rows.append({'benchmark':ep['benchmark'],'subject_id':ep['subject_id'],'budget':n,'base_ridge':r,'A':float(a[:,j].sum()),'C':float(c[:,j].sum())})
        import pandas as pd
        pd.DataFrame(oof_rows).to_csv(folder/'oof_statistics.csv',index=False)
        results.append({'fold':f,'coefficients':{k:final[k] for k in ['independent','monotone','global','grouped']},
            'cv_delta_R25':{m:v['alc_mean']-summary['results']['R25']['alc_mean'] for m,v in summary['results'].items()}})
        print(json.dumps(results[-1]),flush=True)
    freeze(ROOT/'R70/calibration_contract',[Path(__file__),REPO/'research_loop/lexical_peer_common.py',REPO/'research_loop/LEXICAL_PEER_EVIDENCE_SPEC.md',
        *[ROOT/f'R70/calibration/fold{f}/coefficients.json' for f in range(3)]],{'fit':'strict source benchmark OOF; inner-half validation refits only inner fit labels','outer_runs':'predeclared R70 independent/monotone/global and R71 grouped, no selection by outer'})
    write_json(ROOT/'R70/calibration_summary.json',{'folds':results,'seconds':time.perf_counter()-start,'finished':now()})

if __name__=='__main__':main()
