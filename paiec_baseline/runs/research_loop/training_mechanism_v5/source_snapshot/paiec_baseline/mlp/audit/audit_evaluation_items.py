"""R32 counts, repeat-input disagreement and frozen R31 budget decomposition."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from mlp.evaluate import BUDGETS,ALC_WEIGHTS


def main():
    root=Path('runs/research_loop/R32');root.mkdir(parents=True,exist_ok=True);start=time.time()
    r31=json.loads(Path('runs/research_loop/R31/summary.json').read_text());benchmarks=[]
    fields=['fold','benchmark','subject_id','items','responses','evaluation_pool_items','cap_active','repeated_item_groups','max_responses_per_item']
    with (root/'pair_census.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for fold in range(3):
            old=json.loads(Path(f'runs/research_loop/R31/fold{fold}_evaluation.json').read_text())
            for directory,previous in old['by_benchmark'].items():
                folder=Path('D:/paiec_payload/data')/directory
                data=pd.read_parquet(folder/'response.parquet',columns=['subject_id','item_id','response','interactors'])
                data=data[data['response'].isin((0,1))].dropna(subset=['subject_id','item_id'])
                available=set(pd.read_parquet(folder/'subjects.parquet',columns=['subject_id'])['subject_id'])
                records=[];valid=[]
                for sid,g in data.groupby('subject_id',sort=False):
                    counts=g.groupby('item_id',sort=False).size();n=len(counts)
                    if n<80 or sid not in available:continue
                    record=dict(fold=fold,benchmark=directory,subject_id=str(sid),items=n,responses=len(g),evaluation_pool_items=n-n//2,cap_active=int(n-n//2>64),repeated_item_groups=int((counts>1).sum()),max_responses_per_item=int(counts.max()))
                    records.append(record);valid.append(sid);writer.writerow(record)
                assert len(records)==previous['pairs_per_seed'][0]
                active=data[data['subject_id'].isin(valid)].copy();active['interactors']=active['interactors'].fillna('').astype(str)
                itemgroups=active.groupby(['subject_id','item_id'],sort=False).agg(count=('response','size'),success=('response','sum'),interactors=('interactors','nunique'))
                same=active.groupby(['subject_id','item_id','interactors'],sort=False)['response'].agg(['size','sum']);rep=same[same['size']>1]
                repeat_trials=int(rep['size'].sum());empirical=float((rep['sum']*(1-rep['sum']/rep['size'])).sum()/repeat_trials) if repeat_trials else None
                unbiased=float((rep['sum']*(1-rep['sum']/rep['size'])*rep['size']/(rep['size']-1)).sum()/repeat_trials) if repeat_trials else None
                b={'fold':fold,'benchmark':directory,'pairs':len(records),'cap_active_pairs':sum(r['cap_active'] for r in records),
                   'eval_items_quantiles':np.quantile([r['evaluation_pool_items'] for r in records],[0,.5,.9,1]).tolist(),
                   'full_eval_item_count_per_seed':sum(r['evaluation_pool_items'] for r in records),'capped_eval_item_count_per_seed':sum(min(64,r['evaluation_pool_items']) for r in records),
                   'responses':len(active),'repeated_subject_item_groups':int((itemgroups['count']>1).sum()),'multi_interactor_item_groups':int((itemgroups['interactors']>1).sum()),
                   'repeated_same_item_interactor_groups':len(rep),'repeat_trials':repeat_trials,'disagreeing_same_item_interactor_groups':int(((rep['sum']>0)&(rep['sum']<rep['size'])).sum()),
                   'repeat_subset_empirical_within_input_variance':empirical,'repeat_subset_sample_variance_if_iid':unbiased}
                benchmarks.append(b);print(directory,len(records),'capactive',b['cap_active_pairs'],'maxeval',b['eval_items_quantiles'][-1],flush=True)
    diff=np.array(r31['pair_equal_brier']['R25'])-np.array(r31['pair_equal_brier']['smoothed_mean_2']);gain=-float(diff@ALC_WEIGHTS)
    d={'authority':'../doc/loop_2.md','scope':'input-count census plus posthoc descriptive outcome disagreement; no predictor selection',
       'pairs':sum(b['pairs'] for b in benchmarks),'cap_active_pairs':sum(b['cap_active_pairs'] for b in benchmarks),
       'full_eval_item_count_per_seed':sum(b['full_eval_item_count_per_seed'] for b in benchmarks),'capped_eval_item_count_per_seed':sum(b['capped_eval_item_count_per_seed'] for b in benchmarks),
       'repeat_trials':sum(b['repeat_trials'] for b in benchmarks),'by_benchmark':benchmarks,
       'R31_budget_diagnostic':{'budgets':list(BUDGETS),'candidate_minus_baseline_brier':diff.tolist(),'weighted_delta_contributions':(diff*ALC_WEIGHTS).tolist(),'total_gain':gain,'remaining_gap_to_005':.005-gain},'elapsed_seconds':time.time()-start}
    (root/'census.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    refs=['loop_2.md','Research Down–Up Loop.md','High-level 认识维度与相互关系.md','Model-Nature_High-Level_Recognition_Paper_List_v1_1.md','常见的组合思路.md']
    Path('../research_loop/V3_SOURCE_MANIFEST.json').write_text(json.dumps({r:hashlib.sha256((Path('../doc')/r).read_bytes()).hexdigest() for r in refs},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in d.items() if k!='by_benchmark'},indent=2),flush=True)


if __name__=='__main__':main()
