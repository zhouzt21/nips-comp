"""R16 preserve target outcomes; isolate source direction in the ability fit."""
import json
import hashlib
from pathlib import Path
import numpy as np
from .relative_ability import aggregate,fit
from .run_ability_round import corrected
from .run_shared_round import records
from .evaluate import ALC_WEIGHTS
from .summarize_candidate import summarize


def main():
    root=Path('runs/research_loop/R16');root.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(Path('runs/research_loop/R15/label_semantics_inventory.json').read_text(encoding='utf-8'))
    directions={r['directory']:r['direction'] for r in inventory};models=[]
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text())
        allowed=[b for b in split['train_benchmarks'] if directions[b]=='higher_is_better']
        rows=[]
        for b in allowed:rows.extend(aggregate(b))
        model=fit(rows,allowed);models.append(model)
        model['excluded_training_benchmarks']=sorted(set(split['train_benchmarks'])-set(allowed))
        (root/f'fold{fold}_ability.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    strata={}
    for fold,model in enumerate(models):
        old=json.loads(Path(f'runs/research_loop/R15/fold{fold}_evaluation.json').read_text());sel=old['selection']
        names=['filtered_ability','original_ability','peer','base'];total={k:np.zeros(6) for k in names};count=0;by_benchmark={}
        for b in sel['heldout_benchmarks']:
            if b not in old['by_benchmark']:continue
            result=[]
            for seed in [0,1,2]:
                rows=records(b,seed,100);result.append(corrected(rows,b,model,sel['plugin_alpha'],sel['plugin_kappa']).mean(0))
            n=len(rows);count+=n;avg={'filtered_ability':np.mean(result,axis=0)}
            for k,previous in [('original_ability','ability_peer'),('peer','peer'),('base','base')]:avg[k]=np.array(old['by_benchmark'][b]['brier_mean'][previous])
            by_benchmark[b]={'pairs_per_seed':[n]*3,'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():total[k]+=n*v
        output={'selection':sel,'pairs_per_seed':[count]*3,'by_benchmark':by_benchmark,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
        for path,methods in [('R3',{'old_mlp':'original:residual_2','base_seed0':'smoothed_mean_2'}),('R7_verify',{'semantic':'semantic','base_3seed':'smoothed_mean_2'})]:
            data=json.loads(Path(f'runs/research_loop/{path}/fold{fold}_evaluation.json').read_text())
            for b,info in data['by_benchmark'].items():
                direction=directions[b];n=info['pairs_per_seed'][0]
                for dest,src in methods.items():
                    row=strata.setdefault(direction,{}).setdefault(dest,{'pairs':0,'brier_sum':np.zeros(6)})
                    row['pairs']+=n;row['brier_sum']+=n*np.array(info['brier_mean'][src])
    s=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'filtered_ability',names[1:])
    s['scope']='Training-source semantic filter sensitivity; all original target outcomes preserved; reused public folds'
    s['direction_strata']={direction:{k:{'pairs':v['pairs'],'brier':(v['brier_sum']/v['pairs']).tolist(),'alc':float(v['brier_sum']/v['pairs']@ALC_WEIGHTS)} for k,v in group.items()} for direction,group in strata.items()}
    (root/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8')
    sources=['mlp/audit_target_semantics.py','mlp/relative_ability.py','mlp/run_ability_round.py','research_loop/R16_protocol.md']
    snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2),encoding='utf-8')
    print(json.dumps({k:s[k] for k in ['pair_equal_alc','comparisons','direction_strata']},indent=2))


if __name__=='__main__':main()
