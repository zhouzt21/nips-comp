"""R18b frozen encoder, controlled128/256text windows and lexical controls."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from mlp.window_features import WindowFeatures
from mlp.semantic_features import SemanticFeatures
from mlp.run.run_semantic_round import episodes,scores,RIDGES
from mlp.evaluate import ALC_WEIGHTS
from mlp.reporting.summarize_candidate import summarize


def main():
    root=Path('runs/research_loop/R18');root.mkdir(parents=True,exist_ok=True);start=time.time()
    old=SemanticFeatures();control=WindowFeatures(128)
    texts=['A short math question: what is two plus two?', ' '.join(['context question answer hypothesis']*200), '{"question":"Find the valid statement","context":"one two three"}']
    a=old.encode_many(texts);b=control.encode_many(texts);err=max(float(np.max(abs(a[k][m]-b[k][m]))) for k in a for m in ['semantic','text_hash'])
    if err>2e-6:raise RuntimeError(f'128equivalence failed: {err}')
    old.db.close();control.db.close();del old,control
    engine=WindowFeatures(256);selections=[]
    sources=['mlp/window_features.py','mlp/run/run_window_round.py','mlp/run/run_semantic_round.py','../research_loop/R18b_protocol.md']
    snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    for fold in range(3):
        split=json.loads(Path(f'runs/mlp_cv/fold{fold}_split.json').read_text());sums={k:np.zeros(len(RIDGES)) for k in ['semantic','text_hash']};count=0
        for directory in split['train_benchmarks']:
            for row in episodes(directory,73,25,engine):
                for mode in sums:sums[mode]+=[scores(row,mode,r)@ALC_WEIGHTS for r in RIDGES]
                count+=1
        sel={'fold':fold,'train_benchmarks':split['train_benchmarks'],'heldout_benchmarks':split['development_benchmarks'],
             'ridge_grid':RIDGES,'training_pairs':count,'training_alc':{k:(v/count).tolist() for k,v in sums.items()},
             'selected_ridge':{k:RIDGES[int(np.argmin(v))] for k,v in sums.items()}}
        selections.append(sel);(root/f'fold{fold}_selection.json').write_text(json.dumps(sel,indent=2),encoding='utf-8')
        print(f'fold{fold} selected {sel["selected_ridge"]}',flush=True)
    for fold,sel in enumerate(selections):
        historic=json.loads(Path(f'runs/research_loop/R7/fold{fold}_evaluation.json').read_text())
        names=['semantic256','text256','semantic128','text128','base','training_selected_smoothed'];total={k:np.zeros(6) for k in names};by_benchmark={};count=0
        for directory in sel['heldout_benchmarks']:
            rows=episodes(directory,0,100,engine)
            if not rows:continue
            n=len(rows);count+=n;old=historic['by_benchmark'][directory]
            avg={'semantic256':np.mean([scores(r,'semantic',sel['selected_ridge']['semantic']) for r in rows],axis=0),
                 'text256':np.mean([scores(r,'text_hash',sel['selected_ridge']['text_hash']) for r in rows],axis=0)}
            for dest,src in [('semantic128','semantic'),('text128','text_hash'),('base','smoothed_mean_2'),('training_selected_smoothed','training_selected_smoothed')]:avg[dest]=np.array(old['brier_mean'][src])
            np.testing.assert_allclose(np.mean([scores(r,'semantic',None) for r in rows],axis=0),avg['base'],atol=1e-7)
            by_benchmark[directory]={'pairs_per_seed':[n],'brier_mean':{k:v.tolist() for k,v in avg.items()},'alc_mean':{k:float(v@ALC_WEIGHTS) for k,v in avg.items()}}
            for k,v in avg.items():total[k]+=n*v
        result={'selection':sel,'pairs_per_seed':[count],'by_benchmark':by_benchmark,
                'results':{k:{'brier_mean':(v/count).tolist(),'alc_mean':float(v/count@ALC_WEIGHTS)} for k,v in total.items()}}
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    summary=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'semantic256',names[1:]);summary['elapsed_seconds']=time.time()-start;summary['new_embeddings']=engine.encoded
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({'encoder128_max_error':err,'fingerprint':engine.fingerprint,'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}},indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['pair_equal_alc','comparisons']},indent=2))


if __name__=='__main__':main()
