"""R22 matched-compute source-pool expansion of the existing semantic MLP."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from mlp.reporting.summarize_candidate import summarize


def main():
    root=Path('runs/research_loop/R22');root.mkdir(parents=True,exist_ok=True);start=time.time()
    sources=['mlp/train.py','mlp/model.py','mlp/evaluate.py','mlp/semantic_variant.py','mlp/run/run_pool_scale_round.py','../research_loop/R22_protocol.md']
    snap=root/'code';snap.mkdir(exist_ok=True)
    for p in sources:(snap/Path(p).name).write_bytes(Path(p).read_bytes())
    environment=os.environ.copy();environment.update(PYTHONPATH='D:/paiec_research_deps',HF_HUB_OFFLINE='1',HF_HOME='D:/paiec_pretrained/hf_cache',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    def run(command,log):
        began=time.time()
        with log.open('w',encoding='utf-8') as handle:result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=environment)
        record={'command':command,'log':str(log),'started_unix':began,'elapsed_seconds':time.time()-began,'exit_code':result.returncode}
        with (root/'commands.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(record)+'\n')
        if result.returncode:raise RuntimeError(f'command failed; see {log}')
    for fold in range(3):
        folder=root/f'fold{fold}';folder.mkdir(exist_ok=True)
        old=json.loads(Path(f'runs/research_loop/R8/semantic_fold{fold}/weights.json').read_text())
        run([sys.executable,'-m','mlp.train','--data-root','D:/paiec_payload/data','--split-file',f'runs/mlp_cv/fold{fold}_split.json',
             '--out',str(folder/'weights.npz'),'--seed','2026','--max-rows-per-benchmark','2000','--max-validation-rows-per-benchmark','400',
             '--rows-per-epoch',str(old['train_rows']),'--epochs','30','--batch-size','512','--learning-rate','0.001','--weight-decay','0.0001',
             '--feature-mode','semantic_block','--network-kind','mlp','--internal-validation'],folder/'train.log')
        new=json.loads((folder/'weights.json').read_text())
        for key in ['train_benchmarks','validation_benchmarks','validation_rows','validation_pairs']:
            assert new[key]==old[key],(fold,key)
        print('trained',fold,'pool',new['train_rows'],'exposed',new['unique_training_rows_exposed'],'bestcycle',new['best_epoch'],flush=True)
    for fold in range(3):
        run([sys.executable,'-m','mlp.evaluate','--data-root','D:/paiec_payload/data','--split-file',f'runs/mlp_cv/fold{fold}_split.json',
             '--role','development','--seeds','1','--max-evaluation-items','64','--max-pairs-per-benchmark','100','--alphas','2',
             '--weights',f'expanded={root}/fold{fold}/weights.npz','--out',str(root/f'fold{fold}_raw.json')],root/f'fold{fold}_evaluate.log')
        new=json.loads((root/f'fold{fold}_raw.json').read_text());old=json.loads(Path(f'runs/research_loop/R8/fold{fold}_evaluation.json').read_text())
        for target,source in [('results','results')]:
            for k in ['semantic:prior','semantic:residual_2','retained:residual_2','original:residual_2']:
                new[target][k]=old[source][k]
        for directory,row in new['by_benchmark'].items():
            historic=old['by_benchmark'][directory]
            np.testing.assert_allclose(row['brier_mean']['smoothed_mean_2'],historic['brier_mean']['smoothed_mean_2'],atol=1e-12,rtol=0)
            for field in ['brier_mean','alc_mean']:
                for k in ['semantic:prior','semantic:residual_2','retained:residual_2','original:residual_2']:row[field][k]=historic[field][k]
        (root/f'fold{fold}_evaluation.json').write_text(json.dumps(new,indent=2),encoding='utf-8')
    d=summarize([root/f'fold{i}_evaluation.json' for i in range(3)],'expanded:residual_2',
                ['expanded:prior','semantic:prior','semantic:residual_2','retained:residual_2','original:residual_2','smoothed_mean_2'])
    d['elapsed_seconds']=time.time()-start;(root/'summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    (root/'provenance.json').write_text(json.dumps({p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},indent=2))
    print(json.dumps(d['pair_equal_alc'],indent=2),flush=True)


if __name__=='__main__':main()
