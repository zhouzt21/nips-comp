"""Append missing frozen MiniLM vectors required by the predeclared resampling seeds."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mlp.evaluate import _load_directory, _stable_seed
from mlp.reproduction_episodes import DATA_ROOT
from mlp.semantic_features import SemanticFeatures, MODEL_REVISION
from mlp.training_mechanism_data import write_json, digest
from mlp.training_mechanism_guard import sha, ROOT
from mlp.run.run_training_mechanism_round import info


def main():
    start = time.monotonic()
    encoder = SemanticFeatures()
    names = sorted({n for fold in range(3) for half in info(fold)['inner_halves'] for n in half})
    record = {'model_revision': MODEL_REVISION, 'fingerprint': encoder.fingerprint,
              'seeds': list(range(5501,5521)), 'local_files_only': True,
              'model_files': {str(p):sha(p) for p in encoder.root.iterdir() if p.is_file()},
              'benchmarks':{}, 'existing_rows_overwritten':0}
    for name in names:
        response, subjects, items = _load_directory(DATA_ROOT,name)
        groups = response.groupby('subject_id',sort=False)
        eligible = [sid for sid,g in groups if g.item_id.nunique() >= 80 and sid in subjects]
        if len(eligible)>25:
            rng = np.random.default_rng(_stable_seed(2026,name,'pair_sample'))
            eligible = [eligible[i] for i in np.sort(rng.choice(len(eligible),25,replace=False))]
        needed = set()
        for sid in eligible:
            original = pd.unique(groups.get_group(sid).item_id)
            for seed in range(5501,5521):
                ids = original.copy()
                np.random.default_rng(_stable_seed(seed,name,sid)).shuffle(ids)
                needed.update(ids[:31])
                needed.update(ids[len(ids)//2:len(ids)//2+64])
        pending = {}
        for item_id in needed:
            text = items[item_id]['content']
            key = encoder.key(text)
            if encoder.db.execute('SELECT 1 FROM features WHERE key=?',(key,)).fetchone() is None:
                pending[key] = text
        print(json.dumps({'stage':'cache_missing','benchmark':name,'needed_items':len(needed),
                          'missing_unique_texts':len(pending)}),flush=True)
        before = encoder.encoded
        values = list(pending.values())
        for offset in range(0,len(values),256):
            encoder.encode_many(values[offset:offset+256])
            print(json.dumps({'stage':'cache_encoded','benchmark':name,'done':min(offset+256,len(values)),
                              'total':len(values),'seconds':time.monotonic()-start}),flush=True)
        record['benchmarks'][name] = {'needed_items':len(needed),'missing_unique_texts':len(pending),
                                     'encoded':encoder.encoded-before,'added_keys_sha256':digest(*sorted(pending))}
        record['seconds'] = time.monotonic()-start
        record['encoded_total'] = encoder.encoded
        write_json(ROOT/'semantic_cache_preparation.json',record)
    record['status'] = 'complete'
    write_json(ROOT/'semantic_cache_preparation.json',record)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
