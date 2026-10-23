"""Submission orchestration only: frozen feature/prediction routines are separate."""
import threading
from collections import OrderedDict
from .features import CPUItemFeatures
from .adapter_cpu import predict_raw
from .adapter import canonical,ITEM_FIELDS,SUBJECT_FIELDS,subject_key

class Runtime:
    def __init__(self,shared):
        self.shared=shared;self.features=None;self.lock=threading.RLock();self.feature_cache=OrderedDict()
    def run(self,input,labeled=None,visible_subjects=None):
        # Labels/subject contexts are request-local. The bounded cache contains only input features.
        with self.lock:
            if self.features is None:self.features=CPUItemFeatures(threads=2)
            group=input[1].get('benchmark_id')
            def clean(s):return {k:s[k] for k in SUBJECT_FIELDS if k in s}
            subject=clean(input[0]);key=subject_key(subject)
            def relevant(v):
                if not group or v[0][1].get('benchmark_id')!=group:return False
                if self.shared and visible_subjects is not None:return True
                s=clean(v[0][0]);return s==subject if key is None else subject_key(s)==key
            items=[input[1],*[v[0][1] for v in (labeled or []) if relevant(v)]]
            unique={canonical({k:i.get(k) for k in ITEM_FIELDS}):i for i in items}
            pending={k:v for k,v in unique.items() if k not in self.feature_cache}
            vectors=self.features.encode_items(list(pending.values()))
            memo={k:self.feature_cache[k] for k in unique if k in self.feature_cache}
            memo.update(zip(pending,vectors))
            for k,v in memo.items():
                self.feature_cache[k]=v;self.feature_cache.move_to_end(k)
            while len(self.feature_cache)>2048:self.feature_cache.popitem(last=False)
            def encode(item):
                value=memo[canonical({k:item.get(k) for k in ITEM_FIELDS})].copy()
                value[:384]/=2.;value[384:]/=4.
                return value
            return predict_raw(input,labeled,encoder=encode,visible_subjects=visible_subjects if self.shared else None,mode='partial' if self.shared else 'pooled',max_peers=3 if self.shared else 0)
