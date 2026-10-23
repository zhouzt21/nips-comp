"""Read-only bridge to exact R64 identities and R72 legal acquisition records."""
import pandas as pd
from historical_positive_common import *
from mlp.cohort_semantic import CachedSemantic
from mlp.metadata_features import encode as meta_encode

class Benchmark:
    def __init__(self, folder, ridge, engine):
        self.folder=folder;self.name=folder.name+'/formatted_tables';self.ridge=ridge
        self.records=read(ROOT/'R72/contexts'/folder.name/'index.json')
        self.by={(v['subject_id'],v['seed']):v for v in self.records}
        self.indices=[]
        for ip in sorted(folder.glob('index_*.json')):
            for idx in read(ip):
                self.indices.append(idx)
                self.by[(idx['subject_id'],idx['seed'])]['labels']=np.asarray(idx['support_labels'],float)
        it=pd.read_parquet(Path('D:/paiec_payload/data')/self.name/'items.parquet',columns=['item_id','content','item_features']).drop_duplicates('item_id')
        self.items={str(v.item_id):(v.content,v.item_features) for v in it.itertuples(index=False)}
        self.engine=engine;self.vectors={};self.contexts={}
    def vector(self,item):
        if item not in self.vectors:
            content,meta=self.items[item]
            self.vectors[item]=np.r_[self.engine.vector(content)/np.sqrt(self.ridge),meta_encode(meta)/4.]
        return self.vectors[item]
    def context(self,sid,seed,n=31):
        key=(sid,seed)
        if key not in self.contexts:
            v=self.by[key]
            self.contexts[key]={'x':np.stack([self.vector(i) for i in v['support_ids']]),'y':v['labels'],'keys':v['support_keys']}
            assert set(v['support_ids']).isdisjoint(v['query_ids'])
        c=self.contexts[key]
        return {'x':c['x'][:n],'y':c['y'][:n],'keys':c['keys'][:n]}
    def query(self,idx):
        return np.stack([self.vector(idx['query_ids'][i]) for i in idx['query_variant_item_indices']])

def outer_folders():
    return {p.name:p for p in sorted((ROOT/'R64/full/outer').glob('fold*/*'))}

def peak_memory_mb():
    import ctypes
    from ctypes import wintypes
    class Info(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),
          ('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
          ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
          ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    info=Info();info.cb=ctypes.sizeof(info)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(info),info.cb)
    return float(info.PeakWorkingSetSize/1024**2)
