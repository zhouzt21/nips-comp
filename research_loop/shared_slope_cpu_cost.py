"""Synthetic per-request CPU cost at full visible label budget, no cached item lookup."""
import time,ctypes
from ctypes import wintypes
from historical_positive_common import *
from shared_slope_adapter_cpu import predict_raw
from shared_slope_adapter import SUBJECT_FIELDS
from dual_track_cpu_features import CPUItemFeatures

def peak_mb():
    class Info(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
           ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),
           ('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);api=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Info),wintypes.DWORD];api.GetProcessMemoryInfo.restype=wintypes.BOOL
    info=Info();info.cb=ctypes.sizeof(info)
    assert api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(info),info.cb)
    value=info.PeakWorkingSetSize/1024**2;assert value>0;return value

def main():
    engine=CPUItemFeatures(threads=2);encoder=engine.r25_encoder(4.)
    subjects=[{k:('cost-subject-'+str(i) if k=='normalized_name' else 'synthetic' if k=='provider' else '') for k in SUBJECT_FIELDS} for i in range(4)]
    def item(i):return {'item_content':f'R76 synthetic CPU cost fresh item {i}: Evaluate whether a given short numeric inequality follows from the provided assumptions.',
        'item_features':'{}','interactors':'','benchmark_id':'anonymous-cost-group'}
    labels=[[[subjects[j],item(100*j+i)],int((i+j)%3!=0)] for j in range(4) for i in range(31)]
    records=[]
    for peers in [0,3]:
        t=time.perf_counter();v=predict_raw([subjects[0],item(999)],labels[:31*(peers+1)],encoder=encoder,visible_subjects=subjects[:peers+1],mode='partial')
        records.append({'peers':peers,'own_labels':31,'total_visible_labels':31*(peers+1),'seconds':time.perf_counter()-t,'prediction':v['prediction'],'mode':v['mode']})
    root=ROOT/'R76/cpu_adapter_cost';root.mkdir(exist_ok=True)
    write_json(root/'results.json',{'records':records,'cold_model_load_seconds':engine.cold_load_seconds,'lifetime_peak_working_set_mb':peak_mb(),
         'threads':2,'repeats_per_case':1,'limitations':'Synthetic short texts; one timing per case is not a stable latency benchmark; no Hosted guarantee', 'finished':now()})
    freeze(root/'contract',[Path(__file__),REPO/'research_loop/shared_slope_adapter_cpu.py',REPO/'research_loop/shared_slope_adapter.py',REPO/'research_loop/dual_track_cpu_features.py'],{'synthetic':True,'CPU_threads':2})
    print(json.dumps(read(root/'results.json')))

if __name__=='__main__':main()
