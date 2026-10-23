"""Scoped CPU deployment feature replay for plan_8, no model fitting."""
from pathlib import Path
import sys,json,time,hashlib,os
import numpy as np
from dual_track_cpu_features import CPUItemFeatures,BASE
from mlp.cohort_semantic import CachedSemantic
from mlp.metadata_features import encode as meta_encode
REPO=BASE.parent;OUT=BASE/'runs/research_loop/dual_track_v8'
def main():
    begin=time.perf_counter();OUT.mkdir(exist_ok=True);os.chdir(BASE)
    packet=json.loads((BASE/'runs/research_loop/R73/annotation_only/shard0_batch0.json').read_text(encoding='utf-8'))
    items=[{'item_content':r['text'],'item_features':'{}','interactors':''} for r in packet[:4]]
    engine=CPUItemFeatures(threads=2);t=time.perf_counter();encoded=engine.encode_items(items);forward=time.perf_counter()-t
    cache=CachedSemantic();expected=np.stack([np.r_[cache.vector(it['item_content']),meta_encode(it['item_features'])] for it in items])
    error=float(abs(expected-encoded).max());assert error<1e-6,error
    new={'item_content':'Plan8 synthetic unseen-input probe: identify the green triangle among two blue circles.','item_features':'{}','interactors':''}
    t=time.perf_counter();v=engine.raw(new);novel_seconds=time.perf_counter()-t
    assert v.shape==(640,) and np.isfinite(v).all()
    scaled=engine.r25_encoder(4.)(new);assert np.max(abs(scaled-np.r_[v[:384]/2.,v[384:]/4.]))<1e-7
    repeated=engine.raw(new);repeat_error=float(abs(v-repeated).max());assert repeat_error<1e-7
    audit={'status':'passed','reference_items':4,'historical_semantic_metadata_max_abs_error':error,'new_input_finite_640d':True,'repeat_error':repeat_error,'scale_error_max':float(abs(scaled-np.r_[v[:384]/2.,v[384:]/4.]).max()),'cold_model_load_seconds':engine.cold_load_seconds,'first_batch4_seconds':forward,'new_item_forward_seconds':novel_seconds,'total_seconds':time.perf_counter()-begin,'cpu_threads':2,'model_revision':engine.model_revision,'model_local_only':True,'dataset_lookup_for_new_text':False,'historical_cache_open_mode':'ro; reference comparison only, production encoder has no database','mutable_label_state':False,'feature_source_sha256':hashlib.sha256((REPO/'research_loop/dual_track_cpu_features.py').read_bytes()).hexdigest(),'validation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'benchmark_scope':'small synthetic deployment smoke; not hosted or representative latency benchmark','python':sys.version}
    (OUT/'cpu_feature_validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))
if __name__=='__main__':main()
