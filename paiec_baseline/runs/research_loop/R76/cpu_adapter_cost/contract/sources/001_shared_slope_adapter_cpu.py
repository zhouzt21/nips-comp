"""New-item capable CPU entrypoint, wrapping the frozen current-label adapter.

The earlier shared_slope_adapter.make_r25_encoder helper was cache-only. Use this
module's factory for new input. Feature/model state is permitted; labels are never
retained. Selection of at most three sorted visible peers is an engineering policy,
not the random offline cohort assignment used to measure R76.
"""
from shared_slope_adapter import predict_raw as _predict_raw,SUBJECT_FIELDS,ITEM_FIELDS

def predict_raw(input,labeled,*,encoder,visible_subjects=None,own_predictor=None,mode='pooled',max_peers=3):
    """Whitelist current public fields, including on incomplete-configuration fallback."""
    def subject(v):return {k:v[k] for k in SUBJECT_FIELDS if k in v}
    def item(v):return {k:v[k] for k in (*ITEM_FIELDS,'benchmark_id') if k in v}
    clean_input=[subject(input[0]),item(input[1])]
    clean_labels=[[[subject(pair[0]),item(pair[1])],label] for pair,label in (labeled or [])]
    roster=None if visible_subjects is None else [subject(v) for v in visible_subjects]
    return _predict_raw(clean_input,clean_labels,encoder=encoder,visible_subjects=roster,own_predictor=own_predictor,mode=mode,max_peers=max_peers)

def make_r25_encoder(semantic_ridge,model_path='D:/paiec_pretrained/all-MiniLM-L6-v2',threads=2):
    from dual_track_cpu_features import CPUItemFeatures
    engine=CPUItemFeatures(model_path=model_path,threads=threads)
    return engine.r25_encoder(semantic_ridge)
