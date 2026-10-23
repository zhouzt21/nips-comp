"""Real new-text CPU encoding smoke; no measurements database or label cache."""
import time
from historical_positive_common import *
from shared_slope_adapter_cpu import predict_raw
from shared_slope_adapter import SUBJECT_FIELDS
from dual_track_cpu_features import CPUItemFeatures

def main():
    start=time.perf_counter();engine=CPUItemFeatures(threads=2);encoder=engine.r25_encoder(4.)
    def subject(i):return {k:('synthetic-'+str(i) if k=='normalized_name' else 'synthetic-provider' if k=='provider' else '') for k in SUBJECT_FIELDS}
    a,b=subject(0),subject(1)
    def item(text):return {'item_content':text,'item_features':'{}','interactors':'','benchmark_id':'anonymous-current-group'}
    query=[a,item('R76 CPU unseen-item smoke: A violet archivist stores thirteen wooden tokens. How many remain after giving away five?')]
    labeled=[[[a,item('R76 fresh support A: Compare the integers 12 and 19.')],1],[[a,item('R76 fresh support B: State the wavelength of a hypothetical unmapped planet signal.')],0],
             [[b,item('R76 fresh peer support A: Add four pears and seven pears.')],1],[[b,item('R76 fresh peer support B: Explain an intentionally underspecified quantum wiring diagram.')],0]]
    vector=encoder(query[1]);assert vector.shape==(640,) and np.isfinite(vector).all()
    t=time.perf_counter();pooled=predict_raw(query,labeled,encoder=encoder,visible_subjects=[a,b]);seconds=time.perf_counter()-t
    fallback=predict_raw(query,labeled,encoder=encoder);own=predict_raw(query,labeled[:2],encoder=encoder,visible_subjects=[a])
    assert pooled['mode']=='pooled' and pooled['peer_labels']==2 and pooled['own_labels']==2
    assert fallback['mode']=='own_only' and abs(fallback['prediction']-own['prediction'])<1e-12
    import copy
    incomplete_query=copy.deepcopy(query);incomplete_query[0].pop('provider');incomplete_query[0]['subject_id']='hidden-query'
    incomplete_labels=[copy.deepcopy(entry) for entry in labeled[:2]]
    for j,entry in enumerate(incomplete_labels):entry[0][0].pop('provider');entry[0][0]['subject_id']='different-hidden-'+str(j)
    # Cheap deterministic features isolate the whitelist behavior from the model.
    cheap=lambda it:np.array([1.,len(it.get('item_content',''))/100.])
    incomplete=predict_raw(incomplete_query,incomplete_labels,encoder=cheap,visible_subjects=[incomplete_query[0]])
    assert incomplete['mode']=='own_only' and incomplete['own_labels']==2
    for entry in incomplete_labels:entry[0][0]['subject_id']='changed-again'
    assert predict_raw(incomplete_query,incomplete_labels,encoder=cheap,visible_subjects=[incomplete_query[0]])==incomplete
    path=ROOT/'R76/cpu_adapter_smoke';path.mkdir(parents=True,exist_ok=True)
    write_json(path/'results.json',{'status':'passed_new_text_synthetic_only','feature_dim':640,'model_revision':engine.model_revision,
             'cold_load_seconds':engine.cold_load_seconds,'pooled_request_seconds':seconds,'total_seconds':time.perf_counter()-start,
             'pooled':pooled,'fallback':fallback,'own':own,'incomplete_configuration_hidden_ids_ignored':incomplete,'finished':now(),
             'scope':'New synthetic text encoded locally; not Hosted validation or an end-to-end R76 cohort score'})
    freeze(path/'contract',[Path(__file__),REPO/'research_loop/shared_slope_adapter_cpu.py',REPO/'research_loop/shared_slope_adapter.py',
                           REPO/'research_loop/shared_slope_model.py',REPO/'research_loop/dual_track_cpu_features.py'],{'new_item_encoding':True,'CPU_threads':2})
    print(json.dumps({'status':'passed_new_text_synthetic_only','load_seconds':engine.cold_load_seconds,'pooled_request_seconds':seconds,'total_seconds':time.perf_counter()-start}))

if __name__=='__main__':main()
