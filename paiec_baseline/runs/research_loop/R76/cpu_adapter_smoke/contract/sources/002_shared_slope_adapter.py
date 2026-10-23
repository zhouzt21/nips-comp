"""Conservative raw current-input adapter. No cross-call label state.

Flat labeled entries follow the public shape: [ [subject, item], binary_label ].
visible_subjects, when supplied, must be a roster actually visible in this request,
not reconstructed from hidden IDs. Without that uniqueness evidence sharing is off.
"""
import json
import numpy as np
from shared_slope_model import predict

SUBJECT_FIELDS=('normalized_name','provider','release_date','access_date','harness','harness_version','reasoning_effort','subject_features_extra')
ITEM_FIELDS=('item_content','item_features','interactors')

def canonical(v):
    return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),default=str)

def subject_key(subject):
    if not isinstance(subject,dict) or not all(k in subject for k in SUBJECT_FIELDS):return None
    if not subject.get('normalized_name') or not subject.get('provider'):return None
    return canonical({k:subject[k] for k in SUBJECT_FIELDS})

def item_key(item):
    return canonical({k:item.get(k) for k in ITEM_FIELDS})

def predict_raw(input,labeled,*,encoder,visible_subjects=None,own_predictor=None,mode='pooled',max_peers=3):
    """Return prediction plus visible-evidence accounting; a single call owns all labels.

    An anonymous benchmark_id is used only for same-request group equality. It is
    never encoded or mapped to a known Benchmark. Hidden item_id/subject_id ignored.
    """
    if mode not in ('pooled','partial'):raise ValueError('mode must be pooled or partial')
    if not 0<=max_peers<=3:raise ValueError('max_peers must be in [0,3]')
    if len(input)!=2 or not all(isinstance(v,dict) for v in input):raise ValueError('input must be [subject,item]')
    subject,item=input;key=subject_key(subject);group=item.get('benchmark_id');by={};own_entries=[];ambiguous=False
    for entry in labeled or []:
        if len(entry)!=2 or len(entry[0])!=2:raise ValueError('invalid labeled entry')
        (s,it),y=entry
        if y not in (0,1):raise ValueError('labels must be binary')
        if not group or it.get('benchmark_id')!=group:continue
        skey=subject_key(s)
        # Exact visible dictionary equality is sufficient for a conservative own fallback,
        # but never resolves indistinguishable hidden configurations for peer sharing.
        own=(s==subject) if key is None else (skey==key)
        if own:own_entries.append([[s,it],int(y)])
        if skey is None:ambiguous=True;continue
        by.setdefault(skey,[]).append((it,int(y)))
    query=np.asarray(encoder({k:item.get(k) for k in ITEM_FIELDS}),float)[None,:]
    def context(entries):
        return {'x':np.stack([np.asarray(encoder({k:it.get(k) for k in ITEM_FIELDS}),float) for it,_ in entries]) if entries else np.empty((0,query.shape[1])),
                'y':np.asarray([y for _,y in entries],float),'keys':[item_key(it) for it,_ in entries]}
    own=context([(entry[0][1],entry[1]) for entry in own_entries])
    fallback=float(own_predictor(input,own_entries)) if own_predictor else float(predict([own],query)[0])
    reasons=[]
    if key is None:reasons.append('incomplete_subject_configuration')
    if not group:reasons.append('missing_current_group_token')
    if not len(own['y']):reasons.append('no_own_support')
    if visible_subjects is None:reasons.append('visible_subject_roster_unavailable')
    else:
        roster=[subject_key(s) for s in visible_subjects]
        if None in roster or len(set(roster))!=len(roster) or key not in roster:reasons.append('ambiguous_visible_subject_roster')
        if any(k not in roster for k in by):reasons.append('labeled_subject_absent_from_roster')
    if ambiguous:reasons.append('incomplete_labeled_subject_configuration')
    # Repeated identical modeled input within a subject can hide merged subjects or
    # repeated responses. This conservative adapter disables sharing instead of guessing.
    if any(len({item_key(it) for it,_ in entries})!=len(entries) for entries in by.values()):reasons.append('duplicate_or_conflicting_visible_evidence')
    peers=[k for k in sorted(by) if k!=key][:max_peers]
    if not peers:reasons.append('no_peer_support')
    if reasons:return {'prediction':fallback,'mode':'own_only','reasons':reasons,'own_labels':len(own['y']),'peer_labels':0,'peers':0}
    ctx=[own,*[context(by[k]) for k in peers]]
    p=float(predict(ctx,query,partial=(mode=='partial'))[0])
    return {'prediction':p,'mode':mode,'reasons':[],'own_labels':len(own['y']),'peer_labels':sum(len(c['y']) for c in ctx[1:]),'peers':len(peers)}

def make_r25_encoder(semantic_ridge):
    """Use the existing CPU MiniLM cache and unchanged R25 metadata encoder."""
    from mlp.cohort_semantic import CachedSemantic
    from mlp.metadata_features import encode as meta_encode
    engine=CachedSemantic()
    return lambda item:np.r_[engine.vector(item.get('item_content',''))/np.sqrt(semantic_ridge),meta_encode(item.get('item_features',''))/4.]
