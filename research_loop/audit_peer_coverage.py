"""R72 input-only audit: query outcomes are not read for coverage or cohort choice."""
import time,hashlib
from collections import Counter,defaultdict
import pandas as pd
from historical_positive_common import *
from mlp.model import _SUBJECT_FIELDS

def plain(v):
    return '' if v is None or (not isinstance(v,(str,list,dict)) and pd.isna(v)) else str(v)

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def main():
    root=ROOT/'R72';root.mkdir(exist_ok=True);start=time.perf_counter();coverage=[];bm_audits=[]
    freeze(root/'coverage_contract',[Path(__file__),REPO/'research_loop/LEXICAL_PEER_EVIDENCE_SPEC.md',REPO/'research_loop/PAIEC_RUNTIME_EVIDENCE_CONTRACT.json'],
        {'query_outcomes_used':False,'cohort':'hash of exact visible subject; groups of up to4; ambiguous subject keys own-only','scenarios':[0,1,3]})
    for f in range(3):
        for old in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
            name=old.name+'/formatted_tables';folder=root/'contexts'/old.name;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'complete.json').exists():coverage+=read(folder/'coverage.json');bm_audits.append(read(folder/'complete.json'));continue
            data=Path('D:/paiec_payload/data')/name
            it=pd.read_parquet(data/'items.parquet',columns=['item_id','content','item_features']).drop_duplicates('item_id')
            items={str(v.item_id):(plain(v.content),plain(v.item_features)) for v in it.itertuples(index=False)}
            # Read only input strings, never response outcomes, to normalize null interactors.
            inter=pd.read_parquet(data/'response.parquet',columns=['interactors']).interactors.drop_duplicates()
            intermap={hashlib.sha256(str(v).encode()).hexdigest():plain(v) for v in inter}
            sub=pd.read_parquet(data/'subjects.parquet',columns=['subject_id',*_SUBJECT_FIELDS]).drop_duplicates('subject_id').set_index('subject_id').to_dict('index')
            skeys={str(sid):digest({k:plain(v.get(k)) for k in _SUBJECT_FIELDS}) for sid,v in sub.items()}
            # Select only explicitly allowed fields from the frozen indexes; successes/support outcomes unused.
            source_meta=[]
            for p in sorted(old.glob('index_*.json')):
                for idx in read(p):
                    source_meta.append({k:idx[k] for k in ['fold','benchmark','subject_id','seed','support_ids','query_ids',
                        'query_variant_item_indices','query_variant_sha256','support_variant_sha256','counts']})
            sids=sorted({m['subject_id'] for m in source_meta});cnt=Counter(skeys[s] for s in sids)
            eligible=sorted([s for s in sids if cnt[skeys[s]]==1],key=lambda s:skeys[s]);peers={s:[] for s in sids};cohorts=[]
            for a in range(0,len(eligible),4):
                group=eligible[a:a+4];cohorts.append(group)
                for sid in group:peers[sid]=[s for s in group if s!=sid]
            records=[];keyids=defaultdict(set);text_variants=defaultdict(set)
            for m in source_meta:
                def keys(ids,variants):
                    out=[]
                    for iid,h in zip(ids,variants):
                        text,feat=items[iid];vv=intermap[h];key=digest([name,text,feat,vv]);out.append(key)
                        keyids[key].add(iid);text_variants[digest(text)].add(digest([feat,vv]))
                    return out
                qids=[m['query_ids'][i] for i in m['query_variant_item_indices']]
                sk=keys(m['support_ids'],m['support_variant_sha256']);qk=keys(qids,m['query_variant_sha256'])
                records.append({**m,'subject_key':skeys[m['subject_id']],'support_keys':sk,'query_keys':qk,
                    'peer_ids':peers[m['subject_id']],'ambiguous_subject':cnt[skeys[m['subject_id']]]>1})
            by={(m['subject_id'],m['seed']):m for m in records};local=[]
            for m in records:
                assert set(m['support_ids']).isdisjoint(m['query_ids'])
                c=np.array(m['counts'],float);qids=[m['query_ids'][i] for i in m['query_variant_item_indices']]
                for k in [0,1,3]:
                    chosen=[by[(s,m['seed'])] for s in m['peer_ids'][:k]]
                    assert all(v['subject_key']!=m['subject_key'] for v in chosen)
                    for j,n in enumerate(BUDGETS):
                        visible=set();raw=set();ambig=0
                        for peer in chosen:
                            counts=Counter(peer['support_keys'][:n]);ambig+=sum(v-1 for v in counts.values())
                            visible.update(key for key,v in counts.items() if v==1);raw.update(peer['support_ids'][:n])
                        hits=np.array([v in visible for v in m['query_keys']],float)
                        rawhits=np.array([v in raw for v in qids],float)
                        local.append({'fold':f,'benchmark':name,'subject_id':m['subject_id'],'seed':m['seed'],'scenario_peers':k,'budget':n,
                            'actual_peers':len(chosen),'peer_labels':n*len(chosen),'same_input_query_fraction':float(hits.mean()),
                            'same_input_response_fraction':float(c@hits/c.sum()),'same_internal_item_fraction':float(rawhits.mean()),
                            'missing_match_fraction':float(1-hits.mean()),'ambiguous_peer_prefix_entries':ambig,'own_query_overlap':0})
            done={'benchmark':name,'fold':f,'subjects':len(sids),'distinct_visible_subjects':len({skeys[s] for s in sids}),
                'ambiguous_subjects_excluded_from_sharing':sum(cnt[skeys[s]]>1 for s in sids),
                'same_visible_key_multiple_internal_items':sum(len(v)>1 for v in keyids.values()),
                'same_text_multiple_metadata_or_interactors':sum(len(v)>1 for v in text_variants.values()),
                'no_peer_subjects':sum(not peers[s] for s in sids),'cohorts':cohorts,'status':'conditional_simulation','finished':now()}
            write_json(folder/'index.json',records);write_json(folder/'coverage.json',local);write_json(folder/'complete.json',done)
            coverage+=local;bm_audits.append(done);print(json.dumps({k:v for k,v in done.items() if k!='cohorts'}),flush=True)
    df=pd.DataFrame(coverage);df.to_csv(root/'coverage.csv',index=False)
    agg=df.groupby(['scenario_peers','budget']).mean(numeric_only=True).reset_index();agg.to_csv(root/'coverage_summary.csv',index=False)
    write_json(root/'coverage_summary.json',{'benchmarks':bm_audits,'summary':agg.to_dict('records'),'seconds':time.perf_counter()-start,'finished':now(),'query_outcomes_used':False})
    check_freeze(root/'coverage_contract');print(agg[['scenario_peers','budget','actual_peers','same_input_response_fraction']].to_string(index=False),flush=True)

if __name__=='__main__':main()
