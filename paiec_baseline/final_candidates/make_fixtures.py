"""Raw public-input fixtures and independent frozen-cache references, no query outcomes."""
import sys
from common import HERE,BASE,REPO,ART,write,sha
sys.path.insert(0,str(REPO/'research_loop'))
from shared_slope_data import *
from shared_slope_model import predict
from shared_slope_adapter_cpu import predict_raw
from shared_slope_adapter import canonical,ITEM_FIELDS
from mlp.model import _SUBJECT_FIELDS
from audit_peer_coverage import plain,digest

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(2);engine=CachedSemantic();out=HERE/'fixtures';out.mkdir(exist_ok=True);manifest=[]
    for f in range(3):
        for folder in sorted((ROOT/f'R64/full/outer/fold{f}').iterdir()):
            bm=Benchmark(folder,4.,engine);peermap=read(ROOT/f'R76/full/outer/fold{f}'/folder.name/'cohorts.json')['main']
            eligible=[i for i in bm.indices if i['seed']==0 and peermap[i['subject_id']]]
            idx=(eligible or [i for i in bm.indices if i['seed']==0])[0];sid=idx['subject_id'];seed=0;sids=[sid,*peermap[sid]]
            data=Path('D:/paiec_payload/data')/bm.name
            subjects_table=pd.read_parquet(data/'subjects.parquet',columns=['subject_id',*_SUBJECT_FIELDS]).drop_duplicates('subject_id').set_index('subject_id').to_dict('index')
            inter=pd.read_parquet(data/'response.parquet',columns=['interactors']).interactors.drop_duplicates()
            intermap={hashlib.sha256(str(v).encode()).hexdigest():plain(v) for v in inter}
            def item(i,h):
                text,meta=bm.items[i]
                return {'item_content':plain(text),'item_features':plain(meta),'interactors':intermap[h],'benchmark_id':'anonymous-current-group'}
            subjects=[{k:plain(subjects_table[s].get(k)) for k in _SUBJECT_FIELDS} for s in sids]
            assert all(digest(sub)==bm.by[(s,seed)]['subject_key'] for s,sub in zip(sids,subjects))
            contexts=[];memo={}
            for s,sub in zip(sids,subjects):
                rec=bm.by[(s,seed)];labs=[]
                for i,y,h in zip(rec['support_ids'],rec['labels'],rec['support_variant_sha256']):
                    it=item(i,h);memo[canonical({k:it.get(k) for k in ITEM_FIELDS})]=bm.vector(i)
                    labs.append([[sub,it],int(y)])
                contexts.append(labs)
            variants=list(dict.fromkeys(zip([idx['query_ids'][i] for i in idx['query_variant_item_indices']],idx['query_variant_sha256'])))[:4]
            query_ids=[i for i,h in variants];queries=[item(i,h) for i,h in variants]
            for i,it in zip(query_ids,queries):memo[canonical({k:it.get(k) for k in ITEM_FIELDS})]=bm.vector(i)
            def encoder(it):return memo[canonical({k:it.get(k) for k in ITEM_FIELDS})]
            expected=[];fallbacks=[];direct=[]
            q=np.stack([bm.vector(i) for i in query_ids])
            for n in BUDGETS:
                labels=[v for c in contexts for v in c[:n]];ctx=[bm.context(s,0,int(n)) for s in sids]
                ref_own=predict(ctx[:1],q);ref_partial=predict(ctx,q,partial=True) if n>=3 and len(ctx)>1 else ref_own
                direct.append({'R25':ref_own.tolist(),'Partial8':ref_partial.tolist()})
                row=[]
                for j,it in enumerate(queries):
                    r=predict_raw([subjects[0],it],labels,encoder=encoder,visible_subjects=subjects,mode='partial')
                    row.append({'r25':float(ref_own[j]),'partial_adapter':r['prediction'],'partial_operator':float(ref_partial[j]),'mode':r['mode'],'reasons':r['reasons']})
                    if r['mode']=='own_only' and n>=3 and len(ctx)>1:fallbacks.append({'budget':int(n),'query':j,'reasons':r['reasons']})
                    if r['mode']=='partial':assert abs(r['prediction']-ref_partial[j])<1e-10
                expected.append(row)
            payload={'fold':f,'benchmark':bm.name,'identity':idx['identity'],'seed':0,'budgets':list(map(int,BUDGETS)),'subjects':subjects,'contexts':contexts,'queries':queries,'expected':expected,'operator_predictions':direct}
            path=out/(folder.name+'.json');write(path,payload)
            np.savez_compressed(out/(folder.name+'_features.npz'),keys=np.array(list(memo)),vectors=np.stack(list(memo.values())))
            manifest.append({'benchmark':bm.name,'fold':f,'identity':idx['identity'],'path':str(path),'sha256':sha(path),'queries':len(queries),'peers':len(contexts)-1,'fallbacks':fallbacks,'original_index':str(folder),'source_context_sha256':sha(ROOT/'R72/contexts'/folder.name/'index.json')})
    write(HERE/'fixture_manifest.json',{'selection':'first eligible seed0 pair per Benchmark, else first seed0; first four distinct query items; six budgets','fixtures':manifest,'query_outcomes_included':False})
    print(json.dumps({'benchmarks':len(manifest),'queries':sum(v['queries'] for v in manifest),'adapter_fallbacks':sum(len(v['fallbacks']) for v in manifest)}),flush=True)
if __name__=='__main__':main()
