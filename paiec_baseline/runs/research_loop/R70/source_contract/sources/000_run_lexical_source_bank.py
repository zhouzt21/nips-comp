"""Untrained source bank for nested source calibration; no outer target outcomes."""
import time
from lexical_peer_common import *
from run_historical_template_pilot import LexicalOnly
from mlp.cohort_semantic import CachedSemantic
from mlp.context_episodes import episodes

def main():
    from threadpoolctl import threadpool_limits
    threadpool_limits(4);start=time.perf_counter();root=ROOT/'R70/source_bank';root.mkdir(parents=True,exist_ok=True)
    check_freeze(ROOT/'R62/isolation')
    freeze(ROOT/'R70/source_contract',[Path(__file__),REPO/'research_loop/lexical_peer_common.py',REPO/'research_loop/LEXICAL_PEER_EVIDENCE_SPEC.md'],{'ridges':RIDGES,'seed':73,'cap':25,'permutations':4})
    engine=LexicalOnly();semantic=CachedSemantic()
    names=sorted({b for f in range(3) for b in sum(source(f)['inner_halves'],[])})
    for name in names:
        folder=root/name.split('/')[0];folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists():continue
        eps=episodes(name,engine.tokenizer,semantic);bank={};index=[];offset=0
        for ep in eps:
            p=predict_bank(ep['tokens'],ep['old_semantic'],ep['metadata'],ep['labels'],engine,RIDGES,name+'|'+ep['subject_id'])
            q=len(ep['counts']);idx={'benchmark':name,'subject_id':ep['subject_id'],'seed':73,'start':offset,'size':q,
                'identity':__import__('hashlib').sha256((name+'|'+ep['subject_id']+'|73').encode()).hexdigest(),'labels':ep['labels'].tolist()}
            index.append(idx);offset+=q
            for k,v in p.items():bank.setdefault(k,[]).append(v)
            for k in ['counts','successes']:bank.setdefault(k,[]).append(ep[k])
        np.savez_compressed(folder/'predictions.npz',**{k:np.concatenate(v) for k,v in bank.items()})
        write_json(folder/'index.json',index);write_json(folder/'complete.json',{'pairs':len(index),'utc':now()})
        engine.lexical_cache.clear();print(json.dumps({'stage':'source_bank','benchmark':name,'pairs':len(index),'seconds':time.perf_counter()-start}),flush=True)
    write_json(root/'execution.json',{'seconds':time.perf_counter()-start,'finished':now()})
    check_freeze(ROOT/'R70/source_contract');check_freeze(ROOT/'R62/isolation')

if __name__=='__main__':main()
