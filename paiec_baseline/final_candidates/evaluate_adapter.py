"""Account for frozen adapter fallbacks on the same full fixed4 cohort inputs.

Only route between already saved R25/Partial8 probabilities. No new selection,
labels or parameters. Public predict always lacks the required trusted roster.
"""
import sys
from common import HERE,BASE,REPO,ART
sys.path.insert(0,str(REPO/'research_loop'))
from shared_slope_data import *
from mlp.model import _SUBJECT_FIELDS
from audit_peer_coverage import plain

def main():
    out=ART/'adapter_evaluation';out.mkdir(parents=True,exist_ok=True);rows=[];routes=[]
    for f in range(3):
        for folder in sorted((ART/f'evaluation/fold{f}').iterdir()):
            bm=folder.name+'/formatted_tables';records=read(ROOT/'R72/contexts'/folder.name/'index.json');by={(v['subject_id'],v['seed']):v for v in records}
            peers=read(ROOT/f'R76/full/outer/fold{f}'/folder.name/'cohorts.json')['main']
            table=pd.read_parquet(Path('D:/paiec_payload/data')/bm/'subjects.parquet',columns=['subject_id',*_SUBJECT_FIELDS]).drop_duplicates('subject_id').set_index('subject_id').to_dict('index')
            valid={s:bool(plain(v['normalized_name']) and plain(v['provider'])) for s,v in table.items()}
            for ip in sorted(folder.glob('index_*.json')):
                suffix=ip.stem.split('_')[-1];data=dict(np.load(folder/f'predictions_{suffix}.npz'))
                for idx in read(ip):
                    a=idx['start'];b=a+idx['size'];sid=idx['subject_id'];seed=idx['seed'];sids=[sid,*peers[sid]]
                    own=data['release_R25'][a:b];partial=data['release_Partial8'][a:b];routed=own.copy()
                    for j,n in enumerate(BUDGETS):
                        reasons=[]
                        if n==0:reasons.append('no_own_support')
                        if len(sids)==1:reasons.append('no_peer_support')
                        if not all(valid[s] for s in sids):reasons.append('incomplete_subject_configuration')
                        if any(len(set(by[(s,seed)]['support_keys'][:n]))!=n for s in sids):reasons.append('duplicate_or_conflicting_visible_evidence')
                        if not reasons:routed[:,j]=partial[:,j]
                        routes.append({'fold':f,'benchmark':bm,'subject_id':sid,'seed':seed,'budget':int(n),'sharing':not reasons,'reasons':reasons})
                    rows.append(score_episode({**idx,'counts':data['counts'][a:b],'successes':data['successes'][a:b]}, {'R25':own,'Partial8_operator':partial,'Partial8_trusted_roster_adapter':routed,'Partial8_public_interface':own}))
    summarize(rows,out,5242);write_json(out/'routes.json',routes)
    from collections import Counter
    summary={'finished':now(),'trajectories':len(rows),'budget_events':len(routes),'sharing_events':sum(r['sharing'] for r in routes),'fallback_reasons':dict(Counter(reason for r in routes for reason in r['reasons'])),'scope':'same frozen cohort; roster known only in conditional local simulation; public entry always R25'}
    write_json(out/'execution.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
