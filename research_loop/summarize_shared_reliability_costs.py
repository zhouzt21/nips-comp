"""Combine actual R77 scenario accounting, preserving reused R76 main cohort."""
from shared_reliability_common import *
def main():
    frames=[]
    fields=['benchmark','subject_id','seed','budget','own_labels','peer_labels','active_peers','visible_total']
    for p in sorted((ROOT/'R77/full/outer').glob('fold*/*/label_costs.csv')):
        frames.append(pd.read_csv(p,usecols=['scenario',*fields]))
    for p in sorted((ROOT/'R77/alternative/outer').glob('fold*/*/label_costs.csv')):
        frames.append(pd.read_csv(p,usecols=['scenario',*fields]))
    for p in sorted((ROOT/'R76/full/outer').glob('fold*/*/label_costs.csv')):
        d=pd.read_csv(p,usecols=['method',*fields]);d=d[d.method=='R76_peer3'].drop(columns='method');d['scenario']='7501';frames.append(d)
    df=pd.concat(frames,ignore_index=True);df['scenario']=df.scenario.astype(str)
    assert not df.duplicated(['scenario','benchmark','subject_id','seed','budget']).any()
    numerical=['own_labels','peer_labels','active_peers','visible_total']
    df.groupby(['scenario','budget'])[numerical].mean().to_csv(ROOT/'R77/label_cost_summary.csv')
    df.groupby(['scenario','budget'])[numerical].max().to_csv(ROOT/'R77/label_cost_max.csv')
    df.groupby(['benchmark','scenario','budget'])[numerical].mean().to_csv(ROOT/'R77/label_cost_by_benchmark.csv')
    write_json(ROOT/'R77/label_cost_validation.json',{'requests':len(df),'scenarios':df.scenario.nunique(),'trajectories_per_scenario':15726,
        'budgets':6,'maximum_visible_total':int(df.visible_total.max()),'maximum_peer_labels':int(df.peer_labels.max()),'finished':now(),
        'reused_main_costs':'R76/full/outer/*/*/label_costs.csv method R76_peer3','partial_same_information_as_pool':True})
    print('Label accounting scenarios:',df.scenario.nunique(),'requests:',len(df))
if __name__=='__main__':main()
