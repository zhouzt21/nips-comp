"""Measured R33-R35 diagnostics; oracle capacity explicitly separated from usable estimates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    r33=json.loads(Path('runs/research_loop/R33/summary.json').read_text())
    r34=json.loads(Path('runs/research_loop/R34/summary.json').read_text())
    r35=json.loads(Path('runs/research_loop/R35/summary.json').read_text())
    x=np.arange(6);labels=['0','1','3','7','15','31']
    base=np.asarray(r34['pair_equal_brier']['smoothed_mean_2'])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    axes[0].axhline(0,color='.5',linewidth=.8)
    for name,label,color in [('R25','Frozen local predictor','#345995'),('adaptive_scale','Support-only scale (R34)','#d17a22')]:
        axes[0].plot(x,np.asarray(r34['pair_equal_brier'][name])-base,marker='o',label=label,color=color)
    axes[0].plot(x,np.asarray(r33['pair_equal_brier']['oracle_max4'])-base,'--s',color='#7a5ba7',label='Oracle c<=4 (evaluation labels)')
    axes[0].set(xticks=x,xticklabels=labels,xlabel='Revealed-label budget',ylabel='Brier minus smoothed mean 2',title='Potential versus realized correction')
    axes[0].legend(frameon=False,fontsize=8,loc='lower left')
    budgets=[3,7,15,31];values=[r35['outer_by_budget'][str(n)]['rank_correlation'] for n in budgets]
    axes[1].bar(range(4),values,color='#345995',width=.6)
    axes[1].axhline(0,color='.5',linewidth=.8)
    axes[1].set(xticks=range(4),xticklabels=budgets,xlabel='Revealed-label budget',ylabel='Rank correlation',ylim=(-.10,.57),title='Support scale vs oracle scale')
    for i,v in enumerate(values):axes[1].text(i,v+(.018 if v>=0 else -.04),f'{v:.3f}',ha='center',fontsize=9)
    fig.suptitle('Reused public benchmarks: 5,242 pairs, three item-split seeds',fontsize=12)
    folder=Path('../reports/figures');folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/f'v3_scale_identifiability.{ext}',dpi=220)
    plt.close(fig)
    (folder/'v3_scale_identifiability_caption.md').write_text(
        'R33–R35, actual public-data diagnostics. All pair scores retain response weights; each pair uses the historical64queryitems. '
        'R32 separately audits the cap. Dashed oracle uses evaluation labels and is not a deployable candidate. '
        'Correlation pools pair/seed rows descriptively and is not a confidence interval or independent-benchmark generalization claim. '
        'R34 has only0.00004856 additional ALC gain overR25; its benchmark-cluster interval crosses zero.\n',encoding='utf-8')
    print('Saved reports/figures/v3_scale_identifiability.png and .pdf')


if __name__=='__main__':main()
