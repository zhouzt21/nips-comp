"""Keep legal predictor evidence, extra-label diagnostics and source pilots separate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('runs/research_loop');r46=json.loads((root/'R46/summary.json').read_text());r48=json.loads((root/'R48/summary.json').read_text());r49=json.loads((root/'R49/summary.json').read_text())
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(12,3.8),layout='constrained');x=np.arange(6);labels=['0','1','3','7','15','31']
    base=np.array(r46['pair_equal_brier']['smoothed_mean_2'])
    for key,name,color in [('R25','Frozen R25','#345995'),('average','Equal semantic/lexical (R46)','#d17a22')]:
        ax[0].plot(x,np.array(r46['pair_equal_brier'][key])-base,'o-',label=name,color=color)
    ax[0].axhline(0,color='.6',lw=.7);ax[0].set(title='A. Budget-compliant predictors\n5,242 pairs; 64 query items',ylabel='Brier minus smoothed mean 2',xlabel='Allowed labels',xticks=x,xticklabels=labels);ax[0].legend(fontsize=7,frameon=False)
    full=r48['dominant_full_query']['full_brier'];reference=np.array(full['R25'])
    for key,name,color,style in [('crossfit','Opposite-panel calibration','#d17a22','o--'),('oracle_max4','Same-panel oracle','#7a5ba7','s:')]:
        ax[1].plot(x,np.array(full[key])-reference,style,label=name,color=color)
    ax[1].axhline(0,color='.6',lw=.7);ax[1].set(title='B. DIAGNOSTIC: extra labels\n4,278 pairs; one benchmark; full pool',ylabel='Brier minus R25',xlabel='Original support labels (extra labels excluded)',xticks=x,xticklabels=labels);ax[1].legend(fontsize=7,frameon=False)
    delta=[f['pair_equal_alc']['split_calibrated']-f['pair_equal_alc']['R25'] for f in r49['folds']]
    ax[2].bar(range(3),delta,color='#bc4b51',width=.55);ax[2].axhline(0,color='.6',lw=.7)
    ax[2].set(title='C. R49 source-only pilot\nNo outer evaluation: admission failed',ylabel='ALC minus full-support R25',xlabel='Source pool for outer fold',xticks=range(3),xticklabels=['0','1','2'],ylim=(0,max(delta)*1.3))
    for i,v in enumerate(delta):ax[2].text(i,v+max(delta)*.035,f'+{v:.6f}',ha='center',fontsize=8)
    folder=Path('../reports/figures');folder.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(folder/f'v3_oracle_transfer.{ext}',dpi=220)
    plt.close(fig)
    (folder/'v3_oracle_transfer_caption.md').write_text('All panels use actual completed experiments on reused public benchmarks. A: paired three-seed, response-weighted, pair-equal scores; R46 loses primary ALC to R25. B: R48 uses extra labels from the opposite evaluation half (or the scored half for the oracle); neither curve is budget-compliant, eligible for promotion, or a new benchmark generalization claim. C: R49 uses only allowed support labels but its plotted scores are SOURCE pilots, not held-out estimates. It fails all-three-pool admission. Panels have different evaluation scopes and are not one method leaderboard.\n',encoding='utf-8')
    print('Saved reports/figures/v3_oracle_transfer.png and .pdf')


if __name__=='__main__':main()
