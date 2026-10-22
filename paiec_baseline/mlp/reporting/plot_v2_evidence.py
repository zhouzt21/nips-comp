"""Plot measured v2 results; no simulation or missing-value imputation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('runs/research_loop')
    read=lambda p:json.loads((root/p).read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    budgets=[0,1,3,7,15,31];x=np.arange(6)
    for mode,label in [('semantic','Semantic'),('text_hash','Matched lexical')]:
        d=read(f'R10/{mode}/summary.json')['pair_equal_brier']
        item=np.array(d['mean_only'])-d['actual'];mean=np.array(d['baseline'])-d['mean_only']
        axes[0].plot(x,item,marker='o',label=label+' item gain')
        axes[0].plot(x,mean,linestyle='--',label=label+' mean gain')
    axes[0].set_title('R10: measured error components');axes[0].set_ylabel('Brier reduction (positive is better)')
    d=read('R14/summary.json')['pair_equal_brier']
    for key,label in [('local','Local only'),('peer','Peer only'),('combined','Combined')]:
        axes[1].plot(x,np.array(d[key])-d['base'],marker='o',label=label)
    axes[1].set_title('R14: conditional cohort4, 9 draws');axes[1].set_ylabel('Brier difference vs base (lower is better)')
    for key in ['local','peer','base']:
        values=[]
        for seed in [193,271,811]:
            a=read(f'R14/cohort_seed{seed}/summary.json')['pair_equal_alc'];values.append(a['combined']-a[key])
        axes[2].plot([0,1,2],values,marker='o',label='Combined minus '+key)
    axes[2].set_xticks([0,1,2],[193,271,811]);axes[2].set_xlabel('Cohort assignment seed')
    axes[2].set_title('R14: peer composition sensitivity');axes[2].set_ylabel('ALC difference (lower is better)')
    for ax in axes:
        ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in axes[:2]:ax.set_xticks(x,budgets);ax.set_xlabel('Label budget')
    fig.savefig(root/'v2_mechanism_evidence.png',dpi=180);fig.savefig(root/'v2_mechanism_evidence.pdf');plt.close(fig)
    print(root/'v2_mechanism_evidence.png')


if __name__=='__main__':main()
