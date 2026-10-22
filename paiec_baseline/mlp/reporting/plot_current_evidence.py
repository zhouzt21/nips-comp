"""Measured R13/R19/R20 on the same cohort4, three seeds; no simulated points."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('runs/research_loop');read=lambda p:json.loads((root/p).read_text())
    sets=[('Own item slope','R13/cohort4/summary.json','combined'),
          ('Exact peer-item residual','R19_verify/summary.json','matched'),
          ('Shared item slope','R20_verify/summary.json','pooled')]
    fig,axes=plt.subplots(1,2,figsize=(10.6,3.8),layout='constrained');x=np.arange(6)
    colors=['#4771a5','#a76b25','#27785d']
    for i,(label,path,key) in enumerate(sets):
        d=read(path);delta=np.array(d['pair_equal_brier'][key])-d['pair_equal_brier']['peer']
        axes[0].plot(x,delta,marker='o',label=label,color=colors[i])
        c=d['comparisons']['peer'];value=c['pair_equal_delta'];lo,hi=c['cluster_bootstrap_ci95']
        axes[1].errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',capsize=4,color=colors[i])
    axes[0].axhline(0,color='black',lw=.8);axes[0].set_xticks(x,[0,1,3,7,15,31])
    axes[0].set_xlabel('Labels per subject-benchmark pair');axes[0].set_ylabel('Brier difference vs peer mean')
    axes[0].set_title('Where current item evidence helps');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].axvline(0,color='black',lw=.8);axes[1].axvline(-.005,color='#a43d3d',ls='--',lw=1,label='Frozen effect threshold')
    axes[1].set_yticks(range(3),[r[0] for r in sets]);axes[1].invert_yaxis()
    axes[1].set_xlabel('Pair-equal ALC difference vs peer mean')
    axes[1].set_title('Benchmark-cluster 95% intervals');axes[1].grid(axis='x',alpha=.2);axes[1].legend(fontsize=8,loc='upper left')
    fig.suptitle('Measured public CV: cohort4 / assignment193 / seeds0,1,2\n737 pairs, 36 reused benchmarks; conditional evidence scope',fontsize=10)
    fig.savefig(root/'v2_current_evidence.png',dpi=180);fig.savefig(root/'v2_current_evidence.pdf');plt.close(fig)
    print(root/'v2_current_evidence.png')


if __name__=='__main__':main()
