"""Measured R25/R31 curves: sampling sensitivity, not independent tests."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('runs/research_loop')
    capped=json.loads((root/'R25_verify/summary.json').read_text());full=json.loads((root/'R31/summary.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),gridspec_kw={'width_ratios':[1,1,.8]})
    for ax,data,key,title in [(axes[0],capped,'separate','Original proxy: 737 pairs'),(axes[1],full,'R25','Expanded coverage: 5,242 pairs')]:
        for name,label,color in [('smoothed_mean_2','Smoothed mean (2)','#c16b30'),('semantic','Semantic local ridge','#768699'),(key,'Semantic + metadata','#176b87')]:
            ax.plot(range(6),data['pair_equal_brier'][name],marker='o',ms=4,lw=1.8,color=color,label=f"{label}; ALC {data['pair_equal_alc'][name]:.4f}")
        ax.set_xticks(range(6),['0','1','3','7','15','31']);ax.set_xlabel('Revealed labels');ax.set_ylabel('Pair-equal Brier (lower is better)');ax.set_title(title,fontweight='bold');ax.set_ylim(.15,.26);ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper right')
    shares=np.array([100/737,4278/5242]);axes[2].bar(['Original','Expanded'],shares,color='#176b87',label='alignment_faking')
    axes[2].bar(['Original','Expanded'],1-shares,bottom=shares,color='#d8e1e6',label='Other 35 benchmarks')
    for i,v in enumerate(shares):axes[2].text(i,v/2,f'{v:.1%}',ha='center',va='center',color='white',fontweight='bold')
    axes[2].set_ylim(0,1);axes[2].set_ylabel('Share of evaluated pairs');axes[2].set_title('Benchmark composition',fontweight='bold');axes[2].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.14))
    fig.suptitle('Frozen predictors; three seeds; same reused public benchmarks',fontsize=13,y=1.01)
    fig.text(.01,-.04,'Observed experiments, not simulation. Both retain the 64-evaluation-item cap; expanded pairs do not constitute a new blind test.',fontsize=9)
    fig.tight_layout();fig.savefig(root/'v2_coverage_evidence.png',bbox_inches='tight');fig.savefig(root/'v2_coverage_evidence.pdf',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
