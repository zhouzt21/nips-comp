"""Generate standalone R73 annotation-quality figures from saved artifacts."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
REPO=Path(__file__).resolve().parents[1];ROOT=REPO/'paiec_baseline/runs/research_loop/R73'
def main():
    summary=json.loads((ROOT/'quality_summary.json').read_text(encoding='utf-8'))
    repeat=json.loads((ROOT/'repeat_comparisons.json').read_text(encoding='utf-8'))
    dims=['AS','MCr','QLl','KNf'];colors=plt.cm.viridis(np.linspace(.15,.9,6))
    fig,axes=plt.subplots(1,3,figsize=(14.5,4.4),gridspec_kw={'width_ratios':[1.25,1,1]})
    bottom=np.zeros(4)
    for g in range(6):
        counts=np.array([summary['per_rubric'][d]['counts'].get(str(g),0) for d in dims])
        axes[0].bar(dims,counts,bottom=bottom,color=colors[g],label=str(g));bottom+=counts
    axes[0].bar(dims,[summary['per_rubric'][d]['null'] for d in dims],bottom=bottom,color='#c7c7c7',label='null')
    axes[0].set(title='Primary teacher grades (128 items)',ylabel='Number of item-rubric records',ylim=(0,137))
    axes[0].legend(title='Grade',ncol=4,loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=9)
    matrix=np.zeros((6,6),int)
    for r in repeat:
        if r['both_valid']:matrix[r['primary_grade'],r['repeat_grade']]+=1
    axes[1].imshow(matrix,cmap='Blues',vmin=0)
    for i in range(6):
        for j in range(6):
            if matrix[i,j]:axes[1].text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    axes[1].set(title='Independent repeat: 48 numeric pairs',xlabel='Repeat grade',ylabel='Primary grade',xticks=range(6),yticks=range(6))
    axes[1].text(.5,-.32,'Exact: 33/48; difference >1: 1/48\n16 repeated items; same model, no human truth',ha='center',transform=axes[1].transAxes,fontsize=9)
    groups=summary['length_groups'];names=['Within 126\ncontent tokens','Above 126\ncontent tokens']
    complete=[groups[k]['complete_four_grades'] for k in ['within_minilm_window','above_minilm_window']]
    incomplete=[groups[k]['items']-c for k,c in zip(['within_minilm_window','above_minilm_window'],complete)]
    axes[2].bar(names,complete,color='#3288bd',label='Four numeric grades')
    axes[2].bar(names,incomplete,bottom=complete,color='#c7c7c7',label='At least one null')
    for i,(a,b) in enumerate(zip(complete,incomplete)):axes[2].text(i,a+b+1,str(a+b),ha='center')
    axes[2].set(title='Teacher / student text coverage',ylabel='Number of items',ylim=(0,78))
    axes[2].legend(loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=9)
    for ax in [axes[0],axes[2]]:ax.spines[['top','right']].set_visible(False)
    fig.subplots_adjust(bottom=.25,wspace=.45,top=.88)
    out=REPO/'reports/dual_track_figures';out.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out/f'annotation_quality.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(str(out/'annotation_quality.png'))
if __name__=='__main__':main()
