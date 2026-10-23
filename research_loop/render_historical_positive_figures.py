"""Standalone reproducible scientific figures from measured R62/R63(/R64) summaries."""
from historical_positive_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    r62=read(ROOT/'R62/summary.json');r63=read(ROOT/'R63/summary.json')
    entries=[('R54 segment',r62,'R54_segment'),('R56 segment',r62,'R56_segment'),
        ('R61 B1 gate',r62,'R61_support1_gate'),('R43 query',r63,'R43_query_best20'),
        ('R55 resampled query',r63,'R55_resampled_query'),('R55 resampled episode',r63,'R55_resampled_episode')]
    if (ROOT/'R64/full/summary.json').exists():
        entries.append(('R64 template average',read(ROOT/'R64/full/summary.json'),'template_average'))
    fig,axes=plt.subplots(2,1,figsize=(11,9),layout='constrained',gridspec_kw={'height_ratios':[1.2,1]})
    colors=plt.get_cmap('tab10').colors
    for i,(label,s,m) in enumerate(entries):
        c=s['comparisons'][m]['R25'];d=c['delta_alc']*1e3;lo,hi=np.array(c['cluster_ci95'])*1e3
        axes[0].plot([lo,hi],[i,i],color=colors[i],lw=2)
        axes[0].scatter(d,i,color=colors[i],s=45,zorder=3)
        axes[1].plot(range(6),np.array(c['per_budget_delta'])*1e3,'o-',color=colors[i],label=label,lw=1.5)
    axes[0].axvline(0,color='black',lw=.8)
    axes[0].set_yticks(range(len(entries)),[e[0] for e in entries]);axes[0].invert_yaxis()
    axes[0].set_xlabel('ALC difference vs R25 (x 0.001); negative is better')
    axes[0].set_title('Complete public proxy: 5,242 pairs, 3 item-split seeds\n95% paired Benchmark-cluster bootstrap; adaptive reuse is not corrected')
    axes[1].axhline(0,color='black',lw=.8);axes[1].set_xticks(range(6),BUDGETS)
    axes[1].set_xlabel('Revealed-label budget');axes[1].set_ylabel('Brier difference vs R25 (x 0.001)')
    axes[1].legend(ncol=2,fontsize=9,frameon=False)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    folder=REPO/'reports/historical_positive_figures';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(folder/f'outer_comparison.{suffix}',dpi=180)
    plt.close(fig)
    print(str(folder))

if __name__=='__main__':main()
