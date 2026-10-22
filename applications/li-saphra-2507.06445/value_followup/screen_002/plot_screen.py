"""Draw the complete, fixed screen comparison from the analysis artifact."""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    colors={'accepted':'#007C83', 'rejected':'#A45136'}
    fig, axes=plt.subplots(1,2,figsize=(11.5,4.6),gridspec_kw={'width_ratios':[1.35,1]})
    fig.patch.set_facecolor('#FCFBF7')
    for ax in axes:
        ax.set_facecolor('#FCFBF7')
        ax.spines[['top','right']].set_visible(False)
    for name in colors:
        rows=[r for r in report['family_results'] if r['stratum']==name]
        axes[0].scatter([r['native_margin'] for r in rows],[r['anchor_gap'] for r in rows],
                        s=38,alpha=.72,color=colors[name],label=f'{name.capitalize()} (32)')
    axes[0].axvline(8,color='#555555',ls='--',lw=1)
    axes[0].axhline(.202,color='#555555',ls=':',lw=1.3)
    axes[0].set(xlabel='Native margin before intervention (nat)',ylabel='Observed anchor separation (nat)',
                title='A forecast made before the transfers')
    axes[0].legend(frameon=False,fontsize=9,loc='upper left')
    axes[0].text(.02,.02,'Screen: margin < 8  |  Separating: gap > 0.202',
                 transform=axes[0].transAxes,fontsize=8.5,color='#555555')
    p=report['primary']
    for i,name in enumerate(colors):
        group=p['strata'][name]; rate=group['rate']; lo,hi=group['simultaneous_interval']
        axes[1].bar(i,rate,color=colors[name],width=.55,alpha=.85)
        axes[1].errorbar(i,rate,yerr=[[rate-lo],[hi-rate]],fmt='none',color='#222222',capsize=5)
        axes[1].text(i,min(hi+.055,1.12),f"{group['separating']}/32",ha='center',fontsize=12,fontweight='bold')
    axes[1].set(xticks=[0,1],xticklabels=['Accepted','Rejected'],ylim=(0,1.2),
                yticks=[0,.25,.5,.75,1],yticklabels=['0%','25%','50%','75%','100%'],
                ylabel='Families meeting the separation criterion',title='Fresh cases in both groups')
    lo,hi=p['simultaneous_interval']
    fig.suptitle('Does the native-margin screen find more separable cases?',fontsize=15,fontweight='bold',x=.05,ha='left')
    footer=(f"Accepted − rejected: {100*p['enrichment']:.1f} pp; simultaneous interval [{100*lo:.1f}, {100*hi:.1f}] pp. "
            'Error bars: joint coverage ≥95%.\n'
            'One selected head (a9g0io1r, L2/H1). Separation does not establish candidate fit or identify a mechanism.')
    fig.text(.05,.025,footer,fontsize=9,color='#444444',va='bottom')
    fig.subplots_adjust(left=.07,right=.98,bottom=.24,top=.81,wspace=.33)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg']:
        fig.savefig(args.output.with_suffix('.'+extension),dpi=170,facecolor=fig.get_facecolor())


if __name__=='__main__':
    main()
