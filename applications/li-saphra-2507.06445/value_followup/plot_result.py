"""Plot the complete development result; no fitting, inference or selection."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent


def main():
    report=json.loads((HERE/'results/development_report.json').read_text())
    rows=report['family_results']
    eligible=[r for r in rows if r['eligible']]
    fig,axes=plt.subplots(1,3,figsize=(15,5.8),gridspec_kw={'width_ratios':[1,1.1,1]})
    fig.patch.set_facecolor('#faf9f5')
    blue='#236b9a'; orange='#b05732'; gray='#adb5bd'
    ax=axes[0]
    for r in rows:
        c=blue if r['eligible'] else gray
        ax.plot([0,1],[r['balance_contrast'],r['position_contrast']],color=c,alpha=.65,lw=1)
        ax.scatter([0,1],[r['balance_contrast'],r['position_contrast']],color=c,s=17,zorder=3)
    ax.axhline(0,color='#777777',lw=.7)
    ax.set_xticks([0,1],['Change balance\n−2 → +2','Change position\n20 → 28'])
    ax.set_xlim(-.25,1.25)
    ax.set_ylabel('Paired change in rejection margin (nat)')
    ax.set_title('1  Cross the donor properties',loc='left',fontweight='bold')
    ax.text(0,-.32,'Each line is one whole family.\nThe other donor property is averaged within family.',transform=ax.transAxes,fontsize=9,color='#444444')
    ax=axes[1]
    for flag,color,label in [(False,gray,'Insufficient anchor separation'),(True,blue,'Separating family')]:
        select=[r for r in rows if r['eligible']==flag]
        ax.scatter([r['max_prediction_error']['H_state'] for r in select],
                   [r['max_prediction_error']['H_position'] for r in select],c=color,s=35,label=label)
    ax.axvline(.1,color=orange,ls='--',lw=1)
    ax.axhline(.1,color=orange,ls='--',lw=1)
    ax.set_xlim(-.007,.135);ax.set_ylim(-.025,.80)
    ax.set_xlabel('Balance-class account: maximum error (nat)')
    ax.set_ylabel('Position account: maximum error (nat)')
    ax.set_title('2  Score all six predictions',loc='left',fontweight='bold')
    ax.legend(loc='upper left',bbox_to_anchor=(-.05,-.24),frameon=False,fontsize=8)
    ax.text(.015,.62,'7/8 match balance class\n0/8 match position',fontsize=10,color=blue)
    ax=axes[2];ax.axis('off')
    text=("DEVELOPMENT RESULT\n\n"
          f"{report['eligible_families']}/32 families have separating anchors\n"
          "24/32 are outside that eligible subset\n\n"
          "Same-balance, same-position controls:\n"
          "all 32 stay within 0.10 nat\n\n"
          "All intervention and precision checks pass\n"
          "No final answer label changes\n\n"
          "CONFIRMATION NOT STARTED\n"
          "Required: ≥16 separating families\n"
          "and ≥90% definite prediction matches.\n"
          "Observed: 8 families and 87.5%.\n\n"
          "Promising conditional pattern;\n"
          "no confirmed semantic identification.")
    ax.text(.0,.96,text,va='top',fontsize=10,linespacing=1.5,color='#222222')
    fig.suptitle('From within-symbol routing to a test of transferred information',x=.055,ha='left',fontsize=17,fontweight='bold')
    fig.text(.055,.885,'One selected Dyck head · fixed recipient attention · fresh donor prefixes · 32 development families',fontsize=10,color='#555555')
    fig.text(.055,.03,'Scope: balance-class and position invariance on this grid. Sign, exact depth and normalized-depth accounts are not uniquely identified.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.055,right=.985,top=.79,bottom=.30,wspace=.48)
    directory=HERE/'figures';directory.mkdir(exist_ok=True)
    fig.savefig(directory/'value_development.png',dpi=170,facecolor=fig.get_facecolor())
    svg=directory/'value_development.svg'
    fig.savefig(svg,facecolor=fig.get_facecolor())
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':main()
