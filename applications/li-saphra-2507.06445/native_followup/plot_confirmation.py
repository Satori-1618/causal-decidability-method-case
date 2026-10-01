"""Render the two real-model discrimination steps from the verified report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='1aez5d6p')
    p.add_argument('--head',type=int,default=2)
    a=p.parse_args();report=json.loads(a.report.read_text())
    focus=next(r for r in report['tasks'] if r['task']['model_id']==a.model and r['task']['head']==a.head)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14,10),facecolor='#fafaf7')
    gs=fig.add_gridspec(2,2,height_ratios=[1.2,1],width_ratios=[1,1.1],hspace=.78,wspace=.42)
    ax=fig.add_subplot(gs[0,0]);ax.set_axis_off()
    ax.set_title('1  Separate the changes made by a patch',loc='left',fontweight='bold',pad=18)
    labels=['Native N','Full uniform U','Bracket routing R','Group masses G','Within-symbol W','Symbol masses T']
    changes=np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,0],[0,0,1],[0,1,0]])
    cols=['BOS/EOS/bracket\ngroup masses','Open/close\nproportions','Routing among\nidentical symbols']
    for j,label in enumerate(cols):ax.text(.42+j*.25,.99,label,ha='center',va='top',fontsize=9,transform=ax.transAxes)
    for i,label in enumerate(labels):
        y=.76-i*.14
        ax.text(0,y,label,va='center',transform=ax.transAxes)
        for j,val in enumerate(changes[i]):
            ax.scatter(.42+j*.25,y,s=470,marker='s',c='#207d91' if val else '#dfdfd8',transform=ax.transAxes)
            ax.text(.42+j*.25,y,'U' if val else 'N',color='white' if val else '#444444',ha='center',va='center',transform=ax.transAxes,fontsize=9)
    ax.text(0,-.12,'N = native factor   U = uniform factor\nSame model, values, readout and target head.',transform=ax.transAxes,fontsize=9,color='#555555')
    ax=fig.add_subplot(gs[0,1])
    candidates=[('stage1','H_R','Routing alone'),('stage1','H_G','Group masses alone'),('stage2','H_W','Within-symbol routing'),('stage2','H_T','Symbol masses alone')]
    colors={'adequate':'#167044','excluded':'#ac3942','unresolved':'#b07918'}
    for i,(stage,name,label) in enumerate(candidates):
        d=focus['stages'][stage]['candidates'][name];y=3-i;x=d['definite_hits']/d['n']
        ax.plot([d['lower'],d['upper']],[y,y],color=colors[d['status']],lw=4)
        ax.scatter([x],[y],color=colors[d['status']],s=65,zorder=4)
        ax.text(1.025,y,f"{d['definite_hits']}/{d['n']}\n{d['status']}",va='center',fontsize=9,color=colors[d['status']])
    ax.axvline(.8,color='#777777',ls='--',lw=1)
    ax.set_xlim(-.03,1.01);ax.set_ylim(-.6,3.6);ax.set_yticks([3,2,1,0],[v[2] for v in candidates]);ax.set_xticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%'])
    ax.set_xlabel('Families meeting the fixed prediction tolerance')
    ax.set_title('2  Test predictions on fresh families',loc='left',fontweight='bold',pad=18)
    ax.text(.02,-.27,f"{a.model}, head {a.head} • simultaneous 95% bounds\nDashed line: 80% adequacy requirement",transform=ax.transAxes,fontsize=9,color='#555555')
    ax=fig.add_subplot(gs[1,0]);ax.set_axis_off()
    ax.set_title('What was actually narrowed?',loc='left',fontweight='bold')
    ax.text(0,.88,'Full patch changes the output',fontsize=13,fontweight='bold',transform=ax.transAxes,va='top')
    ax.text(0,.68,'↓  Test which component reproduces that change',fontsize=10,transform=ax.transAxes,va='top')
    explanation=('Bracket routing passes its own adequacy test.\nWithin-symbol redistribution then passes the\nseparate test of the bracket-routing effect.' if focus['stages']['stage1']['candidates']['H_R']['status']=='adequate' else 'Within-symbol redistribution reproduces the\nbracket-routing effect in this test.\nStage 1 remains unresolved for routing alone.')
    ax.text(0,.50,explanation,fontsize=11,color='#167044',transform=ax.transAxes,linespacing=1.4,va='top')
    ax.text(0,.12,'Still open: contextual or positional content of the values.\nThe stages have separate conditional populations.',fontsize=9,color='#555555',transform=ax.transAxes,va='top')
    ax=fig.add_subplot(gs[1,1]);transfer=report['summary']['transfer_cohort']['stage2']
    order=['adequate','excluded','unresolved','insufficient_eligible_families','technical_invalidity']
    color={'adequate':'#167044','excluded':'#ac3942','unresolved':'#c28b31','insufficient_eligible_families':'#b5bcc0','technical_invalidity':'#3f4448'}
    for i,name in enumerate(['H_W','H_T']):
        left=0
        for status in order:
            value=transfer[name].get(status,0)
            if value:
                ax.barh(1-i,value,left=left,color=color[status],height=.48)
                ax.text(left+value/2,1-i,str(value),ha='center',va='center',color='white' if status in ['adequate','excluded','technical_invalidity'] else '#222222',fontsize=10)
            left+=value
    ax.set_yticks([1,0],['Within-symbol','Symbol masses']);ax.set_xlim(0,report['summary']['transfer_cohort']['tasks']);ax.set_xlabel('Preselected transfer tasks (finite released cohort)')
    ax.set_title('3  Keep the transfer failures visible',loc='left',fontweight='bold',pad=16)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=color[s],label={'insufficient_eligible_families':'Too few separating families','technical_invalidity':'Technical failure'}.get(s,s.capitalize())) for s in order],loc='upper center',bbox_to_anchor=(.45,-.26),ncol=2,frameon=False,fontsize=8)
    fig.suptitle('From one patch effect to testable causal explanations',x=.07,ha='left',fontsize=18,fontweight='bold',y=.98)
    fig.text(.07,.925,'Li, Saphra et al. released Dyck models  ·  512 new cyclic families  ·  forecasts fixed before hybrid interventions',fontsize=10,color='#555555')
    if a.model!='1aez5d6p':fig.text(.07,.895,'Illustration selected after the cohort run; the full transfer cohort is shown below. Rules and simultaneous bounds were fixed before the run.',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.07,right=.91,top=.81,bottom=.16)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output.with_suffix('.png'),dpi=170,facecolor=fig.get_facecolor())
    fig.savefig(a.output.with_suffix('.pdf'),facecolor=fig.get_facecolor())
    fig.savefig(a.output.with_suffix('.svg'),facecolor=fig.get_facecolor())
    svg = a.output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__=='__main__':main()
