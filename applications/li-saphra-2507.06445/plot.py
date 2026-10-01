"""Draw measured effects and clearly labelled illustrative countermodels."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
with (ROOT/"results/heads.csv").open() as f:
    heads = [r for r in csv.DictReader(f) if int(r["n_layer"]) >= 2 and r["head_type"] == "sign-matching"]
proof = json.loads((ROOT/"results/countermodels.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), gridspec_kw={"width_ratios": [1, 1.15]})
fig.suptitle("A robust replacement benefit leaves two causal explanations open", x=.06, ha="left", fontsize=15, fontweight="bold")
ax = axes[0]
x = [int(r["uniform_change_count"])/10 for r in heads]
y = [int(r["mean_change_count"])/10 for r in heads]
ax.plot([-1,25],[-1,25],color="#b7c3ce",lw=1,ls="--",zorder=0)
ax.axhline(0,color="#bcc5ce",lw=.8); ax.axvline(0,color="#bcc5ce",lw=.8)
ax.scatter(x,y,s=42,color="#187c80",alpha=.8,edgecolor="white",linewidth=.5)
ax.set(xlabel="Uniform replacement: accuracy change (pp)", ylabel="Mean replacement: accuracy change (pp)", xlim=(-1,25), ylim=(-1,25))
ax.set_title("MEASURED · 42 heads in 41 models",loc="left",fontsize=11,pad=12)
ax.text(.04,.93,"40/42 improve by >1 pp under both\nAll 42 retained; no strongest-head selection",transform=ax.transAxes,va="top",fontsize=9)
ax = axes[1]
positions = [0,1,2,3]
for name,color,style,label in [("removal_helps","#2563a6","o-","Removal helps"),("replacement_rescues","#c66c2c","s--","Replacement rescues")]:
    q = proof["witnesses"][name]["correct_counts"]
    vals = [q[k]/10 for k in ("native","uniform","mean","zero")]
    ax.plot(positions,vals,style,color=color,lw=1.8,ms=6,label=label)
for pos,value in zip(positions[:3],[77.9,82.3,82.7]):
    ax.annotate(f"{value:.1f}%",(pos,value),xytext=(0,-18),textcoords="offset points",ha="center",fontsize=9)
ax.annotate("87.9%",(3,87.9),xytext=(-8,6),textcoords="offset points",ha="right",color="#2563a6",fontsize=10)
ax.annotate("67.9%",(3,67.9),xytext=(-8,-15),textcoords="offset points",ha="right",color="#c66c2c",fontsize=10)
ax.axvspan(2.6,3.35,color="#f1e7c9",alpha=.65,zorder=-1)
ax.set(xticks=positions,xticklabels=["Native","Uniform","Mean","Zero output"],ylabel="Accuracy (%)",ylim=(61,94),xlim=(-.25,3.35))
ax.set_title("CONSTRUCTED · same observed scores, opposite effects",loc="left",fontsize=11,pad=12)
ax.legend(loc="lower left",fontsize=9,frameon=False)
ax.text(3,92.5,"Not measured",ha="center",va="top",fontsize=9)
fig.text(.06,.07,"Left: released checkpoint-5 OOD results, Li et al., 2507.06445. Right: exact attention witnesses matched to model 1aez5d6p, L2/H2.",fontsize=9,color="#475569")
fig.text(.06,.03,"The two zero-output values illustrate nonidentification; they are not predictions for the trained model. Retrospective audit.",fontsize=9,color="#475569")
fig.subplots_adjust(left=.07,right=.975,top=.81,bottom=.24,wspace=.34)
(ROOT/"figures").mkdir(exist_ok=True)
fig.savefig(ROOT/"figures/ablation-and-rivals.png",dpi=180,facecolor="white")
fig.savefig(ROOT/"figures/ablation-and-rivals.svg",facecolor="white",metadata={"Date":None})
print("Saved plot from the canonical result files.")
