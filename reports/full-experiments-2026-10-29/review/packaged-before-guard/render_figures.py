"""Render scientific figures exclusively from a verified complete aggregate."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/research-experiments-20261029'
sys.path.insert(0,str(ROOT))
from scripts.resources import require_resources
print(require_resources())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

source=BASE/'aggregate';dest=BASE/'figures'
if dest.exists():raise FileExistsError(dest)
manifest=json.loads((source/'manifest.json').read_text())
assert manifest['complete'] and manifest['primary_runs']==858 and manifest['ablation_runs']==240
for name,expected in manifest['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==expected
table=pd.read_csv(source/'mean-min-max.csv');paired=pd.read_csv(source/'paired-deltas.csv');learning=pd.read_csv(source/'learning-prefixes.csv')
labels={'pilot':'PILOT','linucb':'LinUCB','epoch-greedy':'EpochGreedy','explore-only':'ExploreOnly','random':'Random',
        'fixed-00':'Fixed WizardLM 13B','fixed-01':'Fixed Claude Instant','fixed-02':'Fixed Claude v1',
        'fixed-03':'Fixed GPT-3.5','fixed-04':'Fixed GPT-4','fixed-05':'Fixed Code Llama 34B',
        'fixed-06':'Fixed Llama 2 70B','fixed-07':'Fixed Yi 34B'}
colors=dict(zip(labels,list(plt.get_cmap('tab20').colors)[:len(labels)]));colors['pilot']='#0b716a'
evidence={};dest.mkdir()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
def canvas(title,ylabel,xlabel):
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.7),layout='constrained')
    for ax,domain in zip(axs,['routerbench','mmlu']):
        ax.set_title('RouterBench' if domain=='routerbench' else 'MMLU subset')
        ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.18);ax.set_axisbelow(True)
    fig.suptitle(title+'\nEight-arm local experiment | means of seeds 7, 42, 2026',fontsize=15,fontweight='bold')
    return fig,axs
def save(fig,name,series):
    handles,legend=fig.axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncols=5,fontsize=8,frameon=False)
    for extension in ('png','pdf','svg'):fig.savefig(dest/(name+'.'+extension),dpi=200,bbox_inches='tight')
    plt.close(fig);evidence[name]=series
def curves(name,title,metric,ylabel,*,fraction=1,x='budget_fraction',budget=None):
    fig,axs=canvas(title,ylabel,'Budget / reference budget' if x=='budget_fraction' else 'Fraction of learning stream')
    series=[]
    for ax,domain in zip(axs,['routerbench','mmlu']):
        rows=table[(table.domain==domain)&(table.variant=='main')]
        if fraction is not None:rows=rows[rows.learning_fraction==fraction]
        if budget is not None:rows=rows[rows.budget_fraction==budget]
        for method,label in labels.items():
            r=rows[rows.method==method].sort_values(x)
            xs=r[x].tolist();ys=r[metric+'_mean'].tolist()
            ax.plot(xs,ys,label=label,color=colors[method],marker='o' if method=='pilot' else '.',
                linewidth=2.7 if method=='pilot' else 1.1,alpha=1 if method=='pilot' else .82,
                linestyle='--' if method.startswith('fixed') else '-')
            series.append(dict(domain=domain,method=method,x=xs,y=ys,source_metric=metric+'_mean'))
        if metric in ('service_utility','coverage'):ax.set_ylim(0,1)
    save(fig,name,series)
curves('service-utility','Quality delivered across the requested workload','service_utility','Total reward / requested queries')
curves('coverage','Coverage under the common estimated-budget policy','coverage','Fraction of queries served')
curves('learning-size','Sensitivity to the amount of routing feedback','service_utility','Total reward / requested queries',
       fraction=None,x='learning_fraction',budget=.5)
curves('recorded-cost','Recorded costs are evaluated separately from estimates','actual_recorded_cost','Recorded benchmark cost')

fig,axs=canvas('Retrospective learning regret at the frozen prefix checkpoints','Cumulative empirical oracle regret','Fraction of learning stream')
series=[]
for ax,domain in zip(axs,['routerbench','mmlu']):
    for method,label in labels.items():
        r=learning[(learning.domain==domain)&(learning.variant=='main')&(learning.method==method)]
        r=r.groupby('learning_fraction').cumulative_empirical_oracle_regret.mean().sort_index()
        ax.plot(r.index,r.values,label=label,color=colors[method],linewidth=2.7 if method=='pilot' else 1.1,
                marker='o' if method=='pilot' else '.',linestyle='--' if method.startswith('fixed') else '-')
        series.append(dict(domain=domain,method=method,x=r.index.tolist(),y=r.values.tolist(),source_metric='cumulative_empirical_oracle_regret'))
save(fig,'learning-regret',series)

ablations={'zero-prior-direction':'Zero prior directions','unit-initialization-strength':'Unit initialization strength',
    'prior-only':'Prior only: no routing updates','single-budget-bin':'Single budget bin','projection-dimension-8':'8D projection + prior'}
fig,axs=canvas('Paired controlled ablations relative to full PILOT','Change in reward / requested query','Budget / reference budget')
series=[]
for ax,domain in zip(axs,['routerbench','mmlu']):
    ax.axhline(0,color='#444444',linewidth=.8)
    for variant,label in ablations.items():
        r=paired[(paired.domain==domain)&(paired.comparison==variant)]
        r=r.groupby('budget_fraction').service_utility_delta.agg(['mean','min','max']).sort_index()
        line=ax.plot(r.index,r['mean'],marker='o',linewidth=1.6,label=label)[0]
        ax.fill_between(r.index,r['min'],r['max'],color=line.get_color(),alpha=.08)
        series.append(dict(domain=domain,variant=variant,x=r.index.tolist(),y=r['mean'].tolist(),
            low=r['min'].tolist(),high=r['max'].tolist(),interpretation='paired mean and observed seed range; not a confidence interval'))
save(fig,'ablations',series)
report=dict(kind='verified-aggregate-figure-data',aggregate_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
    source_files=manifest['files'],series=evidence,synthetic=False,paper_equivalent=False,
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()})
with (dest/'figure-data.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({'figures':len(evidence),'directory':str(dest)}))
