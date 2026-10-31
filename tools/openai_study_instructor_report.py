"""Verified Instructor-XL core results paired with prior OpenAI/BGE recipes."""
from pathlib import Path
import json
import math

import pandas as pd

from scripts.experiments import expected_runs,load_artifact,verify_pins
from scripts.resources import require_resources
from tools.course.analyze import verify_package
from tools.openai_study import once,sealed
from tools.openai_study_followup_report import finish
from tools.openai_study_report import normalize_condition,snapshot_record,summarize
from tools.paper_preflight.common import ROOT,sha256

BASE=ROOT/'outputs/instructor-core-20261031'


def main():
    require_resources();protocol=sealed(BASE/'protocol.json');verify_pins(protocol,ROOT)
    expected={r['id']:r for r in expected_runs(protocol)}
    paths={p.parent.relative_to(BASE/'runs/deployment').as_posix():p.parent for p in (BASE/'runs/deployment').rglob('manifest.json')}
    if set(paths)!=set(expected):raise ValueError('Instructor condition inventory incomplete or unexpected')
    groups={};inventory={};prefixes=[];proof_pins={}
    for p in (BASE/'verification').glob('*.json'):
        v=json.loads(p.read_text(encoding='utf-8'));group=(v['domain'],v['seed'],v['method'])
        if group in groups or not v['verified'] or v['script_sha256']!=sha256(ROOT/'tools/research/verify_runs.py'):
            raise ValueError('Instructor replay is duplicate, missing or stale')
        groups[group]=v;proof_pins[p.name]=sha256(p)
        for r in v['inventory']:
            if r['id'] in inventory:raise ValueError('Duplicate replayed condition')
            inventory[r['id']]=r['manifest_sha256']
        folder=BASE/'runs/learning'/group[0]/f'seed-{group[1]}'/'main'/group[2]
        learned=load_artifact(folder)
        if learned['manifest_sha256']!=v['learning_manifest_sha256']:raise ValueError('Learning replay stale')
        for fraction,snapshot in learned['snapshots'].items():prefixes.append(snapshot_record(learned['request']['lineage'],fraction,snapshot))
    if set(groups)!={(r['domain'],r['seed'],r['method']) for r in expected.values()} or set(inventory)!=set(expected):
        raise ValueError('Full78-group independent Instructor replay required')
    verify_package(ROOT/'reports/openai-core-2026-10-31');verify_package(ROOT/'reports/full-experiments-2026-10-29')
    reference=json.loads((ROOT/'reports/openai-core-2026-10-31/gpt4-reference.json').read_text(encoding='utf-8'))
    rows=[]
    for identity,condition in expected.items():
        v=load_artifact(paths[identity]);summary=v['summary'];metrics=summary['metrics'];budget=summary['budget']
        if (v['manifest_sha256']!=inventory[identity] or v['request']['lineage']['condition']!=condition
                or v['request']['lineage']['protocol_sha256']!=sha256(BASE/'protocol.json')
                or summary['deployment_split']['query_ids']!=reference[condition['domain']]['query_ids']
                or v['synthetic'] or budget['estimated_budget_violation'] or not budget['actual_costs_complete']):
            raise ValueError('Instructor result identity, coverage or budget check failed')
        rows.append(normalize_condition(dict(condition,budget=budget['budget'],requested=summary['requested_queries'],
            served=summary['served_queries'],total_reward=metrics['total_reward'],mean_reward_served=metrics['mean_reward_served'],
            estimated_cost=budget['estimated_total'],actual_recorded_cost=budget['actual_total'],
            actual_budget_excess=budget['actual_budget_excess_observed'],learning_rounds=v['request']['lineage']['learning_rounds']),reference[condition['domain']]))
    current=pd.DataFrame(rows);openai=pd.read_csv(ROOT/'reports/openai-core-2026-10-31/per-seed.csv')
    bge=pd.read_csv(ROOT/'reports/openai-core-2026-10-31/bge-core-normalized.csv')
    paired=[]
    for name,other in [('openai',openai),('bge',bge)]:
        if other.id.duplicated().any() or set(other.id)!=set(current.id):raise ValueError('Recipe comparison population differs')
        other=other.set_index('id')
        for row in current.to_dict('records'):
            ref=other.loc[row['id']]
            if any(row[k]!=ref[k] for k in ('domain','seed','method','learning_fraction','budget_fraction')):
                raise ValueError('Paired recipe metadata differs')
            paired.append(dict(id=row['id'],reference=name,domain=row['domain'],seed=row['seed'],method=row['method'],
                learning_fraction=row['learning_fraction'],budget_fraction=row['budget_fraction'],
                **{k+'_delta':row[k]-ref[k] for k in ('quality_fraction_gpt4','cost_fraction_gpt4','coverage')}))
    means=summarize(current);learning=pd.DataFrame(prefixes)
    if len(groups)!=78 or len(current)!=858 or len(learning)!=312:raise ValueError('Instructor core totals differ')
    output=ROOT/'reports/instructor-core-2026-10-31';output.mkdir(parents=True,exist_ok=False)
    for name,frame in [('per-seed',current),('mean-min-max',means),('paired-recipes',pd.DataFrame(paired)),('learning-prefixes',learning)]:
        frame.to_csv(output/(name+'.csv'),index=False,lineterminator='\n')
    once(output/'gpt4-reference.json',reference)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),constrained_layout=True)
    for axis,domain in zip(axes,('routerbench','mmlu')):
        for name,data in [('BGE-M3',bge),('OpenAI',openai),('Instructor-XL',current)]:
            selected=data.loc[(data.domain==domain)&(data.method=='pilot')&(data.learning_fraction==1)].groupby('budget_fraction').mean(numeric_only=True).sort_index()
            axis.plot(100*selected.cost_fraction_gpt4,100*selected.quality_fraction_gpt4,marker='o',markersize=4,label=name)
        axis.set_title(domain.upper());axis.set_xlabel('Recorded cost / same-query GPT4 (%)')
        axis.set_ylabel('True reward / same-query GPT4 (%)');axis.grid(alpha=.2)
    axes[1].legend();fig.suptitle('Local PILOT representation recipes, including retuning; mean three seeds')
    for ext in ('png','svg'):fig.savefig(output/('recipe-comparison.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11.5,8),constrained_layout=True)
    routers={'pilot','linucb','epoch-greedy','explore-only','random'}
    for column,domain in enumerate(('routerbench','mmlu')):
        selected=current.loc[(current.domain==domain)&(current.learning_fraction==1)&current.method.isin(routers)]
        for method,group in selected.groupby('method'):
            group=group.groupby('budget_fraction').mean(numeric_only=True).sort_index()
            axes[0,column].plot(100*group.cost_fraction_gpt4,100*group.quality_fraction_gpt4,marker='o',markersize=3,label=method)
        axes[0,column].set_title(domain.upper());axes[0,column].set_xlabel('Recorded cost / GPT4 (%)');axes[0,column].set_ylabel('True reward / GPT4 (%)')
        for method,group in learning.loc[(learning.domain==domain)&learning.method.isin(routers)].groupby('method'):
            group=group.groupby('rounds').mean(numeric_only=True)
            axes[1,column].plot(group.index,group.cumulative_empirical_oracle_regret,marker='o',markersize=3,label=method)
        axes[1,column].set_xlabel('Observed learning rounds');axes[1,column].set_ylabel('Empirical-oracle regret')
        axes[0,column].grid(alpha=.2);axes[1,column].grid(alpha=.2)
    axes[0,1].legend(fontsize=8);fig.suptitle('Instructor-XL: adaptive/router baselines, unchanged eight-arm pool')
    for ext in ('png','svg'):fig.savefig(output/('instructor-routing.'+ext),dpi=160)
    plt.close(fig)
    focus=[];lines=['# Instructor-XL full core sensitivity','',
        'All858 primary conditions,78 learning streams and312 nested snapshots independently replayed.',
        'Same8 admitted arms, original1245/313 preference split, routing partitions/order, historical cost inputs and same-query GPT4 denominators as OpenAI core.',
        'Real local hkunlp/instructor-xl pinned checkpoint:768D base representation,512-token cap, FP32; learned64D projection with30/40 epochs and original three seeds. Fresh tuning uses only tuning labels.',
        'No additional paid API calls. BGE and OpenAI packages remain unchanged.',
        '', '| Domain | Estimated allocation | PILOT quality/GPT4 | Recorded cost/GPT4 | Coverage | Quality change versus OpenAI (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for domain in ('routerbench','mmlu'):
        for fraction in (.2,.5,1.):
            group=current.loc[(current.domain==domain)&(current.method=='pilot')&(current.learning_fraction==1)&(current.budget_fraction==fraction)]
            old=openai.loc[(openai.domain==domain)&(openai.method=='pilot')&(openai.learning_fraction==1)&(openai.budget_fraction==fraction)]
            quality=group.quality_fraction_gpt4.mean();cost=group.cost_fraction_gpt4.mean();coverage=group.coverage.mean();delta=quality-old.quality_fraction_gpt4.mean()
            lines.append(f'|{domain}|{100*fraction:.0f}%|{100*quality:.4f}%|{100*cost:.4f}%|{100*coverage:.4f}%|{100*delta:+.4f}|')
            focus.append(dict(domain=domain,allocation=fraction,quality=quality,cost=cost,coverage=coverage,quality_delta_openai=delta))
    lines+=['', 'Recipe limits: the routing instruction is a declared local assumption, not confirmed author text. Instructor clips214 routing and52 preference inputs, while the OpenAI recipe uses full text; BGE has its own2048-token tokenizer/cap. These are representation-recipe plus retuning comparisons, not an isolated causal encoder substitution.',
        'Production uses dynamic right padding; pinned official fixed512 padding/pooling/mask replay agrees within the recorded float tolerance. Every vector/role mapping/token length is audited; sampled real GPU outputs are independently checked with official source. This is not a second full neural-model implementation.',
        'Eight-versus-eleven arms, release/86-versus64 tasks, author split/order and unpublished training details remain unresolved. MMLU is nested in RouterBench. Do not report matched-paper reproduction error or claim significance from three seeds.',
        'All13 methods remain in CSV, including eight fixed models. The displayed adaptive curves are the five router families. Unserved questions remain in every denominator; early-stop coverage must accompany quality.',
        '', 'Commands: local encoder prepare/encode -> independent cache verifier -> Instructor study declare/fit/tune/freeze/run ->78 generic independent verify_runs.py groups with the Instructor protocol/index/cache paths -> this report.',
        'Raw runs and this report are immutable. Instructor ablations are not included in this858-condition core sensitivity.']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    metadata=dict(kind='verified-local-instructor-core-study',conditions=858,learning_streams=78,snapshots=312,
        protocol_sha256=sha256(BASE/'protocol.json'),replay_files_sha256=proof_pins,focus_pilot=focus,
        representation_comparison='recipe and retuning, not isolated encoder causality',paper_equivalent=False,additional_paid_api_calls=0)
    print(json.dumps(finish(output,metadata),indent=2))


if __name__=='__main__':main()
