"""All-request GPT-4 normalization and descriptive OpenAI/BGE core comparisons."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from scripts.experiments import expected_runs, load_artifact, verify_pins
from scripts.resources import require_resources
from tools.course.analyze import verify_package
from tools.paper_preflight.common import ROOT, digest, read_json, sha256
from tools.paper_preflight.compare import relative_metrics


def normalize_condition(row,reference):
    if int(row['requested'])!=reference['count']:
        raise ValueError('GPT-4 reference population differs from requested queries')
    return dict(row,**relative_metrics(total_reward=float(row['total_reward']),actual_cost=float(row['actual_recorded_cost']),
        served=int(row['served']),requested=int(row['requested']),gpt4_total_reward=reference['reward'],gpt4_total_cost=reference['cost']),
        allocated_budget_fraction_gpt4=row['budget']/reference['cost'])


def paired_comparison(new,old):
    if (new.id.duplicated().any() or old.id.duplicated().any() or set(new.id)!=set(old.id)):
        raise ValueError('Representation comparison requires identical unique condition IDs')
    old=old.set_index('id');records=[]
    keys=('domain','seed','method','learning_fraction','budget_fraction')
    for row in new.to_dict('records'):
        reference=old.loc[row['id']]
        if any(row[k]!=reference[k] for k in keys):raise ValueError('Paired condition metadata differs')
        result={k:row[k] for k in ('id',)+keys}
        for metric in ('quality_fraction_gpt4','cost_fraction_gpt4','coverage'):
            result.update({metric+'_openai':row[metric],metric+'_bge':reference[metric],
                metric+'_delta':row[metric]-reference[metric]})
        records.append(result)
    return pd.DataFrame(records)


def summarize(frame):
    metrics=['quality_fraction_gpt4','cost_fraction_gpt4','coverage','service_utility',
        'actual_recorded_cost','estimated_cost','actual_budget_excess','allocated_budget_fraction_gpt4']
    keys=['domain','variant','method','learning_fraction','budget_fraction']
    rows=[]
    for key,group in frame.groupby(keys,sort=True):
        if sorted(group.seed)!=[7,42,2026]:raise ValueError('Each summary needs all three seeds')
        row=dict(zip(keys,key),seed_count=3)
        for metric in metrics:
            row.update({metric+'_mean':math.fsum(group[metric])/3,
                metric+'_min':min(group[metric]),metric+'_max':max(group[metric])})
        rows.append(row)
    return pd.DataFrame(rows)


def snapshot_record(lineage,fraction,snapshot):
    if snapshot['rounds']!=snapshot['metrics']['rounds']:raise ValueError('Snapshot rounds differ from metrics')
    return dict(domain=lineage['domain'],seed=lineage['seed'],method=lineage['method'],
        learning_fraction=float(fraction),**snapshot['metrics'])


def execute(base,output):
    preflight=require_resources()
    if output.exists():raise FileExistsError(output)
    protocol=read_json(base/'protocol.json')
    if protocol['manifest_sha256']!=digest({k:v for k,v in protocol.items() if k!='manifest_sha256'}):
        raise ValueError('Frozen protocol checksum mismatch')
    verify_pins(protocol,ROOT)
    expected={r['id']:r for r in expected_runs(protocol)}
    found={p.parent.relative_to(base/'runs/deployment').as_posix():p.parent
        for p in (base/'runs/deployment').rglob('manifest.json')}
    if set(found)!=set(expected):raise ValueError('Incomplete or unexpected deployment inventory')
    replay={};groups=set();proofs={};replay_learning={}
    for p in (base/'verification').glob('*.json'):
        v=read_json(p)
        if not v['verified'] or v['script_sha256']!=sha256(ROOT/'tools/research/verify_runs.py'):
            raise ValueError('Independent replay missing or stale')
        group=(v['domain'],v['seed'],v['method'])
        if group in groups:raise ValueError('Duplicate independent replay')
        groups.add(group);proofs[p.name]=sha256(p)
        replay_learning[group]=v['learning_manifest_sha256']
        for r in v['inventory']:
            if r['id'] in replay:raise ValueError('Duplicate independently verified condition')
            replay[r['id']]=r['manifest_sha256']
    expected_groups={(r['domain'],r['seed'],r['method']) for r in expected.values()}
    if groups!=expected_groups or set(replay)!=set(expected):raise ValueError('Independent replay inventory incomplete')
    folder=ROOT/'data/processed/routerbench_0shot'
    splits=pd.read_csv(folder/'splits.csv')
    outcomes=pd.read_parquet(folder/'outcomes.parquet',columns=['query_id','arm_id','reward','actual_cost'])
    reference={}
    for domain in protocol['primary_plan']['datasets']:
        frame=splits.loc[splits.split=='deployment'].sort_values('stream_position')
        if domain=='mmlu':frame=frame.loc[frame.task.str.startswith('mmlu-')]
        ids=frame.query_id.tolist()
        values=outcomes.loc[(outcomes.arm_id=='gpt-4-1106-preview')&outcomes.query_id.isin(ids)].set_index('query_id').loc[ids]
        reference[domain]=dict(count=len(ids),query_ids=ids,order_sha256=digest(ids),
            reward=float(values.reward.sum()),cost=float(values.actual_cost.sum()))
    records=[];run_hashes={};learning_hashes={};learning_rows=[]
    for identity,row in expected.items():
        value=load_artifact(found[identity]);req=value['request'];s=value['summary'];m=s['metrics'];b=s['budget']
        if (value['manifest_sha256']!=replay[identity] or req['lineage']['condition']!=row
                or req['lineage']['protocol_sha256']!=sha256(base/'protocol.json')
                or s['deployment_split']['query_ids']!=reference[row['domain']]['query_ids']
                or value['synthetic'] or b['estimated_budget_violation'] or not b['actual_costs_complete']):
            raise ValueError('Run identity, lineage, population or budget check failed')
        record=dict(row,budget=b['budget'],requested=s['requested_queries'],served=s['served_queries'],
            total_reward=m['total_reward'],mean_reward_served=m['mean_reward_served'],
            estimated_cost=b['estimated_total'],actual_recorded_cost=b['actual_total'],
            actual_budget_excess=b['actual_budget_excess_observed'],learning_rounds=req['lineage']['learning_rounds'])
        records.append(normalize_condition(record,reference[row['domain']]))
        run_hashes[identity]=value['manifest_sha256']
    learning_files=list((base/'runs/learning').rglob('manifest.json'))
    expected_learning={f'{d}/seed-{s}/main/{m}' for d,s,m in expected_groups}
    if {p.parent.relative_to(base/'runs/learning').as_posix() for p in learning_files}!=expected_learning:
        raise ValueError('Learning identities differ from frozen inventory')
    for p in learning_files:
        value=load_artifact(p.parent);lineage=value['request']['lineage']
        group=(lineage['domain'],lineage['seed'],lineage['method'])
        if value['manifest_sha256']!=replay_learning[group]:raise ValueError('Learning replay proof is stale')
        learning_hashes[p.parent.relative_to(base/'runs/learning').as_posix()]=value['manifest_sha256']
        for fraction,s in value['snapshots'].items():
            learning_rows.append(snapshot_record(lineage,fraction,s))
    if len(learning_hashes)!=78 or len(learning_rows)!=312:raise ValueError('Learning inventory differs')
    old_package=ROOT/'reports/full-experiments-2026-10-29'
    old_proof=verify_package(old_package)
    old=pd.read_csv(old_package/'aggregate/per-seed.csv');old=old.loc[old.variant=='main']
    old=pd.DataFrame([normalize_condition(r,reference[r['domain']]) for r in old.to_dict('records')])
    new=pd.DataFrame(records);paired=paired_comparison(new,old);means=summarize(new)
    output.mkdir(parents=True)
    for name,frame in [('per-seed',new),('mean-min-max',means),('paired-bge',paired),
            ('bge-core-normalized',old),('learning-prefixes',pd.DataFrame(learning_rows))]:
        frame.to_csv(output/(name+'.csv'),index=False,lineterminator='\n')
    pd.DataFrame([dict(method=f'fixed-{i:02d}',arm=arm) for i,arm in enumerate(protocol['arms'])]).to_csv(
        output/'model-catalog.csv',index=False,lineterminator='\n')
    (output/'gpt4-reference.json').write_text(__import__('json').dumps(reference,indent=2)+'\n',encoding='utf-8')
    figures(output,means,summarize(old),pd.DataFrame(learning_rows))
    pilot=new.loc[(new.method=='pilot')&(new.learning_fraction==1)]
    focus=[]
    for domain in ('routerbench','mmlu'):
        for fraction in (.2,.5,1.):
            sample=pilot.loc[(pilot.domain==domain)&(pilot.budget_fraction==fraction)]
            focus.append(dict(domain=domain,budget_fraction=fraction,
                quality=float(sample.quality_fraction_gpt4.mean()),cost=float(sample.cost_fraction_gpt4.mean()),
                coverage=float(sample.coverage.mean())))
    report=dict(version=1,kind='verified-openai-core-study',complete=True,synthetic=False,paper_equivalent=False,
        conditions=858,learning_streams=78,learning_snapshots=312,independent_replays=78,
        protocol_sha256=sha256(base/'protocol.json'),run_manifest_sha256=run_hashes,
        learning_manifest_sha256=learning_hashes,replay_sha256=proofs,old_package=old_proof,
        script_sha256=sha256(__file__),preflight=preflight,additional_paid_api_calls=0,
        focus_pilot=focus,aggregation='Three fixed paired seeds; descriptive mean/min/max, no significance claim')
    report_text(output,report,paired,means,base)
    report['files']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    report['manifest_sha256']=digest(report)
    (output/'manifest.json').write_text(__import__('json').dumps(report,indent=2)+'\n',encoding='utf-8')
    print({k:v for k,v in report.items() if k not in ('run_manifest_sha256','learning_manifest_sha256','files','replay_sha256')})


def figures(output,new,old,learning):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'openai-core-20261031'})
    methods=['pilot','linucb','epoch-greedy','explore-only','random']
    colors=dict(zip(methods,['#1155aa','#cc5500','#008866','#8866aa','#777777']))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis,domain in zip(axes,('routerbench','mmlu')):
        for method in methods:
            values=new.loc[(new.domain==domain)&(new.method==method)&(new.learning_fraction==1)].sort_values('budget_fraction')
            axis.plot(100*values.cost_fraction_gpt4_mean,100*values.quality_fraction_gpt4_mean,
                marker='o',markersize=3,label=method,color=colors[method])
        axis.axhline(100,color='#bbbbbb',linewidth=.8);axis.set_title(domain.upper())
        axis.set_xlabel('Recorded cost / GPT-4 (%)');axis.set_ylabel('Total reward / GPT-4 (%)')
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('OpenAI embeddings: full learning; mean of 3 fixed seeds')
    for ext in ('svg','png'):fig.savefig(output/('quality-cost.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis,domain in zip(axes,('routerbench','mmlu')):
        for frame,label,color in ((new,'OpenAI full text','#1155aa'),(old,'BGE 2048-token recipe','#cc5500')):
            values=frame.loc[(frame.domain==domain)&(frame.method=='pilot')&(frame.learning_fraction==1)].sort_values('budget_fraction')
            axis.errorbar(100*values.cost_fraction_gpt4_mean,100*values.quality_fraction_gpt4_mean,
                yerr=[100*(values.quality_fraction_gpt4_mean-values.quality_fraction_gpt4_min).clip(lower=0),
                      100*(values.quality_fraction_gpt4_max-values.quality_fraction_gpt4_mean).clip(lower=0)],
                marker='o',markersize=4,capsize=2,label=label,color=color)
        axis.set_title(domain.upper());axis.set_xlabel('Recorded cost / GPT-4 (%)')
        axis.set_ylabel('Total reward / GPT-4 (%)');axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle('PILOT representation recipes; bars show seed min-max')
    for ext in ('svg','png'):fig.savefig(output/('pilot-bge-openai.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis,domain in zip(axes,('routerbench','mmlu')):
        for method in methods:
            values=new.loc[(new.domain==domain)&(new.method==method)&(new.budget_fraction==.5)].sort_values('learning_fraction')
            axis.plot(100*values.learning_fraction,100*values.quality_fraction_gpt4_mean,
                marker='o',markersize=3,label=method,color=colors[method])
        axis.set_title(domain.upper());axis.set_xlabel('Nested learning prefix (%)')
        axis.set_ylabel('Total deployment reward / GPT-4 (%)');axis.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Learning-size experiment; fixed 50% estimated-budget allocation')
    for ext in ('svg','png'):fig.savefig(output/('learning-size.'+ext),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis,domain in zip(axes,('routerbench','mmlu')):
        for method in methods:
            values=learning.loc[(learning.domain==domain)&(learning.method==method)]
            group=values.groupby('learning_fraction',sort=True)
            axis.plot(group['rounds'].mean(),group['cumulative_empirical_oracle_regret'].mean(),
                marker='o',markersize=3,label=method,color=colors[method])
        axis.set_title(domain.upper());axis.set_xlabel('Observed learning queries')
        axis.set_ylabel('Cumulative empirical oracle regret');axis.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Learning regret at four nested snapshots; mean of 3 fixed seeds')
    for ext in ('svg','png'):fig.savefig(output/('learning-regret.'+ext),dpi=160)
    plt.close(fig)


def report_text(output,report,paired,means,base):
    lines=['# OpenAI core study — 2026-10-31','',
        'Completed858 primary deployment conditions,78 learning streams and312 snapshots. All78 groups independently replayed.',
        'No additional paid API calls. Historical BGE results are preserved.','',
        'Representation: real text-embedding-3-small,1536D, full original text; projection64D,30/40 epochs; seeds7,42,2026.',
        'Same8 admitted arms, partitions and stream order;1245 preference training and313 calibration pairs. Tuning-only selection.',
        'MMLU is a nested RouterBench subset, so the two domains are not independent datasets.','',
        '| Domain | Estimated budget fraction | PILOT quality / GPT-4 | Recorded cost / GPT-4 | Coverage |',
        '|---|---:|---:|---:|---:|']
    for r in report['focus_pilot']:
        lines.append(f"| {r['domain']} | {r['budget_fraction']:.0%} | {r['quality']:.2%} | {r['cost']:.2%} | {r['coverage']:.2%} |")
    lines += ['', 'Budget fractions refer to tuning-estimated GPT-4 cost; plotted costs use actual historical cost on all deployment queries.',
        'Quality is total reward divided by GPT-4 total reward on the same ALL requested queries. Unserved queries contribute zero reward.','',
        '## Paired representation comparison','',
        'OpenAI minus BGE, for PILOT full-learning conditions; all8 budget fractions and3 seeds are included:','']
    for domain in ('routerbench','mmlu'):
        frame=paired.loc[(paired.domain==domain)&(paired.method=='pilot')&(paired.learning_fraction==1)]
        lines.append(f"- {domain}: mean quality delta {100*frame.quality_fraction_gpt4_delta.mean():+.3f} percentage points; range {100*frame.quality_fraction_gpt4_delta.min():+.3f} to {100*frame.quality_fraction_gpt4_delta.max():+.3f}.")
    lines += ['', '## All-method comparison', '',
        'Ranking by total reward among all13 methods at full learning;3-seed means. Cost and coverage remain available in the CSV.', '',
        '| Domain | Estimated budget fraction | Best method | Best quality / GPT-4 | PILOT rank /13 |',
        '|---|---:|---|---:|---:|']
    for domain in ('routerbench','mmlu'):
        for fraction in (.5,1.):
            values=means.loc[(means.domain==domain)&(means.learning_fraction==1)&(means.budget_fraction==fraction)].sort_values('quality_fraction_gpt4_mean',ascending=False)
            best=values.iloc[0];pilot=values.loc[values.method=='pilot'].iloc[0]
            rank=1+int((values.quality_fraction_gpt4_mean>pilot.quality_fraction_gpt4_mean).sum())
            lines.append(f'| {domain} | {fraction:.0%} | {best.method} | {best.quality_fraction_gpt4_mean:.2%} | {rank} |')
    lines += ['', 'This includes fresh tuning and a different input representation recipe: BGE had a2048-token cap that truncated13 preference inputs; OpenAI uses full text. It does not isolate the encoder alone.',
        '', '## Limits', '',
        '- This remains an8-arm local study. Three model/version aliases remain unverified; no11-arm reproduction claim.',
        '-313 calibration pairs set initialization strengths; they are not an independent preference test.',
        '-86 raw task labels versus64 in the paper remain unresolved. No labels were removed to match the count.',
        '-30/40 training epochs and64D are declared local choices, not verified author hyperparameters.',
        '-Budget controller keeps the declared guarded affordability rule. Estimated-budget safety does not guarantee recorded costs never exceed allocation.',
        '-Three seeds give descriptive variability, not confidence intervals. Paper93%/25% is not a directly matched target.',
        '-No new OpenAI ablations,11-arm trials, Instructor-XL, HybridLLM or RouteLLM runs are included.',
        '', '## Evidence and reproduction', '',
        f'Raw immutable outputs: `{base.relative_to(ROOT).as_posix()}`. Protocol, input/cache/prior/code pins, tuning traces, learning snapshots, deployment traces and independent replay proofs are retained.',
        'Commands (PowerShell; one command at a time, one BLAS thread):', '', '```powershell',
        "$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'",
        '.venv/Scripts/python.exe -X utf8 -B -m tools.openai_study fit',
        '.venv/Scripts/python.exe -X utf8 -B -m tools.openai_study tune',
        '.venv/Scripts/python.exe -X utf8 -B -m tools.openai_study freeze',
        '.venv/Scripts/python.exe -X utf8 -B -m tools.openai_study run',
        '```', '', 'Completed artifacts are validated on resume; changed/partial artifacts are rejected. Reports require a fresh destination.', '']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'outputs/openai-core-20261031')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/openai-core-2026-10-31')
    args=parser.parse_args();execute(args.base.resolve(),args.output.resolve())


if __name__=='__main__':main()
