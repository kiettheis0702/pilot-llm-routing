"""Verified budget-policy tables/figures, retaining all unserved requests."""
import math
from pathlib import Path
import pandas as pd

from scripts.experiments import load_artifact, verify_pins
from scripts.resources import require_resources
from tools.course.analyze import verify_package
from tools.openai_study import checked_pin, sealed
from tools.openai_study_followup_report import finish
from tools.paper_preflight.common import ROOT, read_json, sha256


def main():
    require_resources();base=ROOT/'outputs/openai-followup-20261031/budget'
    output=ROOT/'reports/openai-budget-2026-10-31'
    if output.exists():raise FileExistsError(output)
    protocol=sealed(base/'protocol.json');verify_pins(protocol,ROOT)
    proof=read_json(base/'independent-verification.json')
    if (not proof['verified'] or not proof['complete'] or proof['conditions']!=240
            or proof['protocol_sha256']!=sha256(base/'protocol.json')
            or proof['script_sha256']!=sha256(ROOT/'tools/openai_study_budget_verify.py')):
        raise ValueError('Complete independent budget replay required')
    expected={r['id']:r for r in protocol['conditions']}
    found={p.parent.relative_to(base/'runs').as_posix():p.parent for p in (base/'runs').rglob('manifest.json')}
    verified={r['id']:r['manifest_sha256'] for r in proof['inventory']}
    if set(found)!=set(expected) or set(verified)!=set(expected):raise ValueError('Budget inventory differs')
    rows=[]
    for identity,condition in expected.items():
        value=load_artifact(found[identity]);s=value['summary'];r=s['relative']
        if value['manifest_sha256']!=verified[identity] or value['request']['condition']!=condition:
            raise ValueError('Budget proof/condition mismatch')
        source=read_json(ROOT/value['request']['input']['path'])
        source_hash=proof['input_sha256'][f"{condition['domain']}/{condition['seed']}"]
        if sha256(ROOT/value['request']['input']['path'])!=source_hash:raise ValueError('Budget inputs changed')
        truth={(v['query_id'],v['arm_id']):v['reward'] for v in source['outcomes']}
        mrr=0.
        with (found[identity]/'trace.jsonl').open(encoding='utf-8') as stream:
            import json
            for event in map(json.loads,stream):
                q,a=event['query_id'],event['arm_id']
                if a is not None:mrr+=1/(1+sum(truth[q,b]>truth[q,a] for b in source['arms']))
        rows.append(dict(condition,budget=s['budget'],served=s['served'],requested=s['requested'],
            total_reward=s['total_reward'],actual_recorded_cost=s['actual_cost'],estimated_cost=s['estimated_cost'],
            actual_budget_excess=max(0.,s['actual_cost']-s['budget']),estimated_budget_excess=s['estimated_overspend'],
            quality_fraction_gpt4=r['quality_fraction_gpt4'],cost_fraction_gpt4=r['cost_fraction_gpt4'],coverage=r['coverage'],
            unused_estimated_budget=s['budget']-s['estimated_cost'],empirical_mrr_requested=mrr/s['requested'],
            source='new-frozen-score-simulation',future_queries_used=condition['policy']=='retrospective-lagrange'))
    verify_package(ROOT/'reports/openai-ablations-2026-10-31')
    single=pd.read_csv(ROOT/'reports/openai-ablations-2026-10-31/per-seed.csv')
    single=single.loc[single.variant=='single-budget-bin']
    if len(single)!=48:raise ValueError('Expected48 reused one-bin conditions')
    for r in single.to_dict('records'):
        rows.append(dict(id=r['id'],domain=r['domain'],seed=r['seed'],policy='single-budget-bin',budget_fraction=r['budget_fraction'],
            **{k:r[k] for k in ('budget','served','requested','total_reward','actual_recorded_cost','estimated_cost','actual_budget_excess',
                'quality_fraction_gpt4','cost_fraction_gpt4','coverage')},estimated_budget_excess=0.,
            unused_estimated_budget=r['budget']-r['estimated_cost'],empirical_mrr_requested=None,
            source='reused-verified-ablation',future_queries_used=False))
    frame=pd.DataFrame(rows);pairs=[]
    lookup=frame.loc[frame.policy=='guarded-stop'].set_index(['domain','seed','budget_fraction'])
    for row in rows:
        reference=lookup.loc[row['domain'],row['seed'],row['budget_fraction']]
        pairs.append(dict(id=row['id'],reference_id=reference['id'],domain=row['domain'],seed=row['seed'],
            budget_fraction=row['budget_fraction'],policy=row['policy'],
            **{metric+'_delta':row[metric]-reference[metric] for metric in ('quality_fraction_gpt4','cost_fraction_gpt4','coverage')}))
    means=frame.groupby(['domain','policy','budget_fraction']).agg(
        quality=('quality_fraction_gpt4','mean'),cost=('cost_fraction_gpt4','mean'),coverage=('coverage','mean')).reset_index()
    output.mkdir(parents=True)
    frame.to_csv(output/'per-seed.csv',index=False,lineterminator='\n')
    means.to_csv(output/'means.csv',index=False,lineterminator='\n')
    pd.DataFrame(pairs).to_csv(output/'paired-main.csv',index=False,lineterminator='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis,domain in zip(axes,('routerbench','mmlu')):
        for policy,group in means.loc[means.domain==domain].groupby('policy'):
            group=group.sort_values('budget_fraction')
            axis.plot(100*group.cost,100*group.quality,marker='o',markersize=3,label=policy)
        axis.set_title(domain.upper());axis.set_xlabel('Recorded cost / GPT-4 (%)');axis.set_ylabel('Total reward / GPT-4 (%)');axis.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.suptitle('Fixed PILOT scores; six allocation policies, three paired seeds')
    for ext in ('png','svg'):fig.savefig(output/('allocation-comparison.'+ext),dpi=160)
    plt.close(fig)
    focus=means.loc[(means.domain=='mmlu')&means.budget_fraction.isin([.2,.3])]
    lines=['# OpenAI fixed-score budget comparison','',
        '240 new policy conditions independently replayed, plus48 reused single-bin ablation controls. Same frozen PILOT checkpoint, estimates, order and all-request GPT-4 denominators. No paid API calls.','',
        '| MMLU allocation | Policy | Quality/GPT-4 | Recorded cost/GPT-4 | Coverage |',
        '|---:|---|---:|---:|---:|']
    for r in focus.itertuples(index=False):lines.append(f'| {r.budget_fraction:.0%} | {r.policy} | {r.quality:.2%} | {r.cost:.2%} | {r.coverage:.2%} |')
    lines += ['',
        'Guarded-stop exactly matches all48 immutable core PILOT conditions, including chosen actions. Paper-literal-stop removes only the threshold affordability guard and retains termination.',
        'Guarded-skip preserves current-bin threshold/fallback, records a zero-reward unserved request when no arm qualifies and advances query time. Fixed-per-query caps each request atB/Q without transfer. Single-bin is the earlier controlled ablation, not new training.',
        'Retrospective-lagrange uses all future queries\' predicted scores and estimated costs to choose a feasible P-lambda*C allocation with abstention. It never uses true rewards for selection. This comparator has more information; no exact optimality/oracle guarantee is claimed.',
        'MRR uses historical reward ranks with minimum rank for ties; unserved requests contribute0. It is a declared local diagnostic, not a verified identical author metric.',
        'All variants remain8-arm local experiments. Higher coverage need not increase quality or save cost. Keep adverse results; no retroactive change to the main protocol.']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(finish(output,dict(complete=True,new_conditions=240,reused_conditions=48,independent_verified=True,
        protocol=checked_pin(base/'protocol.json'),verification=checked_pin(base/'independent-verification.json'),
        source_main=checked_pin(ROOT/'reports/openai-core-2026-10-31/package-files.json'),
        source_onebin=checked_pin(ROOT/'reports/openai-ablations-2026-10-31/package-files.json'),
        script_sha256=sha256(__file__),additional_paid_api_calls=0,paper_equivalent=False)))


if __name__=='__main__':main()
