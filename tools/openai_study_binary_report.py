"""Immutable verified binary-router curves and shared-prior provenance."""
from pathlib import Path
import json

import pandas as pd

from scripts.experiments import expected_runs,load_artifact,verify_pins
from scripts.resources import require_resources
from tools.course.analyze import verify_package
from tools.openai_study import sealed
from tools.openai_study_followup_report import finish
from tools.openai_study_report import normalize_condition
from tools.paper_preflight.common import ROOT,sha256


def main():
    require_resources();base=ROOT/'outputs/openai-followup-20261031/binary'
    protocol=sealed(base/'protocol.json');verify_pins(protocol,ROOT)
    expected={r['id']:r for r in expected_runs(protocol)}
    tune=json.loads((base/'tuning-independent.json').read_text(encoding='utf-8'))
    assert tune['verified'] and tune['candidates']==18 and tune['protocol_sha256']==sha256(base/'protocol.json')
    assert tune['verifier_sha256']==sha256(ROOT/'tools/openai_study_binary_verify.py')
    verify_package(ROOT/'reports/openai-core-2026-10-31')
    reference=json.loads((ROOT/'reports/openai-core-2026-10-31/gpt4-reference.json').read_text(encoding='utf-8'))
    inventory={};groups={};prefixes=[]
    for path in (base/'verification').glob('*.json'):
        proof=json.loads(path.read_text(encoding='utf-8'));group=(proof['seed'],proof['method'])
        assert group not in groups and proof['verified'] and proof['verified_deployment_runs']==11
        assert proof['script_sha256']==sha256(ROOT/'tools/openai_study_binary_verify.py')
        assert proof['base_script_sha256']==sha256(ROOT/'tools/research/verify_runs.py') and proof['shared_parent_row_selection']
        learning_dir=base/'runs/learning/routerbench'/f'seed-{group[0]}'/'main'/group[1]
        learned=load_artifact(learning_dir);assert learned['manifest_sha256']==proof['learning_manifest_sha256']
        groups[group]=proof
        for row in proof['inventory']:
            assert row['id'] not in inventory;inventory[row['id']]=row['manifest_sha256']
        for fraction,snapshot in learned['snapshots'].items():
            prefixes.append(dict(seed=group[0],method=group[1],learning_fraction=float(fraction),
                **snapshot['metrics']))
    assert set(groups)=={(s,m) for s in (7,42,2026) for m in protocol['tuning']['routerbench']['selected_parameters']}
    assert set(inventory)==set(expected)
    actual_paths={p.parent.relative_to(base/'runs/deployment').as_posix():p.parent for p in (base/'runs/deployment').rglob('manifest.json')}
    assert set(actual_paths)==set(expected)
    rows=[]
    for identity,condition in expected.items():
        artifact=load_artifact(actual_paths[identity]);summary=artifact['summary'];budget=summary['budget'];metrics=summary['metrics']
        assert artifact['manifest_sha256']==inventory[identity]
        assert artifact['request']['lineage']['condition']==condition
        assert summary['deployment_split']['query_ids']==reference['routerbench']['query_ids']
        rows.append(normalize_condition(dict(condition,budget=budget['budget'],requested=summary['requested_queries'],
            served=summary['served_queries'],total_reward=metrics['total_reward'],actual_recorded_cost=budget['actual_total'],
            estimated_cost=budget['estimated_total'],actual_budget_excess=budget['actual_budget_excess_observed']),reference['routerbench']))
    frame=pd.DataFrame(rows)
    means=frame.groupby(['method','learning_fraction','budget_fraction']).mean(numeric_only=True).reset_index()
    prefix=pd.DataFrame(prefixes).drop(columns=['action_counts'])
    output=ROOT/'reports/openai-binary-2026-10-31';output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/'per-seed.csv',index=False,lineterminator='\n')
    means.to_csv(output/'means.csv',index=False,lineterminator='\n')
    prefix.to_csv(output/'learning-prefixes.csv',index=False,lineterminator='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),constrained_layout=True)
    labels={'fixed-00':'fixed GPT4','fixed-01':'fixed Llama2-70B'}
    for method,group in means.loc[means.learning_fraction==1].groupby('method'):
        group=group.sort_values('budget_fraction')
        axes[0].plot(100*group.cost_fraction_gpt4,100*group.quality_fraction_gpt4,marker='o',markersize=3,label=labels.get(method,method))
    axes[0].set_xlabel('Recorded cost / GPT4 (%)');axes[0].set_ylabel('True reward / GPT4 (%)');axes[0].set_title('Deployment quality versus cost')
    for method,group in means.loc[means.budget_fraction==.5].groupby('method'):
        group=group.sort_values('learning_fraction')
        axes[1].plot(100*group.learning_fraction,100*group.quality_fraction_gpt4,marker='o',markersize=3,label=labels.get(method,method))
    axes[1].set_xlabel('Learning stream used (%)');axes[1].set_ylabel('True reward / GPT4 (%)');axes[1].set_title('At50% estimated allocation')
    for method,group in prefix.groupby('method'):
        group=group.groupby('rounds').mean(numeric_only=True)
        axes[2].plot(group.index,group.cumulative_empirical_oracle_regret,marker='o',markersize=3,label=labels.get(method,method))
    axes[2].set_xlabel('Observed learning rounds');axes[2].set_ylabel('True empirical-oracle regret');axes[2].set_title('Binary-pool oracle')
    for axis in axes:axis.grid(alpha=.2)
    axes[2].legend(fontsize=7);fig.suptitle('RouterBench binary pool: GPT4-1106-preview / Llama2-70B-chat; mean three seeds')
    for ext in ('png','svg'):fig.savefig(output/('binary-routing.'+ext),dpi=160)
    plt.close(fig)
    lines=['# Verified binary routing on RouterBench','',
      'All231 conditions (seven methods x three seeds x eleven learning/budget settings) independently replayed, including all677670 selected-feedback learning rounds,84 nested snapshots and deployment decisions.',
      'Pool: GPT4-1106-preview and Llama2-70B-chat, a verified pair appearing in PILOT Appendix D.4.',
      'Prior: select their two original directions/lambdas from the shared8-arm preference fit; retain the same64D query projection. This is not a fresh two-arm preference fit or an author-identical prior.',
      'Fresh binary tuning:18 candidates on the original1000 tuning questions, independently replayed. PILOT alpha1, LinUCB alpha1, EpochGreedy window500. Shared positive ratio bounds come only from the selected candidates\' tuning predictions.',
      'Three32,270-round learning streams per method, with nested10/25/50/100% snapshots; no restarts for the smaller learning sizes. Same3,227 deployment questions and full-population GPT4 denominators as the primary study.',
      '', '| Method | Mean quality/GPT4 at50% allocation | Mean recorded cost/GPT4 | Mean coverage |',
      '|---|---:|---:|---:|']
    for r in means.loc[(means.learning_fraction==1)&(means.budget_fraction==.5)].itertuples():
        lines.append(f'|{labels.get(r.method,r.method)}|{100*r.quality_fraction_gpt4:.4f}%|{100*r.cost_fraction_gpt4:.4f}%|{100*r.coverage:.4f}%|')
    lines+=['','All compared methods use the same guarded bin controller, unchanged query order and historical prices. Skipped/unserved questions remain in denominators.',
      'Regret is measured against the oracle within these two arms. Do not compare its raw total with an eight-arm oracle total as if the comparator were unchanged.',
      'Original source verifier is preserved. A separate, hash-pinned wrapper selects the two parent theta rows by name and records both reference/wrapper hashes; all reference selection/update/budget equations are reused unchanged.',
      '', 'This is a local D.4 adaptation. It does not reproduce the unresolved GPT4-Mistral/Mixtral pairs or original D.1 HybridLLM comparison. No additional paid API requests.',
      '', 'Commands: tools.openai_study_binary_run declare -> tune -> freeze -> run, tools.openai_study_binary_verify tuning, then the verifier for each of21 seed/method groups. All commands run sequentially with one BLAS thread.']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(finish(output,dict(kind='verified-openai-binary-pair',conditions=231,groups=21,
        arms=protocol['arms'],parent_pretraining_arms=8,protocol_sha256=sha256(base/'protocol.json'),
        tuning_verification_sha256=sha256(base/'tuning-independent.json'),additional_paid_api_calls=0,paper_equivalent=False)),indent=2))


if __name__=='__main__':main()
