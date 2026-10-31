"""Immutable D.5-style local noise report after full independent replay."""
from pathlib import Path
import json

import pandas as pd

from scripts.experiments import load_artifact
from scripts.resources import require_resources
from tools.openai_study_followup_report import finish
from tools.paper_preflight.common import sha256


def main():
    require_resources();root=Path.cwd();base=root/'outputs/openai-followup-20261031/noise'
    protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    proof=json.loads((base/'independent-verification.json').read_text(encoding='utf-8'))
    assert proof['verified'] and proof['complete'] and proof['conditions']==12
    assert proof['protocol_sha256']==sha256(base/'protocol.json')
    assert proof['verifier_sha256']==sha256(root/'tools/openai_study_noise_verify.py')
    verified={r['id']:r['manifest_sha256'] for r in proof['inventory']};rows=[]
    for condition in protocol['conditions']:
        artifact=load_artifact(base/'runs'/condition['id']);summary=artifact['summary'];learning=artifact['learning']
        assert verified[condition['id']]==artifact['manifest_sha256']
        rows.append(dict(condition,requested=summary['requested'],served=summary['served'],
            total_reward=summary['total_reward'],actual_recorded_cost=summary['actual_cost'],
            estimated_cost=summary['accounting']['estimated_total'],
            learning_rounds=learning['rounds'],noise_replaced=learning['noise_replaced'],
            noise_changed=learning['noise_changed'],learning_true_reward=learning['total_reward'],
            learning_true_oracle_regret=learning['cumulative_empirical_oracle_regret'],
            **{k:summary['relative'][k] for k in ('service_utility','quality_fraction_gpt4','cost_fraction_gpt4','coverage')}))
    frame=pd.DataFrame(rows);control=frame.loc[frame.epsilon==0].set_index('seed')
    frame['average_reward_delta_vs_zero']=[r.service_utility-control.loc[r.seed,'service_utility'] for r in frame.itertuples()]
    means=frame.groupby('epsilon').mean(numeric_only=True).reset_index()
    output=root/'reports/openai-noise-2026-10-31';output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/'per-seed.csv',index=False,lineterminator='\n')
    means.to_csv(output/'means.csv',index=False,lineterminator='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),constrained_layout=True)
    for ax,column,title in zip(axes,['service_utility','coverage'],['True reward / all requested','Coverage / all requested']):
        for seed,group in frame.groupby('seed'):
            group=group.sort_values('epsilon');ax.plot(100*group.epsilon,100*group[column],marker='o',label=f'seed{seed}')
        ax.set_title(title);ax.set_xlabel('Selected-feedback replacement probability (%)')
        ax.set_ylabel('Percent');ax.set_ylim(0,100);ax.grid(alpha=.2)
    axes[1].legend();fig.suptitle('OpenAI MMLU: local noise experiment, historical budget1')
    for ext in ('png','svg'):fig.savefig(output/('noise-sensitivity.'+ext),dpi=160)
    plt.close(fig)
    lines=['# Local MMLU feedback-noise sensitivity','',
      'All12 conditions (three seeds x0/1/3/5% replacement) independently reconstructed, including every learning selection, update and deployment budget decision.',
      'The0-noise stream matches the immutable main PILOT state/actions exactly for each seed.',
      'Deployment budget is1 historical accounting dollar. It is neither a paid API request nor1xGPT4 normalized budget.',
      'Only the chosen learning reward is replaced with an independent fair binary value; true outcomes are retained for reward/regret evaluation. Replacement probability differs from flip probability.',
      '', '| Replacement probability | Mean true reward / requested | Mean quality / GPT4 | Mean coverage | Mean true-reward change (pp) |',
      '|---:|---:|---:|---:|---:|']
    for r in means.itertuples():
        lines.append(f'|{100*r.epsilon:.0f}%|{100*r.service_utility:.4f}%|{100*r.quality_fraction_gpt4:.4f}%|{100*r.coverage:.4f}%|{100*r.average_reward_delta_vs_zero:+.4f}|')
    lines+=['', 'A missing/unserved query contributes zero reward and remains in every denominator. Actual replacement and changed-label counts are separately recorded per seed; an unflipped replacement still counts as replaced.',
      'Hyperparameters, prior, original stream order, tuning-only costs and guarded stop controller are unchanged. No noise setting is chosen from deployment quality.',
      'Report individual seeds and coverage alongside means; a curve may reflect an earlier/later budget stop as well as changed ranking.',
      '', 'This is a local adaptation of [PILOT Appendix D.5](https://aclanthology.org/2025.findings-emnlp.1301.pdf), using8 verified arms and local split/order/prices. It does not verify the paper Table5 values or its claimed less-than4% drop.',
      '', 'Commands, sequential with one BLAS thread:',
      '```powershell',
      '.venv/Scripts/python.exe -m tools.openai_study_noise freeze',
      '.venv/Scripts/python.exe -m tools.openai_study_noise run',
      '.venv/Scripts/python.exe -m tools.openai_study_noise_verify',
      '.venv/Scripts/python.exe -m tools.openai_study_noise_report',
      '```','', 'No additional paid API calls. Raw runs and this report are immutable.']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(finish(output,dict(kind='openai-local-mmlu-noise',conditions=12,
        protocol_sha256=sha256(base/'protocol.json'),verification_sha256=sha256(base/'independent-verification.json'),
        zero_noise_matches_main=True,inference_api_calls=0,paper_equivalent=False)),indent=2))


if __name__=='__main__':main()
