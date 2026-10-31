"""Publish a local immutable matched-preference comparison after readback."""
from pathlib import Path
import json

import pandas as pd

from scripts.resources import require_resources
from tools.course.analyze import verify_package
from tools.openai_study_followup_report import finish
from tools.paper_preflight.common import sha256


def main():
    require_resources();root=Path.cwd();base=root/'outputs/openai-followup-20261031/routellm'
    proof=json.loads((base/'independent-verification.json').read_text(encoding='utf-8'))
    protocol=json.loads((base/'protocol.json').read_text(encoding='utf-8'))
    assert proof['verified'] and proof['complete'] and proof['actual_test_label_mutation_weights_identical']
    assert proof['verifier_sha256']==sha256(root/'tools/openai_study_routellm_verify.py')
    assert proof['protocol_sha256']==sha256(base/'protocol.json')
    verify_package(root/'reports/openai-preference500-2026-10-31')
    for folder,p in proof['proofs'].items():
        assert sha256(base/folder/'weights.npz')==p['weights_sha256']
        assert sha256(base/folder/'selection.json')==p['selection_sha256']
    pilot=pd.read_csv(root/'reports/openai-preference500-2026-10-31/per-seed.csv')
    mf=pd.read_csv(base/'per-seed.csv')
    assert set(mf.seed)==set(pilot.seed)=={7,42,2026}
    paired=pilot.merge(mf,on='seed',suffixes=('_pilot','_routellm'),validate='one_to_one')
    paired['accuracy_delta_pp']=100*(paired.accuracy_routellm-paired.accuracy_pilot)
    output=root/'reports/openai-routellm-2026-10-31'
    output.mkdir(parents=True,exist_ok=False)
    paired.to_csv(output/'paired-preference500.csv',index=False,lineterminator='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.4,4.5),constrained_layout=True)
    means=[100*pilot.accuracy.mean(),100*mf.accuracy.mean()]
    ax.bar([0,1],means,color=['#3479b9','#d76c32'],width=.55,alpha=.85)
    for x,data in enumerate([pilot,mf]):
        ax.scatter([x-.09,x,x+.09],100*data.sort_values('seed').accuracy,color='#232d39',zorder=3,s=26)
        ax.text(x,means[x]+2,f'{means[x]:.2f}%',ha='center',fontsize=11)
    ax.set_xticks([0,1],['PILOT preference model','RouteLLM matrix factorization'])
    ax.set_ylim(0,100);ax.set_ylabel('Correct winner predictions (%)')
    ax.set_title('Same local test500 / cached OpenAI vectors / three seeds')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for ext in ('png','svg'):fig.savefig(output/('preference500-comparison.'+ext),dpi=160)
    plt.close(fig)
    lines=['# RouteLLM MF versus PILOT on the same local preference500 split','',
      'This is the paper preference-prediction experiment, not a RouteLLM budgeted routing curve.',
      'Train847 / calibration211 / test500, grouped by prompt, using the same real cached text-embedding-3-small vectors.',
      'The three seeds reuse the same500 pairs; they are not1500 independent test examples.','',
      '| Seed | PILOT correct /500 | RouteLLM MF correct /500 | MF minus PILOT (pp) | Selected MF epoch |',
      '|---:|---:|---:|---:|---:|']
    for row in paired.sort_values('seed').itertuples():
        lines.append(f'|{row.seed}|{row.correct_pilot}|{row.correct_routellm}|{row.accuracy_delta_pp:+.2f}|{row.selected_epoch}|')
    lines+=['',f'Mean PILOT{means[0]:.4f}%; RouteLLM MF{means[1]:.4f}%; difference{means[1]-means[0]:+.4f}percentage points.',
      '', 'The official training class is extracted from pinned RouteLLM commit'+protocol['source_commit']+'.',
      'Architecture:64 model rows,128 latent dimensions, frozen1536D prompt table, normalized model directions, unnormalized linear text projection, unbiased scalar interaction head.',
      'Official example settings:100epochs, batch64, Adam learning rate0.0003, weight decay0.00001, training noise alpha0.1.',
      'Adaptations: matched three-way split instead of upstream95/5; choose minimum noise-free calibration BCE (earliest exact tie), save that checkpoint, and never select on the final test.',
      'The upstream example evaluator calls its noisy forward when calculating loss. This local adaptation evaluates calibration with test=True, stated explicitly.',
      '', 'Independent NumPy reconstruction checks all1558 predictions per seed, selected calibration BCE, source/cache/weight pins and recorded split separation.',
      'An actual seed42 refit after swapping all500 test winner/loser labels produces elementwise identical selected weights; test accuracy340/500 becomes160/500.',
      'Inference/training network connections are denied; additional inference API calls0.',
      '', 'Limits: these500 rows were seen in historical project fits; this is a fresh local resplit, not author-identical or project-wide unseen. The preference pool contains8 verified models rather than the paper11. Do not report a matched reproduction error against the paper accuracy.',
      '', 'Primary source: [official MF training example](https://github.com/lm-sys/RouteLLM/blob/'+protocol['source_commit']+'/routellm/routers/matrix_factorization/train_matrix_factorization.py).',
      '', 'Commands (sequential, one BLAS/Torch thread):',
      '```powershell',
      '.venv/Scripts/python.exe -m tools.openai_study_routellm_train freeze',
      '.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 7',
      '.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 42',
      '.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 2026',
      '.venv-embedding/Scripts/python.exe -m tools.openai_study_routellm_train run --seed 42 --mutation-probe',
      '.venv/Scripts/python.exe -m tools.openai_study_routellm_verify',
      '.venv/Scripts/python.exe -m tools.openai_study_routellm_report',
      '```','', 'Freeze refuses an existing output directory; this report and raw runs are immutable.']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    metadata=dict(kind='routellm-mf-local-matched500',protocol_sha256=sha256(base/'protocol.json'),
        independent_verification_sha256=sha256(base/'independent-verification.json'),
        primary_sources=protocol['sources'],seeds=protocol['seeds'],
        mean_accuracy=dict(pilot=means[0]/100,routellm=means[1]/100),new_api_calls=0,
        adaptations=protocol['adaptations'],test_identity_limit=protocol['historical_test_overlap'])
    print(json.dumps(finish(output,metadata),indent=2))


if __name__=='__main__':main()
