"""Strict readback of OpenAI ablations, preference test and cost counterfactuals."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from scripts.experiments import load_artifact, verify_pins
from scripts.resources import require_resources
from tools.course.analyze import cost_sensitivity, verify_package
from tools.openai_study import checked_pin, once, sealed
from tools.openai_study_followup import BASE, CORE, addon_conditions, validate_parent
from tools.openai_study_report import normalize_condition
from tools.paper_preflight.common import ROOT, digest, read_json, sha256


def finish(output,metadata):
    metadata['files']={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()}
    metadata['manifest_sha256']=digest(metadata);once(output/'manifest.json',metadata)
    once(output/'package-files.json',dict(files={p.relative_to(output).as_posix():sha256(p)
        for p in output.rglob('*') if p.is_file() and p.name!='package-files.json'}))
    return verify_package(output)


def ablations(output):
    if output.exists():raise FileExistsError(output)
    protocol=sealed(BASE/'protocol.json');verify_pins(protocol,ROOT)
    validate_parent(sealed(CORE/'protocol.json'),protocol)
    verify_package(ROOT/'reports/openai-core-2026-10-31')
    expected={r['id']:r for r in addon_conditions(protocol)}
    found={p.parent.relative_to(BASE/'runs/deployment').as_posix():p.parent
        for p in (BASE/'runs/deployment').rglob('manifest.json')}
    if set(found)!=set(expected):raise ValueError('Ablation inventory incomplete or unexpected')
    replay={};groups={}
    for p in (BASE/'verification').glob('*.json'):
        v=read_json(p);group=(v['domain'],v['seed'],v['variant'])
        if group in groups or not v['verified'] or v['script_sha256']!=sha256(ROOT/'tools/research/verify_runs.py'):
            raise ValueError('Duplicate or stale independent replay')
        groups[group]=v
        for r in v['inventory']:
            if r['id'] in replay:raise ValueError('Duplicate replay condition')
            replay[r['id']]=r['manifest_sha256']
    if set(groups)!={(r['domain'],r['seed'],r['variant']) for r in expected.values()} or set(replay)!=set(expected):
        raise ValueError('Required ablation replay missing')
    reference=read_json(ROOT/'reports/openai-core-2026-10-31/gpt4-reference.json')
    main=pd.read_csv(ROOT/'reports/openai-core-2026-10-31/per-seed.csv').set_index('id')
    rows=[];pairs=[];new_rounds=0;learning_proofs={}
    for group,proof in groups.items():
        domain,seed,variant=group;source_variant='main' if variant=='single-budget-bin' else variant
        folder=BASE/'runs/learning'/domain/f'seed-{seed}'/source_variant/'pilot'
        learned=load_artifact(folder)
        if learned['manifest_sha256']!=proof['learning_manifest_sha256']:raise ValueError('Stale learning proof')
        if variant=='prior-only' and (learned['training_mode']!='none' or learned['metrics']['rounds']!=0):
            raise ValueError('Prior-only unexpectedly learned online')
        if variant not in ('single-budget-bin','prior-only'):new_rounds+=learned['metrics']['rounds']
        learning_proofs['/'.join(map(str,group))]=checked_pin(folder/'manifest.json')
    if new_rounds!=401751:raise ValueError('New observed learning-round count differs')
    for identity,condition in expected.items():
        v=load_artifact(found[identity]);request=v['request'];s=v['summary'];b=s['budget'];m=s['metrics']
        if (v['manifest_sha256']!=replay[identity] or request['lineage']['condition']!=condition
                or request['lineage']['protocol_sha256']!=sha256(BASE/'protocol.json')
                or s['deployment_split']['query_ids']!=reference[condition['domain']]['query_ids']
                or v['synthetic'] or b['estimated_budget_violation'] or not b['actual_costs_complete']):
            raise ValueError('Condition identity, population or budget verification failed')
        r=normalize_condition(dict(condition,budget=b['budget'],requested=s['requested_queries'],served=s['served_queries'],
            total_reward=m['total_reward'],actual_recorded_cost=b['actual_total'],estimated_cost=b['estimated_total'],
            actual_budget_excess=b['actual_budget_excess_observed']),reference[condition['domain']])
        rows.append(r)
        ref_id=f"{condition['domain']}/seed-{condition['seed']}/main/pilot/learn-1/budget-{condition['budget_fraction']:g}"
        ref=main.loc[ref_id]
        pairs.append(dict(id=identity,reference_id=ref_id,domain=condition['domain'],seed=condition['seed'],
            variant=condition['variant'],budget_fraction=condition['budget_fraction'],
            **{metric+'_delta':r[metric]-ref[metric] for metric in ('quality_fraction_gpt4','cost_fraction_gpt4','coverage')}))
    frame=pd.DataFrame(rows);paired=pd.DataFrame(pairs)
    means=frame.groupby(['domain','variant','budget_fraction']).agg(
        quality=('quality_fraction_gpt4','mean'),cost=('cost_fraction_gpt4','mean'),coverage=('coverage','mean')).reset_index()
    output.mkdir(parents=True)
    frame.to_csv(output/'per-seed.csv',index=False,lineterminator='\n')
    paired.to_csv(output/'paired-main.csv',index=False,lineterminator='\n')
    means.to_csv(output/'means.csv',index=False,lineterminator='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for axis,domain in zip(axes,('routerbench','mmlu')):
        for variant,group in means.loc[means.domain==domain].groupby('variant'):
            group=group.sort_values('budget_fraction');axis.plot(100*group.cost,100*group.quality,marker='o',markersize=3,label=variant)
        core=main.loc[(main.domain==domain)&(main.method=='pilot')&(main.learning_fraction==1)].groupby('budget_fraction').mean(numeric_only=True)
        axis.plot(100*core.cost_fraction_gpt4,100*core.quality_fraction_gpt4,'k--',label='main64D',linewidth=2)
        axis.set_title(domain.upper());axis.set_xlabel('Recorded cost / GPT-4 (%)');axis.set_ylabel('Total reward / GPT-4 (%)');axis.grid(alpha=.2)
    axes[1].legend(fontsize=7);fig.suptitle('Controlled OpenAI ablations; mean of three seeds')
    for ext in ('png','svg'):fig.savefig(output/('ablations.'+ext),dpi=160)
    plt.close(fig)
    table=paired.loc[paired.budget_fraction==.5].groupby(['domain','variant']).mean(numeric_only=True)
    lines=['# OpenAI controlled ablations','',
        'All240 conditions independently replayed in30 groups.18 new full learning streams,6 initial prior-only snapshots,6 reused main snapshots. New observed learning rounds:401751.',
        'Paired with the same seed/domain/budget main64D OpenAI condition. No ablation retuning; no new paid API calls.','',
        '| Domain | Variant | Quality delta at50% allocation (pp) | Coverage delta (pp) |','|---|---|---:|---:|']
    for (domain,variant),r in table.iterrows():lines.append(f'| {domain} | {variant} | {100*r.quality_fraction_gpt4_delta:+.3f} | {100*r.coverage_delta:+.3f} |')
    lines += ['', 'Zero arm-prior directions retain the preference-learned query projection and calibrated lambdas. Prior-only has zero online learning. Single-bin reuses the exact parent main checkpoint.8D retrains both stages; the base OpenAI vectors are unchanged.',
        'This remains an8-arm local study. Mean/min/max across fixed seeds are descriptive; no matched-paper or significance claim.']
    (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    proof=finish(output,dict(complete=True,conditions=240,independent_replays=30,additional_observed_learning_rounds=new_rounds,
        protocol=checked_pin(BASE/'protocol.json'),parent_core=checked_pin(CORE/'protocol.json'),
        prior_index=checked_pin(BASE/'prior-index.json'),run_manifest_sha256=replay,learning_proofs=learning_proofs,
        additional_paid_api_calls=0,paper_equivalent=False,script_sha256=sha256(__file__)))
    print('Ablation report',proof,flush=True)
    return frame


def preference(output):
    if output.exists():raise FileExistsError(output)
    base=ROOT/'outputs/openai-followup-20261031';proof=read_json(base/'preference500-verification.json')
    if not proof['verified'] or not proof['real_final_test_mutation_invariance']:raise ValueError('Preference evidence incomplete')
    verify_pins(proof,ROOT);output.mkdir(parents=True)
    records=[{k:r[k] for k in ('seed','correct','pairs','accuracy')} for r in proof['records']]
    pd.DataFrame(records).to_csv(output/'per-seed.csv',index=False,lineterminator='\n')
    selected=read_json(base/'preference500/seed-42/manifest.json')
    pd.DataFrame([dict(id=r['id'],projection_epochs=r['projection_epochs'],prior_epochs=r['prior_epochs'],
        calibration_correct=r['calibration']['correct'],calibration_pairs=r['calibration']['pairs'])
        for r in selected['candidates']]).to_csv(output/'calibration-candidates.csv',index=False,lineterminator='\n')
    once(output/'verification.json',proof)
    text=f'''# OpenAI preference500 supplementary evaluation

Train847 / calibration211 / test500, grouped by normalized prompt. Three fixed seeds7/42/2026:313/500,319/500,321/500; mean accuracy{proof['mean_accuracy']:.4%}.

Seed42 calibration: short15/20 and current30/40 tie at146/211; long60/80 gives144/211. Predeclared tie rule selects fewer epochs. Short configuration is frozen for all seeds. No test outcome changes lambda or the selected configuration.

Independent score/loss/lambda checks pass. Flipping all500 final-test labels and refitting gives the byte-identical selected checkpoint; accuracy changes313->187, confirming test-label separation.

Important:400 test pairs were training rows and100 were calibration/holdout rows in the older project. They are held out from these new fits, not unseen across the entire project history. Their identity is not verified against the authors'500 pairs. Do not compare63.53% with paper65% as matched reproduction error.

This supplementary protocol does not replace the main1245/313 protocol or the completed core results. No new API calls. Real OpenAI1536D base vectors are reused.
'''
    (output/'README.md').write_text(text,encoding='utf-8')
    print('Preference report',finish(output,dict(complete=True,seed_count=3,test_pairs=500,
        mean_accuracy=proof['mean_accuracy'],source_verification=checked_pin(base/'preference500-verification.json'),
        paper_equivalent=False,additional_paid_api_calls=0,script_sha256=sha256(__file__))),flush=True)


def costs(output,addon):
    if output.exists():raise FileExistsError(output)
    core=pd.read_csv(ROOT/'reports/openai-core-2026-10-31/per-seed.csv')
    full=pd.concat([core,addon],ignore_index=True)
    pilot=core.loc[(core.method=='pilot')&(core.learning_fraction==1)]
    factors=[.8,1.,1.1,1.25,1.5,2.]
    scratch=json.loads((ROOT/'outputs/openai-followup-20261031/cost-stress-powershell.json').read_text(encoding='utf-8-sig'))
    records=[]
    for name,frame in [('pilot-full',pilot),('all-core',core),('all-openai',full)]:
        values=cost_sensitivity(frame.to_dict('records'),factors)
        for value in values:
            value['population']=name
            if name!='all-openai':
                other=next(r for r in scratch if r['population']==name and r['factor']==value['factor'])
                if other['over_budget_count']!=value['over_budget_count'] or not math.isclose(other['maximum_excess'],value['maximum_excess'],abs_tol=1e-12):
                    raise ValueError('Independent PowerShell/Python cost-shock result mismatch')
            records.append(value)
    output.mkdir(parents=True);pd.DataFrame(records).to_csv(output/'cost-sensitivity.csv',index=False,lineterminator='\n')
    once(output/'cost-sensitivity.json',records)
    (output/'README.md').write_text('# OpenAI fixed-choice cost shocks\n\n'
        'No rerouting or retraining: multiply recorded costs by each factor while keeping chosen arms and rewards fixed. '
        'Population pilot-full has48 conditions; all-core858; all-openai1098. Counts are experiment conditions, not query-level probabilities. '
        'At1.10x,23/48 main PILOT conditions exceed allocation; at1.00x none do. These are benchmark units, not provider charges. '
        'The corresponding historical BGE result25/48 remains separately labelled BGE.\n',encoding='utf-8')
    print('Cost report',finish(output,dict(complete=True,records=len(records),choices_unchanged=True,
        independent_crosscheck=True,source_core=checked_pin(ROOT/'reports/openai-core-2026-10-31/package-files.json'),
        source_ablations=checked_pin(ROOT/'reports/openai-ablations-2026-10-31/package-files.json'),
        additional_paid_api_calls=0,script_sha256=sha256(__file__))),flush=True)


def main():
    require_resources()
    addon=ablations(ROOT/'reports/openai-ablations-2026-10-31')
    preference(ROOT/'reports/openai-preference500-2026-10-31')
    costs(ROOT/'reports/openai-cost-sensitivity-2026-10-31',addon)


if __name__=='__main__':main()
