"""Controlled add-on experiments must preserve their exact core references."""
import copy
import json
from pathlib import Path

import pytest

from scripts.experiments import expected_runs, learn_stream
from scripts.deployment_adapter import FrozenDeploymentRouter
from scripts.policies import LinearUCB
from test_experiments import small_inputs


def declarations():
    core=json.loads(Path('configs/frozen_experiment_protocol.json').read_text(encoding='utf-8'))
    core['ablation_plan']=dict(variants=[],budget_fractions=[],
        inventory=dict(primary_plus_ablation_deployment_runs=858))
    ablation=json.loads(Path('configs/ablation_plan.json').read_text(encoding='utf-8'))
    return core,ablation


def test_addon_inventory_has_all240_conditions_and_no_core_changes():
    from tools.openai_study_followup import declaration, addon_conditions
    core,ablation=declarations();before=copy.deepcopy(core)
    addon=declaration(core,ablation)
    rows=addon_conditions(addon)
    assert len(rows)==240 and len(expected_runs(addon))==1098
    assert {r['family'] for r in rows}=={'ablation'}
    assert len({(r['domain'],r['seed'],r['variant']) for r in rows})==30
    assert all(r['dimension']==(8 if r['variant']=='projection-dimension-8' else 64) for r in rows)
    assert core==before and addon['tuning']==core['tuning']


@pytest.mark.parametrize('dimension',[8,64])
def test_prior_index_selects_exact_seed_and_dimension(dimension):
    from tools.openai_study_followup import select_prior
    entries=[dict(seed=seed,dimension=d,prior={'path':f'{seed}-{d}'}) for seed in (7,42) for d in (8,64)]
    assert select_prior(dict(fits=entries),7,dimension)['prior']['path']==f'7-{dimension}'
    with pytest.raises(ValueError):select_prior(dict(fits=entries),2026,dimension)
    with pytest.raises(ValueError):select_prior(dict(fits=entries+entries),7,dimension)


@pytest.mark.parametrize('field',['arms','tuning','primary_plan'])
def test_addon_cannot_silently_change_main_controls(field):
    from tools.openai_study_followup import declaration, validate_parent
    core,ablation=declarations();addon=declaration(core,ablation)
    validate_parent(core,addon)
    addon[field]=[] if field=='arms' else {}
    with pytest.raises(ValueError):validate_parent(core,addon)


def test_reused_learning_is_byte_identical_resumable_and_rejects_corruption(tmp_path):
    from tools.openai_study_followup import copy_learning
    source=tmp_path/'source';destination=tmp_path/'copy'
    values=small_inputs();values['lineage']['protocol_sha256']='parent-core'
    result=learn_stream(model=LinearUCB(('a','b'),2),output_dir=source,**values)
    before={p.name:p.read_bytes() for p in source.iterdir()}
    copied=copy_learning(source,destination,expected_parent='parent-core')
    assert copied==result and {p.name:p.read_bytes() for p in destination.iterdir()}==before
    frozen=FrozenDeploymentRouter.load_checkpoint(destination/result['snapshots']['1']['file'])
    assert frozen.source_state()['rounds']==5
    assert copy_learning(source,destination,expected_parent='parent-core')==result
    with pytest.raises(ValueError):copy_learning(source,tmp_path/'wrong',expected_parent='other-parent')
    (destination/'trace.jsonl').write_text('corrupted')
    with pytest.raises(ValueError):copy_learning(source,destination,expected_parent='parent-core')
    assert {p.name:p.read_bytes() for p in source.iterdir()}==before


def test_incomplete_reused_learning_is_never_overwritten(tmp_path):
    from tools.openai_study_followup import copy_learning
    source=tmp_path/'source';destination=tmp_path/'copy'
    values=small_inputs();values['lineage']['protocol_sha256']='parent-core'
    learn_stream(model=LinearUCB(('a','b'),2),output_dir=source,**values)
    destination.mkdir();(destination/'trace.jsonl').write_text('partial')
    with pytest.raises(ValueError):copy_learning(source,destination,expected_parent='parent-core')
    assert (destination/'trace.jsonl').read_text()=='partial'
