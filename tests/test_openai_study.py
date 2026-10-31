"""Core-only OpenAI study inventory, independent prior checks and metric safety."""
import copy
import json
from pathlib import Path

import pytest

from scripts import pretraining as pre
from scripts.experiments import expected_runs
from test_pretraining import fixture, ARMS, SIZES, SOURCE


def test_core_inventory_has_858_conditions_without_changing_old_protocol():
    from tools.openai_study import core_declaration
    original = json.loads(Path('configs/frozen_experiment_protocol.json').read_text(encoding='utf-8'))
    before = copy.deepcopy(original)
    core = core_declaration(original)
    rows = expected_runs(core)
    assert len(rows) == 858
    assert {r['family'] for r in rows} == {'primary'}
    assert {r['dimension'] for r in rows} == {64}
    assert len({(r['domain'],r['seed'],r['method']) for r in rows}) == 78
    assert core['primary_plan']['pretraining']['projection_epochs'] == 30
    assert core['primary_plan']['pretraining']['prior_epochs'] == 40
    assert original == before


def test_independent_prior_check_recomputes_losses_and_calibration():
    from tools.openai_study import verify_prior
    cache, rows = fixture()
    prior = pre.fit_preference_prior(cache, rows, ARMS, SIZES, SOURCE, dimension=2,
        projection_epochs=3, prior_epochs=5, seed=7)
    result = verify_prior(prior, cache, rows)
    assert result['verified'] is True
    assert result['calibration_pairs'] == 2
    assert result['train_only_triplets'] is True


@pytest.mark.parametrize('changed', ['calibration', 'loss', 'triplet', 'cache'])
def test_independent_prior_check_rejects_mismatched_evidence(changed):
    from tools.openai_study import verify_prior
    cache, rows = fixture()
    prior = pre.fit_preference_prior(cache, rows, ARMS, SIZES, SOURCE, dimension=2,
        projection_epochs=3, prior_epochs=5, seed=7)
    if changed == 'calibration':
        prior['metrics']['validation']['per_arm'][ARMS[0]]['correct'] += 1
    if changed == 'loss': prior['metrics']['projection_losses'][-1] += .5
    if changed == 'triplet': prior['data']['sampled_triplets'][0][0] = 'q6'
    if changed == 'cache': prior['embedding']['cache_sha256'] = 'not-this-cache'
    with pytest.raises(ValueError): verify_prior(prior, cache, rows)


def test_pins_reject_changed_sources_before_reuse(tmp_path):
    from tools.openai_study import checked_pin
    source = tmp_path/'source.json'
    source.write_text('original')
    pin = checked_pin(source, tmp_path)
    source.write_text('changed')
    from scripts.experiments import verify_pins
    with pytest.raises(ValueError, match='checksum'):
        verify_pins(pin, tmp_path)
    with pytest.raises(ValueError): checked_pin(tmp_path.parent/'outside', tmp_path)


def test_reporting_uses_all_requested_gpt4_queries():
    from tools.openai_study_report import normalize_condition
    row = dict(total_reward=1.,actual_recorded_cost=2.,requested=4,served=1,budget=3.)
    result = normalize_condition(row,dict(count=4,reward=2.,cost=8.))
    assert result['quality_fraction_gpt4'] == .5
    assert result['cost_fraction_gpt4'] == .25
    assert result['coverage'] == .25
    assert result['allocated_budget_fraction_gpt4'] == .375
    with pytest.raises(ValueError): normalize_condition(row,dict(count=3,reward=2.,cost=8.))


def test_representation_comparison_requires_exact_paired_core_conditions():
    from tools.openai_study_report import paired_comparison
    import pandas as pd
    old = pd.DataFrame([dict(id='a',quality_fraction_gpt4=.8,cost_fraction_gpt4=.3,
        coverage=1.,domain='routerbench',method='pilot',seed=7,
        learning_fraction=1.,budget_fraction=.5)])
    new=old.copy();new['quality_fraction_gpt4']=.9
    result=paired_comparison(new,old)
    assert result.iloc[0].quality_fraction_gpt4_delta == pytest.approx(.1)
    with pytest.raises(ValueError):paired_comparison(new,old.iloc[:0])
    with pytest.raises(ValueError):paired_comparison(pd.concat([new,new]),old)


def test_mutated_stage_is_rejected_before_resume(tmp_path):
    from tools.openai_study import sealed
    value=dict(selected_parameters={'pilot':{'alpha':.1}})
    value['manifest_sha256']=pre._digest(value)
    path=tmp_path/'stage.json';path.write_text(json.dumps(value))
    assert sealed(path)==value
    value['selected_parameters']['pilot']['alpha']=10.
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='checksum'):sealed(path)


def test_tuning_lineage_cannot_be_relabelled_after_cache_or_prior_change():
    from tools.openai_study import verify_tuning_lineage
    old={'routing_cache_sha256':'old-cache','prior_file_sha256':'old-prior'}
    stage={'lineage':old};search={'provenance':{'lineage':old}}
    verify_tuning_lineage(stage,search,old)
    with pytest.raises(ValueError,match='lineage'):
        verify_tuning_lineage(stage,search,dict(old,routing_cache_sha256='new-cache'))
    with pytest.raises(ValueError,match='lineage'):
        verify_tuning_lineage(stage,search,dict(old,prior_file_sha256='new-prior'))


def test_report_accepts_actual_snapshot_rounds_field():
    from tools.openai_study_report import snapshot_record
    lineage=dict(domain='routerbench',seed=7,method='pilot')
    snapshot=dict(rounds=3227,metrics=dict(rounds=3227,total_reward=1250.))
    row=snapshot_record(lineage,'0.1',snapshot)
    assert row['rounds']==3227 and row['total_reward']==1250.
    with pytest.raises(ValueError):snapshot_record(lineage,'0.1',dict(snapshot,rounds=3228))
