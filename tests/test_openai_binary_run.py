import copy
import importlib
import numpy as np
import pytest


def parent():
    return dict(arms=['other','gpt-4-1106-preview','meta/llama-2-70b-chat'],dimension=2,
        theta=np.array([[8.,8.],[1.,0.],[0.,1.]]),embedding=dict(cache_sha256='synthetic'),
        metrics=dict(validation=dict(per_arm={'other':{'lambda':7.},
            'gpt-4-1106-preview':{'lambda':2.},'meta/llama-2-70b-chat':{'lambda':3.}})))


def test_binary_policy_never_selects_outside_pair_and_preserves_parent_prior():
    module=importlib.import_module('tools.openai_study_binary_run')
    prior=parent();before=copy.deepcopy(prior)
    model=module.binary_policy(prior,'pilot',{'alpha':0.})
    assert model.arms==('gpt-4-1106-preview','meta/llama-2-70b-chat')
    assert model.select([1.,0.])=='gpt-4-1106-preview'
    assert model.select([0.,1.])=='meta/llama-2-70b-chat'
    np.testing.assert_array_equal(model.A[:,0,0],[2.,3.])
    np.testing.assert_array_equal(model.b,[[2.,0.],[0.,3.]])
    np.testing.assert_array_equal(prior['theta'],before['theta'])
    with pytest.raises(ValueError):module.binary_policy(prior,'fixed-02',{'arm_id':'other'})


def test_binary_declaration_has_all_seven_methods_and_three_seed_conditions():
    module=importlib.import_module('tools.openai_study_binary_run')
    from scripts.experiments import expected_runs
    # A tracked input is required: this test also runs in a clean clone.
    import json
    from pathlib import Path
    core=json.loads(Path('configs/frozen_experiment_protocol.json').read_text(encoding='utf-8'))
    declaration=module.binary_declaration(core)
    conditions=expected_runs(declaration)
    assert len(conditions)==231
    assert {r['method'] for r in conditions}=={'pilot','linucb','epoch-greedy','explore-only','random','fixed-00','fixed-01'}
    assert {r['domain'] for r in conditions}=={'routerbench'}
    assert {r['seed'] for r in conditions}=={7,42,2026}
    assert core['arms']!=declaration['arms']
