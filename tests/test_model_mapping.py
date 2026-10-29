import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.preference_data import prepare_preferences


def mappings():
    return json.loads(Path('configs/model_mapping.json').read_text())['mappings']


def pair(a, b):
    return pd.DataFrame([dict(id=1, model_a=a, model_b=b, winner_model_a=1,
        winner_model_b=0, winner_tie=0, prompt='["question"]',
        response_a='["answer A"]', response_b='["answer B"]')])


def test_aliases_are_opt_in_and_return_routerbench_names():
    frame = pair('claude-1', 'gpt-4-1106-preview')
    assert prepare_preferences(frame, mappings())[0].empty
    result, audit = prepare_preferences(frame, mappings(), mapping_policy='verified_alias')
    assert result.winner_arm.tolist() == ['claude-v1']
    assert audit['mapping_policy'] == 'verified_alias'


def test_identity_matches_need_stronger_explicit_policy():
    frame = pair('yi-34b-chat', 'wizardlm-13b')
    assert prepare_preferences(frame, mappings(), mapping_policy='verified_alias')[0].empty
    result, _ = prepare_preferences(frame, mappings(), mapping_policy='identity_match')
    assert result.arm_b.tolist() == ['WizardLM/WizardLM-13B-V1.2']


def test_unresolved_versions_stay_excluded_in_every_policy():
    for policy in ('exact_id', 'verified_alias', 'identity_match'):
        assert prepare_preferences(pair('claude-2.0', 'mistral-7b-instruct'),
                                   mappings(), mapping_policy=policy)[0].empty


def test_evidence_required_and_unknown_policy_rejected():
    config = copy.deepcopy(mappings())
    entry = next(m for m in config if m['arena'] == 'claude-1')
    entry['evidence'] = []
    with pytest.raises(ValueError, match='evidence'):
        prepare_preferences(pair('claude-1', 'gpt-4-1106-preview'), config,
                            mapping_policy='verified_alias')
    with pytest.raises(ValueError, match='policy'):
        prepare_preferences(pair('a', 'b'), mappings(), mapping_policy='all')
