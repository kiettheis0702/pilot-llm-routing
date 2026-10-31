"""A bounded replay must not accidentally certify an entire published grid."""
import copy
import pytest


def sample():
    published = [dict(id=name, served='2', total_reward='1.5',
        actual_recorded_cost='0.2', estimated_cost='0.25') for name in ('a','b')]
    proof = dict(verified=True, inventory=[dict(id=name, served=2, reward=1.5,
        actual_cost=.2, estimated_cost=.25) for name in ('b','a')])
    return [proof], published


def test_complete_reordered_inventory_matches_without_mutating_inputs():
    from tools.openai_study_full_reproduce import verify_full_inventory
    proofs, published = sample()
    original = copy.deepcopy((proofs,published))
    result = verify_full_inventory(proofs, published)
    assert result['verified'] and result['matches_published'] and result['conditions'] == 2
    assert (proofs,published) == original


@pytest.mark.parametrize('fault', ['missing','duplicate','unknown','unverified',
    'metric','nonfinite','duplicate_reference'])
def test_incomplete_or_inconsistent_replay_cannot_certify_full_grid(fault):
    from tools.openai_study_full_reproduce import verify_full_inventory
    proofs, published = sample()
    rows = proofs[0]['inventory']
    if fault == 'missing': rows.pop()
    elif fault == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif fault == 'unknown': rows[0]['id'] = 'foreign-condition'
    elif fault == 'unverified': proofs[0]['verified'] = False
    elif fault == 'metric': rows[0]['reward'] = 1.4
    elif fault == 'nonfinite': rows[0]['reward'] = float('nan')
    elif fault == 'duplicate_reference': published.append(copy.deepcopy(published[0]))
    with pytest.raises(ValueError):
        verify_full_inventory(proofs, published)
