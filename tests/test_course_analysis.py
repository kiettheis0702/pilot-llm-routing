import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.course.analyze import cost_sensitivity, magnitude_stress, verify_package


def test_cosine_loses_positive_magnitude_with_selected_only_updates():
    result = magnitude_stress(100, 1.0, (0.2, 0.8))
    cosine, linear = result['policies']
    assert cosine['low_count'] == cosine['high_count'] == 50
    assert cosine['regret'] == pytest.approx(30)
    assert linear['high_count'] > 95
    assert linear['regret'] < 3
    for policy in result['policies']:
        assert len(policy['trace']) == 100
        assert policy['total_reward'] == pytest.approx(sum(r['reward'] for r in policy['trace']))
        assert policy['update_count'] == 100
        assert all(set(r) == {'round', 'arm', 'reward', 'regret'} for r in policy['trace'])


@pytest.mark.parametrize('rounds,alpha,rewards', [(0,1,(.2,.8)), (True,1,(.2,.8)),
    (10,float('nan'),(.2,.8)), (10,1,(0,.8)), (10,1,(.9,.2)), (10,1,(.2,1.1))])
def test_invalid_stress_inputs_fail(rounds, alpha, rewards):
    with pytest.raises(ValueError): magnitude_stress(rounds, alpha, rewards)


def test_cost_intervention_preserves_original_data_and_accounts_excess():
    rows = [dict(id='a', budget=10., actual_recorded_cost=9., variant='main',
                 method='pilot', learning_fraction=1.)]
    before = json.dumps(rows)
    result = cost_sensitivity(rows, [1., 1.5])
    assert result[0]['over_budget_count'] == 0
    assert result[1]['over_budget_count'] == 1
    assert result[1]['maximum_excess'] == pytest.approx(3.5)
    assert json.dumps(rows) == before
    with pytest.raises(ValueError): cost_sensitivity(rows, [float('nan')])


def test_package_tampering_and_escape_are_rejected(tmp_path):
    f = tmp_path/'data.txt'
    f.write_text('verified')
    manifest = {'files': {'data.txt': hashlib.sha256(f.read_bytes()).hexdigest()}}
    (tmp_path/'package-files.json').write_text(json.dumps(manifest))
    assert verify_package(tmp_path)['file_count'] == 1
    f.write_text('changed')
    with pytest.raises(ValueError, match='checksum'): verify_package(tmp_path)
    manifest['files'] = {'../escape': '0'*64}
    (tmp_path/'package-files.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='escapes'): verify_package(tmp_path)


def test_unlisted_evidence_cannot_bypass_package_verification(tmp_path):
    (tmp_path/'safe.txt').write_text('ok')
    (tmp_path/'package-files.json').write_text(json.dumps({'files': {
        'safe.txt': hashlib.sha256(b'ok').hexdigest()}}))
    (tmp_path/'analysis.json').write_text('{"unverified": true}')
    with pytest.raises(ValueError, match='inventory'): verify_package(tmp_path)


@pytest.mark.parametrize('key,value', [('seed',7),('lambda',9),('context',[0,1]),
    ('prior_vectors',[[0,0],[0,0]]),('arms',['a','b']),('horizons',[100]),('alphas',[.5])])
def test_protocol_rejects_unexecuted_parameters_and_missing_plot_case(key,value):
    from tools.course import analyze
    protocol=json.loads(Path('configs/course_analysis_protocol.json').read_text())
    protocol['synthetic_magnitude_stress'][key]=value
    with pytest.raises(ValueError): analyze.validate_protocol(protocol)
