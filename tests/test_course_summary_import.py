import json
import math

import pytest

from tools.course.tracking import _payload
from tools.course.import_summaries import build_batch, verify_values, import_batch


def record():
    return _payload(dict(experiment_family='historical-deployment', source_run_id='test/seed-7',
                         historical_import=True, paper_equivalent=False, synthetic=False,
                         counterfactual=False, seed=7),
                    dict(coverage=0.5, requested=10, served=5), 'lab1-reproduction')


def test_batch_preserves_aggregate_values_and_provenance_without_sdk_metadata():
    p = record()
    query, variables = build_batch([p], entity='team', project='project', approved=True)
    value = variables['r0']
    assert 'upsertBucket(input:$r0)' in query
    assert value['entityName'] == 'team' and value['modelName'] == 'project'
    assert value['name'] == p['id'] and value['state'] == 'running'
    assert json.loads(value['summaryMetrics']) == {'coverage': 0.5, 'requested': 10, 'served': 5}
    assert json.loads(value['config'])['seed'] == {'value': 7}
    assert value['groupName'] == 'lab1-reproduction'
    assert not {'host', 'jobProgram', 'jobRepo', 'commit'} & value.keys()


@pytest.mark.parametrize('change', ['approval', 'destination', 'private', 'duplicate'])
def test_invalid_or_unapproved_batch_is_rejected_before_writing(tmp_path, change):
    p = record()
    values = [p]
    if change == 'private': p['config']['api_key'] = 'never-send'
    if change == 'duplicate': values *= 2
    with pytest.raises(ValueError):
        import_batch(values, tmp_path, entity='../escape' if change == 'destination' else 'team',
                     project='project', approved=change != 'approval', execute=None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', ['metric', 'missing', 'config', 'state', 'nan'])
def test_readback_rejects_incomplete_or_changed_server_data(change):
    p = record()
    config = dict(p['config']); metrics = dict(p['metrics']); state = 'finished'
    if change == 'metric': metrics['served'] = 6
    if change == 'missing': del metrics['coverage']
    if change == 'config': config['seed'] = 8
    if change == 'state': state = 'running'
    if change == 'nan': metrics['coverage'] = float('nan')
    with pytest.raises(ValueError): verify_values(p, config, metrics, state)


def test_readback_allows_one_ulp_json_float_rounding():
    p = record()
    p['config']['budget'] = 7.6737414600000005
    config = dict(p['config'], budget=7.67374146)
    verify_values(p, config, p['metrics'], 'finished')


@pytest.mark.parametrize('actual', [7.67374, float('nan'), True, '7.67374146'])
def test_readback_rejects_changed_or_invalid_float_config(actual):
    p = record()
    p['config']['budget'] = 7.6737414600000005
    with pytest.raises(ValueError):
        verify_values(p, dict(p['config'], budget=actual), p['metrics'], 'finished')


def test_readback_rejects_rounding_beyond_one_ulp():
    p = record()
    p['config']['budget'] = 7.6737414600000005
    actual = math.nextafter(math.nextafter(p['config']['budget'], 0), 0)
    with pytest.raises(ValueError):
        verify_values(p, dict(p['config'], budget=actual), p['metrics'], 'finished')


def test_network_failure_retains_stable_planned_identity(tmp_path):
    p = record()
    def fail(query, variables): raise ConnectionError('simulated interruption')
    with pytest.raises(ConnectionError):
        import_batch([p], tmp_path, entity='team', project='project', approved=True,
                     execute=fail, complete=lambda _: None)
    receipt = json.loads((tmp_path/'online/team/project'/f"{p['id']}.json").read_text())
    assert receipt['status'] == 'planned'
    assert receipt['id'] == p['id'] and receipt['payload_sha256'] == p['payload_sha256']


def test_accepted_but_mismatched_result_never_gets_finished_receipt(tmp_path):
    p = record()
    def mismatch(query, variables):
        value = variables['r0']
        return {'r0': {'bucket': {'id': 'server-id', 'name': p['id'], 'state': 'finished',
                      'config': value['config'], 'summaryMetrics': '{"coverage":0.9}'}}}
    with pytest.raises(ValueError):
        import_batch([p], tmp_path, entity='team', project='project', approved=True,
                     execute=mismatch, complete=lambda _: None)
    receipt = json.loads((tmp_path/'online/team/project'/f"{p['id']}.json").read_text())
    assert receipt['status'] == 'accepted' and receipt['server_id'] == 'server-id'
