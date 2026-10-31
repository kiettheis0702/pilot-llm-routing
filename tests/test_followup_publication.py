import json

import pytest

from tools.openai_followup_tracking import make_record
from tools.openai_followup_publish import build_batch, import_batch, assert_capacity


def record():
    return make_record('openai','binary','fixture',dict(seed=42,arm_count=2),
                       dict(coverage=1.,requested=2,served=2),'0'*64)


def response(item, *, bad_metrics=False):
    metrics=dict(item['metrics'])
    if bad_metrics:
        metrics['coverage']=.5
    return {'r0':{'bucket':dict(id='server-identity',name=item['id'],state='running',
        config=json.dumps({k:dict(value=v) for k,v in item['config'].items()}),summaryMetrics=json.dumps(metrics))}}


def test_followup_batch_requires_authorization_and_keeps_recipe_pair_tags():
    item=record()
    with pytest.raises(ValueError):
        build_batch([item],approved=False)
    query,variables=build_batch([item],approved=True)
    assert 'upsertBucket' in query
    assert 'arms-2' in variables['r0']['tags']
    assert 'recipe-openai' in variables['r0']['tags']
    assert 'local-eight-arm' not in variables['r0']['tags']


def test_upload_transport_failure_keeps_planned_receipt_before_any_retry(tmp_path):
    item=record()
    def broken(*args):
        raise ConnectionError('simulated connection failure')
    with pytest.raises(ConnectionError):
        import_batch([item],tmp_path,approved=True,execute=broken,complete=lambda r:None)
    receipt=json.loads((tmp_path/(item['id']+'.json')).read_text())
    assert receipt['status']=='planned' and receipt['payload_sha256']==item['payload_sha256']


def test_server_acceptance_is_saved_before_rejecting_wrong_returned_metrics(tmp_path):
    item=record();completed=[]
    with pytest.raises(ValueError):
        import_batch([item],tmp_path,approved=True,execute=lambda *a:response(item,bad_metrics=True),complete=completed.append)
    receipt=json.loads((tmp_path/(item['id']+'.json')).read_text())
    assert receipt['status']=='accepted' and receipt['server_id']=='server-identity'
    assert completed==[]


def test_retry_failure_does_not_erase_previously_accepted_server_identity(tmp_path):
    item=record()
    with pytest.raises(ValueError):
        import_batch([item],tmp_path,approved=True,
                     execute=lambda *a:response(item,bad_metrics=True),complete=lambda r:None)
    def disconnected(*args):
        raise ConnectionError('simulated retry disconnection')
    with pytest.raises(ConnectionError):
        import_batch([item],tmp_path,approved=True,execute=disconnected,complete=lambda r:None)
    receipt=json.loads((tmp_path/(item['id']+'.json')).read_text())
    assert receipt['server_id']=='server-identity'


def test_completion_transport_does_not_claim_fresh_server_verification(tmp_path):
    item=record();completed=[]
    result=import_batch([item],tmp_path,approved=True,execute=lambda *a:response(item),complete=completed.append)
    assert len(completed)==1 and result[0]['status']=='uploaded'
    assert json.loads((tmp_path/(item['id']+'.json')).read_text())['status']=='uploaded'


class CapacityAPI:
    def __init__(self, *, upgrade=False, storage=1000, expiry='2099-01-01T00:00:00Z'):
        self._service_api=self
        self.upgrade=upgrade;self.storage=storage;self.expiry=expiry

    def execute_graphql(self, query, variables):
        return dict(entity=dict(name=variables['name'],storageBytes=self.storage,
            organization=dict(subscriptions=[dict(status='ENABLED',isAutomaticUpgrade=self.upgrade,
                expiresAt=self.expiry,privileges=json.dumps({'.storage_gigs':1}),
                plan=dict(defaultPrivileges={}))])))


def test_capacity_requires_existing_unexpired_allowance_and_no_automatic_upgrade():
    assert assert_capacity(CapacityAPI(),1000)['existing_allowance_sufficient']
    for api in (CapacityAPI(upgrade=True),CapacityAPI(storage=999_999_999),
                CapacityAPI(expiry='2000-01-01T00:00:00Z')):
        with pytest.raises(ValueError,match='allowance|storage'):
            assert_capacity(api,1000)


def test_capacity_enforces_the_raw_payload_limit_even_with_free_storage():
    with pytest.raises(ValueError,match='limit'):
        assert_capacity(CapacityAPI(),100*1024**2+1)
