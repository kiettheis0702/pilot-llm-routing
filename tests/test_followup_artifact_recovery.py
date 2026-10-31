"""Exercise the actual artifact CLI across pre-submit and accepted interruptions."""
import base64
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import tools.openai_followup_artifact as publication
from tools.openai_followup_tracking import make_record


def setup_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(publication,'ROOT',tmp_path)
    monkeypatch.setattr(publication,'BASE',tmp_path)
    monkeypatch.setenv('TEMP',str(tmp_path))
    monkeypatch.setenv('TMP',str(tmp_path))
    monkeypatch.setattr('scripts.resources.require_resources',lambda:None)
    item=make_record('openai','binary','fixture',dict(seed=42,arm_count=2),
                     dict(coverage=1.,requested=2,served=2),'0'*64)
    raw=b'method,reward\npilot,0.5\n';(tmp_path/'scientific.csv').write_bytes(raw)
    plan=dict(manifest_sha256='a'*64,total_bytes=len(raw),files=[dict(path='scientific.csv',
        artifact_path='scientific.csv',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())])
    payload=dict(runs=[item],payload_sha256='b'*64)
    (tmp_path/'authenticated-verification.json').write_text(json.dumps(dict(
        verified_runs=2498,payload_sha256=payload['payload_sha256'],legacy_runs_preserved=1128)))
    monkeypatch.setattr(publication,'prepare',lambda:(payload,plan))
    monkeypatch.setattr(publication,'assert_capacity',lambda *a:dict(existing_allowance_sufficient=True))
    state=SimpleNamespace(run=SimpleNamespace(config=item['config'],summary=item['metrics'],state='finished'),
        init_calls=0,submissions=0,completions=0,fail_init=False,mark_running_on_init_failure=False,
        fail_wait=False,remote=None,visible=True,graph_import_only=False)
    remote=SimpleNamespace(id='fixed-server-artifact',version='v0',qualified_name='fixture:v0',
        url='https://example.test/fixture',manifest=SimpleNamespace(entries={'scientific.csv':
            SimpleNamespace(size=len(raw),digest=base64.b64encode(hashlib.md5(raw).digest()).decode())}))

    class API:
        _service_api=SimpleNamespace(base_url=publication.HOST)
        def run(self,path):
            assert path==f'{publication.ENTITY}/{publication.PROJECT}/{item["id"]}'
            return state.run
        def artifact(self,name,**kwargs):
            assert name==f'{publication.ENTITY}/{publication.PROJECT}/{publication.NAME}:v0' or name==f'{publication.ENTITY}/{publication.PROJECT}/{publication.NAME}:verified-2026-10-31'
            if state.remote is None or not state.visible:
                raise LookupError('artifact not visible')
            return state.remote

    class Artifact:
        def __init__(self,*args,**kwargs):pass
        def add_file(self,*args,**kwargs):pass

    class Run:
        id=item['id']
        @property
        def summary(self):return state.run.summary
        def log_artifact(self,*args,**kwargs):
            state.submissions+=1;state.remote=remote
            def wait(**kwargs):
                if state.fail_wait:
                    state.fail_wait=False;raise ConnectionError('accepted upload interrupted')
            return SimpleNamespace(id=remote.id,version=remote.version,qualified_name=remote.qualified_name,
                url=remote.url,wait=wait)
        def finish(self):state.run.state='finished'

    def initialize(**kwargs):
        assert kwargs['id']==item['id'] and kwargs['resume'] in ('must','allow')
        state.init_calls+=1
        if state.graph_import_only:
            if kwargs['resume']=='must':
                raise ValueError('Existing GraphQL run has not been initialized by SDK')
            assert kwargs.get('config')==item['config']
            state.run.config=dict(kwargs['config']);state.run.summary={}
        if state.fail_init:
            state.fail_init=False
            if state.mark_running_on_init_failure:state.run.state='running'
            raise ConnectionError('initialization failed before artifact submission')
        state.run.state='running';return Run()

    def complete(record):
        assert record==item;state.completions+=1;state.run.state='finished'

    monkeypatch.setattr(publication,'complete_summary',complete,raising=False)
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(Api=lambda **kw:API(),Settings=lambda **kw:None,
        init=initialize,Artifact=Artifact))
    return state,tmp_path/'artifact-upload-receipt.json'


def test_existing_graph_import_can_be_initialized_without_losing_its_verified_values(tmp_path,monkeypatch):
    state,receipt=setup_cli(tmp_path,monkeypatch);state.graph_import_only=True
    original_config=dict(state.run.config);original_metrics=dict(state.run.summary)
    publication.main(['--approve-upload'])
    assert state.init_calls==1 and state.submissions==1 and state.run.state=='finished'
    assert state.run.config==original_config and state.run.summary==original_metrics


@pytest.mark.parametrize('left_running',[False,True])
def test_initialization_failure_before_submission_can_resume_without_duplicate_artifact(tmp_path,monkeypatch,left_running):
    state,receipt=setup_cli(tmp_path,monkeypatch)
    state.fail_init=True;state.mark_running_on_init_failure=left_running
    with pytest.raises(ConnectionError):publication.main(['--approve-upload'])
    assert state.submissions==0
    publication.main(['--approve-upload'])
    assert state.init_calls==2 and state.submissions==1 and state.run.state=='finished'
    assert json.loads(receipt.read_text())['status']=='finished'


def test_accepted_upload_recovers_manifest_and_finishes_resumed_summary_without_resubmission(tmp_path,monkeypatch):
    state,receipt=setup_cli(tmp_path,monkeypatch);state.fail_wait=True
    with pytest.raises(ConnectionError):publication.main(['--approve-upload'])
    assert state.submissions==1 and state.run.state=='running'
    publication.main(['--approve-upload'])
    assert state.submissions==1 and state.init_calls==1 and state.completions==1
    assert state.run.state=='finished' and json.loads(receipt.read_text())['status']=='finished'


def test_unknown_accepted_upload_is_never_recreated_while_manifest_is_not_visible(tmp_path,monkeypatch):
    state,receipt=setup_cli(tmp_path,monkeypatch);state.fail_wait=True
    with pytest.raises(ConnectionError):publication.main(['--approve-upload'])
    state.visible=False
    with pytest.raises(LookupError,match='not visible'):publication.main(['--approve-upload'])
    assert state.submissions==1 and state.init_calls==1
    state.visible=True;publication.main(['--approve-upload'])
    assert state.submissions==1 and state.run.state=='finished'
