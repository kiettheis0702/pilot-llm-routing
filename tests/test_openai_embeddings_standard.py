import json
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from tools.paper_preflight.embedding_plan import build_plan


def module():
    from tools import openai_embeddings_standard
    return openai_embeddings_standard


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    import socket
    def denied(*args, **kwargs):
        pytest.fail('Unexpected real socket connection')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(module(), 'CLAIM_ROOT', tmp_path / 'claims')


@pytest.fixture
def plan(tmp_path):
    inputs = tmp_path / 'input.json'
    inputs.write_text(json.dumps([dict(query_id='one', text='hello world'),
                                  dict(query_id='two', text='another prompt')]))
    output = tmp_path / 'plan'
    build_plan([inputs], output, batch_size=1, max_batch_tokens=16000)
    return output


def approval(plan, tmp_path, approved=True):
    p = tmp_path / 'approval.json'
    record = module().approval_template(module().inspect_plan(plan))
    record.update(approved=approved, approved_by='synthetic test only')
    p.write_text(json.dumps(record))
    return p


def client(handler):
    return OpenAI(api_key='sk-synthetic-test-only', max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def body(count=1):
    return dict(object='list', model='text-embedding-3-small',
        data=[dict(object='embedding', index=i, embedding=[1.] + [0.] * 1535)
              for i in range(count)], usage=dict(prompt_tokens=2*count, total_tokens=2*count))


def test_unapproved_never_constructs_transport(plan, tmp_path):
    with pytest.raises(ValueError, match='approval'):
        module().run(plan, approval(plan, tmp_path, False), tmp_path / 'run',
            client_factory=lambda: pytest.fail('No transport'))


@pytest.mark.parametrize('field,value',[('max_standard_usd',0),('requests',999),
    ('tokens',999),('mode','batch'),('paid_retries',1)])
def test_scope_and_cost_ceiling_fail_before_transport(plan,tmp_path,field,value):
    a=approval(plan,tmp_path)
    record=json.loads(a.read_text())
    record[field]=value
    a.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='approval|ceiling'):
        module().run(plan,a,tmp_path/'run',client_factory=lambda:pytest.fail('No transport'))


def test_sdk_success_can_resume_without_spending_and_exports_for_importer(plan, tmp_path):
    calls, delays = [], []
    def handler(request):
        assert request.method == 'POST' and request.url.path == '/v1/embeddings'
        sent = json.loads(request.read())
        assert set(sent) == {'model','input','encoding_format'}
        assert sent['model'] == 'text-embedding-3-small'
        calls.append(sent)
        return httpx.Response(200, json=body(len(sent['input'])), headers={
            'x-request-id': f'req_{len(calls)}', 'x-ratelimit-limit-tokens': '1000000',
            'x-ratelimit-limit-requests': '5000'})
    a, run = approval(plan, tmp_path), tmp_path / 'run'
    result = module().run(plan, a, run, client_factory=lambda: client(handler), sleep=delays.append)
    assert result['status'] == 'complete' and result['completed_requests'] == 2
    assert len(calls) == 2 and result['usage_total_tokens'] == 4
    assert result['rate_limits']['tokens_per_minute'] == 1000000
    again = module().run(plan, a, run, resume=True,
        client_factory=lambda: pytest.fail('Completed run must not use transport'))
    assert again['status'] == 'complete'
    from tools.paper_preflight.import_embeddings import import_batch
    receipt = import_batch(plan, [run / 'output.jsonl'], [tmp_path / 'input.json'],
                           tmp_path / 'imported', synthetic=True)
    assert receipt['status'] == 'complete'
    # Same approval cannot start a replacement under a new path.
    with pytest.raises(ValueError, match='used'):
        module().run(plan, a, tmp_path / 'replacement',
            client_factory=lambda: pytest.fail('No replacement'))


def test_ambiguous_network_failure_is_not_resent(plan, tmp_path):
    calls = []
    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout('synthetic lost response', request=request)
    a, run = approval(plan, tmp_path), tmp_path / 'run'
    with pytest.raises(RuntimeError, match='ambiguous'):
        module().run(plan, a, run, client_factory=lambda: client(handler))
    assert len(calls) == 1
    state = json.loads((run / 'state.json').read_text())
    assert state['in_flight']['custom_id'] == 'pilot-embed-000000'
    with pytest.raises(RuntimeError, match='ambiguous'):
        module().run(plan, a, run, resume=True,
            client_factory=lambda: pytest.fail('No automatic retry'))


def test_saved_response_reconciles_after_state_write_failure(plan, tmp_path, monkeypatch):
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(200, json=body(), headers={'x-request-id':f'req_{len(calls)}'})
    a, run = approval(plan, tmp_path), tmp_path / 'run'
    save = module()._save
    def fail_after_saved_response(path, value):
        if Path(path).name == 'state.json' and value.get('completed_requests') == 1:
            raise OSError('synthetic interruption after response persisted')
        return save(path, value)
    monkeypatch.setattr(module(), '_save', fail_after_saved_response)
    with pytest.raises((OSError, RuntimeError)):
        module().run(plan, a, run, client_factory=lambda: client(handler), sleep=lambda delay:None)
    monkeypatch.setattr(module(), '_save', save)
    result = module().run(plan, a, run, resume=True,
        client_factory=lambda: client(handler), sleep=lambda delay:None)
    assert result['status'] == 'complete' and len(calls) == 2


def test_429_is_saved_with_request_id_and_not_retried(plan, tmp_path):
    calls = []
    def handler(request):
        calls.append(1)
        return httpx.Response(429, json={'error':{'type':'insufficient_quota',
            'code':'insufficient_quota','message':'synthetic'}},
            headers={'x-request-id':'req_quota'})
    run = tmp_path / 'run'
    with pytest.raises(RuntimeError):
        module().run(plan, approval(plan,tmp_path), run, client_factory=lambda:client(handler))
    assert len(calls) == 1
    state = json.loads((run / 'state.json').read_text())
    assert state['failure']['request_id'] == 'req_quota'
    assert state['failure']['status_code'] == 429


def test_changed_payload_stops_before_any_paid_request(plan, tmp_path):
    a = approval(plan,tmp_path)
    with (plan / 'requests.jsonl').open('a') as stream:
        stream.write('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        module().run(plan,a,tmp_path/'run',client_factory=lambda:pytest.fail('No transport'))


def test_unicode_line_separators_inside_json_text_are_preserved(tmp_path):
    text='alpha\u2028beta\u0085gamma\u2029delta'
    source=tmp_path/'unicode.json'
    source.write_text(json.dumps([dict(query_id='unicode',text=text)],ensure_ascii=False),encoding='utf-8')
    plan=tmp_path/'unicode-plan'
    build_plan([source],plan,max_batch_tokens=16000)
    requests=module()._requests(plan,module().inspect_plan(plan))
    assert requests[0][0]['body']['input']==[text]


@pytest.mark.parametrize('repeat',[1,2])
def test_second_runner_cannot_enter_the_same_run_between_requests(plan,tmp_path,repeat):
    a=approval(plan,tmp_path)
    run=tmp_path/'run'
    with module().run_lock(run):
        with pytest.raises(RuntimeError,match='active|lock'):
            module().run(plan,a,run,resume=True,client_factory=lambda:pytest.fail('Duplicate transport'))


def test_rate_pacing_uses_smallest_project_limit():
    limits = module().limits_from_headers({'x-ratelimit-limit-tokens':'1000000',
        'x-ratelimit-limit-project-tokens':'40000','x-ratelimit-limit-requests':'100'},
        {'tokens_per_minute':40000,'requests_per_minute':100})
    assert limits['tokens_per_minute'] == 40000
    assert module().pacing_delay(16000,limits) == pytest.approx(30.)


def test_expired_deadline_sends_nothing(plan,tmp_path):
    a=approval(plan,tmp_path)
    result=module().run(plan,a,tmp_path/'run',deadline_seconds=0,
        client_factory=lambda:pytest.fail('Deadline must be checked before transport'))
    assert result['status']=='deadline_reached' and result['completed_requests']==0


@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows process inventory')
def test_resource_guard_detects_standard_embedding_job():
    import subprocess
    import sys
    from scripts.resources import resource_snapshot
    child=subprocess.Popen([sys.executable,'-c',
        "import sys; print('ready',flush=True); sys.stdin.read()",'tools.openai_embeddings_standard'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert child.stdout.readline().strip()=='ready'
        assert any(row['pid']==child.pid or row['parent_pid']==child.pid
                   for row in resource_snapshot()['heavy_jobs'])
    finally:
        child.communicate('',timeout=10)
