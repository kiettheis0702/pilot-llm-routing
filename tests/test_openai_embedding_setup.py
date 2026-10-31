"""Free transport-boundary checks; no real credential or provider connection."""
import json
from pathlib import Path

import pytest

from tools.paper_preflight.embedding_plan import build_plan


@pytest.fixture(autouse=True)
def deny_real_network(monkeypatch, tmp_path):
    import socket
    def denied(*args, **kwargs):
        pytest.fail('A setup test attempted a real network connection')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)
    monkeypatch.setattr(api(), 'CLAIM_ROOT', tmp_path / 'approval-claims')


@pytest.fixture
def plan(tmp_path):
    source = tmp_path / 'inputs.json'
    source.write_text(json.dumps([{'query_id': 'q1', 'text': 'hello world'},
                                  {'query_id': 'q2', 'text': 'different prompt'}]))
    target = tmp_path / 'plan'
    build_plan([source], target, batch_size=1)
    return target


def api():
    from tools import openai_embeddings
    return openai_embeddings


def approve(plan, tmp_path):
    module = api()
    path = tmp_path / 'approval.json'
    record = module.approval_template(module.inspect_plan(plan))
    record.update(approved=True, approved_by='offline-test-only')
    path.write_text(json.dumps(record), encoding='utf-8')
    return path


def test_empty_key_is_not_an_error_for_offline_doctor(plan, tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    env = tmp_path / '.env.openai'
    env.write_text('# paste locally\nOPENAI_API_KEY=\n')
    result = api().doctor(plan, env)
    assert result['key_configured'] is False
    assert result['network_requests'] == 0
    assert result['model'] == 'text-embedding-3-small'
    assert result['unique_texts'] == 2
    assert result['estimated_batch_usd'] == result['estimated_standard_usd'] / 2


def test_key_file_is_local_and_never_printed(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-environment-value')
    env = tmp_path / '.env.openai'
    env.write_text('\ufeffOPENAI_API_KEY="sk-local-test-value"\n', encoding='utf-8')
    assert api().read_key(env) == 'sk-local-test-value'
    env.write_text('OPENAI_API_KEY=\n')
    # An explicitly selected empty file must not silently use another account.
    assert api().read_key(env) is None


@pytest.mark.parametrize('value', ['sk-...', 'YOUR_API_KEY', 'paste-key-here'])
def test_placeholder_is_not_a_configured_key(value, tmp_path):
    env = tmp_path / '.env.openai'
    env.write_text('OPENAI_API_KEY=' + value)
    assert api().read_key(env) is None


def test_production_client_uses_official_host_and_disables_retries(tmp_path, monkeypatch):
    env = tmp_path / '.env.openai'
    env.write_text('OPENAI_API_KEY=sk-offline-test-value\n')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://localhost:9999/v1')
    with api().make_client(env) as client:
        assert str(client.base_url) == 'https://api.openai.com/v1/'
        assert client.max_retries == 0
        assert client._client.trust_env is False


def test_edited_request_file_is_rejected_before_network(plan):
    with (plan / 'requests.jsonl').open('a') as handle:
        handle.write('{}\n')
    with pytest.raises(ValueError, match='checksum'):
        api().inspect_plan(plan)


def test_manifest_cannot_override_price_or_model(plan):
    manifest = plan / 'manifest.json'
    record = json.loads(manifest.read_text())
    record['model'] = 'text-embedding-3-large'
    record['standard_usd_per_million_tokens'] = 0
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='model'):
        api().inspect_plan(plan)


def test_missing_or_unapproved_approval_never_constructs_client(plan, tmp_path):
    def forbidden():
        pytest.fail('Transport must not be constructed')
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps(api().approval_template(api().inspect_plan(plan))))
    with pytest.raises(ValueError, match='approval'):
        api().submit(plan, approval, tmp_path / 'run', client_factory=forbidden)
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('field,value', [('manifest_sha256', '0' * 64),
    ('requests', 99), ('tokens', 99), ('max_standard_usd', 0), ('approved', 'true')])
def test_scope_and_budget_gate(plan, tmp_path, field, value):
    approval = approve(plan, tmp_path)
    record = json.loads(approval.read_text())
    record[field] = value
    approval.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='approval|ceiling'):
        api().submit(plan, approval, tmp_path / 'run',
                     client_factory=lambda: pytest.fail('No transport allowed'))


def test_payload_changed_after_preflight_is_never_uploaded(plan, tmp_path):
    approval = approve(plan, tmp_path)
    def factory():
        with (plan / 'requests.jsonl').open('a') as handle:
            handle.write('{}\n')
        return sdk_client(lambda request: pytest.fail('Mutated payload reached transport'))
    with pytest.raises(ValueError, match='checksum'):
        api().submit(plan, approval, tmp_path / 'run', client_factory=factory)


def sdk_client(handler):
    import httpx
    from openai import OpenAI
    return OpenAI(api_key='sk-offline-test-value', max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def batch(status='validating', **overrides):
    value = dict(id='batch_test', object='batch', endpoint='/v1/embeddings',
        input_file_id='file_test', completion_window='24h', created_at=1,
        status=status, metadata={}, output_file_id=None, error_file_id=None,
        request_counts=dict(total=2, completed=0, failed=0))
    value.update(overrides)
    return value


def test_sdk_submit_persists_ids_and_refuses_same_approval_twice(plan, tmp_path):
    import httpx
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path))
        assert request.url.host == 'api.openai.com'
        if request.url.path == '/v1/files':
            assert b'name="purpose"\r\n\r\nbatch' in request.read()
            return httpx.Response(200, json=dict(id='file_test', object='file',
                bytes=100, created_at=1, filename='requests.jsonl', purpose='batch'),
                headers={'x-request-id': 'req_upload'})
        body = json.loads(request.read())
        assert body['endpoint'] == '/v1/embeddings'
        assert body['completion_window'] == '24h'
        assert body['input_file_id'] == 'file_test'
        return httpx.Response(200, json=batch(metadata=body['metadata']),
                              headers={'x-request-id': 'req_create'})
    approval = approve(plan, tmp_path)
    run = tmp_path / 'run'
    result = api().submit(plan, approval, run, client_factory=lambda: sdk_client(handler))
    saved = json.loads((run / 'state.json').read_text())
    assert result['batch_id'] == saved['batch_id'] == 'batch_test'
    assert saved['upload_request_id'] == 'req_upload'
    assert saved['create_request_id'] == 'req_create'
    assert len(calls) == 2
    with pytest.raises((ValueError, FileExistsError), match='used|exist'):
        api().submit(plan, approval, tmp_path / 'another-run',
                     client_factory=lambda: pytest.fail('No second transport'))
    # Reformatting the same approval or moving its plan is not a new approval.
    copied_approval = tmp_path / 'approval-reformatted.json'
    copied_approval.write_text(json.dumps(json.loads(approval.read_text()), indent=5))
    import shutil
    copied_plan = tmp_path / 'different-parent' / 'plan'
    shutil.copytree(plan, copied_plan)
    with pytest.raises(ValueError, match='used'):
        api().submit(copied_plan, copied_approval, tmp_path / 'another-run',
                     client_factory=lambda: pytest.fail('No replay transport'))


def test_create_timeout_recovers_the_same_job_without_post(plan, tmp_path):
    import httpx
    metadata = {}
    calls = []
    def handler(request):
        calls.append(request.method)
        if request.url.path == '/v1/files':
            return httpx.Response(200, json=dict(id='file_test', object='file', bytes=1,
                created_at=1, filename='requests.jsonl', purpose='batch'))
        metadata.update(json.loads(request.read())['metadata'])
        raise httpx.ReadTimeout('simulated lost create response', request=request)
    run = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='ambiguous|recover'):
        api().submit(plan, approve(plan, tmp_path), run,
                     client_factory=lambda: sdk_client(handler))
    assert calls == ['POST', 'POST']  # SDK retry is disabled.
    state = json.loads((run / 'state.json').read_text())
    assert state['phase'] == 'create_ambiguous'
    assert state['input_file_id'] == 'file_test'
    def recover_handler(request):
        assert request.method == 'GET'
        return httpx.Response(200, json=dict(object='list', has_more=False,
            data=[batch(metadata=metadata)], first_id='batch_test', last_id='batch_test'))
    result = api().recover(run, client_factory=lambda: sdk_client(recover_handler))
    assert result['batch_id'] == 'batch_test'


@pytest.mark.parametrize('http_status', [401, 429, 500])
def test_upload_failure_is_not_retried_or_reported_success(plan, tmp_path, http_status):
    import httpx
    calls = []
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(http_status, json={'error': {'message': 'fake failure',
            'type': 'test_error'}}, headers={'x-request-id': 'req_failed'})
    run = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='recover'):
        api().submit(plan, approve(plan, tmp_path), run,
                     client_factory=lambda: sdk_client(handler))
    assert calls == ['/v1/files']
    assert json.loads((run / 'state.json').read_text())['failure_request_id'] == 'req_failed'


def test_download_streams_output_and_errors_and_can_be_repeated(tmp_path):
    import httpx
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'state.json').write_text(json.dumps({'batch_id': 'batch_test'}))
    calls = []
    def handler(request):
        calls.append(request.url.path)
        assert request.method == 'GET'
        if request.url.path.startswith('/v1/batches/'):
            return httpx.Response(200, json=batch('expired', output_file_id='file_output',
                error_file_id='file_errors', request_counts=dict(total=2, completed=1, failed=1)))
        return httpx.Response(200, content=b'{"custom_id":"synthetic"}\n')
    factory = lambda: sdk_client(handler)
    result = api().download(run, client_factory=factory)
    assert result['status'] == 'expired'
    assert (run / 'output.jsonl').read_bytes() == b'{"custom_id":"synthetic"}\n'
    assert (run / 'errors.jsonl').exists()
    before = len(calls)
    api().download(run, client_factory=factory)
    assert len(calls) == before + 1  # Status refresh only; cached files validated.


def test_status_and_download_do_not_create_jobs(tmp_path):
    import httpx
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'state.json').write_text(json.dumps({'batch_id': 'batch_test'}))
    def handler(request):
        assert request.method == 'GET'
        return httpx.Response(200, json=batch('in_progress'))
    with pytest.raises(ValueError, match='available|ready'):
        api().download(run, client_factory=lambda: sdk_client(handler))


def test_interrupted_download_can_be_retried_without_creating_a_job(tmp_path):
    import httpx
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'state.json').write_text(json.dumps({'batch_id': 'batch_test'}))
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'partial'
            raise httpx.ReadError('simulated connection loss')
    def broken(request):
        assert request.method == 'GET'
        if '/batches/' in request.url.path:
            return httpx.Response(200, json=batch('completed', output_file_id='file_output'))
        return httpx.Response(200, stream=BrokenStream())
    with pytest.raises(httpx.ReadError):
        api().download(run, client_factory=lambda: sdk_client(broken))
    assert not (run / 'output.jsonl').exists()
    assert (run / 'output.jsonl.part').exists()
    def restored(request):
        assert request.method == 'GET'
        if '/batches/' in request.url.path:
            return httpx.Response(200, json=batch('completed', output_file_id='file_output'))
        return httpx.Response(200, content=b'complete\n')
    api().download(run, client_factory=lambda: sdk_client(restored))
    assert (run / 'output.jsonl').read_bytes() == b'complete\n'
    assert not (run / 'output.jsonl.part').exists()


def test_download_recovers_after_final_file_published_before_process_exit(tmp_path, monkeypatch):
    import httpx
    import os
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'state.json').write_text(json.dumps({'batch_id': 'batch_test'}))
    calls = []
    def handler(request):
        calls.append(request.url.path)
        assert request.method == 'GET'
        if '/batches/' in request.url.path:
            return httpx.Response(200, json=batch('completed', output_file_id='file_output'))
        return httpx.Response(200, content=b'complete\n')
    replace = os.replace
    def exit_after_publish(source, target):
        replace(source, target)
        if Path(target).name == 'output.jsonl':
            raise OSError('simulated process exit after publishing')
    factory = lambda: sdk_client(handler)
    monkeypatch.setattr(api().os, 'replace', exit_after_publish)
    with pytest.raises(OSError):
        api().download(run, client_factory=factory)
    assert (run / 'output.jsonl').read_bytes() == b'complete\n'
    monkeypatch.setattr(api().os, 'replace', replace)
    before = len(calls)
    api().download(run, client_factory=factory)
    assert len(calls) == before + 1  # Only status GET; final artifact reconciled.
