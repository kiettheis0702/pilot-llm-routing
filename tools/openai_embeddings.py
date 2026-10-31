"""OpenAI Batch embeddings: offline by default; paid submit requires scoped approval."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from tools.paper_preflight.common import ROOT, read_json, sha256

DEFAULT_PLAN = ROOT / 'outputs/paper-preflight-20261030/embedding-plan'
DEFAULT_ENV = ROOT / '.env.openai'
MODEL = 'text-embedding-3-small'
STANDARD_RATE = .02
CLAIM_ROOT = ROOT / 'outputs/openai-embedding/approval-claims'


def read_key(path=DEFAULT_ENV):
    """An explicit local file takes precedence, including when it is empty."""
    path = Path(path)
    key = os.environ.get('OPENAI_API_KEY', '')
    if path.exists():
        values = []
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, separator, value = line.partition('=')
            if not separator or name.strip() != 'OPENAI_API_KEY':
                raise ValueError('Key file must contain only OPENAI_API_KEY and comments')
            value = value.strip()
            if value.startswith(('"', "'")):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ValueError('Unclosed quote in key file')
                value = value[1:-1]
            values.append(value)
        if len(values) > 1:
            raise ValueError('Duplicate OPENAI_API_KEY in local file')
        key = values[0] if values else ''
    if (not key.startswith('sk-') or len(key) < 12 or '...' in key
            or any(character.isspace() for character in key)):
        return None
    return key


def inspect_plan(path):
    """Recompute payload hashes, identities, limits and tokens without transport."""
    import tiktoken
    path = Path(path)
    manifest = read_json(path / 'manifest.json')
    if (manifest.get('kind') != 'offline-openai-embedding-plan'
            or manifest.get('model') != MODEL or manifest.get('dimensions') != 1536):
        raise ValueError('Unsupported embedding plan/model/dimensions')
    for name in ('requests.jsonl', 'id-map.json'):
        if sha256(path / name) != manifest.get('files', {}).get(name):
            raise ValueError('Embedding plan checksum mismatch')
    if (path / 'requests.jsonl').stat().st_size > 200_000_000:
        raise ValueError('Batch input exceeds 200 MB')
    encoding = tiktoken.get_encoding('cl100k_base')
    request_ids, positions, texts, total, count = set(), {}, set(), 0, 0
    with (path / 'requests.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            custom_id, body = record.get('custom_id'), record.get('body', {})
            if (set(record) != {'custom_id', 'method', 'url', 'body'}
                    or not isinstance(custom_id, str) or not custom_id
                    or custom_id in request_ids or record['method'] != 'POST'
                    or record['url'] != '/v1/embeddings' or not isinstance(body, dict)
                    or set(body) != {'model', 'input', 'encoding_format'}
                    or body['model'] != MODEL or body['encoding_format'] != 'float'
                    or not isinstance(body['input'], list) or not 1 <= len(body['input']) <= 2048):
                raise ValueError('Invalid Batch request/model')
            request_ids.add(custom_id)
            batch_tokens = 0
            for index, text in enumerate(body['input']):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError('Invalid empty embedding input')
                tokens = len(encoding.encode_ordinary(text))
                if not 1 <= tokens <= 8191:
                    raise ValueError('Embedding input token cap exceeded')
                hashed = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if hashed in texts:
                    raise ValueError('Duplicate embedding input would cause duplicate charges')
                texts.add(hashed)
                positions[(custom_id, index)] = hashed
                batch_tokens += tokens
                count += 1
            if batch_tokens > 300000:
                raise ValueError('Per-request token cap exceeded')
            total += batch_tokens
    if not 1 <= len(request_ids) <= 50000 or not 1 <= count <= 50000:
        raise ValueError('Batch request/input count limit exceeded')
    identities = read_json(path / 'id-map.json')
    mapped = set()
    for query, value in identities.items():
        position = (value.get('custom_id'), value.get('index'))
        if (not isinstance(query, str) or not query or type(position[1]) is not int
                or position not in positions or value.get('text_sha256') != positions[position]):
            raise ValueError('Invalid embedding identity map')
        mapped.add(position)
    if mapped != set(positions):
        raise ValueError('Unmapped embedding inputs')
    expected = dict(unique_texts=count, tokens=total, requests=len(request_ids),
                    source_rows=len(identities))
    if any(type(manifest.get(k)) is not int or manifest[k] != v for k, v in expected.items()):
        raise ValueError('Embedding manifest count/token mismatch')
    return dict(model=MODEL, dimensions=1536, **expected,
        manifest_sha256=sha256(path / 'manifest.json'),
        requests_sha256=sha256(path / 'requests.jsonl'),
        estimated_standard_usd=total * STANDARD_RATE / 1_000_000,
        estimated_batch_usd=total * STANDARD_RATE / 2_000_000)


def approval_template(plan):
    return dict(approval_id=uuid4().hex, approved=False, approved_by='', max_batch_jobs=1,
        max_standard_usd=round(plan['estimated_standard_usd'] + .000001, 6),
        **{name: plan[name] for name in ('model', 'manifest_sha256',
            'requests_sha256', 'unique_texts', 'tokens', 'requests')})


def doctor(plan_dir=DEFAULT_PLAN, env_file=DEFAULT_ENV):
    plan = inspect_plan(plan_dir)
    try:
        sdk = version('openai')
    except PackageNotFoundError:
        sdk = None
    return dict(**plan, key_configured=read_key(env_file) is not None,
        sdk_version=sdk, network_requests=0, paid_submission_enabled=False,
        note='Key presence is checked locally; provider access and billing are untested')


def make_client(env_file=DEFAULT_ENV):
    import httpx
    from openai import OpenAI
    key = read_key(env_file)
    if key is None:
        raise ValueError('Missing local OPENAI_API_KEY; paste it into .env.openai')
    # Ignore OPENAI_BASE_URL/proxy settings used by unrelated local tools.
    return OpenAI(api_key=key, base_url='https://api.openai.com/v1', max_retries=0,
        http_client=httpx.Client(trust_env=False, follow_redirects=False, timeout=60))


def _save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _snapshot(state, batch):
    state.update(batch_id=batch.id, status=batch.status, phase='submitted',
        output_file_id=batch.output_file_id, error_file_id=batch.error_file_id,
        request_counts=batch.request_counts.model_dump() if batch.request_counts else None)


def submit(plan_dir, approval_path, run_dir, *, client_factory=make_client):
    plan_dir, approval_path, run_dir = map(Path, (plan_dir, approval_path, run_dir))
    plan = inspect_plan(plan_dir)
    approval = read_json(approval_path)
    template = approval_template(plan)
    approval_id = approval.get('approval_id')
    if (not isinstance(approval_id, str) or len(approval_id) != 32
            or any(c not in '0123456789abcdef' for c in approval_id)
            or approval.get('approved') is not True or not isinstance(approval.get('approved_by'), str)
            or not approval['approved_by'].strip()
            or type(approval.get('max_batch_jobs')) is not int or approval['max_batch_jobs'] != 1
            or any(approval.get(k) != template[k] for k in template
                   if k not in {'approval_id', 'approved', 'approved_by', 'max_standard_usd'})):
        raise ValueError('Explicit approval for this exact plan is required')
    ceiling = approval.get('max_standard_usd')
    if (type(ceiling) not in (int, float) or not math.isfinite(ceiling)
            or ceiling < plan['estimated_standard_usd']):
        raise ValueError('Approval cost ceiling is below conservative Standard estimate')
    # The stable approval ID survives reformatting and copying either input path.
    claim_dir = CLAIM_ROOT
    claim = claim_dir / (approval_id + '.json')
    approval_digest = hashlib.sha256(json.dumps(approval, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if claim.exists():
        raise ValueError('This approval was already used; recover the existing run')
    if run_dir.exists():
        raise FileExistsError('Run directory already exists; use status or recover')
    with client_factory() as client:
        # Freeze exactly the approved bytes, so a later edit cannot change cost/input.
        with (plan_dir / 'requests.jsonl').open('rb') as handle:
            payload = handle.read(200_000_001)
        if (len(payload) > 200_000_000
                or hashlib.sha256(payload).hexdigest() != plan['requests_sha256']):
            raise ValueError('Embedding payload checksum changed after preflight')
        run_dir.mkdir(parents=True, exist_ok=False)
        state_path = run_dir / 'state.json'
        state = dict(version=1, run_token=uuid4().hex, phase='prepared', plan=plan,
            approval_id=approval_id, approval_sha256=approval_digest,
            batch_id=None, input_file_id=None,
            paid_retries=0)
        _save(state_path, state)
        claim_dir.mkdir(parents=True, exist_ok=True)
        with claim.open('x', encoding='utf-8') as handle:
            json.dump(dict(run_dir=str(run_dir.resolve()), run_token=state['run_token']), handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            state['phase'] = 'upload_started'
            _save(state_path, state)
            uploaded = client.files.create(
                file=('requests.jsonl', payload, 'application/jsonl'), purpose='batch')
            state.update(input_file_id=uploaded.id,
                         upload_request_id=getattr(uploaded, '_request_id', None))
            state['phase'] = 'create_started'
            _save(state_path, state)
            created = client.batches.create(input_file_id=uploaded.id,
                endpoint='/v1/embeddings', completion_window='24h',
                metadata=dict(pilot_run=state['run_token'],
                              plan_sha256=plan['manifest_sha256']))
            _snapshot(state, created)
            state['create_request_id'] = getattr(created, '_request_id', None)
            _save(state_path, state)
        except Exception as error:
            state['phase'] = 'create_ambiguous' if state['input_file_id'] else 'upload_ambiguous'
            state['failure_type'] = type(error).__name__
            state['failure_request_id'] = getattr(error, 'request_id', None)
            state['failure_status_code'] = getattr(error, 'status_code', None)
            _save(state_path, state)
            raise RuntimeError('Provider operation failed or is ambiguous; use recover; never resubmit') from None
    return state


def status(run_dir, *, client_factory=make_client):
    state_path = Path(run_dir) / 'state.json'
    state = read_json(state_path)
    if not state.get('batch_id'):
        raise ValueError('No saved batch ID; use recover without submitting a replacement')
    with client_factory() as client:
        batch = client.batches.retrieve(state['batch_id'])
    _snapshot(state, batch)
    state['status_request_id'] = getattr(batch, '_request_id', None)
    _save(state_path, state)
    return state


def recover(run_dir, *, client_factory=make_client):
    state_path = Path(run_dir) / 'state.json'
    state = read_json(state_path)
    if state.get('batch_id'):
        return status(run_dir, client_factory=client_factory)
    if not state.get('input_file_id'):
        raise ValueError('Upload did not return a file ID; no Batch create was attempted. Do not resubmit without new approval')
    with client_factory() as client:
        matches = [batch for batch in client.batches.list(limit=100)
            if (batch.metadata or {}).get('pilot_run') == state['run_token']
            and batch.input_file_id == state['input_file_id']
            and (batch.metadata or {}).get('plan_sha256') == state['plan']['manifest_sha256']]
    if len(matches) != 1:
        raise ValueError('Recovery found no unique Batch; keep this state and inspect the dashboard; do not resubmit')
    _snapshot(state, matches[0])
    _save(state_path, state)
    return state


def download(run_dir, *, client_factory=make_client):
    run_dir = Path(run_dir)
    state = status(run_dir, client_factory=client_factory)
    available = [(field, name) for field, name in
        (('output_file_id', 'output.jsonl'), ('error_file_id', 'errors.jsonl')) if state.get(field)]
    if not available:
        raise ValueError('No output/error files available yet; use status later')
    with client_factory() as client:
        for field, name in available:
            target = run_dir / name
            saved = state.setdefault('downloads', {}).get(name)
            if target.exists():
                if (saved and saved['file_id'] == state[field]
                        and saved['sha256'] == sha256(target)):
                    continue
                raise ValueError('Existing download checksum/identity mismatch; refusing overwrite')
            temporary = target.with_suffix('.jsonl.part')
            with client.files.with_streaming_response.content(state[field]) as response:
                with temporary.open('wb') as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            state['downloads'][name] = dict(file_id=state[field],
                sha256=sha256(temporary), bytes=temporary.stat().st_size,
                request_id=response.headers.get('x-request-id'))
            # Persist identity before publishing: a crash after rename is reconcilable.
            _save(run_dir / 'state.json', state)
            os.replace(temporary, target)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='doctor',
                        choices=['doctor', 'submit', 'status', 'recover', 'download'])
    parser.add_argument('--env-file', type=Path, default=DEFAULT_ENV)
    parser.add_argument('--plan-dir', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--write-approval-template', type=Path)
    args = parser.parse_args(argv)
    try:
        from scripts.resources import require_resources
        require_resources()
        if args.action == 'doctor':
            result = doctor(args.plan_dir, args.env_file)
            if args.write_approval_template:
                args.write_approval_template.parent.mkdir(parents=True, exist_ok=True)
                with args.write_approval_template.open('x', encoding='utf-8') as handle:
                    json.dump(approval_template(result), handle, indent=2)
                    handle.write('\n')
        else:
            if args.run_dir is None:
                raise ValueError('--run-dir is required')
            factory = lambda: make_client(args.env_file)
            if args.action == 'submit':
                if args.approval is None:
                    raise ValueError('Explicit --approval is required; a pasted key is not approval')
                result = submit(args.plan_dir, args.approval, args.run_dir, client_factory=factory)
            else:
                result = globals()[args.action](args.run_dir, client_factory=factory)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(json.dumps(dict(error=str(error))))
        return 1
    except Exception as error:
        # Provider exception strings/bodies may include sensitive content.
        print(json.dumps(dict(error='Operation failed', type=type(error).__name__,
            request_id=getattr(error, 'request_id', None),
            status_code=getattr(error, 'status_code', None))))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
