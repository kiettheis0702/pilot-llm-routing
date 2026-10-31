"""Sequential standard-price embeddings with durable, non-retrying execution."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

from tools.openai_embeddings import (
    CLAIM_ROOT, MODEL, STANDARD_RATE, _save, inspect_plan, make_client,
)
from tools.paper_preflight.common import digest, read_json, sha256
from tools.paper_preflight.import_embeddings import _success


def approval_template(plan):
    return dict(approval_id=uuid4().hex, approved=False, approved_by='',
        mode='standard-sequential-once', max_standard_usd=.15, paid_retries=0,
        **{k:plan[k] for k in ('model','manifest_sha256','requests_sha256',
                              'unique_texts','tokens','requests')})


def limits_from_headers(headers, previous):
    result = dict(previous)
    for key, names in (
        ('tokens_per_minute', ['x-ratelimit-limit-tokens','x-ratelimit-limit-project-tokens']),
        ('requests_per_minute', ['x-ratelimit-limit-requests'])):
        values = []
        for name in names:
            try:
                value = int(headers.get(name, ''))
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        if values:
            result[key] = min(values)
    return result


def pacing_delay(tokens, limits):
    return max(tokens * 60 / (.8 * limits['tokens_per_minute']),
               60 / (.8 * limits['requests_per_minute']))


def _authorize(plan, path):
    approval = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    identifier = approval.get('approval_id')
    if (not isinstance(identifier,str) or len(identifier) != 32
            or any(c not in '0123456789abcdef' for c in identifier)
            or approval.get('approved') is not True
            or not isinstance(approval.get('approved_by'),str) or not approval['approved_by'].strip()
            or approval.get('mode') != 'standard-sequential-once'
            or type(approval.get('paid_retries')) is not int or approval['paid_retries'] != 0
            or any(approval.get(k) != plan[k] for k in
                ('model','manifest_sha256','requests_sha256','unique_texts','tokens','requests'))):
        raise ValueError('Explicit scoped standard-run approval is required')
    ceiling = approval.get('max_standard_usd')
    if (type(ceiling) not in (int,float) or not math.isfinite(ceiling)
            or ceiling < plan['estimated_standard_usd']):
        raise ValueError('Standard approval cost ceiling is insufficient')
    return approval


def _requests(plan_dir, plan):
    import tiktoken
    with (plan_dir / 'requests.jsonl').open('rb') as handle:
        payload = handle.read(200_000_001)
    if (len(payload) > 200_000_000
            or hashlib.sha256(payload).hexdigest() != plan['requests_sha256']):
        raise ValueError('Prepared payload checksum mismatch')
    encoding = tiktoken.get_encoding('cl100k_base')
    # JSONL separates records with LF; Unicode separators may be legitimate prompt text.
    requests = [json.loads(line) for line in payload.split(b'\n') if line.strip()]
    return [(record, sum(len(encoding.encode_ordinary(t)) for t in record['body']['input']))
            for record in requests]


def _accept(state, index, request, path):
    record = read_json(path)
    if record.get('custom_id') != request['custom_id'] or record.get('input_sha256') != digest(request):
        raise ValueError('Stored response input identity mismatch')
    _, request_id, usage = _success(record, len(request['body']['input']))
    hashed = sha256(path)
    old = state['completed'].get(str(index))
    if old and (old['sha256'] != hashed or old['request_id'] != request_id):
        raise ValueError('Stored response checksum/request ID mismatch')
    if any(v['request_id'] == request_id for k,v in state['completed'].items() if k != str(index)):
        raise ValueError('Duplicate provider request ID')
    state['completed'][str(index)] = dict(sha256=hashed, request_id=request_id,
        total_tokens=usage['total_tokens'], prompt_tokens=usage['prompt_tokens'])
    state['completed_requests'] = len(state['completed'])
    state['usage_total_tokens'] = sum(v['total_tokens'] for v in state['completed'].values())
    state['estimated_usage_usd'] = state['usage_total_tokens'] * STANDARD_RATE / 1_000_000
    if (state.get('in_flight') or {}).get('custom_id') == request['custom_id']:
        state['in_flight'] = None


def _export(run_dir, requests, state):
    target = run_dir / 'output.jsonl'
    if not (target.exists() and state.get('output_sha256') == sha256(target)):
        temporary = run_dir / 'output.jsonl.part'
        with temporary.open('wb') as handle:
            for index in range(len(requests)):
                record=read_json(run_dir / 'responses' / f'{index:06d}.json')
                handle.write(json.dumps(record,separators=(',',':'),allow_nan=False).encode('utf-8')+b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,target)
        state['output_sha256'] = sha256(target)
    state['status'] = 'complete'
    state['completed_at'] = time.time()
    _save(run_dir / 'state.json',state)
    return state


@contextmanager
def run_lock(run_dir):
    root=CLAIM_ROOT/'run-locks'
    root.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(os.path.normcase(str(Path(run_dir).resolve())).encode('utf-8')).hexdigest()
    with (root/(identity+'.lock')).open('a+b') as handle:
        handle.seek(0,os.SEEK_END)
        if handle.tell()==0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('Another runner is active for this run; lock refused') from None
        try:
            yield
        finally:
            handle.seek(0)
            if os.name=='nt':
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def run(plan_dir,approval_path,run_dir,*,resume=False,client_factory=make_client,
        sleep=time.sleep,deadline_seconds=36000):
    with run_lock(run_dir):
        return _run_unlocked(plan_dir,approval_path,run_dir,resume=resume,
            client_factory=client_factory,sleep=sleep,deadline_seconds=deadline_seconds)


def _run_unlocked(plan_dir, approval_path, run_dir, *, resume=False, client_factory=make_client,
        sleep=time.sleep, deadline_seconds=36000):
    plan_dir,run_dir = Path(plan_dir),Path(run_dir)
    plan = inspect_plan(plan_dir)
    approval = _authorize(plan,approval_path)
    requests = _requests(plan_dir,plan)  # Freeze exact approved inputs before transport.
    state_path = run_dir / 'state.json'
    claim = CLAIM_ROOT / (approval['approval_id'] + '.json')
    if resume:
        state = read_json(state_path)
        if (state['plan'] != plan or state['approval_sha256'] != digest(approval)
                or state['approval_id'] != approval['approval_id']
                or not claim.is_file() or read_json(claim).get('run_dir') != str(run_dir.resolve())):
            raise ValueError('Resume plan/approval/receipt mismatch')
        for index,(request,_) in enumerate(requests):
            path = run_dir / 'responses' / f'{index:06d}.json'
            if path.exists():
                _accept(state,index,request,path)
            elif str(index) in state['completed']:
                raise ValueError('Previously completed response is missing; refusing regeneration')
        if state.get('in_flight'):
            raise RuntimeError('Previous request outcome is ambiguous; no paid retry is allowed')
        state.pop('failure',None)
        _save(state_path,state)
    else:
        if claim.exists():
            raise ValueError('This approval was already used; resume its existing run')
        if run_dir.exists():
            raise FileExistsError('Run directory already exists; explicit resume is required')
        run_dir.mkdir(parents=True,exist_ok=False)
        (run_dir / 'responses').mkdir()
        state = dict(version=1, kind='standard-openai-embeddings', status='prepared',
            approval_id=approval['approval_id'], approval_sha256=digest(approval), plan=plan,
            started_at=time.time(), deadline_at=time.time()+deadline_seconds,
            completed={}, completed_requests=0, usage_total_tokens=0, estimated_usage_usd=0.,
            in_flight=None, next_allowed_at=0., paid_retries=0,
            rate_limits=dict(tokens_per_minute=40000,requests_per_minute=100))
        _save(state_path,state)
        CLAIM_ROOT.mkdir(parents=True,exist_ok=True)
        with claim.open('x',encoding='utf-8') as handle:
            json.dump(dict(run_dir=str(run_dir.resolve()),approval_sha256=digest(approval)),handle)
            handle.flush()
            os.fsync(handle.fileno())
    if len(state['completed']) == len(requests):
        return _export(run_dir,requests,state)
    if time.time() >= state['deadline_at']:
        state['status']='deadline_reached'
        _save(state_path,state)
        return state
    with client_factory() as client:
        for index,(request,tokens) in enumerate(requests):
            if str(index) in state['completed']:
                continue
            delay = max(0.,state['next_allowed_at']-time.time())
            if time.time()+delay >= state['deadline_at']:
                state['status']='deadline_reached'
                _save(state_path,state)
                return state
            while delay > 0:
                step = min(delay,30.)
                sleep(step)
                delay -= step
            if time.time() >= state['deadline_at']:
                state['status']='deadline_reached'
                _save(state_path,state)
                return state
            if (state['usage_total_tokens']+tokens)*STANDARD_RATE/1_000_000 > approval['max_standard_usd']:
                raise ValueError('Approved conservative usage ceiling would be exceeded')
            state['in_flight']=dict(custom_id=request['custom_id'], index=index,
                input_sha256=digest(request), tokens=tokens, started_at=time.time(), request_id=None)
            state['status']='running'
            _save(state_path,state)
            try:
                with client.embeddings.with_streaming_response.create(**request['body']) as response:
                    headers = response.headers
                    state['in_flight']['request_id'] = headers.get('x-request-id')
                    state['rate_limits']=limits_from_headers(headers,state['rate_limits'])
                    _save(state_path,state)  # Request ID is durable before reading the vector body.
                    result = response.parse()
                    record = dict(custom_id=request['custom_id'], input_sha256=digest(request), error=None,
                        response=dict(status_code=200, request_id=headers.get('x-request-id'),
                                      body=result.model_dump(mode='json')))
                    _success(record,len(request['body']['input']))
                    path=run_dir/'responses'/f'{index:06d}.json'
                    _save(path,record)
                _accept(state,index,request,path)
                state['next_allowed_at']=time.time()+pacing_delay(tokens,state['rate_limits'])
                _save(state_path,state)
                if index == 0 or (index+1)%10 == 0 or index+1 == len(requests):
                    print(json.dumps(dict(completed=state['completed_requests'],total=len(requests),
                        usage_tokens=state['usage_total_tokens'],estimated_usage_usd=state['estimated_usage_usd'],
                        elapsed_seconds=round(time.time()-state['started_at'],1),
                        rate_limits=state['rate_limits'])),flush=True)
            except Exception as error:
                state['status']='stopped_ambiguous'
                state['failure']=dict(type=type(error).__name__,
                    request_id=getattr(error,'request_id',None) or (state.get('in_flight') or {}).get('request_id'),
                    status_code=getattr(error,'status_code',None),
                    code=getattr(error,'code',None))
                _save(state_path,state)
                raise RuntimeError('Request failed or result is ambiguous; saved state, no paid retry') from None
    return _export(run_dir,requests,state)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-dir',type=Path,required=True)
    parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        from scripts.resources import require_resources
        require_resources()
        state=run(args.plan_dir,args.approval,args.run_dir,resume=args.resume)
        print(json.dumps({k:state[k] for k in ('status','completed_requests','usage_total_tokens','estimated_usage_usd')}),flush=True)
        return 0 if state['status']=='complete' else 2
    except Exception as error:
        print(json.dumps(dict(status='stopped',error_type=type(error).__name__,
                             note='Inspect state.json; no automatic paid retry')),flush=True)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
