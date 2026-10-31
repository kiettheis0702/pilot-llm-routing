"""Publish verified aggregates to the existing Team8 project, with recovery receipts.

No online action occurs on import. The default CLI only validates local payloads.
Online writes require --approve-upload; no inference API is involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from tools.course.import_summaries import verify_values, _config_equal
from tools.course.tracking import _save_receipt
from tools.openai_followup_tracking import build_records, digest, validate_record

ROOT = Path(__file__).resolve().parents[1]
ENTITY = 'kiettheis-university-of-science-vnu-hcm'
PROJECT = 'pilot-llm-routing-team8'
HOST = 'https://api.forge.coreweave.com'
PROJECT_URL = f'https://forge.coreweave.com/wandb/{ENTITY}/{PROJECT}'
BASE = ROOT/'outputs/openai-followup-20261031/publication'


def now():
    return datetime.now(timezone.utc).isoformat()


def build_batch(records, *, approved=False):
    if not approved:
        raise ValueError('Explicit upload authorization required')
    if not 1 <= len(records) <= 20 or len({r['id'] for r in records}) != len(records):
        raise ValueError('Expected 1..20 unique follow-up records')
    variables = {}
    for index, record in enumerate(records):
        validate_record(record)
        config = record['config']
        tags = [record['lab'], 'historical-import', 'openai-followup',
                'recipe-'+config['recipe'], config['experiment_family']]
        if 'arm_count' in config:
            tags.append('arms-'+str(config['arm_count']))
        variables[f'r{index}'] = dict(entityName=ENTITY, modelName=PROJECT,
            name=record['id'], displayName=config['source_run_id'], state='running',
            config=json.dumps({k:dict(value=v) for k,v in config.items()},allow_nan=False),
            summaryMetrics=json.dumps(record['metrics'],allow_nan=False),
            groupName=record['lab'],jobType=config['experiment_family'],tags=tags,
            notes='Checksum-verified local historical-data summary. Recipe and arm count '
                  'are explicit; this is not an exact paper reproduction. No raw prompts, '
                  'machine metadata or invented training histories are uploaded.')
    declarations = ','.join(f'${key}:UpsertBucketInput!' for key in variables)
    fields = ' '.join(f'{key}:upsertBucket(input:${key})'
                      '{bucket{id name state config summaryMetrics}}' for key in variables)
    return f'mutation ImportFollowupSummaries({declarations}){{{fields}}}', variables


def import_batch(records, receipt_dir, *, approved=False, execute, complete):
    query, variables = build_batch(records,approved=approved)
    if not callable(execute) or not callable(complete):
        raise ValueError('Both upload and completion transports required')
    folder=Path(receipt_dir);folder.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for record in records:
        path=folder/(record['id']+'.json')
        previous=json.loads(path.read_text()) if path.exists() else {}
        if previous and previous['payload_sha256']!=record['payload_sha256']:
            raise ValueError('Existing receipt conflicts with this payload')
        # A retry is a new transport attempt for the SAME aggregate identity.
        # Retain any earlier server_id even if this attempt disconnects.
        receipt=dict(previous,id=record['id'],payload_sha256=record['payload_sha256'],
            status='planned',planned_at=previous.get('planned_at',now()),
            attempt_planned_at=now(),entity=ENTITY,project=PROJECT,
            url=PROJECT_URL+'/runs/'+record['id'],import_method='aggregate-summary')
        receipts.append((path,receipt))
    for path,receipt in receipts:
        _save_receipt(path,receipt)
    result=execute(query,variables)
    # Preserve ALL accepted server identities before inspecting any metric.
    for index,(path,receipt) in enumerate(receipts):
        bucket=(result.get(f'r{index}') or {}).get('bucket') or {}
        if bucket.get('name')==receipt['id'] and bucket.get('id'):
            receipt.update(status='accepted',server_id=bucket['id'],accepted_at=now())
            _save_receipt(path,receipt)
    for index,(path,receipt) in enumerate(receipts):
        if receipt['status']!='accepted':
            raise ValueError('Missing or incorrect server identity')
        bucket=result[f'r{index}']['bucket']
        config={k:v['value'] for k,v in json.loads(bucket['config']).items()}
        verify_values(records[index],config,json.loads(bucket['summaryMetrics']),bucket['state'],require_finished=False)
        complete(records[index])
        receipt.update(status='uploaded',uploaded_at=now())
        _save_receipt(path,receipt)
    return [receipt for _,receipt in receipts]


def local_payload(path):
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    body={k:v for k,v in payload.items() if k!='payload_sha256'}
    if digest(body)!=payload.get('payload_sha256') or body!=build_records(ROOT):
        raise ValueError('Prepared payload differs from current verified scientific reports')
    for row in payload['runs']:
        validate_record(row)
    return payload


def assert_capacity(api, raw_bytes):
    """Read existing allowance only; never change billing or subscription settings."""
    query='''query($name:String!){entity(name:$name){name storageBytes organization{
      subscriptions{status subscriptionType isAutomaticUpgrade expiresAt
        plan{name displayName planType unitPrice defaultPrivileges} privileges}}}}'''
    entity=api._service_api.execute_graphql(query,dict(name=ENTITY))['entity']
    if entity['name']!=ENTITY or raw_bytes>100*1024**2:
        raise ValueError('Wrong destination or payload exceeds reviewed raw-file limit')
    subscriptions=entity['organization']['subscriptions']
    included=[]
    for item in subscriptions:
        if item['status']!='ENABLED' or item['isAutomaticUpgrade'] is not False:
            continue
        if item.get('expiresAt') and datetime.fromisoformat(item['expiresAt'].replace('Z','+00:00'))<=datetime.now(timezone.utc):
            continue
        for values in (item.get('privileges'),item['plan'].get('defaultPrivileges')):
            if isinstance(values,str):
                values=json.loads(values)
            if isinstance(values,dict):
                included.extend(float(v)*1_000_000_000 for k,v in values.items()
                                if k.lstrip('.')=='storage_gigs' and type(v) in (int,float))
    # Conservatively allow 20x raw payload for summary/index/server overhead.
    if not included or entity['storageBytes']+20*raw_bytes>=max(included):
        raise ValueError('Cannot prove sufficient existing included storage with automatic upgrade disabled')
    return dict(checked_at=now(),existing_allowance_sufficient=True,
                automatic_upgrade=False,raw_bytes=raw_bytes,overhead_multiplier=20,
                billing_changed=False)


def remote_runs(api):
    return {r.id:r for r in api.runs(f'{ENTITY}/{PROJECT}',per_page=100,lazy=False)}


def complete_summary(record, *, session=None):
    """Complete the same verified aggregate identity; never create another run."""
    validate_record(record)
    if session is None:
        import requests
        session=requests.Session()  # Existing host-scoped netrc; never printed.
    metrics=dict(record['metrics'],_step=0,_runtime=0)
    content=[json.dumps(metrics,allow_nan=False)]
    response=session.post(HOST+f'/files/{ENTITY}/{PROJECT}/{record["id"]}/file_stream',
        json=dict(files={'wandb-summary.json':dict(offset=0,content=content),
                         'wandb-history.jsonl':dict(offset=0,content=content)},complete=True,exitcode=0),
        timeout=(10,60),allow_redirects=False)
    response.raise_for_status()
    if not 200<=response.status_code<300:
        raise ValueError('Unexpected redirect completing summary run')


def verify_legacy(remote):
    records=json.loads((ROOT/'reports/course-submission-2026-10-30/tracking-payloads.json').read_text())['runs']
    if len(records)!=1128:
        raise ValueError('Historical record inventory changed')
    for row in records:
        run=remote.get(row['id'])
        if run is None:
            raise ValueError('Missing historical run')
        verify_values(row,run.config,run.summary,run.state)
    return len(records)


def anonymous_readback(records):
    """Fresh unauthenticated GraphQL read, explicitly bypassing netrc and proxies."""
    import requests
    session=requests.Session();session.trust_env=False
    query='''query PublicRuns($entity:String!,$project:String!,$cursor:String){
      project(name:$project,entityName:$entity){name runCount readOnly
        runs(first:100,after:$cursor){edges{node{name state config summaryMetrics}}
          pageInfo{endCursor hasNextPage}}}}'''
    remote={};cursor=None;pages=0
    while True:
        response=session.post(HOST+'/graphql',json=dict(query=query,
            variables=dict(entity=ENTITY,project=PROJECT,cursor=cursor)),
            timeout=(10,60),allow_redirects=False)
        if 'Authorization' in response.request.headers or 'Cookie' in response.request.headers:
            raise ValueError('Anonymous verification unexpectedly authenticated')
        response.raise_for_status();body=response.json()
        if body.get('errors'):
            raise ValueError('Anonymous public query failed')
        data=body['data']['project']
        if not data or data['name']!=PROJECT:
            raise ValueError('Public project unavailable')
        for edge in data['runs']['edges']:
            row=edge['node']
            if row['name'] in remote:
                raise ValueError('Repeated public run identity')
            remote[row['name']]=row
        pages+=1
        if not data['runs']['pageInfo']['hasNextPage']:
            break
        following=data['runs']['pageInfo']['endCursor']
        if not following or following==cursor:
            raise ValueError('Public pagination did not advance')
        cursor=following
    legacy=json.loads((ROOT/'reports/course-submission-2026-10-30/tracking-payloads.json').read_text())['runs']
    for record in legacy+records:
        row=remote.get(record['id'])
        if row is None:
            raise ValueError('Public run missing: '+record['id'])
        config={k:v['value'] for k,v in json.loads(row['config']).items()}
        verify_values(record,config,json.loads(row['summaryMetrics']),row['state'])
    return dict(verified_at=now(),verification_method='anonymous-graphql-readback',
        authentication='none; trust_env=False; no Authorization or Cookie headers',
        project_url=PROJECT_URL,public_verified=True,verified_runs=len(records),
        legacy_runs_preserved=len(legacy),server_runs=len(remote),pages=pages,
        metrics_verified=sum(len(r['metrics']) for r in records),paid_api_calls=0,
        read_only_for_anonymous=data['readOnly'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload',type=Path,default=BASE/'tracking-payloads.json')
    parser.add_argument('--approve-upload',action='store_true')
    parser.add_argument('--anonymous-verify',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--batch-size',type=int,default=20)
    args=parser.parse_args(argv)
    from scripts.resources import require_resources
    require_resources();payload=local_payload(args.payload);records=payload['runs']
    if not 1<=args.batch_size<=20 or (args.limit is not None and args.limit<1):
        raise ValueError('Invalid upload limit/batch size')
    if args.anonymous_verify:
        proof=anonymous_readback(records)
        proof['payload_sha256']=payload['payload_sha256']
        _save_receipt(BASE/'anonymous-verification.json',proof)
        print(json.dumps(proof),flush=True);return
    records=records[:args.limit]
    if not args.approve_upload:
        print(json.dumps(dict(mode='offline-validation',records=len(records),writes=0)),flush=True);return
    build_batch(records[:1],approved=True)
    import wandb
    import requests
    api=wandb.Api(timeout=60)
    if api._service_api.base_url.rstrip('/')!=HOST:
        raise ValueError('Only the reviewed existing Forge project is authorized')
    raw_bytes=args.payload.stat().st_size+sum(r['bytes'] for r in payload['public_files'])
    _save_receipt(BASE/'capacity-check.json',assert_capacity(api,raw_bytes))
    remote=remote_runs(api);verify_legacy(remote);pending=[]
    for record in records:
        run=remote.get(record['id'])
        if run is None:
            pending.append(record)
        elif run.state=='finished':
            verify_values(record,run.config,run.summary,run.state)
        else:
            if any(k in run.config and not _config_equal(v,run.config[k]) for k,v in record['config'].items()):
                raise ValueError('Unfinished remote identity conflicts with verified payload')
            pending.append(record)
    session=requests.Session()  # Existing host-scoped netrc, never printed.
    def complete(record):
        complete_summary(record,session=session)
    done=len(records)-len(pending)
    print(json.dumps(dict(verified_existing=done,pending=len(pending))),flush=True)
    for offset in range(0,len(pending),args.batch_size):
        batch=pending[offset:offset+args.batch_size]
        import_batch(batch,BASE/'receipts',approved=True,execute=api._service_api.execute_graphql,complete=complete)
        done+=len(batch);print(json.dumps(dict(uploaded_or_verified=done,total=len(records))),flush=True)
    remote=remote_runs(wandb.Api(timeout=60));preserved=verify_legacy(remote)
    for record in records:
        run=remote.get(record['id'])
        if run is None:
            raise ValueError('Run missing in fresh authenticated readback')
        verify_values(record,run.config,run.summary,run.state)
        path=BASE/'receipts'/(record['id']+'.json')
        receipt=json.loads(path.read_text()) if path.exists() else dict(id=record['id'],payload_sha256=record['payload_sha256'])
        if receipt['payload_sha256']!=record['payload_sha256']:
            raise ValueError('Receipt payload conflict during readback')
        receipt.update(status='finished',verified_at=now(),url=PROJECT_URL+'/runs/'+record['id'])
        _save_receipt(path,receipt)
    proof=dict(verified_at=now(),project_url=PROJECT_URL,verified_runs=len(records),
        legacy_runs_preserved=preserved,server_runs=len(remote),public_verified=False,
        payload_sha256=payload['payload_sha256'],paid_api_calls=0,
        metrics_verified=sum(len(r['metrics']) for r in records),
        families=dict(Counter(r['config']['experiment_family'] for r in records)))
    _save_receipt(BASE/'authenticated-verification.json',proof)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':
    main()
