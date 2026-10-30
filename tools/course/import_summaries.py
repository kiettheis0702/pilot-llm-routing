"""Import verified historical aggregates as finished W&B runs, in serial batches.

This records scalar summaries, not invented training histories. The GraphQL
schema is supplied by W&B; its pinned SDK supplies authentication/transport.
No wandb.init, machine metadata, source code or model API calls are involved.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from tools.course.analyze import ROOT
from tools.course.tracking import build_payloads, validate_payload, _save_receipt


def build_batch(records, *, entity, project, approved=False):
    """Validate before constructing any external mutation."""
    if not approved:
        raise ValueError('Explicit online upload approval required')
    if not entity or not project or any(not re.fullmatch(r'[A-Za-z0-9_-]+', v)
                                       for v in (entity, project)):
        raise ValueError('Invalid destination')
    if not 1 <= len(records) <= 20 or len({r['id'] for r in records}) != len(records):
        raise ValueError('Batch must contain 1–20 unique records')
    variables = {}
    for index, record in enumerate(records):
        validate_payload(record)
        variables[f'r{index}'] = dict(
            entityName=entity, modelName=project, name=record['id'],
            displayName=record['config']['source_run_id'], state='running',
            config=json.dumps({k: {'value': v} for k, v in record['config'].items()}, allow_nan=False),
            summaryMetrics=json.dumps(record['metrics'], allow_nan=False),
            groupName=record['lab'], jobType=record['config']['experiment_family'],
            tags=[record['lab'], 'historical-import', 'local-eight-arm'],
            notes='Historical aggregate import from checksum-verified local evidence. '
                  'Original execution date is in config. No training history was reconstructed; '
                  'no raw prompts or machine metadata uploaded.')
    declarations = ','.join(f'${key}:UpsertBucketInput!' for key in variables)
    fields = ' '.join(f'{key}:upsertBucket(input:${key})'
                     '{bucket{id name state config summaryMetrics}}' for key in variables)
    return f'mutation ImportHistoricalSummaries({declarations}){{{fields}}}', variables


def _config_equal(expected, actual):
    # W&B JSON readback may round a floating-point value by one representable
    # step, e.g. 7.6737414600000005 -> 7.67374146. Keep other scalars exact.
    if type(expected) is float:
        return (type(actual) in (int, float) and math.isfinite(actual)
                and abs(actual - expected) <= math.ulp(expected))
    return type(actual) is type(expected) and actual == expected


def verify_values(record, config, metrics, state, *, require_finished=True):
    """Reject missing/changed fields or incomplete server state on readback."""
    if require_finished and state != 'finished':
        raise ValueError(f"Run not finished: {record['id']}")
    for key, value in record['config'].items():
        if key not in config or not _config_equal(value, config[key]):
            raise ValueError(f"Config mismatch: {record['id']} / {key}")
    for key, value in record['metrics'].items():
        actual = metrics.get(key)
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(
                actual, value, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Metric mismatch: {record['id']} / {key}")


def import_batch(records, receipt_dir, *, entity, project, approved=False, execute, complete=None):
    """Persist identities before sending; persist acceptance before verification."""
    query, variables = build_batch(records, entity=entity, project=project, approved=approved)
    if not callable(execute) or not callable(complete):
        raise ValueError('Authenticated upload and completion transports required')
    folder = Path(receipt_dir)/'online'/entity/project
    folder.mkdir(parents=True, exist_ok=True)
    receipts = []
    for record in records:
        path = folder/(record['id']+'.json')
        if path.exists() and json.loads(path.read_text())['payload_sha256'] != record['payload_sha256']:
            raise ValueError('Receipt conflicts with verified payload')
        receipt = dict(id=record['id'], payload_sha256=record['payload_sha256'],
                       mode='online', entity=entity, project=project, status='planned',
                       writer_version=3, import_method='aggregate-summary',
                       url=f"https://forge.coreweave.com/wandb/{entity}/{project}/runs/{record['id']}")
        receipts.append((path, receipt))
    for path, receipt in receipts:
        _save_receipt(path, receipt)
    result = execute(query, variables)
    # Store every accepted identity before processing any returned metric data.
    for index, (path, receipt) in enumerate(receipts):
        bucket = (result.get(f'r{index}') or {}).get('bucket') or {}
        if bucket.get('name') != receipt['id'] or not bucket.get('id'):
            continue
        receipt.update(status='accepted', server_id=bucket['id'])
        _save_receipt(path, receipt)
    for index, (path, receipt) in enumerate(receipts):
        if receipt['status'] != 'accepted':
            raise ValueError('Missing or incorrect server run identity')
        bucket = result[f'r{index}']['bucket']
        config = {k: v['value'] for k, v in json.loads(bucket['config']).items()}
        verify_values(records[index], config, json.loads(bucket['summaryMetrics']), bucket['state'],
                      require_finished=False)
        complete(records[index])
        receipt.update(status='uploaded')
        _save_receipt(path, receipt)
    return [receipt for _, receipt in receipts]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entity', required=True)
    parser.add_argument('--project', default='pilot-llm-routing-team8')
    parser.add_argument('--approve-upload', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--receipt-dir', type=Path, default=ROOT/'outputs/course-20261030/wandb-receipts')
    parser.add_argument('--verification', type=Path, required=True)
    args = parser.parse_args(argv)
    records = build_payloads(ROOT/'reports/full-experiments-2026-10-29',
                             ROOT/'reports/course-submission-2026-10-30/analysis')
    frozen = json.loads((ROOT/'reports/course-submission-2026-10-30/tracking-payloads.json').read_text())['runs']
    if records != frozen:
        raise ValueError('Frozen upload payload differs from verified evidence')
    if not 1 <= args.batch_size <= 20 or (args.limit is not None and args.limit < 1):
        raise ValueError('Invalid batch size or limit')
    records = records[:args.limit]
    build_batch(records[:1], entity=args.entity, project=args.project, approved=args.approve_upload)
    from scripts.resources import require_resources
    require_resources()
    import wandb
    api = wandb.Api(timeout=60)
    if api._service_api.base_url.rstrip('/') != 'https://api.forge.coreweave.com':
        raise ValueError('This approved import targets the reviewed W&B cloud only')
    import requests
    session = requests.Session()  # Uses the existing host-scoped _netrc credential.
    def complete(record):
        # W&B marks completion through file_stream, not upsertBucket.
        # Offset 0 also makes an interrupted import of this one aggregate idempotent.
        metrics = dict(record['metrics'], _step=0, _runtime=0)
        response = session.post(
            f'https://api.forge.coreweave.com/files/{args.entity}/{args.project}/{record["id"]}/file_stream',
            json={'files': {
                'wandb-summary.json': {'offset': 0, 'content': [json.dumps(metrics, allow_nan=False)]},
                'wandb-history.jsonl': {'offset': 0, 'content': [json.dumps(metrics, allow_nan=False)]}},
                  'complete': True, 'exitcode': 0}, timeout=(10, 60), allow_redirects=False)
        response.raise_for_status()
        if not 200 <= response.status_code < 300:
            raise ValueError('Unexpected redirect while completing a run')
    destination = f'{args.entity}/{args.project}'
    remote = {run.id: run for run in api.runs(destination, per_page=100, lazy=False)}
    pending = []
    for record in records:
        run = remote.get(record['id'])
        if run is None:
            pending.append(record)
        elif run.state == 'finished':
            verify_values(record, run.config, run.summary, run.state)
        else:
            if any(key in run.config and not _config_equal(value, run.config[key])
                   for key, value in record['config'].items()):
                raise ValueError('Existing unfinished run conflicts with the payload')
            pending.append(record)
    done = len(records)-len(pending)
    print(json.dumps(dict(verified_existing=done, pending=len(pending))), flush=True)
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset:offset+args.batch_size]
        import_batch(batch, args.receipt_dir, entity=args.entity, project=args.project,
                     approved=args.approve_upload, execute=api._service_api.execute_graphql,
                     complete=complete)
        done += len(batch)
        print(json.dumps(dict(uploaded=done, total=len(records))), flush=True)
    # New API instance avoids cached pagination/summary data for independent readback.
    remote = {run.id: run for run in wandb.Api(timeout=60).runs(destination, per_page=100, lazy=False)}
    for record in records:
        run = remote.get(record['id'])
        if run is None:
            raise ValueError('Server is missing an expected run')
        verify_values(record, run.config, run.summary, run.state)
        receipt_path = args.receipt_dir/'online'/args.entity/args.project/(record['id']+'.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['payload_sha256'] != record['payload_sha256']:
                raise ValueError('Receipt conflicts with verified payload')
            receipt.update(status='finished', verified_at=datetime.now(timezone.utc).isoformat())
            _save_receipt(receipt_path, receipt)
    evidence = dict(verified_at=datetime.now(timezone.utc).isoformat(),
                    project_url=f'https://forge.coreweave.com/wandb/{destination}',
                    expected_runs=len(records), verified_runs=len(records), server_runs=len(remote),
                    metrics_verified=sum(len(r['metrics']) for r in records),
                    families=dict(Counter(r['config']['experiment_family'] for r in records)),
                    config_float_tolerance='one ULP; all other scalar types/values exact',
                    public_verified=False, paid_api_calls=0,
                    runs=[dict(id=r['id'], payload_sha256=r['payload_sha256'],
                               url=f"https://forge.coreweave.com/wandb/{destination}/runs/{r['id']}")
                          for r in records])
    args.verification.parent.mkdir(parents=True, exist_ok=True)
    _save_receipt(args.verification, evidence)
    print(json.dumps({k: v for k, v in evidence.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    main()
