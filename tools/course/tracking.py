"""Prepare public-safe W&B records; offline by default, explicit online publication."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from tools.course.analyze import ROOT, read_rows, sha256, verify_package, write_json

CONFIG_KEYS = {'experiment_family', 'source_run_id', 'source_manifest_sha256',
    'source_execution_date', 'historical_import', 'paper_equivalent', 'synthetic',
    'counterfactual', 'domain', 'seed', 'variant', 'method', 'learning_fraction',
    'budget_fraction', 'budget', 'rounds', 'alpha', 'population', 'factor',
    'protocol_sha256', 'description'}
METRIC_KEYS = {'requested', 'served', 'total_reward', 'service_utility', 'mean_reward_served',
    'coverage', 'estimated_cost', 'actual_recorded_cost', 'actual_budget_excess',
    'learning_rounds', 'learning_total_reward', 'learning_empirical_oracle_regret',
    'low_count', 'high_count', 'regret', 'mean_reward', 'conditions',
    'over_budget_count', 'maximum_excess', 'total_excess'}


def digest(value):
    """Hash a strict canonical JSON value for stable tracking identities."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _payload(config, metrics, lab):
    """Wrap a sanitized record with a content-derived stable identity."""
    body = dict(config=config, metrics=metrics, lab=lab)
    fingerprint = digest(body)
    return dict(body, id='pilot-'+fingerprint[:24], payload_sha256=fingerprint)


def validate_payload(payload):
    """Validate the strict allowlist and content identity before SDK import or I/O."""
    if set(payload) != {'id', 'payload_sha256', 'config', 'metrics', 'lab'}:
        raise ValueError('Unexpected tracking fields')
    if set(payload['config'])-CONFIG_KEYS or set(payload['metrics'])-METRIC_KEYS:
        raise ValueError('Unknown or private field in payload')
    if not payload['metrics'] or payload['lab'] not in ('lab1-reproduction', 'lab2-critique'):
        raise ValueError('Missing metrics or invalid lab')
    if not all(type(v) in (int, float) and math.isfinite(v) for v in payload['metrics'].values()):
        raise ValueError('Metrics must be finite numbers')
    if any(type(v) not in (str, bool, int, float) for v in payload['config'].values()):
        raise ValueError('Config must have scalar values')
    body = {k: payload[k] for k in ('config', 'metrics', 'lab')}
    fingerprint = digest(body)
    if fingerprint != payload['payload_sha256'] or payload['id'] != 'pilot-'+fingerprint[:24]:
        raise ValueError('Payload checksum mismatch')
    if any(s in json.dumps(body).lower() for s in ('c:\\\\users', 'api_key', 'prompt_text', '@gmail')):
        raise ValueError('Private content in payload')
    return payload


def build_payloads(package, analysis):
    """Build deterministic aggregate-only records from two checksum-verified packages."""
    package, analysis = Path(package), Path(analysis)
    rows = read_rows(package)
    verify_package(analysis)
    main = json.loads((package/'aggregate/manifest.json').read_text(encoding='utf-8'))
    extra = json.loads((analysis/'analysis.json').read_text(encoding='utf-8'))
    if (extra['source_csv_sha256']!=sha256(package/'aggregate/per-seed.csv')
            or extra['package_integrity']['manifest_sha256']!=sha256(package/'package-files.json')):
        raise ValueError('Supplement/package lineage mismatch')
    executed=datetime.fromisoformat(extra['created_utc'])
    if executed.tzinfo is None or executed.utcoffset() is None:
        raise ValueError('Supplement execution timestamp requires timezone')
    execution_date=executed.astimezone(timezone.utc).date().isoformat()
    records = []
    for row in rows:
        config = {k: row[k] for k in ('domain', 'seed', 'variant', 'method', 'learning_fraction', 'budget_fraction', 'budget')}
        config.update(experiment_family='historical-deployment', source_run_id=row['id'],
            source_manifest_sha256=main['run_manifest_sha256'][row['id']],
            source_execution_date='2026-10-29', historical_import=True,
            paper_equivalent=False, synthetic=False, counterfactual=False,
            protocol_sha256=main['protocol_sha256'])
        records.append(_payload(config, {k: row[k] for k in METRIC_KEYS if k in row},
                         'lab1-reproduction' if row['variant'] == 'main' else 'lab2-critique'))
    source_hash = sha256(analysis/'analysis.json')
    for case in extra['stress']:
        for policy in case['policies']:
            config = dict(experiment_family='synthetic-magnitude', source_execution_date=execution_date,
                source_run_id=f"magnitude/{case['rounds']}/{case['alpha']}/{policy['method']}",
                source_manifest_sha256=source_hash, historical_import=True, synthetic=True,
                counterfactual=False, paper_equivalent=False, method=policy['method'],
                rounds=case['rounds'], alpha=case['alpha'], seed=case['seed'],
                protocol_sha256=extra['protocol_sha256'],
                description='Deterministic construction violating unit-direction reward assumption')
            records.append(_payload(config, {k: policy[k] for k in METRIC_KEYS if k in policy}, 'lab2-critique'))
    for population, cases in extra['cost_sensitivity'].items():
        for case in cases:
            config = dict(experiment_family='cost-accounting', source_execution_date=execution_date,
                source_run_id=f"cost/{population}/{case['factor']}", source_manifest_sha256=source_hash,
                historical_import=True, synthetic=False, counterfactual=case['counterfactual'],
                paper_equivalent=False, population=population, factor=case['factor'],
                protocol_sha256=extra['protocol_sha256'],
                description='Fixed choices; multiply recorded selected costs, not rerouted traffic')
            records.append(_payload(config, {k: case[k] for k in METRIC_KEYS if k in case}, 'lab2-critique'))
    for record in records:
        validate_payload(record)
    if len({p['id'] for p in records}) != len(records):
        raise ValueError('Duplicate tracking identity')
    return records


def _save_receipt(path, value):
    """Atomically persist the latest accepted run state."""
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def record_run(payload, receipt_dir, *, mode='offline', entity=None,
               project='pilot-llm-routing-team8', approved=False, sdk=None):
    """Write one W&B run and retain its identity immediately for safe continuation.

    Online mode requires explicit approval and entity. A receipt records planned,
    accepted and finished states. Fixed content-derived IDs prevent duplicate
    identities after interruption. Returns the receipt, not an assumed public URL.
    """
    validate_payload(payload)
    if mode not in ('offline', 'online'):
        raise ValueError('Unsupported W&B mode')
    if mode == 'online' and not approved:
        raise ValueError('Explicit online publication approval is required')
    if mode == 'online' and not entity:
        raise ValueError('Explicit W&B entity is required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', project) or (entity and not re.fullmatch(r'[A-Za-z0-9_-]+', entity)):
        raise ValueError('Invalid W&B destination')
    receipt_dir = Path(receipt_dir)/mode/(entity or 'local')/project
    receipt_path = receipt_dir/(payload['id']+'.json')
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding='utf-8'))
        if old['payload_sha256'] != payload['payload_sha256']:
            raise ValueError('Receipt conflicts with payload')
        if old['status'] == 'finished' and old.get('writer_version') == 2:
            return old
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = dict(id=payload['id'], payload_sha256=payload['payload_sha256'], mode=mode,
                   entity=entity, project=project, status='planned', url=None, writer_version=2)
    _save_receipt(receipt_path, receipt)
    if sdk is None:
        import wandb as sdk
    settings = sdk.Settings(disable_git=True, disable_code=True, save_code=False,
        x_disable_meta=True, x_disable_stats=True, x_disable_machine_info=True,
        disable_job_creation=True, console='off')
    kwargs = dict(entity=entity, project=project, id=payload['id'], name=payload['config']['source_run_id'],
        group=payload['lab'], job_type=payload['config']['experiment_family'],
        tags=[payload['lab'], 'historical-import', 'local-eight-arm'],
        config=payload['config'], mode=mode, dir=str(receipt_dir),
        save_code=False, settings=settings,
        notes='Imported from verified local evidence. Original execution date is in config. No raw prompts uploaded.')
    if mode == 'online':
        kwargs['resume'] = 'allow'
    run = sdk.init(**kwargs)
    if run.id != payload['id']:
        raise RuntimeError('W&B returned an unexpected run identity')
    receipt.update(status='accepted', url=run.url if mode == 'online' else None)
    _save_receipt(receipt_path, receipt)
    metrics = dict(payload['metrics'])
    quality = [[k,metrics[k]] for k in ('service_utility','coverage','mean_reward_served','mean_reward') if k in metrics]
    if 'over_budget_count' in metrics:
        quality = [['over_budget_fraction', metrics['over_budget_count']/metrics['conditions']]]
    if quality:
        table = sdk.Table(columns=['metric','value'], data=quality)
        metrics['quality_summary'] = sdk.plot.bar(table, 'metric', 'value',
            title='Recorded condition summary (unit scale)')
    run.log(metrics)
    run.finish()
    receipt['status'] = 'finished'
    _save_receipt(receipt_path, receipt)
    return receipt


def main(argv=None):
    """Export or serially record verified conditions, offline unless explicitly approved."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=ROOT/'reports/full-experiments-2026-10-29')
    parser.add_argument('--analysis', type=Path, default=ROOT/'reports/course-submission-2026-10-30/analysis')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--mode', choices=('offline', 'online'), default='offline')
    parser.add_argument('--entity')
    parser.add_argument('--project', default='pilot-llm-routing-team8')
    parser.add_argument('--approve-public-upload', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--receipt-dir', type=Path, default=ROOT/'outputs/course-20261030/wandb-receipts')
    args = parser.parse_args(argv)
    records = build_payloads(args.package, args.analysis)
    if args.limit is not None and args.limit < 1:
        raise ValueError('limit must be positive')
    if args.output:
        write_json(args.output, dict(version=1, public_urls_pending=True, runs=records))
    if args.record:
        from scripts.resources import require_resources
        require_resources()
        for record in records[:args.limit]:
            receipt = record_run(record, args.receipt_dir, mode=args.mode, entity=args.entity,
                project=args.project, approved=args.approve_public_upload)
            print(json.dumps({k: receipt[k] for k in ('id', 'status', 'mode', 'url')}), flush=True)
    else:
        print(json.dumps(dict(mode='dry-run', records=len(records), output=str(args.output))))


if __name__ == '__main__':
    main()
