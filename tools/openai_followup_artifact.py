"""Prepare or publish a separate sanitized follow-up artifact, preserving legacy runs."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path

from tools.course.import_summaries import verify_values
from tools.course.tracking import _save_receipt
from tools.openai_followup_publish import (
    ROOT, BASE, ENTITY, PROJECT, HOST, local_payload, assert_capacity, now, complete_summary, build_batch,
)
from tools.openai_followup_tracking import file_sha

NAME='pilot-team8-openai-followup-results'
PUBLIC=ROOT/'reports/openai-publication-2026-10-31'
PRODUCERS={
    'core':'tools/openai_study.py',
    'ablations':'tools/openai_study_followup.py',
    'budget':'tools/openai_study_budget.py',
    'preference500':'tools/paper_preflight/train.py + tools/openai_study_followup_report.py',
    'routellm-mf':'tools/openai_study_routellm_train.py',
    'noise':'tools/openai_study_robustness.py',
    'binary':'tools/openai_study_binary_run.py',
    'cost-shock':'tools/openai_study_followup_report.py::costs',
    'hybrid-native':'tools/openai_study_hybrid_default_batch.py',
    'hybrid-fixed':'tools/openai_study_hybrid_default_batch.py',
    'task-shift':'tools/openai_study_hybrid_train.py + tools/openai_study_drift.py',
}


def write_once(path, data):
    encoded=json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8')!=encoded:
            raise ValueError('Existing artifact plan differs; inspect rather than replace')
    else:
        with path.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(encoded)


def prepare():
    payload=local_payload(BASE/'tracking-payloads.json')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    mapping=PUBLIC/'experiment-map.csv'
    fields=['recipe','family','source_run_id','source_manifest_sha256','source_csv_sha256','source_csv',
            'producer','wandb_run_id','wandb_url','status']
    csv_paths={r['sha256']:r['path'] for r in payload['public_files'] if r['path'].endswith('.csv')}
    import io
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    for record in payload['runs']:
        config=record['config'];family=config['experiment_family'];producer=PRODUCERS[family]
        if config['recipe']=='instructor':
            producer='tools/openai_study_instructor_study.py'
        if family=='task-shift' and config['recipe']=='deberta':
            producer='tools/openai_study_hybrid_train.py'
        row={k:config[k] for k in ('recipe','source_run_id','source_manifest_sha256','source_csv_sha256')}
        row.update(family=family,source_csv=csv_paths[config['source_csv_sha256']],producer=producer,wandb_run_id=record['id'],wandb_url='',
                   status='prepared; external publication not yet verified')
        writer.writerow(row)
    expected=stream.getvalue()
    if mapping.exists():
        # A later verified registry may add URLs/status, but its identity/provenance
        # fields must remain exactly the original prepared rows.
        with mapping.open(encoding='utf-8',newline='') as handle:
            existing=list(csv.DictReader(handle))
        proposed=list(csv.DictReader(io.StringIO(expected)))
        keys=fields[:-2]
        if [{k:r[k] for k in keys} for r in existing]!=[{k:r[k] for k in keys} for r in proposed]:
            raise ValueError('Existing experiment map provenance differs')
    else:
        with mapping.open('x',encoding='utf-8',newline='') as handle:
            handle.write(expected)
    files=[dict(item,artifact_path=item['path']) for item in payload['public_files']]
    files.append(dict(path=(BASE/'tracking-payloads.json').relative_to(ROOT).as_posix(),
        artifact_path='tracking-payloads.json',sha256=file_sha(BASE/'tracking-payloads.json'),
        bytes=(BASE/'tracking-payloads.json').stat().st_size))
    # Map is intentionally excluded until its external URLs have been verified.
    manifest=dict(version=1,artifact_name=NAME,summary_views=len(payload['runs']),
        payload_sha256=payload['payload_sha256'],paper_equivalent=False,paid_api_calls=0,
        excludes='raw prompts, responses, embeddings, checkpoints, machine/account metadata and presentation files',
        files={item['artifact_path']:dict(sha256=item['sha256'],bytes=item['bytes']) for item in files})
    write_once(BASE/'artifact-manifest.json',manifest)
    files.append(dict(path=(BASE/'artifact-manifest.json').relative_to(ROOT).as_posix(),
        artifact_path='manifest.json',sha256=file_sha(BASE/'artifact-manifest.json'),
        bytes=(BASE/'artifact-manifest.json').stat().st_size))
    if len({item['artifact_path'] for item in files})!=len(files) or sum(item['bytes'] for item in files)>100*1024**2:
        raise ValueError('Duplicate artifact file or reviewed100MiB limit exceeded')
    plan=dict(artifact_name=NAME,files=files,total_bytes=sum(item['bytes'] for item in files),
        manifest_sha256=file_sha(BASE/'artifact-manifest.json'),run_id=payload['runs'][0]['id'])
    write_once(BASE/'artifact-plan.json',plan)
    return payload,plan


def finalize_public_registry():
    """Add URLs only after actual signed-in and anonymous run/manifest readbacks."""
    payload,plan=prepare()
    auth=json.loads((BASE/'authenticated-verification.json').read_text())
    public=json.loads((BASE/'anonymous-verification.json').read_text())
    artifact=json.loads((BASE/'artifact-upload-receipt.json').read_text())
    artifact_public=json.loads((BASE/'anonymous-artifact-verification.json').read_text())
    for proof in (auth,public):
        if (proof['verified_runs']!=2498 or proof['legacy_runs_preserved']!=1128
                or proof['payload_sha256']!=payload['payload_sha256']):
            raise ValueError('Complete current run readback is required for the public registry')
    if (not public['public_verified'] or artifact['status']!='finished'
            or not artifact['remote_manifest_verified'] or artifact['manifest_sha256']!=plan['manifest_sha256']
            or not artifact_public['public_verified'] or artifact_public['manifest_sha256']!=plan['manifest_sha256']
            or artifact_public['artifact_id']!=artifact['artifact_id']):
        raise ValueError('Public readback or exact remote artifact verification is missing')
    mapping=PUBLIC/'experiment-map.csv'
    with mapping.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    for row in rows:
        row.update(wandb_url=f'https://forge.coreweave.com/wandb/{ENTITY}/{PROJECT}/runs/'+row['wandb_run_id'],
                   status='finished; config and metrics verified by anonymous API')
    temporary=mapping.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temporary.replace(mapping)
    _save_receipt(PUBLIC/'public-verification.json',public)
    _save_receipt(PUBLIC/'artifact-verification.json',artifact)
    _save_receipt(PUBLIC/'anonymous-artifact-verification.json',artifact_public)
    text=f'''# Verified OpenAI follow-up publication

The [existing Team8 project](https://forge.coreweave.com/wandb/{ENTITY}/{PROJECT})
now includes2,498 new condition or summary views, with recipe/family namespaces.
All new configs and {public['metrics_verified']:,} metric values were read back
through an unauthenticated API connection. The1,128 historical records were
also checked and preserved. These are historical-data aggregates, not new LLM
calls, independent datasets or2,498 newly trained models.

The [experiment map](experiment-map.csv) joins every new record to its source
CSV/checksum, producer and verified public URL. OpenAI core858+ablations240
remain1,098; Instructor858 and the named supplementary comparisons are separate.
The budget summary includes240 new policy conditions and does not re-import
the48 reused ablation controls. Task-shift windows overlap by design.

The [separate follow-up artifact]({artifact['url']}) contains{artifact['files']}
files ({artifact['bytes']:,} raw bytes). Its entire remote manifest was checked
against the exact prepared local sizes/content digests. It excludes raw prompts,
answers, paid embedding caches, checkpoints, account/machine metadata and all
presentation files. The old19-file artifact is preserved.

Public verification uses anonymous GraphQL with environment credentials,
netrc, cookies and authorization headers disabled. This is API evidence, not a
claim that a browser UI was inspected. The artifact manifest was also fetched
without credentials and every entry was compared. See public-verification.json,
artifact-verification.json and anonymous-artifact-verification.json. This proves
anonymous manifest access; it does not claim every file was downloaded again.

Eight-model scope and unverified author release/task/split/settings remain.
Hybrid/task-shift are declared local single-seed adaptations. Adverse outcomes
are retained. No paid model API requests, billing or visibility changes were made
by this publication. Moodle submission is outside this task.
'''
    (PUBLIC/'README.md').write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps(dict(registry_rows=len(rows),public_verified=True,artifact_files=artifact['files'])),flush=True)


def verify_remote(api, remote, plan):
    entries=remote.manifest.entries
    if set(entries)!={r['artifact_path'] for r in plan['files']}:
        raise ValueError('Remote artifact inventory differs')
    for item in plan['files']:
        path=ROOT/item['path'];entry=entries[item['artifact_path']]
        if file_sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:
            raise ValueError('Prepared local artifact changed')
        md5=base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()
        if entry.size!=item['bytes'] or entry.digest!=md5:
            raise ValueError('Remote artifact size/content digest differs')
    return dict(verified_at=now(),status='finished',artifact_name=NAME,
        artifact_id=remote.id,version=remote.version,qualified_name=remote.qualified_name,
        url=remote.url,manifest_sha256=plan['manifest_sha256'],
        files=len(plan['files']),bytes=plan['total_bytes'],remote_manifest_verified=True,
        paid_api_calls=0)


def anonymous_artifact(plan):
    import requests
    receipt=json.loads((BASE/'artifact-upload-receipt.json').read_text())
    if receipt['status']!='finished' or receipt['manifest_sha256']!=plan['manifest_sha256']:
        raise ValueError('Verified artifact upload required before anonymous check')
    session=requests.Session();session.trust_env=False
    query='query FetchArtifactManifest($id:ID!){artifact(id:$id){id currentManifest{file{directUrl}}}}'
    response=session.post(HOST+'/graphql',json=dict(query=query,variables=dict(id=receipt['artifact_id'])),
                          timeout=(10,60),allow_redirects=False)
    response.raise_for_status();body=response.json()
    if body.get('errors') or not body['data']['artifact']:
        raise ValueError('Anonymous artifact lookup failed')
    artifact=body['data']['artifact']
    if artifact['id']!=receipt['artifact_id']:
        raise ValueError('Anonymous artifact identity differs')
    url=artifact['currentManifest']['file']['directUrl']
    from urllib.parse import urlparse
    if urlparse(url).scheme!='https':
        raise ValueError('Expected HTTPS public manifest')
    downloaded=session.get(url,timeout=(10,60))
    downloaded.raise_for_status()
    for item in (response,downloaded):
        if 'Authorization' in item.request.headers or 'Cookie' in item.request.headers:
            raise ValueError('Artifact read unexpectedly authenticated')
    entries=downloaded.json()['contents']
    if set(entries)!={r['artifact_path'] for r in plan['files']}:
        raise ValueError('Anonymous manifest inventory differs')
    for item in plan['files']:
        path=ROOT/item['path'];entry=entries[item['artifact_path']]
        if file_sha(path)!=item['sha256'] or entry['size']!=item['bytes']:
            raise ValueError('Anonymous manifest/local source size or hash differs')
        if entry['digest']!=base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode():
            raise ValueError('Anonymous manifest content digest differs')
    proof=dict(verified_at=now(),verification_method='anonymous-graphql-and-manifest-fetch',
        authentication='none; trust_env=False; no Authorization or Cookie headers',
        artifact_id=receipt['artifact_id'],manifest_sha256=plan['manifest_sha256'],
        public_verified=True,manifest_entries_verified=len(entries),
        all_files_downloaded_again=False,paid_api_calls=0)
    _save_receipt(BASE/'anonymous-artifact-verification.json',proof)
    print(json.dumps(proof),flush=True)


def finish_artifact_evidence(api,remote,plan,record):
    """Verify the artifact and restore an interrupted resumed summary run."""
    evidence=verify_remote(api,remote,plan)
    current=api.run(f'{ENTITY}/{PROJECT}/{record["id"]}')
    verify_values(record,current.config,current.summary,current.state,require_finished=False)
    if current.state!='finished':
        complete_summary(record)
    import wandb
    fresh=wandb.Api(timeout=60)
    if fresh._service_api.base_url.rstrip('/')!=HOST:
        raise ValueError('Wrong artifact recovery readback host')
    current=fresh.run(f'{ENTITY}/{PROJECT}/{record["id"]}')
    verify_values(record,current.config,current.summary,current.state)
    evidence.update(schema_version=2,submission_attempted=True,run_id=record['id'],
                    run_payload_sha256=record['payload_sha256'],summary_run_verified_finished=True)
    return evidence


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approve-upload',action='store_true')
    parser.add_argument('--finalize-registry',action='store_true')
    parser.add_argument('--anonymous-verify',action='store_true')
    args=parser.parse_args(argv)
    from scripts.resources import require_resources
    require_resources()
    if sum((args.approve_upload,args.finalize_registry,args.anonymous_verify))>1:
        raise ValueError('Keep upload, anonymous readback and local finalization separate')
    if args.finalize_registry:
        if args.approve_upload:
            raise ValueError('Registry finalization is local; keep upload and finalization separate')
        finalize_public_registry();return
    payload,plan=prepare()
    if args.anonymous_verify:
        anonymous_artifact(plan);return
    if not args.approve_upload:
        print(json.dumps(dict(mode='offline-plan',files=len(plan['files']),bytes=plan['total_bytes'],writes=0)),flush=True);return
    proof=json.loads((BASE/'authenticated-verification.json').read_text())
    if proof['verified_runs']!=2498 or proof['payload_sha256']!=payload['payload_sha256'] or proof['legacy_runs_preserved']!=1128:
        raise ValueError('Full authenticated summary readback required before artifact upload')
    import wandb
    api=wandb.Api(timeout=60)
    if api._service_api.base_url.rstrip('/')!=HOST:
        raise ValueError('Wrong publication host')
    _save_receipt(BASE/'artifact-capacity-check.json',assert_capacity(api,plan['total_bytes']))
    record=payload['runs'][0];run=api.run(f'{ENTITY}/{PROJECT}/{record["id"]}')
    receipt_path=BASE/'artifact-upload-receipt.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if receipt:
        if (receipt['manifest_sha256']!=plan['manifest_sha256']
                or receipt.get('run_id',record['id'])!=record['id']
                or receipt.get('run_payload_sha256',record['payload_sha256'])!=record['payload_sha256']):
            raise ValueError('Existing artifact receipt conflicts with plan')
    # Only a matching receipt allows recovery of a run left running/crashed by
    # our earlier resume. Every expected config and metric still has to match.
    verify_values(record,run.config,run.summary,run.state,require_finished=receipt is None)
    if receipt and receipt.get('submission_attempted',True):
        # Never resubmit after an interrupted accepted request. Recover by its
        # fixed name/version. Missing/legacy phase information is conservative:
        # a failed/unknown lookup never authorizes a new artifact submission.
        if receipt.get('version'):
            remote=api.artifact(f'{ENTITY}/{PROJECT}/{NAME}:{receipt["version"]}',type='evaluation-results')
        else:
            remote=api.artifact(f'{ENTITY}/{PROJECT}/{NAME}:verified-2026-10-31',type='evaluation-results')
        evidence=finish_artifact_evidence(api,remote,plan,record)
        _save_receipt(receipt_path,evidence);print(json.dumps(evidence),flush=True);return
    for item in plan['files']:
        path=ROOT/item['path']
        if not path.resolve().is_relative_to(ROOT) or file_sha(path)!=item['sha256']:
            raise ValueError('Artifact input escapes workspace or changed')
    receipt=dict(receipt or {},schema_version=2,status='planned',submission_attempted=False,
        manifest_sha256=plan['manifest_sha256'],run_id=record['id'],
        run_payload_sha256=record['payload_sha256'],planned_at=(receipt or {}).get('planned_at',now()))
    _save_receipt(receipt_path,receipt)
    temporary=ROOT/'.cache/wandb-upload-temp';temporary.mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=str(temporary);os.environ['TMP']=str(temporary)
    settings=wandb.Settings(disable_git=True,disable_code=True,save_code=False,x_disable_meta=True,
        x_disable_stats=True,x_disable_machine_info=True,disable_job_creation=True,console='off')
    labels=build_batch([record],approved=True)[1]['r0']
    # A GraphQL-imported aggregate can exist without SDK initialization metadata.
    # The public API above already proved this exact ID/config/summary exists.
    # Use the documented existing-run artifact recipe (resume='allow'), while
    # explicitly carrying the same verified values and labels into the SDK.
    run=wandb.init(entity=ENTITY,project=PROJECT,id=record['id'],resume='allow',mode='online',
        config=dict(record['config']),allow_val_change=True,name=labels['displayName'],
        group=labels['groupName'],job_type=labels['jobType'],tags=labels['tags'],notes=labels['notes'],
        dir=str(ROOT/'outputs/wandb-artifacts'),save_code=False,settings=settings)
    if run.id!=record['id']:
        raise ValueError('Artifact SDK resumed wrong run identity')
    receipt.update(status='initialized',initialized_at=now());_save_receipt(receipt_path,receipt)
    run.summary.update(record['metrics'])
    artifact=wandb.Artifact(NAME,type='evaluation-results',
        description='Verified local OpenAI/Instructor/DeBERTa follow-up aggregate evidence. '
                    '2498 condition or summary views; not exact paper reproduction. No raw prompts.',
        metadata=dict(summary_views=2498,paper_equivalent=False,manifest_sha256=plan['manifest_sha256']))
    for item in plan['files']:
        artifact.add_file(str(ROOT/item['path']),name=item['artifact_path'],policy='immutable')
    # Persist BEFORE invoking the submission transport. An exception from the
    # call can mean acceptance with a lost response; it must not be resubmitted.
    receipt.update(status='submitting',submission_attempted=True,submission_started_at=now())
    _save_receipt(receipt_path,receipt)
    logged=run.log_artifact(artifact,aliases=['verified-2026-10-31'])
    receipt.update(status='accepted',accepted_at=now());_save_receipt(receipt_path,receipt)
    logged.wait(timeout=120)
    receipt.update(artifact_id=logged.id,version=logged.version,qualified_name=logged.qualified_name,url=logged.url)
    _save_receipt(receipt_path,receipt);run.finish()
    api=wandb.Api(timeout=60)
    remote=api.artifact(f'{ENTITY}/{PROJECT}/{NAME}:{logged.version}',type='evaluation-results')
    evidence=finish_artifact_evidence(api,remote,plan,record)
    _save_receipt(receipt_path,evidence);print(json.dumps(evidence),flush=True)


if __name__=='__main__':
    main()
