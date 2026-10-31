"""Assemble local evidence/configs after the numerical report and QA are complete."""
import json
import shutil
from pathlib import Path
from tools.paper_preflight.common import ROOT, read_json, sha256, digest
from tools.openai_study import once
from scripts.experiments import expected_runs
from tools.course.analyze import verify_package

base=ROOT/'outputs/openai-core-20261031'
output=ROOT/'reports/openai-core-2026-10-31'
report=read_json(output/'manifest.json')
assert report['complete'] and report['conditions']==858 and report['independent_replays']==78
assert digest({k:v for k,v in report.items() if k!='manifest_sha256'})==report['manifest_sha256']
for name,want in report['files'].items():assert sha256(output/name)==want
evidence=output/'evidence';evidence.mkdir(exist_ok=False)
sources=['protocol.json','prior-index.json','embedding-verification.json','negative-control-results.json',
    'priors/seed-7/verification.json','priors/seed-42/verification.json','priors/seed-2026/verification.json',
    'tuning/routerbench/manifest.json','tuning/routerbench/calibration.json','tuning/routerbench/estimator.json',
    'tuning/mmlu/manifest.json','tuning/mmlu/calibration.json','tuning/mmlu/estimator.json',
    'verify-all.ps1','full-tests-final.log','full-tests-final-failed-winerror5.log',
    'cache-targeted-rerun.log','readback-final.json','readback.py','diagnose_lowbudget.py',
    'readback-failed-invocation.log','package_report.py']
for name in sources:
    target=evidence/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(base/name,target)
    assert sha256(target)==sha256(base/name)
protocol=read_json(base/'protocol.json');index=read_json(base/'prior-index.json')
entries={r['seed']:r for r in index['fits']}
with (output/'tracking-run-configs.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
    for condition in expected_runs(protocol):
        d=condition['domain'];seed=condition['seed']
        config=dict(condition,embedding_model='text-embedding-3-small',embedding_dimension=1536,
            projection_epochs=30,prior_epochs=40,arms=protocol['arms'],
            parameters=protocol['tuning'][d]['selected_parameters'][condition['method']],
            budget_bounds=protocol['tuning'][d]['bounds'],bin_size=100,
            absolute_budget=protocol['tuning'][d]['absolute_reference_budget']*condition['budget_fraction'],
            frozen_counts=protocol['primary_plan']['frozen_counts'][d],prior=entries[seed]['prior'],
            protocol_sha256=report['protocol_sha256'],synthetic=False,paper_equivalent=False,
            status='verified-local-only',uploaded=False)
        stream.write(json.dumps(config,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
once(output/'assembly.json',dict(script_sha256=sha256(__file__),local_only=True,
    extra_paid_api_calls=0,tracking_configurations=858))
once(output/'package-files.json',dict(files={p.relative_to(output).as_posix():sha256(p)
    for p in sorted(output.rglob('*')) if p.is_file() and p.name!='package-files.json'}))
print(verify_package(output))
