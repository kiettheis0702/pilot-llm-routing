param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('inputs','preference','mutation','parents','ablations','budget','noise','binary','summary')]
    [string]$Stage
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArguments)
    & .venv/Scripts/python.exe @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Python stage failed with exit code $LASTEXITCODE" }
}
Invoke-CheckedPython -c 'from scripts.resources import require_resources; print(require_resources())'
$core = 'outputs/openai-core-20261031'
$followup = 'outputs/openai-followup-20261031'
$readback = 'outputs/ci-reproduction/readback'
New-Item -ItemType Directory -Path $readback -Force | Out-Null
switch ($Stage) {
    'inputs' {
        $release = Get-Content configs/openai-reproduction-release.json -Raw | ConvertFrom-Json
        $archive = Join-Path '.cache' $release.filename
        if (-not (Test-Path -LiteralPath $archive)) { Invoke-WebRequest -Uri $release.url -OutFile $archive }
        Invoke-CheckedPython -m tools.reproduction_bundle verify --archive $archive
        Invoke-CheckedPython -m tools.reproduction_bundle restore --archive $archive
        Invoke-CheckedPython -m tools.reproduction_bundle restore --archive $archive
        Invoke-CheckedPython -m tools.openai_study_reproduce prepare
        Invoke-CheckedPython -m tools.openai_study_reproduce prepare
        Invoke-CheckedPython -m tools.openai_followup_assets --sources
        Invoke-CheckedPython -m tools.openai_study_routellm_train freeze
        Invoke-CheckedPython -m tools.openai_study_reproduce costs
    }
    'preference' {
        foreach ($seed in @(42,7,2026)) { Invoke-CheckedPython -m tools.openai_study_reproduce preference --seed $seed }
    }
    'mutation' { Invoke-CheckedPython -m tools.openai_study_reproduce mutation }
    'parents' {
        Invoke-CheckedPython -m tools.openai_study_reproduce parents
        foreach ($domain in @('routerbench','mmlu')) {
            foreach ($seed in @(7,42,2026)) {
                Invoke-CheckedPython tools/research/verify_runs.py --domain $domain --seed $seed --method pilot --base "$core/runs" --protocol "$core/protocol.json" --index "$core/prior-index.json" --cache-dir outputs/openai-standard-20261030/imported/caches/source-0 --output "$readback/core-$domain-$seed.json"
            }
        }
    }
    'ablations' {
        Invoke-CheckedPython -m tools.openai_study_followup fit8
        Invoke-CheckedPython -m tools.openai_study_followup freeze
        Invoke-CheckedPython -m tools.openai_study_followup run --limit-groups 5
        $variants = (Get-Content configs/ablation_plan.json -Raw | ConvertFrom-Json).variants
        foreach ($variant in $variants) {
            Invoke-CheckedPython tools/research/verify_runs.py --domain routerbench --seed 7 --method pilot --variant $variant.id --base "$followup/ablations/runs" --protocol "$followup/ablations/protocol.json" --index "$followup/ablations/prior-index.json" --cache-dir outputs/openai-standard-20261030/imported/caches/source-0 --output "$readback/ablation-$($variant.id).json"
        }
    }
    'budget' {
        Invoke-CheckedPython -m tools.openai_study_budget freeze
        Invoke-CheckedPython -m tools.openai_study_budget run --limit 40
        Invoke-CheckedPython -m tools.openai_study_budget_verify --limit 40
    }
    'noise' {
        Invoke-CheckedPython -m tools.openai_study_noise freeze
        Invoke-CheckedPython -m tools.openai_study_noise run --limit 2
        Invoke-CheckedPython -m tools.openai_study_noise_verify --limit 2
    }
    'binary' {
        foreach ($action in @('declare','tune','freeze')) { Invoke-CheckedPython -m tools.openai_study_binary_run $action }
        Invoke-CheckedPython -m tools.openai_study_binary_verify tuning
        # The frozen producer iterates seven methods per seed. Its first method
        # is epoch-greedy, so a one-group limit does not produce a PILOT trace.
        $protocol = Get-Content "$followup/binary/protocol.json" -Raw | ConvertFrom-Json
        $methods = @($protocol.tuning.routerbench.selected_parameters.PSObject.Properties.Name)
        if ($methods.Count -ne 7 -or 'pilot' -notin $methods) { throw 'Unexpected binary method inventory' }
        Invoke-CheckedPython -m tools.openai_study_binary_run run --limit 7
        foreach ($method in $methods) {
            Invoke-CheckedPython -m tools.openai_study_binary_verify --domain routerbench --seed 7 --method $method --output "$readback/binary-$method-seed7.json"
        }
    }
    'summary' {
        @'
import csv, hashlib, json, math, os, sys
from pathlib import Path
from tools.openai_study_reproduce import record_ci_summary, sha
root=Path.cwd()
def read(path): return json.loads(path.read_text(encoding='utf-8'))
checks=root/'outputs/followup-reproduction-checks'
readback=root/'outputs/ci-reproduction/readback'
summary=dict(commit=os.environ.get('GITHUB_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'),
    python=sys.version,paid_api_calls=0,full_grid_rerun=False,fresh_cuda_environment_tested=False)
for family,pattern,package,expected in [('core','core-*.json','openai-core',6),
    ('ablations','ablation-*.json','openai-ablations',5),('binary','binary-*.json','openai-binary',7)]:
    proofs=[read(p) for p in sorted(readback.glob(pattern))]
    assert len(proofs)==expected and all(p['verified'] for p in proofs)
    with (root/f'reports/{package}-2026-10-31/per-seed.csv').open(encoding='utf-8',newline='') as stream:
        published={r['id']:r for r in csv.DictReader(stream)}
    conditions=0
    for proof in proofs:
        for result in proof['inventory']:
            original=published[result['id']]
            for key,column in [('served','served'),('reward','total_reward'),('actual_cost','actual_recorded_cost'),('estimated_cost','estimated_cost')]:
                assert math.isclose(result[key],float(original[column]),rel_tol=1e-10,abs_tol=1e-9),(family,result['id'],key)
            conditions+=1
    summary[family]=dict(groups=len(proofs),deployment_conditions=conditions,
        learning_rounds=sum(p['learning_rounds'] for p in proofs),all_independently_replayed=True,matches_published=True)
summary['preference']=[read(checks/f'preference-seed-{s}.json') for s in (7,42,2026)]
summary['mutation']=read(checks/'preference-mutation.json')
summary['cost_shocks']=dict(conditions=len(read(checks/'costs.json')['records']),verified=True)
for name,count in [('budget',40),('noise',2)]:
    proof=read(root/f'outputs/openai-followup-20261031/{name}/independent-smoke-{count}.json')
    assert proof['verified'] and proof['conditions']==count and not proof['complete']
    summary[name]=dict(conditions=count,verified=True,full_inventory_complete=False)
packages=0
for manifest in (root/'reports').glob('*/package-files.json'):
    for name,want in read(manifest)['files'].items():
        with (manifest.parent/name).open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
        assert actual==want
    packages+=1
summary['unchanged_report_packages']=packages
summary['source_sha256']={name:sha(root/name) for name in (
    'tools/openai_study_reproduce.py','tools/reproduction-ci.ps1',
    'configs/hybrid-batch-profile-2026-10-31.json','uv.lock')}
record_ci_summary(summary)
print('REPRODUCTION_EVIDENCE_JSON='+json.dumps(summary,separators=(',',':')))
'@ | & .venv/Scripts/python.exe -X utf8 -
        if ($LASTEXITCODE -ne 0) { throw "Evidence summary failed with exit code $LASTEXITCODE" }
    }
}
