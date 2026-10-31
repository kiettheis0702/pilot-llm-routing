param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('core','followups','mf')]
    [string]$Scope
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
function Invoke-CorePython {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArguments)
    & .venv/Scripts/python.exe @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Core Python command failed: $PythonArguments" }
}
function Invoke-NeuralPython {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArguments)
    & .venv-neural/Scripts/python.exe @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Neural Python command failed: $PythonArguments" }
}
Invoke-CorePython -c 'from scripts.resources import require_resources; print(require_resources())'
$core = 'outputs/openai-core-20261031'
$followup = 'outputs/openai-followup-20261031'
$readback = 'outputs/full-reproduction/readback'
New-Item -ItemType Directory -Path $readback -Force | Out-Null
switch ($Scope) {
    'core' {
        # Keep every restored input before refitting at the original relative paths.
        $root = [IO.Path]::GetFullPath((Get-Location).Path) + [IO.Path]::DirectorySeparatorChar
        $original = [IO.Path]::GetFullPath((Join-Path $root $core))
        $backup = [IO.Path]::GetFullPath((Join-Path $root 'outputs/full-reproduction/original-core-inputs'))
        if (-not $original.StartsWith($root,[StringComparison]::OrdinalIgnoreCase) -or
            -not $backup.StartsWith($root,[StringComparison]::OrdinalIgnoreCase)) { throw 'Core move must stay inside this checkout' }
        if (Test-Path -LiteralPath $backup) { throw 'Original core inputs already preserved; use a fresh checkout' }
        if (-not (Test-Path -LiteralPath "$original/embedding-verification.json")) { throw 'Restore core inputs first' }
        Move-Item -LiteralPath $original -Destination $backup
        New-Item -ItemType Directory -Path $original | Out-Null
        Copy-Item -LiteralPath "$backup/embedding-verification.json" -Destination "$original/embedding-verification.json"
        foreach ($phase in @('fit','tune','freeze')) { Invoke-CorePython -m tools.openai_study $phase }
        Invoke-CorePython -m tools.openai_study_full_reproduce core-inputs
        Invoke-CorePython -m tools.openai_study run
        $protocol = Get-Content "$core/protocol.json" -Raw | ConvertFrom-Json
        foreach ($domain in @('routerbench','mmlu')) {
            $methods = @($protocol.tuning.$domain.selected_parameters.PSObject.Properties.Name)
            if ($methods.Count -ne 13) { throw 'Unexpected core method inventory' }
            foreach ($seed in @(7,42,2026)) {
                foreach ($method in $methods) {
                    Invoke-CorePython tools/research/verify_runs.py --domain $domain --seed $seed --method $method --base "$core/runs" --protocol "$core/protocol.json" --index "$core/prior-index.json" --cache-dir outputs/openai-standard-20261030/imported/caches/source-0 --output "$readback/core-$domain-$seed-$method.json"
                }
            }
        }
    }
    'followups' {
        Invoke-CorePython -m tools.openai_study_reproduce parents
        foreach ($phase in @('fit8','freeze','run')) { Invoke-CorePython -m tools.openai_study_followup $phase }
        $variants = (Get-Content configs/ablation_plan.json -Raw | ConvertFrom-Json).variants
        foreach ($domain in @('routerbench','mmlu')) {
            foreach ($seed in @(7,42,2026)) {
                foreach ($variant in $variants) {
                    Invoke-CorePython tools/research/verify_runs.py --domain $domain --seed $seed --method pilot --variant $variant.id --base "$followup/ablations/runs" --protocol "$followup/ablations/protocol.json" --index "$followup/ablations/prior-index.json" --cache-dir outputs/openai-standard-20261030/imported/caches/source-0 --output "$readback/ablation-$domain-$seed-$($variant.id).json"
                }
            }
        }
        foreach ($phase in @('freeze','run')) { Invoke-CorePython -m tools.openai_study_budget $phase }
        Invoke-CorePython -m tools.openai_study_budget_verify
        foreach ($phase in @('freeze','run')) { Invoke-CorePython -m tools.openai_study_noise $phase }
        Invoke-CorePython -m tools.openai_study_noise_verify
        foreach ($phase in @('declare','tune','freeze')) { Invoke-CorePython -m tools.openai_study_binary_run $phase }
        Invoke-CorePython -m tools.openai_study_binary_verify tuning
        Invoke-CorePython -m tools.openai_study_binary_run run
        $protocol = Get-Content "$followup/binary/protocol.json" -Raw | ConvertFrom-Json
        $methods = @($protocol.tuning.routerbench.selected_parameters.PSObject.Properties.Name)
        if ($methods.Count -ne 7) { throw 'Unexpected binary method inventory' }
        foreach ($seed in @(7,42,2026)) {
            foreach ($method in $methods) {
                Invoke-CorePython -m tools.openai_study_binary_verify --domain routerbench --seed $seed --method $method --output "$readback/binary-$seed-$method.json"
            }
        }
    }
    'mf' {
        if (Test-Path -LiteralPath .venv-neural) { throw 'Use a fresh checkout for neural installation validation' }
        uv venv .venv-neural --python 3.11.18
        if ($LASTEXITCODE -ne 0) { throw 'Neural venv creation failed' }
        Invoke-CorePython -c 'from scripts.resources import require_resources; print(require_resources())'
        uv pip install --python .venv-neural/Scripts/python.exe torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) { throw 'Pinned PyTorch installation failed' }
        Invoke-CorePython -c 'from scripts.resources import require_resources; print(require_resources())'
        uv pip install --python .venv-neural/Scripts/python.exe -e '.[cost]' -r configs/followup-neural-requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Pinned neural requirements installation failed' }
        foreach ($seed in @(7,42,2026)) { Invoke-NeuralPython -m tools.openai_study_routellm_train run --seed $seed }
        Invoke-NeuralPython -m tools.openai_study_routellm_train run --seed 42 --mutation-probe
        Invoke-CorePython -m tools.openai_study_routellm_verify
    }
}
Invoke-CorePython -m tools.openai_study_full_reproduce $Scope
