$ErrorActionPreference = 'Stop'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$base='outputs/openai-core-20261031'
$protocol=Get-Content -LiteralPath "$base/protocol.json" -Raw | ConvertFrom-Json
$scriptHash=(Get-FileHash -Algorithm SHA256 -LiteralPath tools/research/verify_runs.py).Hash.ToLowerInvariant()
$count=0
foreach ($domain in @('routerbench','mmlu')) {
  foreach ($seed in @(7,42,2026)) {
    foreach ($method in $protocol.tuning.$domain.selected_parameters.PSObject.Properties.Name) {
      $destination="$base/verification/$domain-seed$seed-$method-main.json"
      if (Test-Path -LiteralPath $destination) {
        $existing=Get-Content -LiteralPath $destination -Raw | ConvertFrom-Json
        $learn=Get-Content -LiteralPath "$base/runs/learning/$domain/seed-$seed/main/$method/manifest.json" -Raw | ConvertFrom-Json
        if (-not $existing.verified -or $existing.script_sha256 -ne $scriptHash -or $existing.learning_manifest_sha256 -ne $learn.manifest_sha256) {
          throw "Stale verification: $destination"
        }
      } else {
        & .venv/Scripts/python.exe -X utf8 -B tools/research/verify_runs.py --domain $domain --seed $seed --method $method --base "$base/runs" --protocol "$base/protocol.json" --index "$base/prior-index.json" --cache-dir outputs/openai-standard-20261030/imported/caches/source-0 --output $destination
        if ($LASTEXITCODE -ne 0) { throw "Replay failed for $domain / $seed / $method" }
      }
      $count++
      Write-Output "Replay complete $count/78 $domain seed$seed $method"
    }
  }
}
