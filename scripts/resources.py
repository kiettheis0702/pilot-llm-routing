"""Read-only local preflight. Never terminate or suspend another process."""
from __future__ import annotations

import json
import math
import os
import subprocess


def resource_snapshot():
    if os.name != 'nt':
        raise RuntimeError('Automatic resource preflight is currently implemented for Windows only')
    command = r"""
$ram = Get-CimInstance Win32_OperatingSystem
$processes = @(Get-CimInstance Win32_Process)
$own = $processes | Where-Object ProcessId -eq OWN_PID
$parent = $processes | Where-Object ProcessId -eq $own.ParentProcessId
$ignored = @(OWN_PID)
# Windows venv launchers wait on an interpreter with the same arguments.
$executable = '^(?:"[^"]+"|\S+)\s*'
if ($parent.Name -match '^python(w)?\.exe$' -and
    ($parent.CommandLine -replace $executable,'') -eq ($own.CommandLine -replace $executable,'')) {
  $ignored += $parent.ProcessId
}
$jobs = @($processes | Where-Object {
  $_.ProcessId -notin $ignored -and $_.Name -match '^(python|pythonw|node|pytest|uv)\.exe$' -and
  $_.CommandLine -match '(pytest|vitest|jest|playwright|torchrun|run_baselines|run_toy_pipeline|research_pipeline|local_embeddings|input_audit|cost_inputs|scripts\.(?:tuning|experiments)|\buv(?:\.exe)?["\s]+(?:sync|pip (?:install|sync)))'
} | Select-Object @{n='pid';e={$_.ProcessId}},@{n='parent_pid';e={$_.ParentProcessId}},Name)
@{available_gib=$ram.FreePhysicalMemory/1MB;heavy_jobs=$jobs} | ConvertTo-Json -Depth 4 -Compress
""".replace('OWN_PID', str(os.getpid()))
    result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
                            capture_output=True,text=True,check=True)
    return json.loads(result.stdout)


def require_resources(*, snapshot=None, minimum_gib=8.0):
    if isinstance(minimum_gib, bool) or not math.isfinite(minimum_gib) or minimum_gib < 8:
        raise ValueError('Minimum RAM threshold must be at least 8 GiB')
    snapshot = resource_snapshot() if snapshot is None else snapshot
    available = snapshot.get('available_gib')
    if not isinstance(available, (int,float)) or not math.isfinite(available):
        raise RuntimeError('Unable to measure physical RAM')
    if available < minimum_gib:
        raise RuntimeError(f'RAM preflight: {available:.2f} GiB available; need {minimum_gib:.2f} GiB')
    if snapshot.get('heavy_jobs'):
        raise RuntimeError('Another heavy job is running: ' + json.dumps(snapshot['heavy_jobs']))
    return snapshot
