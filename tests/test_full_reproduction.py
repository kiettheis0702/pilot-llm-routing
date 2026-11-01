"""A bounded replay must not accidentally certify an entire published grid."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import venv
import pytest


def sample():
    published = [dict(id=name, served='2', total_reward='1.5',
        actual_recorded_cost='0.2', estimated_cost='0.25') for name in ('a','b')]
    proof = dict(verified=True, inventory=[dict(id=name, served=2, reward=1.5,
        actual_cost=.2, estimated_cost=.25) for name in ('b','a')])
    return [proof], published


def test_complete_reordered_inventory_matches_without_mutating_inputs():
    from tools.openai_study_full_reproduce import verify_full_inventory
    proofs, published = sample()
    original = copy.deepcopy((proofs,published))
    result = verify_full_inventory(proofs, published)
    assert result['verified'] and result['matches_published'] and result['conditions'] == 2
    assert (proofs,published) == original


@pytest.mark.parametrize('fault', ['missing','duplicate','unknown','unverified',
    'metric','nonfinite','duplicate_reference'])
def test_incomplete_or_inconsistent_replay_cannot_certify_full_grid(fault):
    from tools.openai_study_full_reproduce import verify_full_inventory
    proofs, published = sample()
    rows = proofs[0]['inventory']
    if fault == 'missing': rows.pop()
    elif fault == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif fault == 'unknown': rows[0]['id'] = 'foreign-condition'
    elif fault == 'unverified': proofs[0]['verified'] = False
    elif fault == 'metric': rows[0]['reward'] = 1.4
    elif fault == 'nonfinite': rows[0]['reward'] = float('nan')
    elif fault == 'duplicate_reference': published.append(copy.deepcopy(published[0]))
    with pytest.raises(ValueError):
        verify_full_inventory(proofs, published)


def restored_core(root):
    core = root / 'outputs/openai-core-20261031'
    core.mkdir(parents=True)
    files = []
    for name in ('embedding-verification.json', 'prior-index.json'):
        path = core / name
        path.write_bytes(b'{}')
        files.append(dict(path=path.relative_to(root).as_posix(), size=2,
                          sha256=hashlib.sha256(b'{}').hexdigest()))
    (root / 'configs').mkdir()
    (root / 'configs/openai-reproduction-files.json').write_text(
        json.dumps(dict(files=files)), encoding='utf-8')
    return core


def test_pristine_core_check_accepts_only_restored_inputs_without_moving_them(tmp_path):
    from tools.openai_study_full_reproduce import validate_pristine_core_inputs
    core = restored_core(tmp_path)
    validate_pristine_core_inputs(tmp_path)
    assert (core / 'prior-index.json').read_bytes() == b'{}'
    assert not (tmp_path / 'outputs/full-reproduction/original-core-inputs').exists()


@pytest.mark.parametrize('fault', ['completed_runs','extra','missing','changed'])
def test_nonfresh_core_is_rejected_before_any_scientific_tree_moves(tmp_path, fault):
    from tools.openai_study_full_reproduce import validate_pristine_core_inputs
    core = restored_core(tmp_path)
    if fault == 'completed_runs':
        (core / 'runs/completed').mkdir(parents=True)
        (core / 'runs/completed/evidence.json').write_bytes(b'scientific evidence')
    elif fault == 'extra': (core / 'unlisted.json').write_bytes(b'keep me')
    elif fault == 'missing': (core / 'prior-index.json').unlink()
    elif fault == 'changed': (core / 'prior-index.json').write_bytes(b'changed')
    before = {p.relative_to(core):p.read_bytes() for p in core.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        validate_pristine_core_inputs(tmp_path)
    assert before == {p.relative_to(core):p.read_bytes() for p in core.rglob('*') if p.is_file()}
    assert not (tmp_path / 'outputs/full-reproduction/original-core-inputs').exists()


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell/Windows orchestration boundary')
@pytest.mark.parametrize('exit_code', [0,7])
def test_mf_child_hides_cuda_and_restores_callers_device_selection(tmp_path, exit_code):
    # Exercise the actual PowerShell function and a real child interpreter,
    # without installing torch or executing a fit in this regression test.
    venv.EnvBuilder(with_pip=False).create(tmp_path / '.venv-neural')
    source = Path(__file__).resolve().parents[1] / 'tools/reproduction-full.ps1'
    script = tmp_path / 'probe.ps1'
    quoted_source = str(source).replace("'", "''")
    script.write_text("""$ErrorActionPreference='Stop'
$tokens=$null; $parseErrors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile('SOURCE',[ref]$tokens,[ref]$parseErrors)
if ($parseErrors) { throw 'Parse error' }
$definition=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Invoke-NeuralPython'},$true)
if (-not $definition) { throw 'Missing neural invocation boundary' }
. ([scriptblock]::Create($definition.Extent.Text))
$env:CUDA_VISIBLE_DEVICES='caller-device-choice'
$failed=$false
try { Invoke-NeuralPython -c 'import os,sys; print("CHILD_CUDA="+str(os.environ.get("CUDA_VISIBLE_DEVICES"))); sys.exit(EXIT_CODE)' }
catch { $failed=$true }
Write-Output ('CALLER_CUDA='+$env:CUDA_VISIBLE_DEVICES)
Write-Output ('CHILD_FAILED='+$failed)
""".replace('SOURCE', quoted_source).replace('EXIT_CODE', str(exit_code)), encoding='utf-8')
    result = subprocess.run(['pwsh','-NoProfile','-NonInteractive','-File',str(script)],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert 'CHILD_CUDA=-1' in result.stdout, (result.stdout, result.stderr)
    assert 'CALLER_CUDA=caller-device-choice' in result.stdout
    assert f'CHILD_FAILED={str(exit_code != 0)}' in result.stdout
