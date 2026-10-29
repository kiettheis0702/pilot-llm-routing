"""Evidence tools must reject optimized Python before any import or output."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[1] / 'tools/research'
NAMES = [path.name for path in sorted(TOOLS.glob('*.py'))]


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('mode', ['flag', 'environment'])
def test_evidence_tool_rejects_optimized_python_before_io(tmp_path, name, mode):
    target = tmp_path / 'tools/research' / name
    target.parent.mkdir(parents=True)
    shutil.copy2(TOOLS / name, target)
    env = dict(os.environ, PYTHONOPTIMIZE='1' if mode == 'environment' else '0')
    command = [sys.executable, '-S'] + (['-O'] if mode == 'flag' else [])
    result = subprocess.run(command + [str(target), '--help'], env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'Evidence verification requires Python assertions' in result.stderr
    assert not (tmp_path / 'outputs').exists()


def test_optimized_deliverable_verifier_cannot_publish_empty_invalid_evidence(tmp_path):
    target = tmp_path / 'tools/research/verify_deliverables.py'
    target.parent.mkdir(parents=True)
    shutil.copy2(TOOLS / target.name, target)
    # The empty PDF list never uses this optional parser. Keep this probe stdlib-only.
    (target.parent / 'pdfplumber.py').write_text('# unused optional dependency\n')
    base = tmp_path / 'outputs/research-experiments-20261029'
    for folder in ('aggregate', 'figures', 'deliverables', 'verification', 'verification-reproduction'):
        (base / folder).mkdir(parents=True)
    for name in ('per-seed.csv', 'learning-prefixes.csv'):
        (base / 'aggregate' / name).write_text('invalid_header\n')
    (base / 'figures/figure-data.json').write_text(json.dumps({'files': {}, 'source_files': {}, 'series': {}}))
    (base / 'deliverables/manifest.json').write_text(json.dumps({
        'aggregate_manifest_sha256': 'invalid', 'figure_manifest_sha256': 'invalid', 'files': {}}))
    for folder in ('verification', 'verification-reproduction'):
        (base / folder / 'routerbench-seed7-pilot-main.json').write_text(json.dumps({
            'script_sha256': 'invalid', 'preflight': {}}))
    result = subprocess.run([sys.executable, '-S', '-O', str(target)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0, result.stdout
    assert 'Evidence verification requires Python assertions' in result.stderr
    assert not (base / 'deliverable-checks.json').exists()
