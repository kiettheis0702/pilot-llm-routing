"""Certify complete native replay inventories without changing frozen reports."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

from tools.openai_study_reproduce import read, sha

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/full-reproduction'


def validate_pristine_core_inputs(root):
    """Refuse to relocate a scientific tree that contains more than restored inputs."""
    root = Path(root).resolve()
    core = root / 'outputs/openai-core-20261031'
    if not core.is_dir() or core.resolve() != core:
        raise ValueError('Core inputs must be a real directory in this fresh checkout')
    entries = [row for row in read(root / 'configs/openai-reproduction-files.json')['files']
               if row['path'].startswith('outputs/openai-core-20261031/')]
    files = {Path(row['path']).relative_to('outputs/openai-core-20261031'):row for row in entries}
    if not files or len(files) != len(entries):
        raise ValueError('Empty or duplicate restored core input inventory')
    allowed = set(files)
    for relative in files:
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Restored core input escapes its directory')
        allowed.update(parent for parent in relative.parents if parent != Path('.'))
    actual = set()
    for path in core.rglob('*'):
        if path.resolve() != path:
            raise ValueError('Linked core inputs are not a fresh restored tree')
        actual.add(path.relative_to(core))
    if actual != allowed:
        raise ValueError('Use a fresh restored-input-only core tree; existing results are preserved')
    for relative, entry in files.items():
        path = core / relative
        if not path.is_file() or path.stat().st_size != entry['size'] or sha(path) != entry['sha256']:
            raise ValueError('Restored core input differs: ' + relative.as_posix())
    return dict(verified=True, restored_core_files=len(files), scientific_outputs_moved=False)


def verify_full_inventory(proofs, published):
    expected = {row['id']: row for row in published}
    if not expected or len(expected) != len(published):
        raise ValueError('Empty or duplicate reference inventory')
    measured = {}
    for proof in proofs:
        if proof.get('verified') is not True:
            raise ValueError('Independent proof is not verified')
        for row in proof['inventory']:
            if row['id'] in measured:
                raise ValueError('Duplicate replay condition')
            measured[row['id']] = row
    if measured.keys() != expected.keys():
        raise ValueError('Replay inventory differs from the complete published inventory')
    columns = dict(served='served', reward='total_reward', actual_cost='actual_recorded_cost',
                   estimated_cost='estimated_cost')
    for identifier, row in measured.items():
        for key, column in columns.items():
            actual, wanted = float(row[key]), float(expected[identifier][column])
            if (not math.isfinite(actual) or not math.isfinite(wanted)
                    or not math.isclose(actual, wanted, rel_tol=1e-10, abs_tol=1e-9)):
                raise ValueError(f'Published metric differs: {identifier}/{key}: {actual} != {wanted}')
    return dict(verified=True, matches_published=True, conditions=len(measured),
                inventory=[measured[name] for name in sorted(measured)])


def table(package, filename='per-seed.csv'):
    from tools.course.analyze import verify_package
    folder = ROOT / 'reports' / package
    verify_package(folder)
    with (folder / filename).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def groups(family, count, package):
    files = sorted((OUTPUT / 'readback').glob(f'{family}-*.json'))
    if len(files) != count:
        raise ValueError(f'{family}: expected {count} independent group proofs, found {len(files)}')
    proofs = [read(path) for path in files]
    result = verify_full_inventory(proofs, table(package))
    result.update(groups=len(proofs), replayed_learning_rounds=sum(p['learning_rounds'] for p in proofs))
    return result


def core_inputs():
    original = ROOT / 'outputs/full-reproduction/original-core-inputs'
    current = ROOT / 'outputs/openai-core-20261031'
    entries = [row for row in read(ROOT / 'configs/openai-reproduction-files.json')['files']
               if row['path'].startswith('outputs/openai-core-20261031/')]
    checks = []
    for entry in entries:
        relative = Path(entry['path']).relative_to('outputs/openai-core-20261031')
        if sha(original / relative) != entry['sha256']:
            raise ValueError('Preserved original core input differs: ' + str(relative))
        actual = sha(current / relative)
        checks.append(dict(path=relative.as_posix(), original=entry['sha256'], actual=actual,
                           byte_identical=actual == entry['sha256']))
    if not entries:
        raise ValueError('Missing original core input inventory')
    result = dict(fresh_prior_seeds=[7,42,2026], fresh_tuning_domains=['routerbench','mmlu'],
                  preserved_original_inputs=True, all_bytes_identical=all(r['byte_identical'] for r in checks),
                  files=checks)
    from tools.openai_study import once
    once(OUTPUT / 'core-inputs.json', result)
    return result


def simulation(family, count):
    from scripts.experiments import load_artifact
    base = ROOT / f'outputs/openai-followup-20261031/{family}'
    proof = read(base / 'independent-verification.json')
    if not proof['verified'] or not proof['complete'] or proof['conditions'] != count:
        raise ValueError('Incomplete independent simulation inventory: ' + family)
    inventory = []
    for entry in proof['inventory']:
        artifact = load_artifact(base / 'runs' / entry['id'])
        if artifact['manifest_sha256'] != entry['manifest_sha256']:
            raise ValueError('Reader and artifact disagree: ' + entry['id'])
        row = artifact['summary']
        inventory.append(dict(id=entry['id'], served=row['served'], reward=row['total_reward'],
            actual_cost=row['actual_cost'], estimated_cost=(row['estimated_cost'] if family == 'budget'
                                                          else row['accounting']['estimated_total'])))
    published = table(f'openai-{family}-2026-10-31')
    if family == 'budget':
        # The48 one-bin controls are independently covered by the full ablation inventory.
        published = [row for row in published if row['policy'] != 'single-budget-bin']
    if len(published) != count:
        raise ValueError('Unexpected published simulation inventory')
    return verify_full_inventory([dict(verified=True, inventory=inventory)], published)


def mf_cpu():
    base = ROOT / 'outputs/openai-followup-20261031/routellm'
    proof = read(base / 'independent-verification.json')
    if not proof['verified'] or not proof['complete'] or not proof['actual_test_label_mutation_weights_identical']:
        raise ValueError('Incomplete MF independent verification')
    published = {int(r['seed']):r for r in table('openai-routellm-2026-10-31', 'paired-preference500.csv')}
    results = []
    for seed in (7,42,2026):
        selection = read(base / f'seed-{seed}/selection.json')
        if selection['environment']['device'] != 'cpu':
            raise ValueError('This scope records the deliberately selected CPU replay only')
        actual = proof['proofs'][f'seed-{seed}']['scores']['test']['correct']
        original = int(published[seed]['correct_routellm'])
        results.append(dict(seed=seed, correct=actual, pairs=500, original_cuda_correct=original,
                            difference_correct=actual-original, environment=selection['environment']))
    return dict(verified=True, complete=True, execution_device='cpu', original_cuda_run_repeated=False,
                actual_test_label_mutation_weights_identical=True, mutation_correct=proof['mutation_correct'],
                results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', choices=['core-preflight','core-inputs','core','followups','mf'])
    args = parser.parse_args()
    from scripts.resources import require_resources
    from tools.course.analyze import verify_package
    from tools.openai_study import once
    preflight = require_resources()
    if args.scope == 'core-preflight':
        result = validate_pristine_core_inputs(ROOT)
    elif args.scope == 'core-inputs':
        result = core_inputs()
    else:
        result = dict(scope=args.scope, commit=os.environ.get('GITHUB_SHA'), run_id=os.environ.get('GITHUB_RUN_ID'),
                      paid_api_calls=0, fresh_cuda_environment_tested=False, preflight=preflight)
        if args.scope == 'core':
            result.update(core=groups('core',78,'openai-core-2026-10-31'),
                          refit=read(OUTPUT / 'core-inputs.json'))
        elif args.scope == 'followups':
            result.update(ablations=groups('ablation',30,'openai-ablations-2026-10-31'),
                binary=groups('binary',21,'openai-binary-2026-10-31'),
                budget=simulation('budget',240), noise=simulation('noise',12))
        else:
            result['mf_cpu'] = mf_cpu()
        packages = sorted((ROOT / 'reports').glob('*/package-files.json'))
        for manifest in packages:
            verify_package(manifest.parent)
        result['unchanged_report_packages'] = len(packages)
        result['source_sha256'] = {name:sha(ROOT / name) for name in
            ('tools/openai_study_full_reproduce.py','tools/reproduction-full.ps1','uv.lock')}
        once(OUTPUT / f'{args.scope}-summary.json', result)
    import json
    print('FULL_REPRODUCTION_EVIDENCE_JSON='+json.dumps(result, separators=(',',':')), flush=True)


if __name__ == '__main__':
    main()
