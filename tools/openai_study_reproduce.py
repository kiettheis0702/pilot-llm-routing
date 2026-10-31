"""Fill missing offline follow-up preparation and analysis entrypoints.

Scientific producers and their frozen source hashes remain unchanged.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import tempfile
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
PROFILE_SOURCE = ROOT / 'configs/hybrid-batch-profile-2026-10-31.json'
PROFILE_TARGET = 'outputs/openai-followup-20261031/hybrid-batch-profile.json'
PROFILE_SHA256 = 'a62acd18cdc179d25f871c578f0b8ac8bf6f38713dbc9a0f288b62e42d7f1d51'
SPLIT_TARGET = 'outputs/paper-preflight-20261030/input-audit'
SPLIT_SHA256 = 'd78956d9687822df8dcb4d0043f9f6e0de96889dc91412f9cc3f13700430963c'
FOLLOWUP = ROOT / 'outputs/openai-followup-20261031'
CHECKS = ROOT / 'outputs/followup-reproduction-checks'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish(root, files):
    """Check all destinations before atomically publishing any missing file."""
    from tools.reproduction_bundle import safe_path
    destinations = {name: safe_path(root, name) for name in files}
    for name, path in destinations.items():
        if path.exists() and (not path.is_file() or path.read_bytes() != files[name]):
            raise FileExistsError('Existing reproduction evidence differs: ' + str(path))
        if any(p.exists() and not p.is_dir() for p in path.parents):
            raise FileExistsError('Output parent is not a directory: ' + str(path))
    for name, path in destinations.items():
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.prepare-' + uuid4().hex)
        try:
            with temporary.open('xb') as stream:
                stream.write(files[name]); stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return destinations


def profile_bytes(source):
    if sha(source) != PROFILE_SHA256:
        raise ValueError('Historical hardware profile checksum differs')
    return Path(source).read_bytes()


def install_profile(root, *, source=PROFILE_SOURCE):
    return publish(root, {PROFILE_TARGET: profile_bytes(source)})[PROFILE_TARGET]


def prepare(root=ROOT):
    from tools.reproduction_bundle import members, safe_path, check_file
    from tools.paper_preflight.audit import execute
    root = Path(root).resolve()
    entries = members(read(ROOT / 'configs/openai-reproduction-files.json'))
    for name, entry in entries.items():
        check_file(safe_path(root, name), entry)
    files = {PROFILE_TARGET: profile_bytes(PROFILE_SOURCE)}
    (root / 'outputs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='followup-prepare-', dir=root / 'outputs') as scratch:
        stage = Path(scratch) / 'audit'
        execute(root / 'data/processed/routerbench_0shot/queries.parquet',
                root / 'data/processed/routerbench_0shot/splits.csv',
                root / 'outputs/local-pipeline-verification-20261029/prepared-identity/comparisons.json',
                root / 'outputs/research-input-audit-20261029/inputs/preference-inputs.json', stage)
        if sha(stage / 'preference-three-way.json') != SPLIT_SHA256:
            raise ValueError('Reconstructed preference split differs from original experiment')
        files.update({SPLIT_TARGET + '/' + path.name: path.read_bytes() for path in stage.iterdir()})
        publish(root, files)
    return dict(verified_core_files=len(entries), prepared_files=len(files),
                split_sha256=SPLIT_SHA256, historical_profile_sha256=PROFILE_SHA256,
                new_hardware_profile_measured=False, paid_api_calls=0)


def selected_recipe(manifest):
    if manifest['seed'] != 42 or not manifest['candidates']:
        raise ValueError('Seed42 calibration selection must precede replicate fits')
    candidates = manifest['candidates']
    for row in candidates:
        score = row['calibration']
        if (score['pairs'] != 211 or not 0 <= score['correct'] <= 211
                or score['accuracy'] != score['correct'] / 211):
            raise ValueError('Invalid calibration selection statistics')
    best = min(candidates, key=lambda row: (-row['calibration']['accuracy'],
               row['projection_epochs'] + row['prior_epochs'], row['id']))
    if manifest['selected_candidate'] != best['id']:
        raise ValueError('Saved selection disagrees with calibration-only tie rule')
    return {key: best[key] for key in ('id', 'dimension', 'projection_epochs', 'prior_epochs')}


def verify_preference(folder, seed, candidates, *, comparisons=None):
    import numpy as np
    from scripts.embedding_store import DiskEmbeddingCache
    from scripts.pretraining import load_checkpoint
    from tools.openai_study import verify_prior
    from tools.course.analyze import verify_package
    manifest = read(folder / 'manifest.json')
    pair_path = comparisons or ROOT / SPLIT_TARGET / 'preference-three-way.json'
    rows = read(pair_path)
    sizes = ROOT / 'outputs/research-cost-inputs-20261029/cost-size-identity_match.json'
    cache = DiskEmbeddingCache.load(ROOT / 'outputs/openai-standard-20261030/imported/caches/source-1')
    if (manifest['seed'] != seed or manifest['synthetic'] or manifest['status'] != 'ready'
            or manifest['cache_fingerprint'] != cache.fingerprint()
            or manifest['source_sha256'] != dict(comparisons=sha(pair_path), sizes=sha(sizes))
            or manifest['counts'] != dict(train=847, calibration=211, test=500)
            or [{k:r[k] for k in ('id','dimension','projection_epochs','prior_epochs')}
                for r in manifest['candidates']] != candidates):
        raise ValueError('Preference fit input/recipe identity differs')
    for candidate in manifest['candidates']:
        if sha(folder / candidate['checkpoint']) != candidate['sha256']:
            raise ValueError('Preference candidate checksum differs')
    selected = folder / 'selected-prior.json'
    chosen = next(r for r in manifest['candidates'] if r['id'] == manifest['selected_candidate'])
    if sha(selected) != manifest['selected_checkpoint_sha256'] or sha(selected) != chosen['sha256']:
        raise ValueError('Selected preference checkpoint checksum differs')
    if sha(folder / 'final-test.json') != manifest['final_test_sha256']:
        raise ValueError('Final-test file checksum differs')
    prior = load_checkpoint(selected)
    fit_rows = [dict(r, split='holdout' if r['split'] == 'calibration' else 'train')
                for r in rows if r['split'] != 'test']
    fit = verify_prior(prior, cache, fit_rows)
    if (chosen['calibration']['correct'] != fit['correct_pairs']
            or chosen['calibration']['pairs'] != fit['calibration_pairs']):
        raise ValueError('Selected candidate calibration differs from independent scoring')
    correct = 0
    directions = np.asarray(prior['theta'])
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    positions = {arm:i for i,arm in enumerate(prior['arms'])}
    for row in rows:
        if row['split'] != 'test':
            continue
        x = prior['W'] @ cache.get(row['query_id']) + prior['b']
        x = x / np.linalg.norm(x)
        correct += int(x @ directions[positions[row['winner']]] > x @ directions[positions[row['loser']]])
    final = read(folder / 'final-test.json')
    if final['correct'] != correct or final['pairs'] != 500 or final['used_for_selection']:
        raise ValueError('Independent final-test score disagrees with producer')
    report = ROOT / 'reports/openai-preference500-2026-10-31'
    verify_package(report)
    import csv
    with (report / 'per-seed.csv').open(encoding='utf-8', newline='') as stream:
        expected = next(r for r in csv.DictReader(stream) if int(r['seed']) == seed)
    expected_correct = int(expected['correct']) if comparisons is None else 500 - int(expected['correct'])
    if correct != expected_correct:
        raise ValueError('Reproduced preference accuracy differs from frozen report')
    return dict(verified=True, seed=seed, correct=correct, pairs=500, fit=fit,
                checkpoint_sha256=sha(selected), matches_published=comparisons is None,
                test_labels_flipped=comparisons is not None, paid_api_calls=0)


def preference(seed):
    from tools.paper_preflight.train import run_training
    from tools.openai_study import once
    if sha(ROOT / SPLIT_TARGET / 'preference-three-way.json') != SPLIT_SHA256:
        raise ValueError('Run verified preparation before preference fitting')
    settings = read(ROOT / 'configs/paper_preflight_protocol.json')['pretraining']
    base = FOLLOWUP / 'preference500'
    if seed == 42:
        candidates = settings['candidates']
    else:
        selection = read(base / 'seed-42/manifest.json')
        verify_preference(base / 'seed-42', 42, settings['candidates'])
        candidates = [selected_recipe(selection)]
    folder = base / f'seed-{seed}'
    if not folder.exists():
        run_training(ROOT / 'outputs/openai-standard-20261030/imported/caches/source-1',
            ROOT / SPLIT_TARGET / 'preference-three-way.json',
            ROOT / 'outputs/research-cost-inputs-20261029/cost-size-identity_match.json', folder,
            candidates=candidates, seed=seed, learning_rate=settings['learning_rate'], margin=settings['margin'])
    if seed == 42:
        selected_recipe(read(folder / 'manifest.json'))
    result = verify_preference(folder, seed, candidates)
    once(CHECKS / f'preference-seed-{seed}.json', result)
    return result


def mutation():
    """Repeat all seed42 candidates with only final-test labels flipped."""
    from tools.openai_study import once
    from tools.paper_preflight.train import run_training
    baseline = preference(42)
    rows = read(ROOT / SPLIT_TARGET / 'preference-three-way.json')
    changed = [dict(row, winner=row['loser'], loser=row['winner']) if row['split'] == 'test'
               else row for row in rows]
    pairs = CHECKS / 'mutated-final-test.json'
    once(pairs, changed)
    settings = read(ROOT / 'configs/paper_preflight_protocol.json')['pretraining']
    folder = FOLLOWUP / 'preference500/mutation-seed-42'
    if not folder.exists():
        run_training(ROOT / 'outputs/openai-standard-20261030/imported/caches/source-1', pairs,
            ROOT / 'outputs/research-cost-inputs-20261029/cost-size-identity_match.json', folder,
            candidates=settings['candidates'], seed=42,
            learning_rate=settings['learning_rate'], margin=settings['margin'])
    selected_recipe(read(folder / 'manifest.json'))
    result = verify_preference(folder, 42, settings['candidates'], comparisons=pairs)
    if result['checkpoint_sha256'] != baseline['checkpoint_sha256']:
        raise ValueError('Test-label mutation changed the fitted preference checkpoint')
    proof = dict(verified=True, actual_refit=True, selected_checkpoint_byte_identical=True,
                 baseline_correct=baseline['correct'], mutated_correct=result['correct'],
                 checkpoint_sha256=result['checkpoint_sha256'], paid_api_calls=0)
    once(CHECKS / 'preference-mutation.json', proof)
    return proof


def parents():
    """Run the six exact PILOT groups consumed by allocation/ablation studies."""
    import gc
    from scripts.experiments import execute_group, verify_pins
    from scripts.resources import require_resources
    from tools.openai_study import DEFAULT_BASE, SEEDS, load_data, sealed
    protocol = sealed(DEFAULT_BASE / 'protocol.json')
    verify_pins(protocol, ROOT)
    completed = []
    for domain in ('routerbench', 'mmlu'):
        for seed in SEEDS:
            require_resources()
            bundle = load_data(DEFAULT_BASE, domain, seed)
            bundle['protocol'] = protocol
            bundle['lineage'].update(protocol_sha256=sha(DEFAULT_BASE / 'protocol.json'),
                prior_index_sha256=sha(DEFAULT_BASE / 'prior-index.json'),
                tuning_stage_sha256=protocol['tuning'][domain]['stage']['sha256'])
            execute_group(bundle, DEFAULT_BASE / 'runs', method='pilot', resume=True)
            completed.append(dict(domain=domain, seed=seed, method='pilot'))
            print(json.dumps(completed[-1]), flush=True)
            del bundle; gc.collect()
    return dict(completed=completed, full_core_grid_complete=False, paid_api_calls=0)


def costs():
    import pandas as pd
    from tools.course.analyze import cost_sensitivity, verify_package
    from tools.openai_study import once
    for name in ('openai-core','openai-ablations','openai-cost-sensitivity'):
        verify_package(ROOT / f'reports/{name}-2026-10-31')
    core = pd.read_csv(ROOT / 'reports/openai-core-2026-10-31/per-seed.csv')
    addon = pd.read_csv(ROOT / 'reports/openai-ablations-2026-10-31/per-seed.csv')
    expected = read(ROOT / 'reports/openai-cost-sensitivity-2026-10-31/cost-sensitivity.json')
    populations = {'pilot-full': core.loc[(core.method == 'pilot') & (core.learning_fraction == 1)],
                   'all-core': core, 'all-openai': pd.concat([core, addon], ignore_index=True)}
    result = []
    for name, frame in populations.items():
        rows = frame.to_dict('records')
        for item in cost_sensitivity(rows, [.8, 1., 1.1, 1.25, 1.5, 2.]):
            item['population'] = name
            excess = [max(Decimal(0), Decimal(str(item['factor'])) * Decimal(str(r['actual_recorded_cost']))
                          - Decimal(str(r['budget']))) for r in rows]
            independent = dict(over_budget_count=sum(x > Decimal('1e-9') for x in excess),
                               maximum_excess=float(max(excess)), total_excess=float(sum(excess)))
            saved = next(r for r in expected if (r['population'], r['factor']) == (name, item['factor']))
            for other in (independent, saved):
                for key, value in other.items():
                    if isinstance(value, float):
                        same = math.isclose(item[key], value, rel_tol=1e-11, abs_tol=1e-10)
                    else:
                        same = item[key] == value
                    if not same:
                        raise ValueError('Cost replay differs: ' + name + '/' + key)
            result.append(item)
    proof = dict(verified=True, records=result, independent_decimal_crosscheck=True,
                 matches_published=True, choices_unchanged=True, paid_api_calls=0)
    once(CHECKS / 'costs.json', proof)
    return dict(verified=True, conditions=len(result), matches_published=True, paid_api_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'preference', 'mutation', 'parents', 'costs'])
    parser.add_argument('--seed', type=int, choices=[7, 42, 2026])
    args = parser.parse_args()
    if args.stage == 'preference' and args.seed is None:
        parser.error('preference requires --seed; run seed42 first')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    from scripts.resources import require_resources
    require_resources()
    def deny(*args, **kwargs):
        raise RuntimeError('Network disabled during offline follow-up reproduction')
    socket.socket.connect = deny
    result = preference(args.seed) if args.stage == 'preference' else globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
