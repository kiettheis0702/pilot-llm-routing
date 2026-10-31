"""Fit on train, select/calibrate on calibration, evaluate final test separately."""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from scripts.embedding_store import DiskEmbeddingCache
from scripts.pretraining import (fit_preference_prior, evaluate_prior, load_checkpoint,
    save_checkpoint, validate_comparisons, _integer, _positive, _sizes)
from tools.paper_preflight.common import ROOT, read_json, sha256, write_json


def _accuracy(statistics):
    support = sum(v['support'] for v in statistics['per_arm'].values())
    correct = sum(v['correct'] for v in statistics['per_arm'].values())
    return dict(correct=correct//2, pairs=support//2, accuracy=correct/support if support else None)


def run_training(cache_dir, comparisons_path, sizes_path, output_dir, *, candidates,
                 seed=42, allow_synthetic=False, learning_rate=.03, margin=.2):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    cache = DiskEmbeddingCache.load(cache_dir)
    if cache.provenance['synthetic'] and not allow_synthetic:
        raise ValueError('Explicit permission required for synthetic workflow fixtures')
    _integer(seed, 'seed', 0)
    _positive(learning_rate, 'learning rate')
    _positive(margin, 'margin')
    sizes = read_json(sizes_path)
    if not isinstance(sizes, dict) or type(sizes.get('synthetic')) is not bool:
        raise ValueError('Cost sizes require an explicit Boolean synthetic flag')
    synthetic = cache.provenance['synthetic'] or sizes['synthetic']
    if synthetic and not allow_synthetic:
        raise ValueError('Explicit permission required for synthetic cost sizes or embeddings')
    arms = sorted(sizes['sizes'])
    _sizes(arms, sizes['sizes'], sizes['provenance'])
    rows, groups, battles = read_json(comparisons_path), {}, set()
    hashes = dict(zip(cache.ids, cache.prompt_hashes))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty three-way comparison list required')
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'battle_id', 'query_id', 'winner', 'loser', 'split'}
                or not all(isinstance(v, str) and v.strip() for v in row.values())
                or row['battle_id'] in battles or row['query_id'] not in hashes
                or row['split'] not in ('train', 'calibration', 'test')
                or row['winner'] not in arms or row['loser'] not in arms or row['winner'] == row['loser']):
            raise ValueError('Invalid three-way comparison identity, arm or split')
        battles.add(row['battle_id'])
        group = hashes[row['query_id']]
        if group in groups and groups[group] != row['split']:
            raise ValueError('Prompt group leaks across preference partitions')
        groups[group] = row['split']
    counts = dict(Counter(row['split'] for row in rows))
    if set(counts) != {'train', 'calibration', 'test'}:
        raise ValueError('All three preference partitions are required')
    fit_rows = [dict(row, split='holdout' if row['split'] == 'calibration' else 'train')
                for row in rows if row['split'] != 'test']
    test_rows = [dict(row, split='holdout' if row['split'] == 'test' else 'train')
                for row in rows if row['split'] != 'calibration']
    validate_comparisons(cache, fit_rows, arms)
    validate_comparisons(cache, test_rows, arms)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('At least one training candidate is required')
    names = set()
    for candidate in candidates:
        if (not isinstance(candidate, dict) or set(candidate) != {'id', 'dimension', 'projection_epochs', 'prior_epochs'}
                or not isinstance(candidate['id'], str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', candidate['id'])
                or candidate['id'] in names):
            raise ValueError('Invalid or duplicate training candidate')
        names.add(candidate['id'])
        for key in ('dimension', 'projection_epochs', 'prior_epochs'):
            _integer(candidate[key], key)
    output.mkdir(parents=True)
    results = []
    for candidate in candidates:
        fit = fit_preference_prior(cache, fit_rows, arms, sizes['sizes'], sizes['provenance'],
            **{k: candidate[k] for k in ('dimension', 'projection_epochs', 'prior_epochs')},
            seed=seed, learning_rate=learning_rate, margin=margin)
        target = output/f"candidate-{candidate['id']}.json"
        save_checkpoint(target, fit)
        statistics = fit['metrics']['validation']
        score = _accuracy(statistics)
        results.append(dict(candidate, calibration=score, all_lambdas_ready=statistics['all_lambdas_ready'],
            checkpoint=target.name, sha256=sha256(target)))
        print(f"Fit {candidate['id']}: calibration {score['correct']}/{score['pairs']}", flush=True)
    selected = min(results, key=lambda r: (-r['calibration']['accuracy'],
                   r['projection_epochs']+r['prior_epochs'], r['id']))
    prior = load_checkpoint(output/selected['checkpoint'])
    save_checkpoint(output/'selected-prior.json', prior)
    test = evaluate_prior(cache, test_rows, arms, prior['W'], prior['b'], prior['theta'])
    final = dict(_accuracy(test), role='final_test_only', used_for_selection=False,
        statistics=test, selected_candidate=selected['id'], synthetic=synthetic)
    write_json(output/'final-test.json', final)
    result = dict(version=1, kind='three-way-preference-training',
        status='ready' if selected['all_lambdas_ready'] else 'prior_not_ready', counts=counts,
        selected_candidate=selected['id'], candidates=results, seed=seed,
        learning_rate=learning_rate, margin=margin,
        selection='calibration accuracy; ties fewer total epochs then candidate ID',
        lambda_source='calibration only; final-test accuracy never replaces checkpoint lambda',
        synthetic=synthetic, api_calls_performed=0,
        paper_equivalent=False, interpretation='local training protocol; not recovered author settings',
        cache_fingerprint=cache.fingerprint(),
        source_sha256=dict(comparisons=sha256(comparisons_path), sizes=sha256(sizes_path)),
        selected_checkpoint_sha256=sha256(output/'selected-prior.json'),
        final_test_sha256=sha256(output/'final-test.json'))
    write_json(output/'manifest.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--cost-sizes', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/paper_preflight_protocol.json')
    parser.add_argument('--allow-synthetic', action='store_true')
    args = parser.parse_args(argv)
    from scripts.resources import require_resources
    require_resources()
    settings = read_json(args.protocol)['pretraining']
    result = run_training(args.cache_dir, args.comparisons, args.cost_sizes, args.output_dir,
        candidates=settings['candidates'], seed=settings['candidate_seed'], allow_synthetic=args.allow_synthetic,
        learning_rate=settings.get('learning_rate', .03), margin=settings.get('margin', .2))
    print({k: result[k] for k in ('status', 'selected_candidate', 'synthetic', 'api_calls_performed')})


if __name__ == '__main__':
    main()
