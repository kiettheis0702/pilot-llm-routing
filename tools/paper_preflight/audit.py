"""Audit existing streams and propose a separate grouped preference protocol."""
from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.data_gate import query_hash
from tools.paper_preflight.common import ROOT, digest, read_json, sha256, write_json


def summarize_stream(queries, splits):
    if (queries.empty or queries.query_id.duplicated().any()
            or splits.query_id.duplicated().any()
            or set(queries.query_id) != set(splits.query_id)):
        raise ValueError('Query identities must cover the split exactly')
    if not queries['query'].map(query_hash).eq(queries.query_hash).all():
        raise ValueError('Query hash mismatch')
    joined = splits.merge(queries[['query_id', 'query_hash', 'task']], on='query_id',
                          suffixes=('', '_source'), validate='one_to_one')
    if not joined.query_hash.eq(joined.query_hash_source).all() or not joined.task.eq(joined.task_source).all():
        raise ValueError('Split metadata differs from source queries')
    if set(splits.split) != {'tuning', 'learning', 'deployment'}:
        raise ValueError('Expected tuning, learning and deployment')
    streams = {}
    for phase, frame in splits.groupby('split'):
        if (frame.stream_position.duplicated().any()
                or not frame.stream_position.map(lambda n: isinstance(n, int) and n >= 0).all()):
            raise ValueError('Stream positions must be distinct nonnegative integers')
        ids = frame.sort_values('stream_position').query_id.tolist()
        streams[phase] = dict(count=len(ids), order_sha256=digest(ids),
                             mmlu_count=int(frame.task.str.startswith('mmlu-').sum()))
    groups = splits.groupby('query_hash').split.nunique()
    return dict(rows=len(queries), raw_task_count=int(queries.task.nunique()),
        task_counts={str(k): int(v) for k, v in queries.task.value_counts().items()},
        mmlu_raw_task_count=int(queries.loc[queries.task.str.startswith('mmlu-'), 'task'].nunique()),
        duplicate_prompt_excess=int(queries.query_hash.duplicated().sum()),
        cross_split_prompt_groups=int((groups > 1).sum()), streams=streams,
        stream_order='existing stream_position, never reshuffled',
        paper_64_task_equivalence='unresolved; all raw labels retained')


def split_preferences_three_way(comparisons, inputs, *, test_pairs=500,
                                 calibration_fraction=.2, seed=42):
    if (type(test_pairs) is not int or test_pairs < 1 or type(seed) is not int
            or isinstance(calibration_fraction, bool)
            or not isinstance(calibration_fraction, (int, float))
            or not math.isfinite(calibration_fraction) or not 0 < calibration_fraction < 1):
        raise ValueError('Invalid preference partition settings')
    texts = {}
    for row in inputs:
        q, text = row.get('query_id'), row.get('text')
        if not isinstance(q, str) or not q.strip() or q in texts or not isinstance(text, str) or not text.strip():
            raise ValueError('Invalid preference identity or text')
        hashed = query_hash(text)
        if row.get('prompt_hash') != hashed:
            raise ValueError('Preference prompt hash mismatch')
        texts[q] = hashed
    battles, groups = set(), {}
    for row in comparisons:
        battle, q = row.get('battle_id'), row.get('query_id')
        if (not isinstance(battle, str) or not battle.strip() or battle in battles or q not in texts
                or not all(isinstance(row.get(k), str) and row[k].strip() for k in ('winner', 'loser'))
                or row['winner'] == row['loser']):
            raise ValueError('Invalid preference comparison')
        battles.add(battle)
        groups.setdefault(texts[q], []).append(row)
    if set(texts) != {r['query_id'] for r in comparisons}:
        raise ValueError('Preference input/comparison identities differ')
    order = sorted(groups, key=lambda h: (hashlib.sha256(f'{seed}:test:{h}'.encode()).hexdigest(), h))
    # Subset sum keeps complete duplicate-prompt groups, including when counts >1.
    reachable = {0: ()}
    for group in order:
        size = len(groups[group])
        for total, chosen in list(reachable.items()):
            target = total + size
            if target <= test_pairs and target not in reachable:
                reachable[target] = chosen + (group,)
        if test_pairs in reachable:
            break
    if test_pairs not in reachable:
        raise ValueError('Cannot reserve exact test size without splitting prompt groups')
    test = set(reachable[test_pairs])
    remaining = [h for h in order if h not in test]
    if len(remaining) < 2:
        raise ValueError('Not enough prompt groups left for train and calibration')
    remaining.sort(key=lambda h: (hashlib.sha256(f'{seed}:calibration:{h}'.encode()).hexdigest(), h))
    count = min(len(remaining) - 1, max(1, math.ceil(len(remaining) * calibration_fraction)))
    calibration = set(remaining[:count])
    return [dict(row, split='test' if texts[row['query_id']] in test else
        'calibration' if texts[row['query_id']] in calibration else 'train')
        for row in sorted(comparisons, key=lambda row: row['battle_id'])]


def execute(queries_path, splits_path, comparisons_path, inputs_path, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    queries = pd.read_parquet(queries_path)
    splits = pd.read_csv(splits_path, keep_default_na=False)
    original, inputs = read_json(comparisons_path), read_json(inputs_path)
    stream = summarize_stream(queries, splits)
    proposal = split_preferences_three_way(original, inputs)
    support = {}
    arms = sorted({r[k] for r in original for k in ('winner', 'loser')})
    for phase in ('train', 'calibration', 'test'):
        counts = Counter(r[k] for r in proposal if r['split'] == phase for k in ('winner', 'loser'))
        support[phase] = {a: counts[a] for a in arms}
    original_roles = {r['battle_id']: r['split'] for r in original}
    test_prior_roles = Counter(original_roles[r['battle_id']] for r in proposal if r['split'] == 'test')
    report = dict(version=1, kind='free-paper-input-audit', routing=stream,
        preference=dict(arms=arms, original_pairs=len(original),
            original_roles=dict(Counter(r['split'] for r in original)),
            proposed_roles=dict(Counter(r['split'] for r in proposal)), support=support,
            all_arms_supported=all(n > 0 for counts in support.values() for n in counts.values()),
            new_test_previous_roles=dict(test_prior_roles),
            old_holdout_role='initialization coefficient calibration and reported accuracy; not a separate final test',
            proposal_status='LOCAL_RESPLIT_NOT_HISTORICALLY_UNSEEN_NOT_AUTHOR_SPLIT',
            grouping='whitespace-normalized prompt; seeded SHA256 subset-sum; calibration by remaining prompt groups',
            seed=42, calibration_group_fraction=.2),
        source_sha256={Path(p).name: sha256(p) for p in (queries_path, splits_path, comparisons_path, inputs_path)},
        paid_api_calls=0, paper_equivalent=False, eleven_model_ready=False)
    output.mkdir(parents=True)
    write_json(output/'audit.json', report)
    write_json(output/'preference-three-way.json', proposal)
    write_json(output/'fit-comparisons.json', [dict(r, split='holdout' if r['split'] == 'calibration' else 'train')
        for r in proposal if r['split'] != 'test'])
    write_json(output/'test-comparisons.json', [r for r in proposal if r['split'] == 'test'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, default=ROOT/'outputs/course-20261030/rebuilt-inputs')
    args = parser.parse_args(argv)
    from scripts.resources import require_resources
    require_resources()
    result = execute(ROOT/'data/processed/routerbench_0shot/queries.parquet',
        ROOT/'data/processed/routerbench_0shot/splits.csv', args.input_root/'prepared/comparisons.json',
        args.input_root/'inputs/preference-inputs.json', args.output_dir)
    print({k: result[k] for k in ('kind', 'paid_api_calls', 'eleven_model_ready')})


if __name__ == '__main__':
    main()
