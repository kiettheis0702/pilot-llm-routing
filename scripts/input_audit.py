"""Full local input export and bounded token audit; no model or provider calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from scripts.fetch_data import ROOT, sha256
from scripts.local_embeddings import MODEL, REVISION, validate_rows
from scripts.pretraining import _digest
from scripts.research_pipeline import load_prepared, load_routing, write_json

PHASES = {'routing:tuning', 'routing:learning', 'routing:deployment',
          'preference:train', 'preference:holdout'}
CAPS = (512, 1024, 2048, 4096, 8192)


def _publish_json(path, value):
    temporary = path.with_name(f'.{path.name}-{uuid4().hex}.tmp')
    try:
        write_json(temporary, value)
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_inputs(data_dir, prepared_dir, output_dir, *, root=ROOT, tuning_size=1000):
    """Export all frozen routing phases, independent of the prior smoke subset."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _, queries, splits, hashes = load_routing(data_dir, root, tuning_size)
    prepared = load_prepared(prepared_dir)
    prior = prepared['manifest']
    if hashes != prior['source_hashes']:
        raise ValueError('Preparation/source lineage mismatch')
    query_index = queries.set_index('query_id')
    routing, phases = [], {}
    for phase in ('tuning', 'learning', 'deployment'):
        for query_id in splits.select(phase).query_ids:
            row = query_index.loc[query_id]
            routing.append(dict(query_id=query_id, text=row['query'], prompt_hash=row['query_hash']))
            phases[query_id] = 'routing:' + phase
    preferences = prepared['preference-inputs']
    comparisons = prepared['comparisons']
    ids = [r['query_id'] for r in preferences]
    if (len(comparisons) != len(ids) or {r['query_id'] for r in comparisons} != set(ids)
            or any(r['split'] not in ('train', 'holdout') for r in comparisons)):
        raise ValueError('Preference phases do not cover inputs exactly')
    phases.update({r['query_id']: 'preference:' + r['split'] for r in comparisons})
    rows = routing + preferences
    validate_rows(rows)
    if set(phases) != {r['query_id'] for r in rows}:
        raise ValueError('Incomplete input phases')
    routing_hashes = set(queries.query_hash)
    manifest = dict(version=1, kind='full-research-inputs', source_hashes=hashes,
        preparation_sha256=sha256(Path(prepared_dir) / 'manifest.json'),
        mapping_policy=prior['mapping_policy'], mapping_sha256=prior['mapping_sha256'],
        arena_sha256=prior['arena_sha256'], arms=prior['arms'],
        phase_counts=dict(Counter(phases.values())),
        task_counts={str(k): int(v) for k, v in queries.task.value_counts().items()},
        preference_routing_prompt_overlap=sum(r['prompt_hash'] in routing_hashes for r in preferences),
        paper_equivalent=False, research_blockers=prior['research_blockers'])
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in [('routing-inputs.json', routing), ('preference-inputs.json', preferences),
                        ('phases.json', phases)]:
        _publish_json(output_dir / name, value)
    manifest['files'] = {name: sha256(output_dir / name) for name in
                         ('routing-inputs.json', 'preference-inputs.json', 'phases.json')}
    manifest['manifest_sha256'] = _digest(manifest)
    _publish_json(output_dir / 'manifest.json', manifest)
    return rows, phases, manifest


class LocalBGECounter:
    """Use the exact cached fast-tokenizer graph, without importing torch."""
    def __init__(self, cache_dir='.cache/huggingface'):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import tokenizers
        location = Path(cache_dir) / 'models--BAAI--bge-m3/snapshots' / REVISION
        names = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')
        hashes = {name: sha256(location / name) for name in names}
        self.tokenizer = tokenizers.Tokenizer.from_file(str(location / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.metadata = dict(model=MODEL, revision=REVISION, file_sha256=hashes,
            library='tokenizers', library_version=tokenizers.__version__, special_tokens=True,
            truncation=False, padding=False, network=False)

    def __call__(self, texts):
        return [len(item.ids) for item in self.tokenizer.encode_batch(texts, add_special_tokens=True)]


def _stats(lengths, caps):
    values = np.asarray(lengths, dtype=np.int64)
    return dict(count=len(lengths), total_tokens=int(values.sum()), max=int(values.max()),
        quantiles={str(p): float(np.quantile(values, p / 100, method='linear'))
                   for p in (50, 90, 95, 99, 99.9)},
        over_cap={str(cap): int((values > cap).sum()) for cap in caps},
        truncated_tokens={str(cap): int(np.maximum(values - cap, 0).sum()) for cap in caps})


def audit_inputs(rows, phases, counter, output_dir, *, sources, caps=CAPS,
                 batch_size=32, coverage=0.99):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    ids, _ = validate_rows(rows)
    if (not isinstance(phases, dict) or set(phases) != set(ids)
            or any(p not in PHASES for p in phases.values())):
        raise ValueError('Invalid input phases')
    if (not caps or any(type(c) is not int or not 8 <= c <= 8192 for c in caps)
            or tuple(sorted(set(caps))) != tuple(caps)):
        raise ValueError('Caps must be unique increasing integers in 8..8192')
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError('Batch size must be 1..32')
    if isinstance(coverage, bool) or not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError('Coverage must be in (0,1]')
    if not isinstance(sources, dict) or not sources or not isinstance(counter.metadata, dict) or not counter.metadata:
        raise ValueError('Source and tokenizer provenance are required')
    output_dir.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    by_phase = defaultdict(list)
    selected = defaultdict(list)
    with (output_dir / 'lengths.jsonl').open('x', encoding='utf-8') as stream:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            lengths = counter([r['text'] for r in batch])
            if len(lengths) != len(batch) or any(type(n) is not int or n < 1 for n in lengths):
                raise ValueError('Invalid tokenizer lengths')
            for row, length in zip(batch, lengths):
                phase = phases[row['query_id']]
                by_phase[phase].append(length)
                if phase in ('routing:learning', 'routing:tuning'):
                    selected['routing'].append(length)
                elif phase == 'preference:train':
                    selected['preference'].append(length)
                record = dict(query_id=row['query_id'], prompt_hash=row['prompt_hash'], phase=phase,
                    text_sha256=hashlib.sha256(row['text'].encode('utf-8')).hexdigest(), tokens=length)
                stream.write(json.dumps(record, sort_keys=True) + '\n')
            if start and start % (batch_size * 100) == 0:
                print(json.dumps({'audited': start + len(batch), 'total': len(rows)}), flush=True)
    group_stats = {key: _stats(values, caps) for key, values in sorted(selected.items())}
    eligible = [cap for cap in caps if group_stats and all(
        1 - stats['over_cap'][str(cap)] / stats['count'] >= coverage
        for stats in group_stats.values())]
    report = dict(version=1, kind='input-length-audit', rows=len(rows),
        input_sha256=_digest({'rows': rows, 'phases': phases}), tokenizer=counter.metadata,
        source_provenance=sources, caps=list(caps), batch_size=batch_size,
        by_phase={key: _stats(values, caps) for key, values in sorted(by_phase.items())},
        selection_statistics=group_stats,
        decision=dict(cap=eligible[0] if eligible else None, coverage_target=coverage,
            coverage_target_met=bool(eligible), selection_groups={k: len(v) for k, v in sorted(selected.items())},
            rule='smallest cap meeting coverage separately in routing learning+tuning and preference train; holdouts excluded'),
        lengths_sha256=sha256(output_dir / 'lengths.jsonl'), seconds=time.perf_counter() - began)
    report['manifest_sha256'] = _digest(report)
    _publish_json(output_dir / 'manifest.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    from scripts.resources import require_resources
    preflight = require_resources()
    counter = LocalBGECounter()
    rows, phases, bundle = export_inputs(args.data_dir, args.prepared_dir, args.output_dir / 'inputs')
    report = audit_inputs(rows, phases, counter, args.output_dir / 'audit',
        sources={'input_bundle_sha256': bundle['manifest_sha256'], 'preflight': preflight},
        batch_size=args.batch_size)
    print(json.dumps({'rows': report['rows'], 'decision': report['decision'],
                      'output_dir': str(args.output_dir), 'seconds': report['seconds']}))


if __name__ == '__main__':
    main()
