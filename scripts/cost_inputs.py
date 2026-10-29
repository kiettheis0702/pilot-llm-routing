"""Offline RouterBench accounting with explicit source/dataset scale calibration."""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import date
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re

VOCAB_URL = 'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken'
VOCAB_CACHE_KEY = hashlib.sha1(VOCAB_URL.encode()).hexdigest()
VOCAB_SHA256 = '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be positive and finite')
    return float(value)


def prices_from_spec(spec):
    from scripts.cost_policy import TokenPrices
    required = {'version', 'currency', 'source_url', 'source_revision', 'source_date',
                'source_file_sha256', 'source_divisor', 'dataset_multiplier',
                'calibration_provenance', 'rates'}
    if not isinstance(spec, dict) or set(spec) != required or type(spec['version']) is not int or spec['version'] != 1:
        raise ValueError('Invalid price specification')
    for field in ('currency', 'source_url', 'source_revision', 'source_date', 'source_file_sha256', 'calibration_provenance'):
        if not isinstance(spec[field], str) or not spec[field].strip():
            raise ValueError('Missing price provenance: ' + field)
    if (not spec['source_url'].startswith('https://')
            or not re.fullmatch('[0-9a-f]{40}', spec['source_revision'])
            or not re.fullmatch('[0-9a-f]{64}', spec['source_file_sha256'])):
        raise ValueError('Invalid price source URL, revision or checksum')
    date.fromisoformat(spec['source_date'])
    scale = 1e6 * _positive(spec['dataset_multiplier'], 'Dataset multiplier') / _positive(spec['source_divisor'], 'Source divisor')
    if not isinstance(spec['rates'], dict) or not spec['rates']:
        raise ValueError('Missing arm rates')
    result = {}
    for arm, rates in spec['rates'].items():
        if not isinstance(arm, str) or not arm.strip() or not isinstance(rates, dict) or set(rates) != {'input', 'output'}:
            raise ValueError('Invalid arm rate')
        result[arm] = TokenPrices(scale * _positive(rates['input'], 'Input rate'),
                                 scale * _positive(rates['output'], 'Output rate'))
    return result


def triplet_cost_sizes(spec, arms):
    prices = prices_from_spec(spec)
    if not arms or len(set(arms)) != len(arms) or any(a not in prices for a in arms):
        raise ValueError('Unknown or duplicate arm in cost-size order')
    return dict(sizes={a: prices[a].input_per_million + prices[a].output_per_million for a in arms},
        provenance='Declared 1:1 input/output reference-token mix; sum of calibrated per-million rates. '
                   'This scalar cost-order convention is a local choice, not a documented PILOT mix. '
                   + spec['calibration_provenance'],
        source_urls=[spec['source_url']], synthetic=False)


class HistoricalCounter:
    """Pinned cl100k graph loaded directly from local vocabulary; no network path."""
    def __init__(self, cache_dir='.cache/tiktoken'):
        from scripts.fetch_data import sha256
        path = Path(cache_dir) / VOCAB_CACHE_KEY
        if sha256(path) != VOCAB_SHA256:
            raise ValueError('Pinned vocabulary checksum mismatch')
        if version('tiktoken') != '0.5.2':
            raise ValueError('Historical counter requires tiktoken==0.5.2; install the cost extra')
        import tiktoken
        ranks = {}
        for line in path.read_bytes().splitlines():
            token, rank = line.split()
            ranks[base64.b64decode(token, validate=True)] = int(rank)
        # Exact pattern and special IDs from tiktoken 0.5.2 openai_public.py.
        pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self.encoding = tiktoken.Encoding(name='cl100k_base', pat_str=pattern,
            mergeable_ranks=ranks, special_tokens={'<|endoftext|>':100257, '<|fim_prefix|>':100258,
                '<|fim_middle|>':100259, '<|fim_suffix|>':100260, '<|endofprompt|>':100276})
        self.metadata = dict(implementation='tokencost 0.0.5 count_string_tokens semantics',
            tokenizer='cl100k_base', tiktoken_version=version('tiktoken'),
            vocabulary_sha256=VOCAB_SHA256, vocabulary_url=VOCAB_URL,
            special_token_behavior='reject reserved markers, matching encoding.encode defaults',
            segment_behavior='count each original segment separately, then sum',
            limitation='Benchmark fallback for non-OpenAI model names; not their native billing tokenizer',
            network=False, synthetic=False)

    def __call__(self, text):
        if not isinstance(text, str):
            raise ValueError('Token counter requires text')
        return len(self.encoding.encode(text))


def _segments(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError('Expected serialized text segments') from exc
    if not isinstance(value, list) or any(not isinstance(s, str) for s in value):
        raise ValueError('Expected a list of text segments')
    return value


def count_segments(value, counter):
    counts = [counter(text) for text in _segments(value)]
    if any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('Invalid token count')
    return sum(counts)


def build_tables(raw, queries, tuning, spec, counter):
    from scripts.cost_policy import TuningTokenCostEstimator
    from scripts.data_gate import query_hash
    from scripts.splits import SplitSelection
    prices = prices_from_spec(spec)
    if not isinstance(tuning, SplitSelection) or tuning.split != 'tuning':
        raise ValueError('Only a verified tuning selection is allowed')
    if not isinstance(counter.metadata, dict) or not counter.metadata:
        raise ValueError('Missing tokenizer provenance')
    required = {'query_id', 'query', 'query_hash', 'query_segments_json'}
    raw_columns = {'sample_id', 'prompt'} | {a + suffix for a in prices for suffix in ('|model_response', '|total_cost')}
    if (not queries.columns.is_unique or not raw.columns.is_unique or not required <= set(queries)
            or not raw_columns <= set(raw) or queries.empty or raw.empty
            or queries.query_id.duplicated().any() or raw.sample_id.duplicated().any()
            or set(raw.sample_id) != set(queries.query_id) or not set(tuning.query_ids) <= set(queries.query_id)):
        raise ValueError('Query/source coverage mismatch')
    input_counts, parts = {}, {}
    for row in queries.itertuples():
        try:
            segments = _segments(json.loads(row.query_segments_json))
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid prompt segments') from exc
        if (not segments or not any(s.strip() for s in segments)
                or '\n\n'.join(segments) != row.query or query_hash(row.query) != row.query_hash):
            raise ValueError('Verbatim prompt/segments/hash mismatch')
        parts[row.query_id] = segments
        input_counts[row.query_id] = count_segments(segments, counter)
    # Select before accessing any response or realized cost; all other outcomes
    # can even be malformed without influencing calibration or fitted estimates.
    selected = raw.set_index('sample_id').loc[list(tuning.query_ids)]
    outputs = {}
    errors = []
    per_arm = {a: {'pairs':0, 'matched':0, 'max_absolute_error':0.0} for a in prices}
    for query_id, row in selected.iterrows():
        if _segments(row['prompt']) != parts[query_id]:
            raise ValueError('Raw/prepared tuning prompt mismatch')
        outputs[query_id] = {}
        for arm, price in prices.items():
            count = count_segments(row[arm+'|model_response'], counter)
            outputs[query_id][arm] = count
            actual = row[arm+'|total_cost']
            if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < 0:
                raise ValueError('Invalid tuning cost calibration value')
            expected = (input_counts[query_id]*price.input_per_million + count*price.output_per_million)/1e6
            error = abs(actual - expected)
            matched = math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
            stats = per_arm[arm]
            stats['pairs'] += 1
            stats['matched'] += int(matched)
            stats['max_absolute_error'] = max(stats['max_absolute_error'], error)
            if not matched:
                errors.append((query_id, arm))
    if errors:
        raise ValueError(f'Tuning cost calibration mismatch on {len(errors)} pairs; first={errors[0]}; no automatic rescaling')
    estimator = TuningTokenCostEstimator.fit(tuning, outputs, prices,
        token_provenance=json.dumps(counter.metadata, sort_keys=True),
        price_provenance=json.dumps(spec, sort_keys=True))
    return dict(input_counts=input_counts, output_counts=outputs, estimator=estimator.metadata(),
        calibration=dict(all_matched=True, matched_pairs=len(tuning.query_ids)*len(prices),
            tuning_queries=len(tuning.query_ids), per_arm=per_arm, source_divisor=spec['source_divisor'],
            dataset_multiplier=spec['dataset_multiplier'], relative_tolerance=1e-10, absolute_tolerance=1e-12))


def export_cost_inputs(data_dir, output_dir, spec, *, root=None, tuning_size=1000,
                       counter=None, allow_synthetic=False):
    from dataclasses import asdict
    import pandas as pd
    from scripts.fetch_data import ROOT, sha256, verify
    from scripts.input_audit import _publish_json
    from scripts.model_mapping import POLICIES, resolve_mappings
    from scripts.pretraining import _digest
    from scripts.research_pipeline import load_routing
    root = ROOT if root is None else Path(root)
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    prices_from_spec(spec)
    source = json.loads((root/'configs/data_sources.json').read_text(encoding='utf-8'))['routerbench_0shot']
    raw_path = data_dir/'routerbench_0shot'/source['file']
    verify(raw_path, source)
    _, queries, splits, source_hashes = load_routing(data_dir, root, tuning_size)
    raw = pd.read_pickle(raw_path)  # Only the size/hash-pinned source is deserialized.
    counter = HistoricalCounter(root/'.cache/tiktoken') if counter is None else counter
    synthetic = counter.metadata.get('synthetic')
    if type(synthetic) is not bool or (synthetic and not allow_synthetic):
        raise ValueError('Real cost export refuses synthetic or undeclared tokenizer provenance')
    tuning = splits.select('tuning')
    tables = build_tables(raw, queries, tuning, spec, counter)
    del raw
    mappings_path = root/'configs/model_mapping.json'
    mappings = json.loads(mappings_path.read_text(encoding='utf-8'))['mappings']
    files = {'input-counts.json':tables['input_counts'],
             'tuning-output-counts.json':tables['output_counts'],
             'estimator.json':tables['estimator']}
    for policy in POLICIES:
        arms = sorted(resolve_mappings(mappings, policy).values())
        sizes = triplet_cost_sizes(spec, arms)
        sizes['synthetic'] = synthetic
        files[f'cost-size-{policy}.json'] = sizes
    manifest = dict(version=1, kind='research-cost-inputs', rows=len(queries),
        arms=sorted(spec['rates']), tuning_selection=asdict(tuning), source_hashes=source_hashes,
        raw_source_sha256=source['sha256'], mapping_sha256=sha256(mappings_path),
        price_spec_sha256=_digest(spec), price_spec=spec, tokenizer=counter.metadata,
        calibration=tables['calibration'], synthetic=synthetic, paper_equivalent=False,
        limitations=['Source formula is scaled explicitly by tuning-only dataset calibration',
            'cl100k_base is the historical benchmark fallback for non-OpenAI arms',
            'Triplet scalar order uses a declared local 1:1 input/output mix',
            'Exact PILOT data/shot release and three Arena model bridges remain unresolved'])
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        _publish_json(output_dir/name, value)
    manifest['files'] = {name: sha256(output_dir/name) for name in files}
    manifest['manifest_sha256'] = _digest(manifest)
    _publish_json(output_dir/'manifest.json', manifest)
    return manifest


def load_cost_inputs(path):
    from scripts.fetch_data import sha256
    from scripts.model_mapping import POLICIES
    from scripts.pretraining import _digest
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    names = {'input-counts.json', 'tuning-output-counts.json', 'estimator.json'}
    names.update(f'cost-size-{policy}.json' for policy in POLICIES)
    if (manifest.get('kind') != 'research-cost-inputs' or manifest.get('version') != 1
            or set(manifest.get('files', {})) != names):
        raise ValueError('Invalid cost-input manifest')
    if manifest.get('manifest_sha256') != _digest({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('Cost-input manifest checksum mismatch')
    result = {'manifest':manifest}
    for name in sorted(names):
        if sha256(path/name) != manifest['files'][name]:
            raise ValueError('Cost-input checksum mismatch: ' + name)
        result[name.removesuffix('.json')] = json.loads((path/name).read_text(encoding='utf-8'))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prices', type=Path, default=Path('configs/routerbench_costs.json'))
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    # Preflight precedes heavy dependencies/data loading. This module imports
    # only the standard library until a stage is actually requested.
    from scripts.resources import require_resources
    print(json.dumps({'preflight':require_resources()}), flush=True)
    report = export_cost_inputs(args.data_dir, args.output_dir,
        json.loads(args.prices.read_text(encoding='utf-8')))
    print(json.dumps({'output_dir':str(args.output_dir), 'rows':report['rows'],
        'calibration':report['calibration'], 'manifest_sha256':report['manifest_sha256']}))


if __name__ == '__main__':
    main()
