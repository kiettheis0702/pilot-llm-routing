"""Staged local research runs; scientific readiness is distinct from execution."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import numpy as np

from scripts.fetch_data import ROOT, sha256, verify
from scripts.model_mapping import POLICIES, resolve_mappings
from scripts.preference_data import prepare_preferences, split_preferences, pair_coverage
from scripts.splits import load_split_manifest
from scripts.pretraining import _digest, _write_json, _read_json

BLOCKERS = [
    'Authors exact RouterBench release and 0-shot/5-shot choice unconfirmed',
    '86 observed raw task labels versus 64 in paper; no invented grouping',
    'Authors exact Arena preference release/preprocessing unconfirmed',
    'Three provider checkpoint aliases unresolved; full eleven-arm prior unavailable',
    'BGE-M3 is an embedding substitution; final input/truncation recipe needs length audit',
    'Historical per-arm token prices/tokenizer counts and triplet cost-size order need provenance',
    'Budget evaluation needs tuning-only token counts, bounds and common baseline deployment scores',
]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load_routing(data_dir, root=ROOT, tuning_size=1000):
    prepared = Path(data_dir) / 'processed/routerbench_0shot'
    hashes = json.loads((Path(root) / 'reports/data-gate.json').read_text())['prepared']['file_sha256']
    for name in ('queries.parquet', 'outcomes.parquet', 'splits.csv'):
        if sha256(prepared / name) != hashes[name]:
            raise ValueError(f'Prepared data checksum mismatch: {name}')
    queries = pd.read_parquet(prepared / 'queries.parquet')
    splits = load_split_manifest(prepared / 'splits.csv', queries,
        expected_sha256=hashes['splits.csv'], tuning_size=tuning_size)
    return prepared, queries, splits, hashes


def prepare_data(data_dir, output_dir, *, root=ROOT, mapping_policy='exact_id',
                 rows=128, seed=42, tuning_size=1000, require_all_arms=False):
    root, data_dir, output_dir = Path(root), Path(data_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    mappings = json.loads((root / 'configs/model_mapping.json').read_text())['mappings']
    admitted = resolve_mappings(mappings, mapping_policy)
    excluded = [m['routerbench'] for m in mappings if m['arena'] not in admitted]
    if require_all_arms and excluded:
        raise ValueError('Unresolved model mappings: ' + ', '.join(excluded))
    source = json.loads((root / 'configs/data_sources.json').read_text())['arena_55k']
    arena = data_dir / 'arena_55k/train.csv'
    verify(arena, source)
    _, queries, splits, source_hashes = load_routing(data_dir, root, tuning_size)
    selection = splits.select('learning', limit=rows, subset_seed=seed)
    chosen = queries.set_index('query_id').loc[list(selection.query_ids)].reset_index()
    accepted, seen, primary = [], set(), Counter()
    source_rows = 0
    for chunk in pd.read_csv(arena, chunksize=500):
        ids = set(chunk.id.astype(str))
        if seen.intersection(ids):
            raise ValueError('Duplicate battle identity across chunks')
        seen.update(ids)
        pairs, audit = prepare_preferences(chunk, mappings, mapping_policy=mapping_policy)
        accepted.append(pairs)
        source_rows += len(chunk)
        primary.update(audit['primary_exclusion_counts'])
    pairs = pd.concat(accepted, ignore_index=True)
    pairs = split_preferences(pairs, seed=seed)
    arms = sorted(admitted.values())
    comparisons = [dict(battle_id=r.battle_id, query_id='arena:' + r.battle_id,
                        winner=r.winner_arm, loser=r.loser_arm, split=r.split)
                   for r in pairs.itertuples()]
    pref_rows = [dict(query_id='arena:' + r.battle_id, prompt_hash=r.prompt_hash, text=r.prompt)
                 for r in pairs.itertuples()]
    route_rows = [dict(query_id=r.query_id, prompt_hash=r.query_hash, text=r.query)
                  for r in chosen.itertuples()]
    coverage = {phase: pair_coverage(pairs[pairs.split == phase], arms) for phase in ('train', 'holdout')}
    unsupported = {phase: [a for a, count in coverage[phase]['per_arm'].items() if not count['comparisons']]
                   for phase in coverage}
    report = dict(version=1, kind='local-preparation', mapping_policy=mapping_policy, arms=arms,
        excluded_arms=excluded, source_rows=source_rows, eligible_preferences=len(pairs),
        routing_queries=len(chosen), observed_tasks=int(queries.task.nunique()),
        coverage=coverage, unsupported_arms=unsupported, primary_exclusions=dict(primary),
        preference_routerbench_exact_prompt_overlap=int(pairs.prompt_hash.isin(queries.query_hash).sum()),
        full_pool_ready=not excluded and not any(unsupported.values()), paper_equivalent=False,
        research_blockers=BLOCKERS, selection=asdict(selection), source_hashes=source_hashes,
        arena_sha256=source['sha256'], mapping_sha256=sha256(root/'configs/model_mapping.json'),
        mapping_evidence=mappings)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in [('comparisons.json', comparisons), ('preference-inputs.json', pref_rows),
                          ('routing-inputs.json', route_rows)]:
        write_json(output_dir / name, content)
    report['files'] = {name: sha256(output_dir / name) for name in
                       ('comparisons.json', 'preference-inputs.json', 'routing-inputs.json')}
    report['manifest_sha256'] = _digest(report)
    write_json(output_dir / 'manifest.json', report)
    return report


def load_prepared(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    names = {'comparisons.json', 'preference-inputs.json', 'routing-inputs.json'}
    if manifest.get('kind') != 'local-preparation' or manifest.get('version') != 1 or set(manifest.get('files', {})) != names:
        raise ValueError('Invalid preparation manifest')
    if manifest.get('manifest_sha256') != _digest({k:v for k,v in manifest.items() if k!='manifest_sha256'}):
        raise ValueError('Preparation manifest checksum mismatch')
    result = {'manifest': manifest}
    for name in names:
        if sha256(path / name) != manifest['files'][name]:
            raise ValueError('Prepared input checksum mismatch: ' + name)
        result[name.removesuffix('.json')] = json.loads((path / name).read_text(encoding='utf-8'))
    return result


def _aligned_cache(cache_path, inputs, *, allow_synthetic=False):
    from scripts.embedding_store import DiskEmbeddingCache
    from scripts.local_embeddings import validate_rows, validate_recipe, provenance
    ids, hashes = validate_rows(inputs)
    cache = DiskEmbeddingCache.load(cache_path)
    if cache.ids != ids or cache.prompt_hashes != hashes:
        raise ValueError('Cache IDs/prompt hashes differ from prepared inputs')
    if cache.provenance['synthetic'] and not allow_synthetic:
        raise ValueError('Real stage refuses synthetic vectors')
    if not cache.provenance['synthetic']:
        if cache.provenance['model'] == 'text-embedding-3-small':
            from scripts.openai_cache import validate_inputs
            validate_inputs(cache, inputs)
        else:
            recipe = validate_recipe(cache.metadata.get('recipe'))
            if (cache.metadata.get('input_recipe_sha256') != _digest({'rows':inputs,'recipe':recipe})
                    or cache.provenance != provenance(recipe)):
                raise ValueError('Cache verbatim inputs or recipe differ from preparation')
    return cache


def fit_prepared(prepared_dir, cache_path, cost_sizes, output_dir, *, dimension=8,
                 projection_epochs=2, prior_epochs=5, seed=42, allow_synthetic=False):
    from scripts.pretraining import fit_preference_prior, save_checkpoint
    output_dir = Path(output_dir)
    if output_dir.exists():raise FileExistsError(output_dir)
    data = load_prepared(prepared_dir)
    manifest = data['manifest']
    cache = _aligned_cache(cache_path, data['preference-inputs'], allow_synthetic=allow_synthetic)
    fields = {'sizes','provenance','source_urls','synthetic'}
    if (not isinstance(cost_sizes,dict) or set(cost_sizes)!=fields
            or not isinstance(cost_sizes['provenance'],str) or not cost_sizes['provenance'].strip()
            or not isinstance(cost_sizes['source_urls'],list) or not cost_sizes['source_urls']
            or any(not isinstance(url,str) or not url.startswith('https://') for url in cost_sizes['source_urls'])
            or type(cost_sizes['synthetic']) is not bool):
        raise ValueError('Cost-size order needs provenance, source URLs and synthetic flag')
    if cost_sizes['synthetic'] and not allow_synthetic:
        raise ValueError('Real stage refuses synthetic cost-size order')
    for value, upper in ((dimension,128),(projection_epochs,100),(prior_epochs,200)):
        if type(value) is not int or not 1<=value<=upper:
            raise ValueError('Training exceeds configured bounded experiment limits')
    result = fit_preference_prior(cache,data['comparisons'],manifest['arms'],cost_sizes['sizes'],
        json.dumps({'provenance':cost_sizes['provenance'],'source_urls':cost_sizes['source_urls']},sort_keys=True),
        dimension=dimension,projection_epochs=projection_epochs,prior_epochs=prior_epochs,seed=seed)
    output_dir.mkdir(parents=True,exist_ok=False)
    save_checkpoint(output_dir/'prior.json',result)
    report = dict(version=1,kind='local-fit',prepared_sha256=sha256(Path(prepared_dir)/'manifest.json'),
        checkpoint_sha256=sha256(output_dir/'prior.json'),arms=manifest['arms'],
        prior_ready=result['metrics']['validation']['all_lambdas_ready'],metrics=result['metrics'],
        synthetic=cache.provenance['synthetic'] or cost_sizes['synthetic'],
        paper_equivalent=False,research_blockers=manifest['research_blockers'],cost_sizes=cost_sizes)
    _write_json(output_dir/'fit-manifest.json',report)
    return report


def learn_prepared(prepared_dir, data_dir, cache_path, fit_dir, output_dir, *,
                   root=ROOT, tuning_size=1000, seed=42, allow_synthetic=False):
    from scripts.bandit_contracts import ContextTable
    from scripts.evaluation import run_learning, write_run
    from scripts.pilot import PILOT
    from scripts.policies import RandomPolicy, LinearUCB, ExploreOnly, EpochGreedy, FixedPolicy
    from scripts.pretraining import load_checkpoint, project_cache
    output_dir, fit_dir = Path(output_dir), Path(fit_dir)
    if output_dir.exists():raise FileExistsError(output_dir)
    data = load_prepared(prepared_dir); manifest = data['manifest']
    fit = _read_json(fit_dir/'fit-manifest.json')
    if (fit.get('kind')!='local-fit' or fit.get('version')!=1
            or fit['prepared_sha256']!=sha256(Path(prepared_dir)/'manifest.json')
            or fit['checkpoint_sha256']!=sha256(fit_dir/'prior.json') or fit['arms']!=manifest['arms']):
        raise ValueError('Fit/preparation lineage mismatch')
    if fit['synthetic'] and not allow_synthetic:raise ValueError('Real stage refuses synthetic fit')
    prior = load_checkpoint(fit_dir/'prior.json')
    if prior['arms']!=manifest['arms']:raise ValueError('Prior arm catalog mismatch')
    if not prior['metrics']['validation']['all_lambdas_ready']:
        raise ValueError('Prior validation is blocked; no lambda fallback')
    cache = _aligned_cache(cache_path,data['routing-inputs'],allow_synthetic=allow_synthetic)
    projected = project_cache(prior,cache)
    folder,queries,splits,hashes = load_routing(data_dir,root,tuning_size)
    if hashes!=manifest['source_hashes']:raise ValueError('Routing source lineage mismatch')
    saved = manifest['selection']
    selection = splits.select('learning',limit=saved['limit'],subset_seed=saved['subset_seed'])
    if _digest(asdict(selection))!=_digest(saved):raise ValueError('Routing selection mismatch')
    arms,dimension=tuple(prior['arms']),prior['dimension']
    contexts=ContextTable(dict(zip(cache.ids,projected)),label='LOCAL DEVELOPMENT: learned projection of pinned embeddings')
    queries=queries[queries.query_id.isin(selection.query_ids)].copy()
    outcomes=pd.read_parquet(folder/'outcomes.parquet')
    outcomes=outcomes[outcomes.query_id.isin(selection.query_ids)&outcomes.arm_id.isin(arms)].copy()
    validation=prior['metrics']['validation']['per_arm']
    methods={'pilot':PILOT(arms,dimension,priors=dict(zip(arms,prior['theta'])),
        lambdas={a:validation[a]['lambda'] for a in arms},provenance='prior sha256='+fit['checkpoint_sha256']),
        'random':RandomPolicy(arms,dimension),'linucb':LinearUCB(arms,dimension),
        'explore-only':ExploreOnly(arms,dimension),'epoch-greedy':EpochGreedy(arms,dimension,window=10)}
    methods.update({f'fixed-{i:02d}':FixedPolicy(arms,dimension,arm_id=a) for i,a in enumerate(arms)})
    output_dir.mkdir(parents=True,exist_ok=False)
    summaries={}
    for name,policy in methods.items():
        run=run_learning(policy,queries,outcomes,contexts,selection,seed=seed)
        run.summary.update(paper_equivalent=False,research_blockers=manifest['research_blockers'])
        write_run(run,output_dir/name)
        if name=='pilot':policy.save_checkpoint(output_dir/'learned-pilot.json')
        summaries[name]=run.summary['metrics']
    report=dict(methods=summaries,rows_per_method=len(selection.query_ids),arms=list(arms),
        seed=seed,selection=asdict(selection),context_sha256=cache.fingerprint(),
        prior_sha256=fit['checkpoint_sha256'],paper_equivalent=False,synthetic=fit['synthetic'],
        scope='learning-only development comparison; no deployment budget figure',
        research_blockers=manifest['research_blockers'])
    write_json(output_dir/'comparison.json',report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--data-dir', type=Path, required=True)
    prepare.add_argument('--output-dir', type=Path, required=True)
    prepare.add_argument('--mapping-policy', choices=POLICIES, default='exact_id')
    prepare.add_argument('--rows', type=int, default=128)
    prepare.add_argument('--seed', type=int, default=42)
    prepare.add_argument('--require-all-arms', action='store_true')
    fit = sub.add_parser('fit')
    fit.add_argument('--prepared-dir',type=Path,required=True)
    fit.add_argument('--cache-dir',type=Path,required=True)
    fit.add_argument('--cost-sizes',type=Path,required=True)
    fit.add_argument('--output-dir',type=Path,required=True)
    fit.add_argument('--dimension',type=int,default=8)
    fit.add_argument('--projection-epochs',type=int,default=2)
    fit.add_argument('--prior-epochs',type=int,default=5)
    fit.add_argument('--seed',type=int,default=42)
    learn = sub.add_parser('learn')
    for name in ('prepared-dir','data-dir','cache-dir','fit-dir','output-dir'):
        learn.add_argument('--'+name,type=Path,required=True)
    learn.add_argument('--seed',type=int,default=42)
    args = parser.parse_args(argv)
    if args.stage == 'prepare':
        report = prepare_data(args.data_dir, args.output_dir, mapping_policy=args.mapping_policy,
            rows=args.rows, seed=args.seed, require_all_arms=args.require_all_arms)
        print(json.dumps({key: report[key] for key in
              ('mapping_policy', 'eligible_preferences', 'routing_queries', 'arms', 'full_pool_ready')}))
    elif args.stage=='fit':
        from scripts.resources import require_resources
        require_resources()
        result=fit_prepared(args.prepared_dir,args.cache_dir,json.loads(args.cost_sizes.read_text()),
            args.output_dir,dimension=args.dimension,projection_epochs=args.projection_epochs,
            prior_epochs=args.prior_epochs,seed=args.seed)
        print(json.dumps({'output_dir':str(args.output_dir),'prior_ready':result['prior_ready']}))
    elif args.stage=='learn':
        from scripts.resources import require_resources
        require_resources()
        result=learn_prepared(args.prepared_dir,args.data_dir,args.cache_dir,args.fit_dir,args.output_dir,seed=args.seed)
        print(json.dumps({'output_dir':str(args.output_dir),'methods':len(result['methods'])}))


if __name__ == '__main__':
    main()
