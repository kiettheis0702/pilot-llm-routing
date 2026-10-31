"""Offline core study using completed OpenAI caches; no provider client or key access."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.bandit_contracts import ContextTable
from scripts.cost_inputs import load_cost_inputs
from scripts.cost_policy import TokenPrices, TuningTokenCostEstimator
from scripts.experiments import execute_group, expected_runs, load_artifact, verify_pins
from scripts.pretraining import _digest, _read_json, load_checkpoint
from scripts.research_pipeline import _aligned_cache, fit_prepared, load_prepared, load_routing
from scripts.resources import require_resources
from scripts.tuning import run_search, load_search, calibrate_budget
from tools.paper_preflight.common import ROOT, read_json, sha256, write_json

DEFAULT_BASE = ROOT/'outputs/openai-core-20261031'
EMBEDDINGS = ROOT/'outputs/openai-standard-20261030/imported/caches'
PREPARED = ROOT/'outputs/local-pipeline-verification-20261029/prepared-identity'
INPUTS = ROOT/'outputs/research-input-audit-20261029/inputs'
COSTS = ROOT/'outputs/research-cost-inputs-20261029'
SEEDS = (7, 42, 2026)


def checked_pin(path, root=ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Pinned source must be an existing project file')
    return dict(path=path.relative_to(root).as_posix(), sha256=sha256(path))


def once(path, value):
    path = Path(path)
    if path.exists():
        if _digest(read_json(path)) != _digest(value):
            raise ValueError('Immutable study artifact differs: '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    return value


def sealed(path):
    value=read_json(path)
    if value.get('manifest_sha256')!=_digest({k:v for k,v in value.items() if k!='manifest_sha256'}):
        raise ValueError('Study stage checksum mismatch: '+str(path))
    return value


def core_declaration(original):
    value = dict(version=1, kind='frozen-openai-core-study', synthetic=False,
        paper_equivalent=False, arms=copy.deepcopy(original['arms']),
        primary_plan=copy.deepcopy(original['primary_plan']),
        ablation_plan=dict(variants=[], budget_fractions=[],
            inventory=dict(primary_plus_ablation_deployment_runs=858)))
    value['primary_plan']['embedding_recipe'] = 'completed OpenAI text-embedding-3-small1536D; full-text'
    value['primary_plan']['scope'] = 'OpenAI eight-arm core study; no new ablation or eleven-arm equivalence claim'
    value['primary_plan']['ablation_inventory_status'] = 'excluded from this858-condition core study'
    return value


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('Invalid projected or prior direction')
    return vector/norm


def verify_prior(prior, cache, comparisons):
    if (prior['embedding']['cache_sha256'] != cache.fingerprint()
            or prior['embedding']['recipe'] != cache.provenance
            or prior['input_dimension'] != cache.dimension):
        raise ValueError('Prior/cache identity mismatch')
    W, bias, theta = (np.asarray(prior[k]) for k in ('W', 'b', 'theta'))
    if not all(np.isfinite(v).all() for v in (W, bias, theta)):
        raise ValueError('Nonfinite prior parameters')
    projected = {q:W@cache.get(q)+bias for q in cache.ids}
    directions = {q:_unit(v) for q,v in projected.items()}
    arms = prior['arms']; arm_dirs = {a:_unit(theta[i]) for i,a in enumerate(arms)}
    train = {r['query_id']:r for r in comparisons if r['split']=='train'}
    hashes = dict(zip(cache.ids, cache.prompt_hashes)); sizes = prior['data']['cost_sizes']
    losses = []
    for anchor,positive,negative in prior['data']['sampled_triplets']:
        if any(q not in train for q in (anchor,positive,negative)) or len({hashes[q] for q in (anchor,positive,negative)}) != 3:
            raise ValueError('Triplet leaks or repeats a prompt group')
        a,p,n = (train[q] for q in (anchor,positive,negative))
        if (a['winner'] != p['winner'] or n['loser'] != a['winner']
                or not sizes[n['winner']] < sizes[a['winner']]):
            raise ValueError('Triplet violates preference/cost-size rule')
        losses.append(max(0., prior['protocol']['margin']-directions[anchor]@directions[positive]
                          +directions[anchor]@directions[negative]))
    if not losses or not math.isclose(float(np.mean(losses)), prior['metrics']['projection_losses'][-1], rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError('Independent projection loss mismatch')
    pair_losses=[]; stats={a:dict(support=0,correct=0) for a in arms}; count=0
    for row in comparisons:
        q,w,l=row['query_id'],row['winner'],row['loser']
        sw,sl = float(directions[q]@arm_dirs[w]),float(directions[q]@arm_dirs[l])
        if row['split']=='train': pair_losses.append(float(np.logaddexp(0.,sl-sw)))
        else:
            count+=1
            for a in (w,l): stats[a]['support']+=1;stats[a]['correct']+=int(sw>sl)
    if not math.isclose(float(np.mean(pair_losses)), prior['metrics']['prior_losses'][-1],rel_tol=1e-9,abs_tol=1e-10):
        raise ValueError('Independent preference loss mismatch')
    for a, measured in stats.items():
        saved=prior['metrics']['validation']['per_arm'][a]
        if any(measured[k]!=saved[k] for k in ('support','correct')):
            raise ValueError('Independent calibration counts mismatch')
        want=measured['support']/measured['correct'] if measured['correct'] else None
        if saved['lambda']!=want:raise ValueError('Calibration lambda mismatch')
    return dict(verified=True,calibration_pairs=count,correct_pairs=sum(s['correct'] for s in stats.values())//2,
        train_only_triplets=True,triplets=len(losses),dimension=prior['dimension'],input_dimension=cache.dimension,
        projection_loss=float(np.mean(losses)),preference_loss=float(np.mean(pair_losses)),
        synthetic=cache.provenance['synthetic'],cache_fingerprint=cache.fingerprint())


def fit_priors(base):
    data=load_prepared(PREPARED)
    cache=_aligned_cache(EMBEDDINGS/'source-1',data['preference-inputs'])
    if cache.provenance['model']!='text-embedding-3-small' or cache.dimension!=1536:
        raise ValueError('This study requires the completed real OpenAI cache')
    entries=[]
    for seed in (42,7,2026):
        require_resources()
        folder=base/'priors'/f'seed-{seed}'
        if not folder.exists():
            fit_prepared(PREPARED,EMBEDDINGS/'source-1',read_json(COSTS/'cost-size-identity_match.json'),folder,
                dimension=64,projection_epochs=30,prior_epochs=40,seed=seed)
        fit=_read_json(folder/'fit-manifest.json');prior=load_checkpoint(folder/'prior.json')
        if (fit['checkpoint_sha256']!=sha256(folder/'prior.json') or fit['prepared_sha256']!=sha256(PREPARED/'manifest.json')
                or fit['synthetic'] or not fit['prior_ready'] or prior['dimension']!=64
                or prior['protocol']['seed']!=seed or prior['protocol']['projection_epochs']!=30
                or prior['protocol']['prior_epochs']!=40):
            raise ValueError('Existing prior does not match the declared core fit')
        proof=verify_prior(prior,cache,data['comparisons'])
        once(folder/'verification.json',proof)
        entries.append(dict(seed=seed,dimension=64,prior=checked_pin(folder/'prior.json'),
            fit=checked_pin(folder/'fit-manifest.json'),verification=checked_pin(folder/'verification.json')))
        print(f"Verified OpenAI prior seed {seed}: {proof['correct_pairs']}/{proof['calibration_pairs']} calibration",flush=True)
    index=dict(version=1,kind='openai-core-prior-index',fits=sorted(entries,key=lambda r:r['seed']),synthetic=False)
    index['manifest_sha256']=_digest(index)
    once(base/'prior-index.json',index)
    return index


def load_data(base,domain,seed):
    index=sealed(base/'prior-index.json');verify_pins(index,ROOT)
    entry=next(e for e in index['fits'] if e['seed']==seed)
    prior=load_checkpoint(ROOT/entry['prior']['path'])
    prepared=load_prepared(PREPARED)
    preference_cache=_aligned_cache(EMBEDDINGS/'source-1',prepared['preference-inputs'])
    if prior['embedding']['cache_sha256']!=preference_cache.fingerprint():
        raise ValueError('Prior preference cache lineage differs')
    inputs=read_json(INPUTS/'manifest.json')
    if inputs['manifest_sha256']!=_digest({k:v for k,v in inputs.items() if k!='manifest_sha256'}):
        raise ValueError('Input bundle checksum mismatch')
    for name,want in inputs['files'].items():
        if sha256(INPUTS/name)!=want:raise ValueError('Input bundle member changed')
    cache=_aligned_cache(EMBEDDINGS/'source-0',read_json(INPUTS/'routing-inputs.json'))
    if cache.provenance!=prior['embedding']['recipe'] or cache.dimension!=prior['input_dimension']:
        raise ValueError('Prior/routing OpenAI recipe mismatch')
    folder,queries,splits,hashes=load_routing(ROOT/'data')
    costs=load_cost_inputs(COSTS)
    if inputs['source_hashes']!=hashes or prepared['manifest']['source_hashes']!=hashes or costs['manifest']['source_hashes']!=hashes:
        raise ValueError('Study source lineage differs')
    selections={p:splits.select(p) for p in ('tuning','learning','deployment')}
    if domain=='mmlu':
        members=set(queries.loc[queries.task.str.startswith('mmlu-'),'query_id'])
        selections={p:replace(s,query_ids=tuple(q for q in s.query_ids if q in members),
            full_size=sum(q in members for q in s.query_ids)) for p,s in selections.items()}
    elif domain!='routerbench':raise ValueError('Unknown study domain')
    ids=tuple(q for p in ('tuning','learning','deployment') for q in selections[p].query_ids)
    contexts=ContextTable({q:prior['W']@cache.get(q)+prior['b'] for q in ids},
        label=f"REAL {cache.provenance['model']}1536D with verified64D preference projection seed{seed}")
    arms=prior['arms'];meta=costs['estimator'];tune=selections['tuning']
    estimator=TuningTokenCostEstimator.fit(tune,{q:{a:costs['tuning-output-counts'][q][a] for a in arms} for q in tune.query_ids},
        {a:TokenPrices(**meta['prices'][a]) for a in arms},token_provenance=meta['token_provenance'],price_provenance=meta['price_provenance'])
    outcomes=pd.read_parquet(folder/'outcomes.parquet',columns=['query_id','arm_id','reward','actual_cost'])
    outcomes=outcomes.loc[outcomes.arm_id.isin(arms)].copy()
    lineage=dict(prior_file_sha256=entry['prior']['sha256'],prior_verification_sha256=entry['verification']['sha256'],
        input_bundle_sha256=inputs['manifest_sha256'],routing_cache_sha256=cache.fingerprint(),
        preference_cache_sha256=prior['embedding']['cache_sha256'],source_hashes=hashes,
        cost_manifest_sha256=costs['manifest']['manifest_sha256'],synthetic=False,paper_equivalent=False,
        code_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted((ROOT/'scripts').glob('*.py'))})
    return dict(prior=prior,queries=queries,outcomes=outcomes,contexts=contexts,selections=selections,
        estimator=estimator,input_counts={q:costs['input-counts'][q] for q in selections['deployment'].query_ids},
        tuning_input_counts={q:costs['input-counts'][q] for q in tune.query_ids},
        input_token_provenance=meta['token_provenance'],lineage=lineage,domain=domain,seed=seed)


def verify_tuning_lineage(stage,search,current):
    if stage['lineage']!=current or search['provenance']['lineage']!=current:
        raise ValueError('Tuning lineage differs from current verified cache, prior or code')


def validate_tuning(base,domain):
    folder=base/'tuning'/domain
    stage=sealed(folder/'manifest.json');verify_pins(stage,ROOT)
    search=load_search(folder/'search');bundle=load_data(base,domain,42)
    verify_tuning_lineage(stage,search,bundle['lineage'])
    selection=bundle['selections']['tuning']
    if (search['grid']!=read_json(ROOT/'configs/tuning_grid.json')
            or search['selection']['query_ids']!=list(selection.query_ids)
            or search['provenance']['context_sha256']!=bundle['contexts'].fingerprint(selection.query_ids)
            or read_json(folder/'estimator.json')!=bundle['estimator'].metadata()
            or read_json(folder/'calibration.json')!=calibrate_budget(folder/'search',bundle['tuning_input_counts'],bundle['estimator'],reference_arm='gpt-4-1106-preview')
            or stage['selected_parameters']!={m:c['parameters'] for m,c in search['selected'].items()}):
        raise ValueError('Tuning grid, context, estimator or calibrated bounds differ')
    return stage


def tune(base):
    for domain in ('routerbench','mmlu'):
        require_resources();folder=base/'tuning'/domain
        if (folder/'manifest.json').exists():
            validate_tuning(base,domain)
            continue
        if folder.exists():raise ValueError('Incomplete tuning folder; retain evidence and use a new destination')
        bundle=load_data(base,domain,42)
        result=run_search(bundle['prior'],bundle['queries'],bundle['outcomes'],bundle['contexts'],
            bundle['selections']['tuning'],read_json(ROOT/'configs/tuning_grid.json'),folder/'search',lineage=bundle['lineage'])
        calibration=calibrate_budget(folder/'search',bundle['tuning_input_counts'],bundle['estimator'],reference_arm='gpt-4-1106-preview')
        once(folder/'calibration.json',calibration);once(folder/'estimator.json',bundle['estimator'].metadata())
        stage=dict(version=1,kind='openai-core-tuning',domain=domain,synthetic=False,
            tuning_count=len(bundle['selections']['tuning'].query_ids),selected_parameters={m:v['parameters'] for m,v in result['selected'].items()},
            search=checked_pin(folder/'search/manifest.json'),calibration=checked_pin(folder/'calibration.json'),
            estimator=checked_pin(folder/'estimator.json'),lineage=bundle['lineage'])
        stage['manifest_sha256']=_digest(stage);once(folder/'manifest.json',stage)
        print(f"Tuned {domain}: {len(result['candidates'])} candidates on {stage['tuning_count']} queries",flush=True)
        del bundle;gc.collect()


def freeze(base):
    original=read_json(ROOT/'configs/frozen_experiment_protocol.json')
    protocol=core_declaration(original)
    protocol['prior_index']=checked_pin(base/'prior-index.json')
    protocol['embedding_verification']=checked_pin(base/'embedding-verification.json')
    pins=[ROOT/'configs/model_mapping.json',ROOT/'configs/tuning_grid.json',PREPARED/'manifest.json',INPUTS/'manifest.json',COSTS/'manifest.json']
    for source in ('source-0','source-1'):pins += [EMBEDDINGS/source/'manifest.json',EMBEDDINGS/source/'vectors.npy']
    protocol['source_pins']=[checked_pin(p) for p in pins]
    protocol['tuning']={}
    for domain in ('routerbench','mmlu'):
        folder=base/'tuning'/domain;stage=validate_tuning(base,domain);cal=read_json(folder/'calibration.json')
        if stage['tuning_count']!=protocol['primary_plan']['frozen_counts'][domain]['tuning']:
            raise ValueError('Tuning population differs')
        protocol['tuning'][domain]=dict(stage=checked_pin(folder/'manifest.json'),
            selected_parameters=stage['selected_parameters'],bounds={k:cal[k] for k in ('lower_bound','upper_bound')},
            reference_estimated_cost_per_query=cal['reference_estimated_cost_per_query'],
            absolute_reference_budget=cal['reference_estimated_cost_per_query']*protocol['primary_plan']['frozen_counts'][domain]['deployment'])
    protocol['code_pins']=[checked_pin(p) for p in sorted((ROOT/'scripts').glob('*.py'))]+[
        checked_pin(Path(__file__)),checked_pin(ROOT/'tools/openai_study_report.py'),
        checked_pin(ROOT/'tools/research/verify_runs.py'),checked_pin(ROOT/'tools/paper_preflight/compare.py')]
    protocol['manifest_sha256']=_digest(protocol)
    verify_pins(protocol,ROOT)
    once(base/'protocol.json',protocol)
    once(base/'inventory.json',dict(conditions=expected_runs(protocol)))
    print('Frozen858 primary conditions before deployment',flush=True)
    return protocol


def run(base,limit_groups=None):
    protocol=read_json(base/'protocol.json')
    if protocol['manifest_sha256']!=_digest({k:v for k,v in protocol.items() if k!='manifest_sha256'}):
        raise ValueError('Study protocol checksum mismatch')
    verify_pins(protocol,ROOT)
    groups=0
    for domain in ('routerbench','mmlu'):
        for seed in SEEDS:
            require_resources();bundle=load_data(base,domain,seed);bundle['protocol']=protocol
            bundle['lineage'].update(protocol_sha256=sha256(base/'protocol.json'),prior_index_sha256=sha256(base/'prior-index.json'),
                tuning_stage_sha256=protocol['tuning'][domain]['stage']['sha256'])
            for method in protocol['tuning'][domain]['selected_parameters']:
                require_resources()
                execute_group(bundle,base/'runs',method=method,resume=True)
                groups+=1
                write_json(base/'progress.json',dict(completed_groups=groups,expected_groups=78,last_group=f'{domain}/{seed}/{method}',
                    stage='running' if groups<78 else 'complete',additional_paid_api_calls=0))
                print(f'Completed group {groups}/78 {domain} seed{seed} {method}',flush=True)
                if limit_groups and groups>=limit_groups:return
            del bundle;gc.collect()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('fit','tune','freeze','run','all'))
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--limit-groups',type=int)
    args=parser.parse_args(argv)
    base=args.base.resolve()
    if not base.is_relative_to(ROOT.resolve()):raise ValueError('Study output must remain inside this project')
    require_resources();base.mkdir(parents=True,exist_ok=True)
    if args.stage in ('fit','all'):fit_priors(base)
    if args.stage in ('tune','all'):tune(base)
    if args.stage in ('freeze','all'):freeze(base)
    if args.stage in ('run','all'):run(base,args.limit_groups)


if __name__=='__main__':main()
