"""Instructor-XL sensitivity with frozen local vectors and the full core design.

Adds no API client and changes no source pinned by earlier OpenAI/BGE studies.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from scripts.bandit_contracts import ContextTable
from scripts.cost_inputs import load_cost_inputs
from scripts.cost_policy import TokenPrices,TuningTokenCostEstimator
from scripts.embedding_store import DiskEmbeddingCache
from scripts.experiments import execute_group,expected_runs,verify_pins
from scripts.pretraining import _digest,_read_json,_write_json,fit_preference_prior,load_checkpoint,save_checkpoint
from scripts.research_pipeline import load_prepared,load_routing
from scripts.resources import require_resources
from scripts.tuning import calibrate_budget,load_search,run_search
from tools.openai_study import COSTS,INPUTS,PREPARED,checked_pin,core_declaration,once,sealed,verify_prior
from tools.openai_study_instructor import MODEL,REVISION
from tools.paper_preflight.common import ROOT,read_json,sha256

ENCODING=ROOT/'outputs/openai-followup-20261031/instructor'
BASE=ROOT/'outputs/instructor-core-20261031'
SEEDS=(7,42,2026)


def instructor_declaration(original):
    value=core_declaration(original)
    value['kind']='frozen-instructor-xl-core-study'
    value['primary_plan']['embedding_recipe']='Pinned local Instructor-XL768D, instruction-masked mean/Dense/L2,512-token cap,FP32'
    value['primary_plan']['scope']='Instructor-XL eight-arm sensitivity; same original core design, fresh priors and tuning'
    return value


def aligned_instructor(path,rows,role,completion):
    cache=DiskEmbeddingCache.load(path)
    if (not completion.get('complete') or role not in completion['role_caches']
            or cache.fingerprint()!=completion['role_caches'][role]['fingerprint']
            or len(rows)!=completion['role_caches'][role]['rows']
            or cache.ids!=tuple(r['query_id'] for r in rows)
            or cache.prompt_hashes!=tuple(r['prompt_hash'] for r in rows)
            or cache.dimension!=768 or cache.provenance['model']!=MODEL
            or cache.provenance['version']!=REVISION or cache.provenance['synthetic']):
        raise ValueError('Instructor completed cache/input/recipe identity mismatch')
    return cache


def verified_encoding():
    completion=read_json(ENCODING/'completion.json');proof=read_json(ENCODING/'independent-verification.json')
    if (not completion['complete'] or not proof['verified'] or not proof['complete']
            or proof['completion_sha256']!=sha256(ENCODING/'completion.json')
            or completion['protocol_sha256']!=sha256(ENCODING/'encoding-protocol.json')
            or proof['verifier_sha256']!=sha256(ROOT/'tools/openai_study_instructor_verify.py')):
        raise ValueError('Complete independently verified Instructor encoding required')
    protocol=read_json(ENCODING/'encoding-protocol.json')
    for path,fingerprint in protocol['source_pins'].items():
        if sha256(ROOT/path)!=fingerprint:raise ValueError('Frozen encoding dependency changed')
    if 'recovery_overlay_sha256' in completion:
        if proof.get('recovery_overlay_sha256')!=completion['recovery_overlay_sha256'] or sha256(ENCODING/'recovery-overlay.json')!=completion['recovery_overlay_sha256']:
            raise ValueError('Recovery overlay lineage changed')
        overlay=read_json(ENCODING/'recovery-overlay.json')
        if sha256(ROOT/'tools/openai_study_instructor_tail.py')!=overlay['producer_sha256']:
            raise ValueError('Recovery producer source changed')
    return completion


def declare():
    verified_encoding()
    original=read_json(ROOT/'configs/frozen_experiment_protocol.json')
    protocol=instructor_declaration(original)
    value=dict(version=1,kind='instructor-study-prerun-declaration',study=protocol,
        representation='full512-token Instructor recipe; local instruction assumption; not isolated encoder causality',
        encoding=[checked_pin(ENCODING/name) for name in ('completion.json','independent-verification.json','encoding-protocol.json')],
        source=checked_pin(Path(__file__)),grid=checked_pin(ROOT/'configs/tuning_grid.json'),
        seeds=list(SEEDS),additional_paid_api_calls=0)
    value['manifest_sha256']=_digest(value);once(BASE/'declaration.json',value)


def fit():
    declaration=sealed(BASE/'declaration.json');verify_pins(declaration,ROOT);completion=verified_encoding()
    prepared=load_prepared(PREPARED)
    cache=aligned_instructor(ENCODING/'caches/source-1',prepared['preference-inputs'],'source-1',completion)
    sizes=read_json(COSTS/'cost-size-identity_match.json');entries=[]
    provenance=json.dumps({k:sizes[k] for k in ('provenance','source_urls')},sort_keys=True)
    for seed in (42,7,2026):
        require_resources();folder=BASE/'priors'/f'seed-{seed}'
        if not folder.exists():
            prior=fit_preference_prior(cache,prepared['comparisons'],prepared['manifest']['arms'],sizes['sizes'],provenance,
                dimension=64,projection_epochs=30,prior_epochs=40,learning_rate=.03,margin=.2,seed=seed)
            folder.mkdir(parents=True,exist_ok=False);save_checkpoint(folder/'prior.json',prior)
            _write_json(folder/'fit-manifest.json',dict(version=1,kind='instructor-preference-fit',
                prepared_sha256=sha256(PREPARED/'manifest.json'),checkpoint_sha256=sha256(folder/'prior.json'),
                arms=prior['arms'],prior_ready=prior['metrics']['validation']['all_lambdas_ready'],synthetic=False,
                cost_sizes=sizes,declaration_sha256=sha256(BASE/'declaration.json'),metrics=prior['metrics']))
        prior=load_checkpoint(folder/'prior.json');manifest=_read_json(folder/'fit-manifest.json')
        if (manifest['checkpoint_sha256']!=sha256(folder/'prior.json') or not manifest['prior_ready'] or manifest['synthetic']
                or manifest['prepared_sha256']!=sha256(PREPARED/'manifest.json')
                or manifest['declaration_sha256']!=sha256(BASE/'declaration.json')
                or (prior['dimension'],prior['input_dimension'],prior['protocol']['seed'])!=(64,768,seed)):
            raise ValueError('Prior blocked or Instructor fit identity changed')
        proof=verify_prior(prior,cache,prepared['comparisons']);once(folder/'verification.json',proof)
        entries.append(dict(seed=seed,dimension=64,prior=checked_pin(folder/'prior.json'),
            fit=checked_pin(folder/'fit-manifest.json'),verification=checked_pin(folder/'verification.json')))
        print(f'Instructor prior seed{seed}: {proof["correct_pairs"]}/{proof["calibration_pairs"]} calibration',flush=True)
    index=dict(version=1,kind='instructor-prior-index',fits=sorted(entries,key=lambda r:r['seed']),synthetic=False)
    index['manifest_sha256']=_digest(index);once(BASE/'prior-index.json',index)


def load_data(domain,seed):
    completion=verified_encoding();index=sealed(BASE/'prior-index.json');verify_pins(index,ROOT)
    entry=next(e for e in index['fits'] if e['seed']==seed);prior=load_checkpoint(ROOT/entry['prior']['path'])
    prepared=load_prepared(PREPARED)
    preference=aligned_instructor(ENCODING/'caches/source-1',prepared['preference-inputs'],'source-1',completion)
    if prior['embedding']['cache_sha256']!=preference.fingerprint():raise ValueError('Instructor prior/preference lineage mismatch')
    inputs=read_json(INPUTS/'manifest.json')
    if inputs['manifest_sha256']!=_digest({k:v for k,v in inputs.items() if k!='manifest_sha256'}):raise ValueError('Input checksum mismatch')
    for name,want in inputs['files'].items():
        if sha256(INPUTS/name)!=want:raise ValueError('Full input source changed')
    cache=aligned_instructor(ENCODING/'caches/source-0',read_json(INPUTS/'routing-inputs.json'),'source-0',completion)
    if cache.provenance!=prior['embedding']['recipe']:raise ValueError('Instructor routing/preference recipe mismatch')
    folder,queries,splits,hashes=load_routing(ROOT/'data');costs=load_cost_inputs(COSTS)
    if any(source!=hashes for source in (inputs['source_hashes'],prepared['manifest']['source_hashes'],costs['manifest']['source_hashes'])):
        raise ValueError('Instructor source/cost lineage mismatch')
    selections={phase:splits.select(phase) for phase in ('tuning','learning','deployment')}
    if domain=='mmlu':
        members=set(queries.loc[queries.task.str.startswith('mmlu-'),'query_id'])
        selections={p:replace(s,query_ids=tuple(q for q in s.query_ids if q in members),full_size=sum(q in members for q in s.query_ids)) for p,s in selections.items()}
    elif domain!='routerbench':raise ValueError('Unknown study domain')
    ids=tuple(q for phase in ('tuning','learning','deployment') for q in selections[phase].query_ids)
    contexts=ContextTable({q:prior['W']@cache.get(q)+prior['b'] for q in ids},
        label=f'REAL pinned InstructorXL768D with verified64D preference projection seed{seed}')
    arms=prior['arms'];meta=costs['estimator'];tuning=selections['tuning']
    estimator=TuningTokenCostEstimator.fit(tuning,
        {q:{a:costs['tuning-output-counts'][q][a] for a in arms} for q in tuning.query_ids},
        {a:TokenPrices(**meta['prices'][a]) for a in arms},token_provenance=meta['token_provenance'],price_provenance=meta['price_provenance'])
    outcomes=pd.read_parquet(folder/'outcomes.parquet',columns=['query_id','arm_id','reward','actual_cost'])
    outcomes=outcomes.loc[outcomes.arm_id.isin(arms)].copy()
    lineage=dict(prior_file_sha256=entry['prior']['sha256'],prior_verification_sha256=entry['verification']['sha256'],
        input_bundle_sha256=inputs['manifest_sha256'],routing_cache_sha256=cache.fingerprint(),
        preference_cache_sha256=prior['embedding']['cache_sha256'],source_hashes=hashes,
        cost_manifest_sha256=costs['manifest']['manifest_sha256'],synthetic=False,paper_equivalent=False,
        encoding_verification_sha256=sha256(ENCODING/'independent-verification.json'),
        code_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted((ROOT/'scripts').glob('*.py'))})
    return dict(prior=prior,queries=queries,outcomes=outcomes,contexts=contexts,selections=selections,estimator=estimator,
        input_counts={q:costs['input-counts'][q] for q in selections['deployment'].query_ids},
        tuning_input_counts={q:costs['input-counts'][q] for q in tuning.query_ids},
        input_token_provenance=meta['token_provenance'],lineage=lineage,domain=domain,seed=seed)


def validate_tuning(domain):
    folder=BASE/'tuning'/domain;stage=sealed(folder/'manifest.json');verify_pins(stage,ROOT)
    search=load_search(folder/'search');data=load_data(domain,42)
    if (stage['lineage']!=data['lineage'] or search['provenance']['lineage']!=data['lineage']
            or search['grid']!=read_json(ROOT/'configs/tuning_grid.json')
            or search['selection']['query_ids']!=list(data['selections']['tuning'].query_ids)
            or read_json(folder/'estimator.json')!=data['estimator'].metadata()
            or read_json(folder/'calibration.json')!=calibrate_budget(folder/'search',data['tuning_input_counts'],data['estimator'],reference_arm='gpt-4-1106-preview')
            or stage['selected_parameters']!={m:c['parameters'] for m,c in search['selected'].items()}):
        raise ValueError('Frozen Instructor tuning dependencies changed')
    return stage


def tune():
    declaration=sealed(BASE/'declaration.json');verify_pins(declaration,ROOT)
    for domain in ('routerbench','mmlu'):
        require_resources();folder=BASE/'tuning'/domain
        if (folder/'manifest.json').exists():validate_tuning(domain);continue
        if folder.exists():raise ValueError('Incomplete Instructor tuning artifact; retain and inspect')
        data=load_data(domain,42)
        search=run_search(data['prior'],data['queries'],data['outcomes'],data['contexts'],data['selections']['tuning'],
            read_json(ROOT/'configs/tuning_grid.json'),folder/'search',lineage=data['lineage'])
        calibration=calibrate_budget(folder/'search',data['tuning_input_counts'],data['estimator'],reference_arm='gpt-4-1106-preview')
        once(folder/'calibration.json',calibration);once(folder/'estimator.json',data['estimator'].metadata())
        stage=dict(version=1,kind='instructor-core-tuning',domain=domain,synthetic=False,
            tuning_count=len(data['selections']['tuning'].query_ids),selected_parameters={m:c['parameters'] for m,c in search['selected'].items()},
            search=checked_pin(folder/'search/manifest.json'),calibration=checked_pin(folder/'calibration.json'),
            estimator=checked_pin(folder/'estimator.json'),lineage=data['lineage'])
        stage['manifest_sha256']=_digest(stage);once(folder/'manifest.json',stage)
        print(f'Instructor tuning {domain}: {len(search["candidates"])} candidates',flush=True)
        del data;gc.collect()


def freeze():
    declaration=sealed(BASE/'declaration.json');verify_pins(declaration,ROOT);verified_encoding()
    protocol=copy.deepcopy(declaration['study']);protocol['declaration']=checked_pin(BASE/'declaration.json')
    protocol['prior_index']=checked_pin(BASE/'prior-index.json');protocol['encoding_verification']=checked_pin(ENCODING/'independent-verification.json')
    pins=[ROOT/'configs/model_mapping.json',ROOT/'configs/tuning_grid.json',PREPARED/'manifest.json',INPUTS/'manifest.json',COSTS/'manifest.json']
    for role in ('source-0','source-1'):pins +=[ENCODING/'caches'/role/'manifest.json',ENCODING/'caches'/role/'vectors.npy']
    protocol['source_pins']=[checked_pin(p) for p in pins];protocol['tuning']={}
    for domain in ('routerbench','mmlu'):
        folder=BASE/'tuning'/domain;stage=validate_tuning(domain);calibration=read_json(folder/'calibration.json')
        if stage['tuning_count']!=protocol['primary_plan']['frozen_counts'][domain]['tuning']:raise ValueError('Tuning population changed')
        protocol['tuning'][domain]=dict(stage=checked_pin(folder/'manifest.json'),selected_parameters=stage['selected_parameters'],
            bounds={k:calibration[k] for k in ('lower_bound','upper_bound')},
            reference_estimated_cost_per_query=calibration['reference_estimated_cost_per_query'],
            absolute_reference_budget=calibration['reference_estimated_cost_per_query']*protocol['primary_plan']['frozen_counts'][domain]['deployment'])
    protocol['code_pins']=[checked_pin(p) for p in sorted((ROOT/'scripts').glob('*.py'))]+[
        checked_pin(Path(__file__)),checked_pin(ROOT/'tools/openai_study.py'),
        checked_pin(ROOT/'tools/openai_study_instructor_report.py'),checked_pin(ROOT/'tools/openai_study_instructor_verify.py'),
        checked_pin(ROOT/'tools/openai_study_instructor_tail.py'),
        checked_pin(ROOT/'tools/research/verify_runs.py'),checked_pin(ROOT/'tools/paper_preflight/compare.py')]
    protocol['manifest_sha256']=_digest(protocol);verify_pins(protocol,ROOT)
    once(BASE/'protocol.json',protocol);once(BASE/'inventory.json',dict(conditions=expected_runs(protocol)))
    print('Frozen858 Instructor conditions',flush=True)


def run(limit_groups=None):
    protocol=sealed(BASE/'protocol.json');verify_pins(protocol,ROOT);groups=0
    for domain in ('routerbench','mmlu'):
        for seed in SEEDS:
            require_resources();data=load_data(domain,seed);data['protocol']=protocol
            data['lineage'].update(protocol_sha256=sha256(BASE/'protocol.json'),prior_index_sha256=sha256(BASE/'prior-index.json'),
                tuning_stage_sha256=protocol['tuning'][domain]['stage']['sha256'])
            for method in protocol['tuning'][domain]['selected_parameters']:
                require_resources();execute_group(data,BASE/'runs',method=method,resume=True);groups+=1
                print(f'Instructor group{groups}/78 {domain} seed{seed} {method}',flush=True)
                if limit_groups and groups>=limit_groups:return
            del data;gc.collect()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['declare','fit','tune','freeze','run'])
    parser.add_argument('--limit-groups',type=int);args=parser.parse_args();require_resources()
    if args.stage=='run':run(args.limit_groups)
    else:globals()[args.stage]()


if __name__=='__main__':main()
