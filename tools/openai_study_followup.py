"""Additive OpenAI ablations; completed core results remain immutable."""
from __future__ import annotations

import argparse
import copy
import gc
import shutil
from dataclasses import replace
from pathlib import Path

from scripts.bandit_contracts import ContextTable
from scripts.deployment_adapter import FrozenDeploymentRouter
from scripts.experiments import (deploy_run, execute_group, expected_runs, load_artifact, verify_pins)
from scripts.pretraining import _digest, _read_json, load_checkpoint
from scripts.research_pipeline import fit_prepared, load_prepared, _aligned_cache
from scripts.resources import require_resources
from tools.openai_study import (DEFAULT_BASE as CORE, EMBEDDINGS, PREPARED, COSTS, SEEDS,
    checked_pin, load_data, once, sealed, verify_prior)
from tools.paper_preflight.common import ROOT, read_json, sha256, write_json

BASE=ROOT/'outputs/openai-followup-20261031/ablations'


def declaration(core,ablation):
    value=copy.deepcopy(core)
    value.pop('manifest_sha256',None)
    value.update(kind='openai-controlled-ablation-addon',ablation_plan=copy.deepcopy(ablation),
        addon_scope='240 new ablations;858 primary conditions are immutable parent references, not reruns')
    for item in value['ablation_plan']['variants']:
        item['interpretation']=item['interpretation'].replace('BGE-M3','the same OpenAI base embeddings')
    return value


def addon_conditions(protocol):
    return [r for r in expected_runs(protocol) if r['family']=='ablation']


def validate_parent(core,addon):
    if any(core[k]!=addon[k] for k in ('arms','tuning','primary_plan')):
        raise ValueError('Add-on changes a frozen parent control')


def select_prior(index,seed,dimension):
    entries=[r for r in index['fits'] if (r['seed'],r['dimension'])==(seed,dimension)]
    if len(entries)!=1:raise ValueError('Prior index needs one exact seed/dimension match')
    return entries[0]


def copy_learning(source,destination,*,expected_parent):
    source,destination=Path(source),Path(destination)
    value=load_artifact(source)
    if (value['kind']!='learning-stream' or '1' not in value['snapshots']
            or value['request']['lineage'].get('protocol_sha256')!=expected_parent):
        raise ValueError('Reused learning is not from the frozen parent')
    if destination.exists():
        copied=load_artifact(destination)
        if copied!=value or sha256(destination/'manifest.json')!=sha256(source/'manifest.json'):
            raise ValueError('Reused learning differs from parent')
    else:
        shutil.copytree(source,destination)
        copied=load_artifact(destination)
        if copied!=value:raise ValueError('Learning copy failed validation')
    return copied


def fit8(base):
    verify_pins(sealed(CORE/'protocol.json'),ROOT)
    index=sealed(CORE/'prior-index.json');verify_pins(index,ROOT)
    entries=copy.deepcopy(index['fits'])
    prepared=load_prepared(PREPARED)
    cache=_aligned_cache(EMBEDDINGS/'source-1',prepared['preference-inputs'])
    for seed in SEEDS:
        require_resources();folder=base/'priors'/f'seed-{seed}-dim8'
        if not folder.exists():
            fit_prepared(PREPARED,EMBEDDINGS/'source-1',read_json(COSTS/'cost-size-identity_match.json'),folder,
                dimension=8,projection_epochs=30,prior_epochs=40,seed=seed)
        fit=_read_json(folder/'fit-manifest.json');prior=load_checkpoint(folder/'prior.json')
        protocol=prior['protocol']
        if (fit['checkpoint_sha256']!=sha256(folder/'prior.json') or fit['prepared_sha256']!=sha256(PREPARED/'manifest.json')
                or fit['synthetic'] or not fit['prior_ready'] or prior['dimension']!=8
                or protocol['seed']!=seed or protocol['projection_epochs']!=30 or protocol['prior_epochs']!=40
                or protocol['learning_rate']!=.03 or protocol['margin']!=.2):
            raise ValueError('8D fit violates controlled pretraining settings')
        once(folder/'verification.json',verify_prior(prior,cache,prepared['comparisons']))
        entries.append(dict(seed=seed,dimension=8,prior=checked_pin(folder/'prior.json'),
            fit=checked_pin(folder/'fit-manifest.json'),verification=checked_pin(folder/'verification.json')))
        print(f'Verified8D prior seed{seed}',flush=True)
    result=dict(version=1,kind='openai-six-prior-addon-index',synthetic=False,
        parent=checked_pin(CORE/'prior-index.json'),fits=sorted(entries,key=lambda r:(r['dimension'],r['seed'])))
    result['manifest_sha256']=_digest(result);once(base/'prior-index.json',result)
    return result


def freeze(base):
    core=sealed(CORE/'protocol.json');verify_pins(core,ROOT)
    index=sealed(base/'prior-index.json');verify_pins(index,ROOT)
    if {(e['seed'],e['dimension']) for e in index['fits']}!={(s,d) for s in SEEDS for d in (8,64)}:
        raise ValueError('Six exact priors are required')
    value=declaration(core,read_json(ROOT/'configs/ablation_plan.json'))
    validate_parent(core,value)
    value.update(parent_core=checked_pin(CORE/'protocol.json'),addon_prior_index=checked_pin(base/'prior-index.json'),
        addon_code_pins=[checked_pin(Path(__file__)),checked_pin(ROOT/'configs/ablation_plan.json')])
    value['manifest_sha256']=_digest(value)
    once(base/'protocol.json',value)
    once(base/'inventory.json',dict(conditions=addon_conditions(value)))
    print('Frozen240 controlled ablations',flush=True)


def single_bin(bundle,base):
    domain,seed=bundle['domain'],bundle['seed']
    source=CORE/'runs/learning'/domain/f'seed-{seed}'/'main/pilot'
    destination=base/'runs/learning'/domain/f'seed-{seed}'/'main/pilot'
    learned=copy_learning(source,destination,expected_parent=sha256(CORE/'protocol.json'))
    snap=learned['snapshots']['1'];checkpoint=destination/snap['file']
    parent=learned['request']['lineage']
    if (parent['prior_file_sha256']!=bundle['lineage']['prior_file_sha256']
            or snap['rounds']!=len(bundle['selections']['learning'].query_ids)):
        raise ValueError('Single-bin source prior or full-learning count differs')
    learning=replace(bundle['selections']['learning'],limit=snap['rounds'],subset_seed=None)
    deployment=bundle['selections']['deployment'];tuning=bundle['protocol']['tuning'][domain]
    for row in addon_conditions(bundle['protocol']):
        if (row['domain'],row['seed'],row['variant'])!=(domain,seed,'single-budget-bin'):continue
        lineage=dict(bundle['lineage'],condition=row,
            learning_manifest_sha256=sha256(destination/'manifest.json'),
            learning_checkpoint_sha256=sha256(checkpoint),learning_rounds=snap['rounds'],learning_metrics=snap['metrics'],
            learning_selection_role=learned['learning_selection_role'],reused_parent_manifest=checked_pin(source/'manifest.json'))
        deploy_run(router=FrozenDeploymentRouter.load_checkpoint(checkpoint),queries=bundle['queries'],
            outcomes=bundle['outcomes'],contexts=bundle['contexts'],learning_selection=learning,
            deployment_selection=deployment,estimator=bundle['estimator'],input_tokens=bundle['input_counts'],
            input_token_provenance=bundle['input_token_provenance'],
            budget=tuning['absolute_reference_budget']*row['budget_fraction'],bin_size=len(deployment.query_ids),
            **tuning['bounds'],lineage=lineage,output_dir=base/'runs/deployment'/row['id'],resume=True)


def run(base,limit_groups=None):
    protocol=sealed(base/'protocol.json');core=sealed(CORE/'protocol.json')
    verify_pins(protocol,ROOT);validate_parent(core,protocol)
    index=sealed(base/'prior-index.json');verify_pins(index,ROOT)
    groups=0
    for domain in ('routerbench','mmlu'):
        for seed in SEEDS:
            require_resources();bundle=load_data(CORE,domain,seed)
            bundle['protocol']=protocol
            bundle['lineage'].update(protocol_sha256=sha256(base/'protocol.json'),
                prior_index_sha256=sha256(base/'prior-index.json'),parent_core_sha256=sha256(CORE/'protocol.json'),
                tuning_stage_sha256=protocol['tuning'][domain]['stage']['sha256'])
            for variant in (v['id'] for v in protocol['ablation_plan']['variants']):
                require_resources()
                if variant=='single-budget-bin':single_bin(bundle,base)
                elif variant=='projection-dimension-8':
                    entry=select_prior(index,seed,8);small=dict(bundle)
                    small['prior']=load_checkpoint(ROOT/entry['prior']['path'])
                    prior=small['prior'];cache=_aligned_cache(EMBEDDINGS/'source-0',read_json(ROOT/'outputs/research-input-audit-20261029/inputs/routing-inputs.json'))
                    ids=tuple(q for p in ('tuning','learning','deployment') for q in bundle['selections'][p].query_ids)
                    small['contexts']=ContextTable({q:prior['W']@cache.get(q)+prior['b'] for q in ids},
                        label=f'REAL OpenAI1536D controlled8D projection seed{seed}')
                    small['lineage']=dict(bundle['lineage'],prior_file_sha256=entry['prior']['sha256'],
                        prior_verification_sha256=entry['verification']['sha256'])
                    execute_group(small,base/'runs',method='pilot',variant=variant,resume=True)
                    del small,cache;gc.collect()
                else:execute_group(bundle,base/'runs',method='pilot',variant=variant,resume=True)
                groups+=1
                write_json(base/'progress.json',dict(completed_groups=groups,expected_groups=30,
                    last_group=f'{domain}/{seed}/{variant}',stage='complete' if groups==30 else 'running',additional_paid_api_calls=0))
                print(f'Completed add-on group{groups}/30 {domain} seed{seed} {variant}',flush=True)
                if limit_groups and groups>=limit_groups:return
            del bundle;gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('fit8','freeze','run'))
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--limit-groups',type=int)
    args=parser.parse_args();base=args.base.resolve()
    if not base.is_relative_to(ROOT.resolve()):raise ValueError('Add-on output must remain in the project')
    require_resources();base.mkdir(parents=True,exist_ok=True)
    if args.stage=='fit8':fit8(base)
    elif args.stage=='freeze':freeze(base)
    else:run(base,args.limit_groups)


if __name__=='__main__':main()
