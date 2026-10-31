"""Two verified arms, shared eight-arm preference prior, fresh binary tuning."""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict,replace
from pathlib import Path

from scripts.cost_policy import TokenPrices,TuningTokenCostEstimator
from scripts.deployment_adapter import FrozenDeploymentRouter
from scripts.experiments import deploy_run,expected_runs,fraction_key,learn_stream,load_artifact,verify_pins
from scripts.pilot import PILOT
from scripts.policies import LinearUCB,ExploreOnly,EpochGreedy,RandomPolicy,FixedPolicy
from scripts.resources import require_resources
from scripts.tuning import SCORE_RULE,calibrate_budget,candidate_specs,load_search
from tools.openai_study import DEFAULT_BASE as CORE,COSTS,checked_pin,load_data,once,sealed
from tools.paper_preflight.common import ROOT,digest,read_json,sha256

ARMS=('gpt-4-1106-preview','meta/llama-2-70b-chat')
BASE=ROOT/'outputs/openai-followup-20261031/binary'


def binary_policy(parent,method,parameters):
    arms=ARMS;dimension=parent['dimension']
    if not set(arms)<=set(parent['arms']):raise ValueError('Missing verified binary arm')
    if method=='pilot':
        return PILOT(arms,dimension,priors={a:parent['theta'][parent['arms'].index(a)] for a in arms},
            lambdas={a:parent['metrics']['validation']['per_arm'][a]['lambda'] for a in arms},
            provenance='shared8-arm preference cache sha256='+parent['embedding']['cache_sha256'],**parameters)
    if method=='linucb':return LinearUCB(arms,dimension,**parameters)
    if method=='epoch-greedy':return EpochGreedy(arms,dimension,**parameters)
    if method=='explore-only':return ExploreOnly(arms,dimension)
    if method=='random':return RandomPolicy(arms,dimension)
    if method in ('fixed-00','fixed-01') and parameters.get('arm_id')==arms[int(method[-2:])]:
        return FixedPolicy(arms,dimension,**parameters)
    raise ValueError('Method/arm outside the verified binary catalog')


def binary_declaration(core):
    value=copy.deepcopy(core);value.pop('manifest_sha256',None)
    value['kind']='openai-verified-binary-study';value['arms']=list(ARMS)
    value['primary_plan']['datasets']=['routerbench']
    value['primary_plan']['scope']='PILOT D.4 verified GPT4-Llama2 pair, local split, shared8-arm preference prior'
    value['ablation_plan']=dict(variants=[],budget_fractions=[],inventory=dict(primary_plus_ablation_deployment_runs=231))
    return value


def bundle(seed):
    value=load_data(CORE,'routerbench',seed);selection=value['selections']['tuning']
    costs=read_json(COSTS/'estimator.json');outputs=read_json(COSTS/'tuning-output-counts.json')
    value['estimator']=TuningTokenCostEstimator.fit(selection,
        {q:{a:outputs[q][a] for a in ARMS} for q in selection.query_ids},
        {a:TokenPrices(**costs['prices'][a]) for a in ARMS},
        token_provenance=costs['token_provenance'],price_provenance=costs['price_provenance'])
    value['outcomes']=value['outcomes'].loc[value['outcomes'].arm_id.isin(ARMS)].copy()
    value['lineage']=dict(value['lineage'],binary_arms=list(ARMS),pretraining_pool='shared original8 admitted arms')
    return value


def declare():
    core=sealed(CORE/'protocol.json');verify_pins(core,ROOT)
    value=dict(version=1,kind='binary-prerun-declaration',parent=checked_pin(CORE/'protocol.json'),
        code=checked_pin(Path(__file__)),grid=checked_pin(ROOT/'configs/tuning_grid.json'),arms=list(ARMS),
        study=binary_declaration(core),tuning='fresh18 candidates on1000 tuning queries seed42; maximum prequential total reward',
        prior='two arm directions/lambdas from unchanged original8-arm preference fit; shared64D projection',
        exact_author_prior=False,inference_api_calls=0)
    value['manifest_sha256']=digest(value);once(BASE/'declaration.json',value)


def tune():
    declaration=sealed(BASE/'declaration.json');verify_pins(declaration,ROOT);require_resources()
    value=bundle(42);prior=value['prior'];selection=value['selections']['tuning'];ids=list(selection.query_ids)
    grid=read_json(ROOT/'configs/tuning_grid.json');folder=BASE/'tuning/search'
    if folder.exists():return load_search(folder)
    lookup={(r.query_id,r.arm_id):(float(r.reward),float(r.actual_cost)) for r in value['outcomes'].itertuples(index=False)}
    candidates=[];files={}
    for spec in candidate_specs(ARMS,grid):
        model=binary_policy(prior,spec['method'],spec['parameters']);model.reset(grid['seed'])
        destination=folder/spec['id'];destination.mkdir(parents=True,exist_ok=False);rewards=[]
        with (destination/'trace.jsonl').open('x',encoding='utf-8') as stream:
            for t,q in enumerate(ids,1):
                x=value['contexts'].get(q);predictions=model.predict_rewards(x);a=model.select(x)
                reward,cost=lookup[q,a]
                row=dict(round=t,query_id=q,arm_id=a,reward=reward,actual_cost=cost,predictions=predictions,rounds_before_feedback=model.rounds)
                model.observe(a,x,reward);rewards.append(reward)
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        frozen=FrozenDeploymentRouter(model,seed=grid['seed']);frozen.save_checkpoint(destination/'router.json')
        candidate=dict(spec,rounds=len(ids),total_reward=math.fsum(rewards),mean_reward=math.fsum(rewards)/len(ids),
            trace=spec['id']+'/trace.jsonl',checkpoint=spec['id']+'/router.json',
            final_state_sha256=frozen.metadata()['source_state_sha256'])
        candidates.append(candidate)
        for key in ('trace','checkpoint'):files[candidate[key]]=sha256(folder/candidate[key])
    selected={}
    for candidate in candidates:
        method=candidate['method']
        if method not in selected or candidate['total_reward']>selected[method]['total_reward']:selected[method]=candidate
    result=dict(version=1,kind='tuning-search',phase='tuning',selection={**asdict(selection),'query_ids':ids},
        arms=list(ARMS),grid=grid,score_rule=SCORE_RULE,candidates=candidates,selected=selected,
        provenance=dict(lineage=value['lineage'],declaration_sha256=sha256(BASE/'declaration.json')),
        synthetic=False,paper_equivalent=False,files=files)
    result['manifest_sha256']=digest(result);once(folder/'manifest.json',result);load_search(folder)
    calibration=calibrate_budget(folder,value['tuning_input_counts'],value['estimator'],reference_arm=ARMS[0])
    once(BASE/'tuning/calibration.json',calibration);once(BASE/'tuning/estimator.json',value['estimator'].metadata())
    print('Completed18 binary tuning candidates',flush=True)


def freeze():
    declaration=sealed(BASE/'declaration.json');verify_pins(declaration,ROOT)
    search=load_search(BASE/'tuning/search');calibration=read_json(BASE/'tuning/calibration.json')
    value=declaration['study'];parent=sealed(CORE/'protocol.json')
    value['tuning']={'routerbench':dict(selected_parameters={m:c['parameters'] for m,c in search['selected'].items()},
        bounds={k:calibration[k] for k in ('lower_bound','upper_bound')},
        reference_estimated_cost_per_query=calibration['reference_estimated_cost_per_query'],
        absolute_reference_budget=calibration['reference_estimated_cost_per_query']*value['primary_plan']['frozen_counts']['routerbench']['deployment'])}
    value['binary_prerun']=checked_pin(BASE/'declaration.json')
    value['binary_tuning']=[checked_pin(BASE/name) for name in ('tuning/search/manifest.json','tuning/calibration.json','tuning/estimator.json')]
    value['binary_source']=checked_pin(Path(__file__));value['parent']=checked_pin(CORE/'protocol.json')
    value['manifest_sha256']=digest(value);expected_runs(value);once(BASE/'protocol.json',value)


def run(limit=None):
    protocol=sealed(BASE/'protocol.json');verify_pins(protocol,ROOT);conditions=expected_runs(protocol);groups=0
    for seed in (7,42,2026):
        require_resources();data=bundle(seed);learning=data['selections']['learning'];deployment=data['selections']['deployment']
        controls=protocol['tuning']['routerbench']
        for method,parameters in controls['selected_parameters'].items():
            require_resources();lineage=dict(data['lineage'],domain='routerbench',seed=seed,method=method,variant='main',binary_protocol_sha256=sha256(BASE/'protocol.json'))
            learning_dir=BASE/'runs/learning/routerbench'/f'seed-{seed}'/'main'/method
            trained=learn_stream(model=binary_policy(data['prior'],method,parameters),queries=data['queries'],outcomes=data['outcomes'],
                contexts=data['contexts'],selection=learning,seed=seed,fractions=protocol['primary_plan']['learning_fractions'],
                lineage=lineage,output_dir=learning_dir,resume=True)
            for row in [r for r in conditions if r['seed']==seed and r['method']==method]:
                snap=trained['snapshots'][fraction_key(row['learning_fraction'])];checkpoint=learning_dir/snap['file']
                selected=replace(learning,query_ids=learning.query_ids[:row['learning_count']],limit=row['learning_count'],subset_seed=None)
                members=set(selected.query_ids+deployment.query_ids)
                deploy_lineage=dict(data['lineage'],condition=row,learning_manifest_sha256=sha256(learning_dir/'manifest.json'),
                    learning_checkpoint_sha256=sha256(checkpoint),learning_rounds=snap['rounds'],learning_metrics=snap['metrics'],
                    learning_selection_role=trained['learning_selection_role'],binary_protocol_sha256=sha256(BASE/'protocol.json'))
                deploy_run(router=FrozenDeploymentRouter.load_checkpoint(checkpoint),
                    queries=data['queries'].loc[data['queries'].query_id.isin(members)],
                    outcomes=data['outcomes'].loc[data['outcomes'].query_id.isin(members)],contexts=data['contexts'],
                    learning_selection=selected,deployment_selection=deployment,estimator=data['estimator'],
                    input_tokens=data['input_counts'],input_token_provenance=data['input_token_provenance'],
                    budget=controls['absolute_reference_budget']*row['budget_fraction'],bin_size=protocol['primary_plan']['bin_size'],
                    **controls['bounds'],lineage=deploy_lineage,output_dir=BASE/'runs/deployment'/row['id'],resume=True)
            groups+=1;print(f'Completed binary{groups}/21 seed{seed} {method}',flush=True)
            if limit and groups>=limit:return


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['declare','tune','freeze','run'])
    parser.add_argument('--limit',type=int);args=parser.parse_args();require_resources()
    if args.action=='run':run(args.limit)
    else:globals()[args.action]()


if __name__=='__main__':main()
