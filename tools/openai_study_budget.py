"""Declared budget/exhaustion counterfactuals using fixed historical scores."""
from __future__ import annotations

import math
import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from tools.paper_preflight.compare import relative_metrics, simulate_budget_stream

POLICIES=('guarded-stop','paper-literal-stop','guarded-skip','fixed-per-query','retrospective-lagrange')


def condition_inventory(core):
    plan=core['primary_plan'];rows=[]
    for domain in plan['datasets']:
        for seed in plan['replicate_seeds']:
            for policy in POLICIES:
                for fraction in plan['budget_fractions']:
                    rows.append(dict(domain=domain,seed=seed,policy=policy,budget_fraction=fraction,
                        id=f'{domain}/seed-{seed}/pilot/{policy}/budget-{fraction:g}'))
    return rows


def _validated(values):
    ids,arms=tuple(values['query_ids']),tuple(values['arms'])
    if (not ids or len(set(ids))!=len(ids) or not arms or len(set(arms))!=len(arms)
            or values['reference_arm'] not in arms or type(values['bin_size']) is not int or values['bin_size']<1
            or set(values['predictions'])!=set(ids) or set(values['estimates'])!=set(ids)
            or set(values['outcomes'])!={(q,a) for q in ids for a in arms}):
        raise ValueError('Invalid budget experiment identities or stream')
    for name in ('budget','lower_bound','upper_bound'):
        value=values[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Budget/bounds must be finite nonnegative numbers')
    if not 0<values['lower_bound']<=values['upper_bound']:raise ValueError('Invalid ratio bounds')
    for q in ids:
        if set(values['predictions'][q])!=set(arms) or set(values['estimates'][q])!=set(arms):
            raise ValueError('Incomplete prediction/cost grid')
        for a in arms:
            score,cost=values['predictions'][q][a],values['estimates'][q][a]
            reward,actual=values['outcomes'][q,a]
            if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                    for v in (score,cost,reward,actual)) or cost<0 or actual<0 or not 0<=reward<=1):
                raise ValueError('Invalid prediction, estimated cost or outcome')
    return ids,arms


def _retrospective(ids,arms,predictions,estimates,budget):
    scores=np.asarray([[predictions[q][a] for a in arms] for q in ids],dtype=float)
    costs=np.asarray([[estimates[q][a] for a in arms] for q in ids],dtype=float)
    def choose(lam):
        value=scores-lam*costs;chosen=np.argmax(value,axis=1)
        selected=[None if value[i,a]<=0 else arms[int(a)] for i,a in enumerate(chosen)]
        spent=sum((Fraction(str(estimates[q][a])) for q,a in zip(ids,selected) if a is not None),Fraction(0))
        return selected,spent
    selected,spent=choose(0.)
    if spent<=budget:return selected,0.
    positive=costs[costs>0]
    high=float(max(0.,scores.max())/positive.min()+1.) if len(positive) else 1.
    best,total=choose(high)
    if total>budget:raise ValueError('Cannot bracket a feasible retrospective allocation')
    low=0.
    for _ in range(80):
        middle=(low+high)/2;selected,spent=choose(middle)
        if spent<=budget:high=middle;best=selected
        else:low=middle
    return best,high


def simulate_allocation(*,policy,query_ids,arms,predictions,estimates,outcomes,reference_arm,
                        budget,bin_size,lower_bound,upper_bound):
    if policy not in POLICIES:raise ValueError('Unknown allocation policy')
    args=dict(query_ids=query_ids,arms=arms,predictions=predictions,estimates=estimates,
        outcomes=outcomes,reference_arm=reference_arm,budget=budget,bin_size=bin_size,
        lower_bound=lower_bound,upper_bound=upper_bound)
    ids,arms=_validated(args)
    if policy in ('guarded-stop','paper-literal-stop'):
        result,trace=simulate_budget_stream(**args,affordability_mode='guarded' if policy=='guarded-stop' else 'paper_literal')
        spent=result['accounting']['estimated_total']
        return dict(result,policy=policy,estimated_cost=spent,estimated_overspend=max(0.,spent-budget)),trace
    total=Fraction(str(budget));spent=Fraction(0);actual=Fraction(0);reward=0.;served=0
    bin_spent=Fraction(0);base=total/math.ceil(len(ids)/bin_size);trace=[]
    choices=lam=None
    if policy=='retrospective-lagrange':choices,lam=_retrospective(ids,arms,predictions,estimates,total)
    for t,q in enumerate(ids):
        costs={a:Fraction(str(estimates[q][a])) for a in arms};scores=predictions[q]
        if t%bin_size==0:bin_spent=Fraction(0)
        available=base*(t//bin_size+1)-spent
        remaining=min(bin_size-t%bin_size,len(ids)-t)
        if choices is not None:
            arm=choices[t];branch='retrospective-lagrange';eligible=[]
        elif policy=='fixed-per-query':
            eligible=[a for a in arms if costs[a]<=min(total/len(ids),total-spent)]
            arm=max(eligible,key=scores.get) if eligible else None;branch='fixed-per-query'
        else:
            utilization=float(bin_spent/base) if base else 0.
            logphi=math.log(lower_bound)-1+utilization*(math.log(upper_bound)-math.log(lower_bound)+1)
            eligible=[a for a in arms if costs[a]<=available and scores[a]>=0 and
                (costs[a]==0 or scores[a]>0 and math.log(scores[a])-math.log(float(costs[a]))>=logphi)]
            branch='threshold'
            if not eligible:
                eligible=[a for a in arms if costs[a]<=available/remaining];branch='fallback'
            arm=max(eligible,key=scores.get) if eligible else None
        if arm is not None:
            spent+=costs[arm];bin_spent+=costs[arm]
            observed,price=outcomes[q,arm];reward+=observed;actual+=Fraction(str(price));served+=1
        if spent>total:raise ValueError('Guarded allocation violated its estimated budget')
        trace.append(dict(query_id=q,arm_id=arm,status='selected' if arm is not None else 'skipped',
            branch=branch,bin_index=t//bin_size+1,available_in_bin_before=float(available),
            eligible_arms=eligible,reward=outcomes[q,arm][0] if arm is not None else 0.,
            estimated_cost=float(costs[arm]) if arm is not None else 0.,
            actual_cost=outcomes[q,arm][1] if arm is not None else 0.,cumulative_estimated_cost=float(spent)))
    relative=relative_metrics(total_reward=reward,actual_cost=float(actual),served=served,requested=len(ids),
        gpt4_total_reward=math.fsum(outcomes[q,reference_arm][0] for q in ids),
        gpt4_total_cost=math.fsum(outcomes[q,reference_arm][1] for q in ids))
    return dict(policy=policy,status='complete' if served==len(ids) else 'partial',requested=len(ids),served=served,
        total_reward=reward,actual_cost=float(actual),budget=budget,estimated_cost=float(spent),
        estimated_overspend=0.,actual_overspend=float(max(Fraction(0),actual-total)),
        globally_unused_estimated_budget=float(total-spent),retrospective_lambda=lam,
        future_query_information_used=choices is not None,relative=relative),trace


def freeze(base):
    from scripts.experiments import verify_pins, load_artifact
    from tools.openai_study import DEFAULT_BASE as CORE, checked_pin, once, sealed
    from tools.paper_preflight.common import ROOT, digest
    core=sealed(CORE/'protocol.json');verify_pins(core,ROOT)
    sources=[]
    for domain in core['primary_plan']['datasets']:
        for seed in core['primary_plan']['replicate_seeds']:
            folder=CORE/'runs/learning'/domain/f'seed-{seed}'/'main/pilot'
            value=load_artifact(folder)
            sources.extend([checked_pin(folder/'manifest.json'),checked_pin(folder/value['snapshots']['1']['file'])])
    value=dict(version=1,kind='frozen-openai-allocation-comparison',core=checked_pin(CORE/'protocol.json'),
        sources=sources,code=checked_pin(Path(__file__)),conditions=condition_inventory(core),
        policies=list(POLICIES),synthetic=False,paper_equivalent=False,
        selection='Frozen main PILOT scores and tuning-only costs; no rerouting/training changes',
        fixed_per_query='capB/Q without transfer; skip if no affordable arm',
        retrospective='P-lambda*C; full future scores/cost estimates,80-step feasible search, abstain0; not guaranteed optimum',
        skip='keep current-bin threshold/fallback; advance time on unserved queries; never borrow unreleased bin funds')
    value['manifest_sha256']=digest(value);once(base/'protocol.json',value)
    print('Frozen240 policy comparisons',flush=True)


def run(base,limit=None):
    from scripts.deployment_adapter import FrozenDeploymentRouter
    from scripts.experiments import load_artifact, verify_pins
    from scripts.resources import require_resources
    from tools.openai_study import DEFAULT_BASE as CORE, checked_pin, load_data, once, sealed
    from tools.paper_preflight.common import ROOT, digest, read_json, sha256, write_json
    protocol=sealed(base/'protocol.json');verify_pins(protocol,ROOT)
    completed=0
    for domain in ('routerbench','mmlu'):
        for seed in (7,42,2026):
            require_resources();bundle=load_data(CORE,domain,seed)
            folder=CORE/'runs/learning'/domain/f'seed-{seed}'/'main/pilot'
            learning=load_artifact(folder);snap=folder/learning['snapshots']['1']['file']
            router=FrozenDeploymentRouter.load_checkpoint(snap)
            ids=list(bundle['selections']['deployment'].query_ids);arms=list(router.arms)
            predictions={q:router.predict_rewards(bundle['contexts'].get(q)) for q in ids}
            estimates={q:bundle['estimator'].estimate(bundle['input_counts'][q]) for q in ids}
            truth=bundle['outcomes'].loc[bundle['outcomes'].query_id.isin(ids)].to_dict('records')
            outcomes={(r['query_id'],r['arm_id']):(float(r['reward']),float(r['actual_cost'])) for r in truth}
            core=read_json(CORE/'protocol.json');controls=core['tuning'][domain]
            source=dict(domain=domain,seed=seed,query_ids=ids,arms=arms,predictions=predictions,
                estimates=estimates,outcomes=truth,checkpoint=checked_pin(snap),
                parent_learning=checked_pin(folder/'manifest.json'),core=protocol['core'],
                estimator=bundle['estimator'].metadata(),cost_policy_bounds=controls['bounds'],synthetic=False)
            input_path=base/'inputs'/domain/f'seed-{seed}.json';once(input_path,source)
            for policy in POLICIES:
                require_resources()
                for condition in protocol['conditions']:
                    if (condition['domain'],condition['seed'],condition['policy'])!=(domain,seed,policy):continue
                    destination=base/'runs'/condition['id']
                    request=dict(condition=condition,input=checked_pin(input_path),
                        protocol_sha256=sha256(base/'protocol.json'),core=protocol['core'])
                    if destination.exists():load_artifact(destination,request=request)
                    else:
                        result,trace=simulate_allocation(policy=policy,query_ids=ids,arms=arms,predictions=predictions,
                            estimates=estimates,outcomes=outcomes,reference_arm='gpt-4-1106-preview',
                            budget=controls['absolute_reference_budget']*condition['budget_fraction'],
                            bin_size=core['primary_plan']['bin_size'],**controls['bounds'])
                        if policy=='guarded-stop':
                            reference=CORE/'runs/deployment'/domain/f'seed-{seed}'/'main/pilot/learn-1'/f"budget-{condition['budget_fraction']:g}"
                            original=load_artifact(reference);s=original['summary']
                            if (result['served']!=s['served_queries']
                                    or not math.isclose(result['total_reward'],s['metrics']['total_reward'],abs_tol=1e-9)
                                    or not math.isclose(result['actual_cost'],s['budget']['actual_total'],abs_tol=1e-9)):
                                raise ValueError('Guarded baseline does not match immutable main')
                            with (reference/'trace.jsonl').open(encoding='utf-8') as stream:
                                old_actions=[(r['query_id'],r['decision']['arm_id']) for r in map(json.loads,stream)]
                            if [(r['query_id'],r['arm_id']) for r in trace]!=old_actions:
                                raise ValueError('Guarded action trace differs from main')
                        destination.mkdir(parents=True,exist_ok=False)
                        once(destination/'summary.json',result)
                        with (destination/'trace.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
                            for row in trace:stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
                        manifest=dict(version=1,kind='frozen-score-allocation-evaluation',request=request,summary=result,
                            synthetic=False,paper_equivalent=False,
                            files={name:sha256(destination/name) for name in ('summary.json','trace.jsonl')})
                        manifest['manifest_sha256']=digest(manifest);once(destination/'manifest.json',manifest)
                    completed+=1
                    write_json(base/'progress.json',dict(completed=completed,expected=240,last_condition=condition['id'],
                        complete=completed==240,additional_paid_api_calls=0))
                    if limit and completed>=limit:return
                print(f'Completed budget{completed}/240 {domain} seed{seed} {policy}',flush=True)


def main():
    from scripts.resources import require_resources
    from tools.paper_preflight.common import ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--base',type=Path,default=ROOT/'outputs/openai-followup-20261031/budget')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();base=args.base.resolve()
    if not base.is_relative_to(ROOT.resolve()):raise ValueError('Budget output must stay in project')
    require_resources();base.mkdir(parents=True,exist_ok=True)
    if args.stage=='freeze':freeze(base)
    else:run(base,args.limit)


if __name__=='__main__':main()
