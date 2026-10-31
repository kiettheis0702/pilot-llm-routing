"""Paper D.5 uniform feedback-replacement experiment on cached OpenAI MMLU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.openai_study_robustness import apply_noise

LEVELS=(0.,.01,.03,.05)


def draw_for_query(seed,query_id):
    if type(seed) is not int or seed<0 or not isinstance(query_id,str) or not query_id:
        raise ValueError('Invalid noise seed or query identity')
    data=hashlib.sha256(f'pilot-feedback-noise-v1:{seed}:{query_id}'.encode('utf-8')).digest()
    return (int.from_bytes(data[:8],'big')>>11)/2**53,data[8]&1


def learn_noise(model,query_ids,contexts,feedback,oracle,*,epsilon,draws):
    if len(set(query_ids))!=len(query_ids) or set(draws)!=set(query_ids) or set(oracle)!=set(query_ids):
        raise ValueError('Invalid learning identities/draws')
    trace=[];reward=cost=regret=0.;replaced=changed=0
    for t,q in enumerate(query_ids,1):
        x=contexts.get(q);arm=model.select(x);true,price=feedback(q,arm)
        noisy=apply_noise(true,epsilon,*draws[q]);model.observe(arm,x,noisy['feedback'])
        reward+=true;cost+=price;gap=float(oracle[q])-true;regret+=gap
        replaced+=int(noisy['replaced']);changed+=int(noisy['changed'])
        trace.append(dict(round=t,query_id=q,arm_id=arm,reward=true,feedback=noisy['feedback'],
            actual_cost=price,oracle_reward=float(oracle[q]),empirical_oracle_regret=gap,
            noise_replaced=noisy['replaced'],noise_changed=noisy['changed'],
            cumulative_reward=reward,cumulative_actual_cost=cost,cumulative_empirical_oracle_regret=regret))
    return dict(rounds=len(query_ids),total_reward=reward,total_actual_cost=cost,
        cumulative_empirical_oracle_regret=regret,noise_replaced=replaced,noise_changed=changed),trace


def freeze(base):
    from scripts.experiments import verify_pins
    from tools.openai_study import DEFAULT_BASE as CORE, checked_pin, once, sealed
    from tools.paper_preflight.common import ROOT,digest
    core=sealed(CORE/'protocol.json');verify_pins(core,ROOT)
    conditions=[dict(seed=seed,epsilon=epsilon,id=f'seed-{seed}/noise-{epsilon:g}')
        for seed in (7,42,2026) for epsilon in LEVELS]
    protocol=dict(version=1,kind='openai-mmlu-uniform-feedback-noise',core=checked_pin(CORE/'protocol.json'),
        sources=[checked_pin(Path(__file__)),checked_pin(ROOT/'tools/openai_study_robustness.py')],
        conditions=conditions,domain='mmlu',method='pilot',deployment_budget=1.,
        budget_interpretation='1 historical token-price accounting dollar, no API spending; not normalized GPT4 fraction',
        selection='Use unchanged main prior/alpha/tuning costs/bounds/order; corrupt selected learning feedback only',
        noise_rng='SHA256 domain-separated seed/query, top53 bits uniform and next byte lowbit; same draws across epsilon',
        evaluation='Uncorrupted historical outcomes, true empirical oracle regret; all requested queries in denominator',
        equivalence='local D.5 adaptation:8 verified arms, unresolved author release/split/order; not exact paper Table5',
        synthetic=False,paper_equivalent=False)
    protocol['manifest_sha256']=digest(protocol);once(base/'protocol.json',protocol)


def run(base,limit=None):
    from scripts.deployment_adapter import FrozenDeploymentRouter,policy_state
    from scripts.experiments import load_artifact,verify_pins
    from scripts.resources import require_resources
    from scripts.tuning import build_policy
    from tools.openai_study import DEFAULT_BASE as CORE,checked_pin,load_data,once,sealed
    from tools.paper_preflight.common import ROOT,digest,sha256
    from tools.paper_preflight.compare import simulate_budget_stream
    protocol=sealed(base/'protocol.json');verify_pins(protocol,ROOT);core=sealed(CORE/'protocol.json')
    completed=0
    for seed in (7,42,2026):
        require_resources();bundle=load_data(CORE,'mmlu',seed);arms=bundle['prior']['arms']
        ids=list(bundle['selections']['learning'].query_ids);deployment=list(bundle['selections']['deployment'].query_ids)
        deployment_members=set(deployment)
        outcome_records=bundle['outcomes'].to_dict('records')
        truth={(r['query_id'],r['arm_id']):(float(r['reward']),float(r['actual_cost'])) for r in outcome_records}
        if any(truth[q,a][0] not in (0.,1.) for q in ids+deployment for a in arms):
            raise ValueError('D.5 noise requires truly binary MMLU rewards')
        oracle={q:max(truth[q,a][0] for a in arms) for q in ids}
        draws={q:draw_for_query(seed,q) for q in ids};controls=core['tuning']['mmlu']
        estimates={q:bundle['estimator'].estimate(bundle['input_counts'][q]) for q in deployment}
        for condition in [r for r in protocol['conditions'] if r['seed']==seed]:
            destination=base/'runs'/condition['id']
            request=dict(condition=condition,protocol_sha256=sha256(base/'protocol.json'),lineage=bundle['lineage'])
            if destination.exists():load_artifact(destination,request=request)
            else:
                model=build_policy(bundle['prior'],'pilot',controls['selected_parameters']['pilot']);model.reset(seed)
                result,trace=learn_noise(model,ids,bundle['contexts'],lambda q,a:truth[q,a],oracle,
                    epsilon=condition['epsilon'],draws=draws)
                if condition['epsilon']==0.:
                    parent=CORE/'runs/learning/mmlu'/f'seed-{seed}'/'main/pilot'
                    original=load_artifact(parent)
                    source=FrozenDeploymentRouter.load_checkpoint(parent/original['snapshots']['1']['file'])
                    if digest(policy_state(model))!=digest(source.source_state()):
                        raise ValueError('Zero-noise learned state differs from immutable main')
                    with (parent/'trace.jsonl').open(encoding='utf-8') as stream:
                        for old,new in zip(map(json.loads,stream),trace,strict=True):
                            if any(old[k]!=new[k] for k in ('round','query_id','arm_id','reward','actual_cost')):
                                raise ValueError('Zero-noise trace differs from immutable main')
                router=FrozenDeploymentRouter(model,seed=seed)
                predictions={q:router.predict_rewards(bundle['contexts'].get(q)) for q in deployment}
                evaluation,deploy_trace=simulate_budget_stream(query_ids=deployment,arms=arms,predictions=predictions,
                    estimates=estimates,outcomes={k:v for k,v in truth.items() if k[0] in deployment_members},
                    reference_arm='gpt-4-1106-preview',budget=protocol['deployment_budget'],
                    bin_size=core['primary_plan']['bin_size'],**controls['bounds'],affordability_mode='guarded')
                destination.mkdir(parents=True,exist_ok=False)
                router.save_checkpoint(destination/'checkpoint.json')
                once(destination/'learning-summary.json',result);once(destination/'deployment-summary.json',evaluation)
                for name,rows in [('learning-trace.jsonl',trace),('deployment-trace.jsonl',deploy_trace)]:
                    with (destination/name).open('x',encoding='utf-8') as stream:
                        for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                names=['checkpoint.json','learning-summary.json','deployment-summary.json','learning-trace.jsonl','deployment-trace.jsonl']
                artifact=dict(version=1,kind='noise-learning-and-deployment',request=request,
                    learning=result,summary=evaluation,synthetic=False,paper_equivalent=False,
                    files={name:sha256(destination/name) for name in names})
                artifact['manifest_sha256']=digest(artifact);once(destination/'manifest.json',artifact)
            completed+=1
            print(f'Completed noise{completed}/12 {condition["id"]}',flush=True)
            if limit and completed>=limit:return


def main():
    from scripts.resources import require_resources
    from tools.paper_preflight.common import ROOT
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--limit',type=int);args=parser.parse_args();require_resources()
    base=ROOT/'outputs/openai-followup-20261031/noise'
    if args.action=='freeze':base.mkdir(parents=True,exist_ok=False);freeze(base)
    else:run(base,args.limit)


if __name__=='__main__':main()
