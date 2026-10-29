"""Independent full trace replay: reads files, never imports routing math."""
if not __debug__:
    raise RuntimeError("Evidence verification requires Python assertions; remove -O/-OO and PYTHONOPTIMIZE")

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--domain',default='routerbench')
parser.add_argument('--seed',type=int,default=7)
parser.add_argument('--method',default='pilot')
parser.add_argument('--variant',default='main')
parser.add_argument('--output',required=True)
args=parser.parse_args()
destination=ROOT/args.output
if destination.exists():raise FileExistsError(destination)
from scripts.resources import require_resources
preflight=require_resources()
import numpy as np
import pandas as pd

BASE=ROOT/'outputs/research-experiments-20261029'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def envelope(p):
    v=read(p);assert set(v)=={'payload','sha256'} and digest(v['payload'])==v['sha256'];return v['payload']
def manifest(p):
    v=read(p);assert digest({k:w for k,w in v.items() if k!='manifest_sha256'})==v['manifest_sha256']
    for n,s in v['files'].items():assert filehash(Path(p).parent/n)==s
    return v
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11),(a,b)
def unit(x):
    scale=np.abs(x).max();return (x/scale)/np.linalg.norm(x/scale) if scale else np.zeros_like(x)
def rows(p):
    with Path(p).open(encoding='utf-8') as f:
        for s in f:yield json.loads(s)

protocol=read(ROOT/'configs/frozen_experiment_protocol.json');plan=protocol['primary_plan'];arms=protocol['arms'];K=len(arms)
assert digest({k:v for k,v in protocol.items() if k!='manifest_sha256'})==protocol['manifest_sha256']
index=read(ROOT/'configs/research_prior_index.json');dimension=8 if args.variant=='projection-dimension-8' else 64
entry=next(v for v in index['fits'] if (v['dimension'],v['seed'])==(dimension,args.seed))
assert filehash(ROOT/entry['prior']['path'])==entry['prior']['sha256']
prior=envelope(ROOT/entry['prior']['path']);W=np.array(prior['W']);bias=np.array(prior['b'])
cache_dir=ROOT/'outputs/research-embeddings-20261029/routing-full';cache=envelope(cache_dir/'manifest.json')
assert filehash(cache_dir/'vectors.npy')==cache['vector_sha256']
vectors=np.load(cache_dir/'vectors.npy',mmap_mode='r');positions={q:i for i,q in enumerate(cache['query_ids'])}
data=ROOT/'data/processed/routerbench_0shot';queries=pd.read_parquet(data/'queries.parquet')
outcomes=pd.read_parquet(data/'outcomes.parquet',columns=['query_id','arm_id','reward','actual_cost'])
splits=pd.read_csv(data/'splits.csv');ids={}
for phase in ('learning','deployment','tuning'):
    selected=splits.loc[splits.split==phase].sort_values('stream_position')
    if args.domain=='mmlu':selected=selected.loc[selected.task.str.startswith('mmlu-')]
    ids[phase]=selected.query_id.tolist();assert len(ids[phase])==plan['frozen_counts'][args.domain][phase]
contexts={q:W@vectors[positions[q]]+bias for q in ids['learning']+ids['deployment']}
truth=outcomes.loc[outcomes.arm_id.isin(arms)].set_index(['query_id','arm_id'])
oracle=outcomes.loc[outcomes.arm_id.isin(arms)].groupby('query_id').reward.max().to_dict()
cost_dir=ROOT/'outputs/research-cost-inputs-20261029';input_counts=read(cost_dir/'input-counts.json')
output_counts=read(cost_dir/'tuning-output-counts.json');prices=read(cost_dir/'estimator.json')['prices']
means={a:Fraction(sum(output_counts[q][a] for q in ids['tuning']),len(ids['tuning'])) for a in arms}
def estimated(q):
    # The estimator emits a float; the cost policy reserves its decimal value.
    return {a:Fraction(str(float((input_counts[q]*Fraction(str(prices[a]['input_per_million']))+
         means[a]*Fraction(str(prices[a]['output_per_million'])))/1000000))) for a in arms}
variant='main' if args.variant=='single-budget-bin' else args.variant
learning_dir=BASE/'learning'/args.domain/f'seed-{args.seed}'/variant/args.method
learn=manifest(learning_dir/'manifest.json');request=learn['request'];initial=request['initial_state']
assert request['selection']['query_ids']==ids['learning'] and request['seed']==args.seed
assert request['lineage']['prior_file_sha256']==entry['prior']['sha256']
for p,s in request['lineage']['code_sha256'].items():assert filehash(ROOT/p)==s
kind=initial['policy_type'];parameters=initial['parameters'];rng=np.random.default_rng(args.seed)
assert initial['rounds']==0 and initial['arms']==arms and initial['dimension']==dimension
assert initial['rng_state']==rng.bit_generator.state
if kind=='PILOT':
    lambdas=np.array([prior['metrics']['validation']['per_arm'][a]['lambda'] for a in arms])
    theta=np.array(prior['theta'])
    if variant=='zero-prior-direction':theta[:]=0
    if variant=='unit-initialization-strength':lambdas[:]=1
    assert np.array_equal(parameters['priors'],theta) and np.array_equal(parameters['lambdas'],lambdas)
    assert parameters['alpha']==protocol['tuning'][args.domain]['selected_parameters']['pilot']['alpha']
    A=lambdas[:,None,None]*np.eye(dimension)[None,:,:];b=lambdas[:,None]*theta
elif kind in ('LinearUCB','ExploreOnly','EpochGreedy'):
    A=np.repeat(np.eye(dimension)[None,:,:],K,axis=0);b=np.zeros((K,dimension))
else:A=b=None
if A is not None:
    assert np.array_equal(initial['A'],A) and np.array_equal(initial['b'],b)
snapshots={};reward=cost=regret=0.;count=0;actions=dict.fromkeys(arms,0)
def check_snapshot(t):
    for key,s in learn['snapshots'].items():
        if s['rounds']!=t:continue
        state=envelope(learning_dir/s['file'])['state']
        assert state['rounds']==t and state['rng_state']==rng.bit_generator.state
        assert state['parameters']==parameters and digest(state)==s['source_state_sha256']
        if A is not None:
            assert np.allclose(state['A'],A,rtol=1e-12,atol=1e-12) and np.allclose(state['b'],b,rtol=1e-12,atol=1e-12)
        metrics=s['metrics'];assert metrics['rounds']==t and metrics['action_counts']==actions
        close(metrics['total_reward'],reward);close(metrics['total_actual_cost'],cost)
        close(metrics['cumulative_empirical_oracle_regret'],regret)
        snapshots[key]=state
if variant=='prior-only':
    assert learn['training_mode']=='none' and learn['learning_selection_role']=='unused_disjoint_reference'
    assert request['outcomes_sha256'] is None
    check_snapshot(0)
else:
    assert learn['training_mode']=='full'
    for t,row in enumerate(rows(learning_dir/'trace.jsonl')):
        q=ids['learning'][t];assert row['query_id']==q and row['round']==t+1
        x=unit(contexts[q]) if kind=='PILOT' else contexts[q]
        if A is not None:
            theta=np.linalg.solve(A,b[...,None])[...,0]
            scores=np.clip(np.array([unit(v)@x for v in theta]),-1,1) if kind=='PILOT' else theta@x
            uncertainty=np.sqrt(np.maximum(np.linalg.solve(A,np.broadcast_to(x,b.shape)[...,None])[...,0]@x,0))
            scores=uncertainty if kind=='ExploreOnly' else scores+parameters['alpha']*uncertainty
            chosen=int(rng.integers(K)) if kind=='EpochGreedy' and t%parameters['window']==0 else int(np.argmax(scores))
        elif kind=='RandomPolicy':chosen=int(rng.integers(K))
        else:chosen=arms.index(parameters['arm_id'])
        arm=arms[chosen];assert row['arm_id']==arm
        actual=truth.loc[(q,arm)];close(row['reward'],float(actual.reward));close(row['actual_cost'],float(actual.actual_cost))
        if A is not None:A[chosen]+=np.outer(x,x);b[chosen]+=float(actual.reward)*x
        reward+=float(actual.reward);cost+=float(actual.actual_cost);gap=float(oracle[q])-float(actual.reward);regret+=gap;actions[arm]+=1;count+=1
        close(row['oracle_reward'],float(oracle[q]));close(row['empirical_oracle_regret'],gap)
        close(row['cumulative_reward'],reward);close(row['cumulative_actual_cost'],cost);close(row['cumulative_empirical_oracle_regret'],regret)
        check_snapshot(t+1)
    assert count==len(ids['learning'])
assert set(snapshots)==set(learn['snapshots'])
assert learn['metrics']['rounds']==count
fractions=plan['learning_fractions'] if args.variant=='main' else [1.]
inventory=[];tuning=protocol['tuning'][args.domain]
for fraction in fractions:
    key=format(fraction,'.12g');state=snapshots[key]
    theta=np.linalg.solve(np.array(state['A']),np.array(state['b'])[...,None])[...,0] if state['A'] is not None else None
    budgets=plan['budget_fractions'] if fraction==1 else [plan['learning_size_budget_fraction']]
    for budget_fraction in budgets:
        folder=BASE/'deployment'/args.domain/f'seed-{args.seed}'/args.variant/args.method/('learn-'+key)/('budget-'+format(budget_fraction,'.12g'))
        result=manifest(folder/'manifest.json');req=result['request'];summary=result['summary'];trace=list(rows(folder/'trace.jsonl'))
        assert read(folder/'summary.json')==summary and not result['synthetic']
        assert summary['model']['sha256']==digest(state)
        assert req['lineage']['learning_checkpoint_sha256']==filehash(learning_dir/learn['snapshots'][key]['file'])
        assert req['lineage']['learning_manifest_sha256']==filehash(learning_dir/'manifest.json')
        assert req['lineage']['learning_rounds']==state['rounds']
        assert summary['deployment_split']['query_ids']==ids['deployment']
        assert summary['learning_split']['query_ids']==ids['learning'][:math.ceil(fraction*len(ids['learning']))]
        amount=tuning['absolute_reference_budget']*budget_fraction;assert req['budget']==amount
        bin_size=len(ids['deployment']) if args.variant=='single-budget-bin' else plan['bin_size']
        assert req['bin_size']==bin_size
        assert req['estimator']['tuning_query_ids']==ids['tuning']
        for a in arms:close(req['estimator']['mean_output_tokens'][a],float(means[a]))
        bins=math.ceil(len(ids['deployment'])/bin_size);base=Fraction(str(amount))/bins
        spent=bin_spent=actual_spent=Fraction(0);reward=regret=0.;served=0
        for t,q in enumerate(ids['deployment']):
            x=contexts[q]
            if kind=='PILOT':scores=np.clip(np.array([unit(v)@unit(x) for v in theta]),-1,1)
            elif theta is not None:scores=theta@x
            elif kind=='RandomPolicy':scores=np.full(K,1/K)
            else:scores=np.array([float(a==parameters['arm_id']) for a in arms])
            prediction=dict(zip(arms,scores));costs=estimated(q);bin_index=t//bin_size+1
            if t%bin_size==0:bin_spent=Fraction(0)
            available=base*bin_index-spent;remaining=min(bin_size-t%bin_size,len(ids['deployment'])-t)
            utilization=float(bin_spent/base) if base else 0.
            lb=tuning['bounds']['lower_bound'];ub=tuning['bounds']['upper_bound']
            log_phi=math.log(lb)-1+utilization*(math.log(ub)-math.log(lb)+1)
            allowed=[parameters['arm_id']] if kind=='FixedPolicy' else arms
            eligible=[a for a in arms if a in allowed and costs[a]<=available and prediction[a]>=0 and
                (costs[a]==0 or prediction[a]>0 and math.log(prediction[a])-math.log(float(costs[a]))>=log_phi)]
            branch='threshold'
            if not eligible:eligible=[a for a in arms if a in allowed and costs[a]<=available/remaining];branch='fallback'
            order=arms
            if kind=='RandomPolicy':
                r=np.random.default_rng(int(digest({'seed':args.seed,'query_id':q,'arms':arms}),16))
                order=[arms[int(i)] for i in r.permutation(K)]
            chosen=max((a for a in order if a in eligible),key=prediction.get) if eligible else None
            if chosen is None:
                assert len(trace)==t and summary['terminal_decision']['query_id']==q
                assert summary['terminal_decision']['decision']['status']=='insufficient_budget'
                break
            row=trace[t];decision=row['decision'];assert row['query_id']==q and decision['arm_id']==chosen
            assert decision['branch']==branch and decision['eligible_arms']==eligible
            close(decision['predicted_reward'],float(prediction[chosen]));close(decision['estimated_cost'],float(costs[chosen]))
            close(decision['available_before'],float(available));close(decision['utilization_before'],utilization)
            spent+=costs[chosen];bin_spent+=costs[chosen];actual=truth.loc[(q,chosen)]
            close(row['reward'],float(actual.reward));close(row['actual_cost'],float(actual.actual_cost))
            reward+=float(actual.reward);gap=float(oracle[q])-float(actual.reward);regret+=gap;actual_spent+=Fraction(str(float(actual.actual_cost)));served+=1
            close(row['cumulative_reward'],reward);close(row['empirical_oracle_regret'],gap)
            close(row['cumulative_empirical_oracle_regret'],regret)
            close(decision['cumulative_estimated_cost'],float(spent));close(decision['available_after'],float(available-costs[chosen]))
        assert served==len(trace)==summary['served_queries'] and summary['unserved_queries']==len(ids['deployment'])-served
        close(summary['budget']['estimated_total'],float(spent));close(summary['budget']['actual_total'],float(actual_spent))
        assert spent<=Fraction(str(amount))
        close(summary['budget']['actual_budget_excess_observed'],float(max(Fraction(0),actual_spent-Fraction(str(amount)))))
        close(summary['metrics']['total_reward'],reward);close(summary['metrics']['reward_per_requested_query'],reward/len(ids['deployment']))
        close(summary['metrics']['served_fraction'],served/len(ids['deployment']))
        if served:close(summary['metrics']['mean_reward_served'],reward/served)
        else:assert summary['metrics']['mean_reward_served'] is None
        inventory.append(dict(id=req['lineage']['condition']['id'],manifest_sha256=result['manifest_sha256'],
            served=served,reward=reward,actual_cost=float(actual_spent),estimated_cost=float(spent)))
report=dict(verified=True,kind='independent-full-condition-replay',domain=args.domain,seed=args.seed,
    method=args.method,variant=args.variant,learning_rounds=count,verified_deployment_runs=len(inventory),
    verified_served_deployment_rounds=sum(v['served'] for v in inventory),inventory=inventory,
    learning_manifest_sha256=learn['manifest_sha256'],script_sha256=filehash(__file__),preflight=preflight,
    synthetic=False,paper_equivalent=False)
destination.parent.mkdir(parents=True,exist_ok=True)
with destination.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({k:v for k,v in report.items() if k!='inventory'}))
