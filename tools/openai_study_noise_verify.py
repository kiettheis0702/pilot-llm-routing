"""Independent selected-feedback and budget replay for all noise conditions."""
if not __debug__:raise RuntimeError('Verification requires assertions, remove -O')

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.embedding_store import DiskEmbeddingCache
from scripts.resources import require_resources
from tools.paper_preflight.common import ROOT


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def envelope(p):
    v=read(p);assert digest(v['payload'])==v['sha256'];return v['payload']
def sealed(p):
    v=read(p);assert digest({k:r for k,r in v.items() if k!='manifest_sha256'})==v['manifest_sha256'];return v
def close(a,b):assert math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10),(a,b)
def unit(x):
    m=np.max(np.abs(x));return np.zeros_like(x) if m==0 else (x/m)/np.linalg.norm(x/m)
def lines(p):
    with Path(p).open(encoding='utf-8') as stream:
        for row in stream:yield json.loads(row)


def verify(limit=None):
    preflight=require_resources();base=ROOT/'outputs/openai-followup-20261031/noise'
    protocol=sealed(base/'protocol.json');core=sealed(ROOT/protocol['core']['path'])
    assert sha(ROOT/protocol['core']['path'])==protocol['core']['sha256']
    for pin in protocol['sources']:assert sha(ROOT/pin['path'])==pin['sha256']
    index=sealed(ROOT/'outputs/openai-core-20261031/prior-index.json')
    cache=DiskEmbeddingCache.load(ROOT/'outputs/openai-standard-20261030/imported/caches/source-0')
    data=ROOT/'data/processed/routerbench_0shot';split=pd.read_csv(data/'splits.csv')
    raw=pd.read_parquet(data/'outcomes.parquet',columns=['query_id','arm_id','reward','actual_cost'])
    truth={(r.query_id,r.arm_id):(float(r.reward),float(r.actual_cost)) for r in raw.itertuples(index=False)}
    ids={phase:split.loc[(split.split==phase)&split.task.str.startswith('mmlu-')].sort_values('stream_position').query_id.tolist()
         for phase in ('tuning','learning','deployment')}
    stage_pin=core['tuning']['mmlu']['stage']
    assert sha(ROOT/stage_pin['path'])==stage_pin['sha256']
    stage=sealed(ROOT/stage_pin['path']);estimator_pin=stage['estimator']
    assert sha(ROOT/estimator_pin['path'])==estimator_pin['sha256']
    estimator=read(ROOT/estimator_pin['path']);prices=estimator['prices']
    counts=ROOT/'outputs/research-cost-inputs-20261029'
    input_counts=read(counts/'input-counts.json');out_counts=read(counts/'tuning-output-counts.json')
    means={a:Fraction(sum(out_counts[q][a] for q in estimator['tuning_query_ids']),len(estimator['tuning_query_ids']))
        for a in core['arms']}
    estimates={q:{a:Fraction(str(float((input_counts[q]*Fraction(str(prices[a]['input_per_million']))
        +means[a]*Fraction(str(prices[a]['output_per_million'])))/1000000))) for a in core['arms']}
        for q in ids['deployment']}
    proofs=[];contexts={}
    for condition in protocol['conditions']:
        if limit and len(proofs)>=limit:break
        seed=condition['seed'];epsilon=condition['epsilon'];folder=base/'runs'/condition['id']
        artifact=sealed(folder/'manifest.json');assert artifact['request']['condition']==condition
        assert artifact['request']['protocol_sha256']==sha(base/'protocol.json')
        for name,h in artifact['files'].items():assert sha(folder/name)==h
        lineage=artifact['request']['lineage']
        for name,h in lineage['source_hashes'].items():assert sha(data/name)==h
        assert cache.fingerprint()==lineage['routing_cache_sha256']
        entry=next(r for r in index['fits'] if r['seed']==seed)
        assert sha(ROOT/entry['prior']['path'])==entry['prior']['sha256']
        prior=envelope(ROOT/entry['prior']['path']);arms=prior['arms'];dim=prior['dimension']
        assert arms==core['arms']
        if seed not in contexts:
            W=np.asarray(prior['W']);bias=np.asarray(prior['b'])
            contexts[seed]={q:unit(W@cache.get(q)+bias) for phase in ('learning','deployment') for q in ids[phase]}
        lambdas=np.array([prior['metrics']['validation']['per_arm'][a]['lambda'] for a in arms])
        A=lambdas[:,None,None]*np.eye(dim)[None,:,:];b=lambdas[:,None]*np.asarray(prior['theta'])
        reward=price=regret=0.;replaced=changed=0
        for t,(q,row) in enumerate(zip(ids['learning'],lines(folder/'learning-trace.jsonl'),strict=True),1):
            x=contexts[seed][q]
            theta=np.linalg.solve(A,b[...,None])[...,0]
            variance=np.linalg.solve(A,np.broadcast_to(x,b.shape)[...,None])[...,0]@x
            scores=np.clip(np.array([unit(v)@x for v in theta]),-1,1)+core['tuning']['mmlu']['selected_parameters']['pilot']['alpha']*np.sqrt(np.maximum(variance,0))
            chosen=int(np.argmax(scores));a=arms[chosen];true,cost=truth[q,a]
            h=hashlib.sha256(('pilot-feedback-noise-v1:'+str(seed)+':'+q).encode('utf-8')).digest()
            u=(int.from_bytes(h[:8],'big')>>11)/9007199254740992;bit=h[8]&1
            replace=u<epsilon;observed=float(bit) if replace else true
            assert row['round']==t and row['query_id']==q and row['arm_id']==a
            assert row['noise_replaced']==replace and row['noise_changed']==(observed!=true)
            close(row['reward'],true);close(row['feedback'],observed);close(row['actual_cost'],cost)
            gap=max(truth[q,aa][0] for aa in arms)-true
            reward+=true;price+=cost;regret+=gap;replaced+=int(replace);changed+=int(observed!=true)
            for key,value in [('oracle_reward',true+gap),('empirical_oracle_regret',gap),('cumulative_reward',reward),
                    ('cumulative_actual_cost',price),('cumulative_empirical_oracle_regret',regret)]:close(row[key],value)
            A[chosen]+=np.outer(x,x);b[chosen]+=observed*x
        summary=read(folder/'learning-summary.json')
        assert summary['rounds']==len(ids['learning']) and summary['noise_replaced']==replaced and summary['noise_changed']==changed
        close(summary['total_reward'],reward);close(summary['total_actual_cost'],price)
        close(summary['cumulative_empirical_oracle_regret'],regret)
        state=envelope(folder/'checkpoint.json')['state']
        np.testing.assert_allclose(A,state['A'],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(b,state['b'],atol=1e-10,rtol=1e-10)
        assert state['rounds']==len(ids['learning'])
        theta=np.linalg.solve(A,b[...,None])[...,0];directions=np.array([unit(v) for v in theta])
        trace=list(lines(folder/'deployment-trace.jsonl'));result=read(folder/'deployment-summary.json')
        assert result==artifact['summary'];B=Fraction(str(protocol['deployment_budget']))
        S=core['primary_plan']['bin_size'];basebin=B/math.ceil(len(ids['deployment'])/S)
        spent=bin_spent=actual=Fraction(0);reward=0.;served=0
        bounds=core['tuning']['mmlu']['bounds']
        for t,q in enumerate(ids['deployment']):
            if t%S==0:bin_spent=Fraction(0)
            available=basebin*(t//S+1)-spent;utilization=float(bin_spent/basebin)
            logphi=math.log(bounds['lower_bound'])-1+utilization*(math.log(bounds['upper_bound'])-math.log(bounds['lower_bound'])+1)
            score=dict(zip(arms,np.clip(directions@contexts[seed][q],-1,1)));costs=estimates[q]
            eligible=[a for a in arms if costs[a]<=available and score[a]>=0 and
                (costs[a]==0 or score[a]>0 and math.log(score[a])-math.log(float(costs[a]))>=logphi)]
            if not eligible:
                remaining=min(S-t%S,len(ids['deployment'])-t)
                eligible=[a for a in arms if costs[a]<=available/remaining]
            chosen=max(eligible,key=score.get) if eligible else None
            if chosen is None:
                assert len(trace)==t and result['terminal']['query_id']==q;break
            row=trace[t];assert row['query_id']==q and row['arm_id']==chosen
            observed,cost=truth[q,chosen];close(row['reward'],observed);close(row['actual_cost'],cost)
            spent+=costs[chosen];bin_spent+=costs[chosen];actual+=Fraction(str(cost));reward+=observed;served+=1
        assert spent<=B and served==len(trace)==result['served'] and result['requested']==len(ids['deployment'])
        close(result['total_reward'],reward);close(result['actual_cost'],float(actual));close(result['accounting']['estimated_total'],float(spent))
        close(result['relative']['service_utility'],reward/len(ids['deployment']))
        close(result['relative']['coverage'],served/len(ids['deployment']))
        close(result['relative']['quality_fraction_gpt4'],reward/sum(truth[q,'gpt-4-1106-preview'][0] for q in ids['deployment']))
        close(result['relative']['cost_fraction_gpt4'],float(actual)/sum(truth[q,'gpt-4-1106-preview'][1] for q in ids['deployment']))
        proofs.append(dict(id=condition['id'],manifest_sha256=artifact['manifest_sha256'],noise_replaced=replaced,
            noise_changed=changed,served=served,average_reward=reward/len(ids['deployment'])))
        print('Independent noise replay',len(proofs),condition['id'],flush=True)
    result=dict(verified=True,complete=len(proofs)==12,conditions=len(proofs),inventory=proofs,
        protocol_sha256=sha(base/'protocol.json'),verifier_sha256=sha(__file__),preflight=preflight,inference_api_calls=0)
    output=base/('independent-verification.json' if len(proofs)==12 else f'independent-smoke-{len(proofs)}.json')
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args();verify(args.limit)
