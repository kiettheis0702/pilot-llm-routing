"""Independent score/cost/allocation replay; never calls a routing policy."""
if not __debug__:
    raise RuntimeError('Independent verification requires assertions; remove -O')

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
def filehash(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def envelope(p):
    value=read(p);assert digest(value['payload'])==value['sha256'];return value['payload']
def manifest(folder):
    value=read(folder/'manifest.json');assert value['manifest_sha256']==digest({k:v for k,v in value.items() if k!='manifest_sha256'})
    for name,want in value['files'].items():
        path=(folder/name).resolve();assert path.is_relative_to(folder.resolve()) and filehash(path)==want
    return value
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11),(a,b)
def unit(x):
    scale=np.max(np.abs(x));return (x/scale)/np.linalg.norm(x/scale)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'outputs/openai-followup-20261031/budget')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();base=args.base.resolve();preflight=require_resources()
    protocol=read(base/'protocol.json')
    assert protocol['manifest_sha256']==digest({k:v for k,v in protocol.items() if k!='manifest_sha256'})
    assert filehash(ROOT/protocol['code']['path'])==protocol['code']['sha256']
    assert filehash(ROOT/protocol['core']['path'])==protocol['core']['sha256']
    core=read(ROOT/protocol['core']['path'])
    assert core['manifest_sha256']==digest({k:v for k,v in core.items() if k!='manifest_sha256'})
    index=read(ROOT/'outputs/openai-core-20261031/prior-index.json')
    cache=DiskEmbeddingCache.load(ROOT/'outputs/openai-standard-20261030/imported/caches/source-0')
    costs=ROOT/'outputs/research-cost-inputs-20261029'
    inputs=read(costs/'input-counts.json');output_counts=read(costs/'tuning-output-counts.json')
    data=ROOT/'data/processed/routerbench_0shot'
    hashes=read(ROOT/'reports/data-gate.json')['prepared']['file_sha256']
    assert filehash(data/'outcomes.parquet')==hashes['outcomes.parquet']
    outcomes=pd.read_parquet(data/'outcomes.parquet',columns=['query_id','arm_id','reward','actual_cost'])
    truth={(r.query_id,r.arm_id):(float(r.reward),float(r.actual_cost)) for r in outcomes.itertuples(index=False)}
    verified_inputs={};proofs=[]
    for condition in protocol['conditions']:
        if args.limit and len(proofs)>=args.limit:break
        d,s,policy=condition['domain'],condition['seed'],condition['policy']
        source_path=base/'inputs'/d/f'seed-{s}.json';source=read(source_path)
        ids,arms=source['query_ids'],source['arms']
        if (d,s) not in verified_inputs:
            assert filehash(ROOT/source['checkpoint']['path'])==source['checkpoint']['sha256']
            state=envelope(ROOT/source['checkpoint']['path'])['state']
            theta=np.linalg.solve(np.array(state['A']),np.array(state['b'])[...,None])[...,0]
            directions=[unit(v) for v in theta]
            entry=next(e for e in index['fits'] if e['seed']==s and e['dimension']==64)
            assert filehash(ROOT/entry['prior']['path'])==entry['prior']['sha256']
            prior=envelope(ROOT/entry['prior']['path']);W=np.asarray(prior['W']);bias=np.asarray(prior['b'])
            metadata=source['estimator'];tune=metadata['tuning_query_ids'];prices=metadata['prices']
            means={a:Fraction(sum(output_counts[q][a] for q in tune),len(tune)) for a in arms}
            for q in ids:
                x=unit(W@cache.get(q)+bias)
                for a,theta_a in zip(arms,directions):
                    close(source['predictions'][q][a],float(np.clip(theta_a@x,-1,1)))
                    expected=float((inputs[q]*Fraction(str(prices[a]['input_per_million']))+
                        means[a]*Fraction(str(prices[a]['output_per_million'])))/1000000)
                    close(source['estimates'][q][a],expected)
            for row in source['outcomes']:
                expected=truth[row['query_id'],row['arm_id']]
                close(row['reward'],expected[0]);close(row['actual_cost'],expected[1])
            verified_inputs[d,s]=filehash(source_path)
        folder=base/'runs'/condition['id'];value=manifest(folder);result=value['summary']
        assert value['request']['condition']==condition
        assert value['request']['input']['sha256']==verified_inputs[d,s]
        assert value['request']['protocol_sha256']==filehash(base/'protocol.json')
        assert read(folder/'summary.json')==result
        with (folder/'trace.jsonl').open(encoding='utf-8') as stream:trace=[json.loads(line) for line in stream]
        amount=core['tuning'][d]['absolute_reference_budget']*condition['budget_fraction']
        close(result['budget'],amount)
        S=core['primary_plan']['bin_size'];B=Fraction(str(amount));per_bin=B/math.ceil(len(ids)/S)
        spent=Fraction(0);bin_spent=Fraction(0);actual=Fraction(0);reward=0.;served=0;processed=0
        bounds=core['tuning'][d]['bounds'];lam=result.get('retrospective_lambda')
        for t,q in enumerate(ids):
            if t%S==0:bin_spent=Fraction(0)
            cost={a:Fraction(str(source['estimates'][q][a])) for a in arms}
            score=source['predictions'][q];available=per_bin*(t//S+1)-spent
            if policy=='retrospective-lagrange':
                assert lam is not None and lam>=0
                utility={a:score[a]-lam*float(cost[a]) for a in arms}
                best=max(arms,key=utility.get);chosen=best if utility[best]>0 else None
            elif policy=='fixed-per-query':
                eligible=[a for a in arms if cost[a]<=min(B/len(ids),B-spent)]
                chosen=max(eligible,key=score.get) if eligible else None
            else:
                utilization=float(bin_spent/per_bin) if per_bin else 0.
                logphi=math.log(bounds['lower_bound'])-1+utilization*(math.log(bounds['upper_bound'])-math.log(bounds['lower_bound'])+1)
                eligible=[a for a in arms if (policy=='paper-literal-stop' or cost[a]<=available)
                    and score[a]>=0 and (cost[a]==0 or score[a]>0 and math.log(score[a])-math.log(float(cost[a]))>=logphi)]
                if not eligible:
                    remaining=min(S-t%S,len(ids)-t)
                    eligible=[a for a in arms if cost[a]<=available/remaining]
                chosen=max(eligible,key=score.get) if eligible else None
            if chosen is None and policy in ('guarded-stop','paper-literal-stop'):
                assert len(trace)==t and result['terminal']['query_id']==q;break
            row=trace[t];assert row['query_id']==q and row['arm_id']==chosen
            processed+=1
            if chosen is not None:
                spent+=cost[chosen];bin_spent+=cost[chosen]
                observed,price=truth[q,chosen];reward+=observed;actual+=Fraction(str(price));served+=1
                close(row['reward'],observed);close(row['actual_cost'],price)
            else:close(row['reward'],0.);close(row['actual_cost'],0.)
        assert processed==len(trace) and served==result['served'] and result['requested']==len(ids)
        if policy!='paper-literal-stop':assert spent<=B
        close(result['total_reward'],reward);close(result['estimated_cost'],float(spent));close(result['actual_cost'],float(actual))
        close(result['estimated_overspend'],float(max(Fraction(0),spent-B)))
        relative=result['relative'];ref_reward=math.fsum(truth[q,'gpt-4-1106-preview'][0] for q in ids)
        ref_cost=math.fsum(truth[q,'gpt-4-1106-preview'][1] for q in ids)
        close(relative['quality_fraction_gpt4'],reward/ref_reward)
        close(relative['cost_fraction_gpt4'],float(actual)/ref_cost);close(relative['coverage'],served/len(ids))
        proofs.append(dict(id=condition['id'],manifest_sha256=value['manifest_sha256'],served=served))
        if len(proofs)%40==0:print(f'Independent allocation replay {len(proofs)}/240',flush=True)
    report=dict(verified=True,complete=len(proofs)==240,conditions=len(proofs),inventory=proofs,
        input_sha256={f'{d}/{s}':v for (d,s),v in verified_inputs.items()},protocol_sha256=filehash(base/'protocol.json'),
        script_sha256=filehash(__file__),preflight=preflight,additional_paid_api_calls=0)
    output=base/('independent-verification.json' if len(proofs)==240 else f'independent-smoke-{len(proofs)}.json')
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print('Independent allocation verification',len(proofs),'complete',report['complete'])


if __name__=='__main__':main()
