"""Full independent replay, reusing the pinned mathematical reference verifier.

Adapt its single parent-theta read to select the two named rows. The original
verifier file is unchanged; ROOT and proof metadata are adapted for this wrapper.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/openai-followup-20261031/binary'
REFERENCE_SHA256='8cec19439a77b94bb5a4e14e7e59ee53e734edb2a308425b08df4aec9b40cfd9'


def adapt_shared_prior(source):
    original="    theta=np.array(prior['theta'])\n"
    if source.count(original)!=1:raise ValueError('Independent verifier theta-read boundary changed')
    return source.replace(original,"    theta=np.array(prior['theta'])[[prior['arms'].index(a) for a in arms]]\n")


def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def verify_tuning():
    import math
    from fractions import Fraction
    import numpy as np
    import pandas as pd
    from scripts.embedding_store import DiskEmbeddingCache
    from scripts.resources import require_resources
    preflight=require_resources();protocol=read(BASE/'protocol.json');search=read(BASE/'tuning/search/manifest.json')
    assert digest({k:v for k,v in search.items() if k!='manifest_sha256'})==search['manifest_sha256']
    for name,h in search['files'].items():assert sha(BASE/'tuning/search'/name)==h
    core=read(ROOT/'outputs/openai-core-20261031/protocol.json')
    index=read(ROOT/'outputs/openai-core-20261031/prior-index.json')
    entry=next(r for r in index['fits'] if r['seed']==42)
    assert sha(ROOT/entry['prior']['path'])==entry['prior']['sha256']
    wrapper=read(ROOT/entry['prior']['path']);assert digest(wrapper['payload'])==wrapper['sha256'];prior=wrapper['payload']
    arms=protocol['arms'];dim=prior['dimension'];rows=[prior['arms'].index(a) for a in arms]
    assert arms==['gpt-4-1106-preview','meta/llama-2-70b-chat']
    cache=DiskEmbeddingCache.load(ROOT/'outputs/openai-standard-20261030/imported/caches/source-0')
    split=pd.read_csv(ROOT/'data/processed/routerbench_0shot/splits.csv')
    ids=split.loc[split.split=='tuning'].sort_values('stream_position').query_id.tolist()
    assert ids==search['selection']['query_ids'] and len(ids)==1000
    raw=pd.read_parquet(ROOT/'data/processed/routerbench_0shot/outcomes.parquet')
    truth={(r.query_id,r.arm_id):(float(r.reward),float(r.actual_cost)) for r in raw.loc[raw.query_id.isin(ids)&raw.arm_id.isin(arms)].itertuples(index=False)}
    W=np.asarray(prior['W']);bias=np.asarray(prior['b']);contexts={q:W@cache.get(q)+bias for q in ids}
    def unit(x):
        m=np.max(np.abs(x));return np.zeros_like(x) if m==0 else (x/m)/np.linalg.norm(x/m)
    selected={};trace_rows={}
    for candidate in search['candidates']:
        method=candidate['method'];parameters=candidate['parameters'];rng=np.random.default_rng(42)
        if method=='pilot':
            lambdas=np.array([prior['metrics']['validation']['per_arm'][a]['lambda'] for a in arms])
            A=lambdas[:,None,None]*np.eye(dim)[None,:,:];b=lambdas[:,None]*np.asarray(prior['theta'])[rows]
        else:A=np.repeat(np.eye(dim)[None,:,:],2,axis=0);b=np.zeros((2,dim))
        with (BASE/'tuning/search'/candidate['trace']).open(encoding='utf-8') as stream:trace=[json.loads(s) for s in stream]
        total=0.
        for t,(q,row) in enumerate(zip(ids,trace,strict=True)):
            x=unit(contexts[q]) if method=='pilot' else contexts[q]
            theta=np.linalg.solve(A,b[...,None])[...,0]
            if method=='random':scores=np.full(2,.5);chosen=int(rng.integers(2))
            elif method.startswith('fixed-'):
                chosen=arms.index(parameters['arm_id']);scores=np.array([float(i==chosen) for i in range(2)])
            else:
                scores=np.clip(np.array([unit(v)@x for v in theta]),-1,1) if method=='pilot' else theta@x
                variance=np.linalg.solve(A,np.broadcast_to(x,b.shape)[...,None])[...,0]@x
                uncertainty=np.sqrt(np.maximum(variance,0))
                selection=uncertainty if method=='explore-only' else scores+parameters.get('alpha',0)*uncertainty
                chosen=int(rng.integers(2)) if method=='epoch-greedy' and t%parameters['window']==0 else int(np.argmax(selection))
            a=arms[chosen];true,cost=truth[q,a]
            assert row['query_id']==q and row['arm_id']==a and row['round']==t+1
            assert row['reward']==true and row['actual_cost']==cost
            np.testing.assert_allclose([row['predictions'][a] for a in arms],scores,atol=1e-11,rtol=1e-10)
            if method not in ('random','fixed-00','fixed-01'):A[chosen]+=np.outer(x,x);b[chosen]+=true*x
            total+=true
        assert total==candidate['total_reward']
        if method not in selected or total>selected[method]['total_reward']:selected[method]=candidate
        trace_rows[candidate['id']]=trace
    assert selected==search['selected']
    assert protocol['tuning']['routerbench']['selected_parameters']=={m:c['parameters'] for m,c in selected.items()}
    estimator=read(BASE/'tuning/estimator.json');counts=ROOT/'outputs/research-cost-inputs-20261029'
    inputs=read(counts/'input-counts.json');outputs=read(counts/'tuning-output-counts.json')
    means={a:Fraction(sum(outputs[q][a] for q in ids),len(ids)) for a in arms};prices=estimator['prices']
    estimates={q:{a:float((inputs[q]*Fraction(str(prices[a]['input_per_million']))+means[a]*Fraction(str(prices[a]['output_per_million'])))/1000000) for a in arms} for q in ids}
    ratios=[r['predictions'][a]/estimates[r['query_id']][a] for c in selected.values() for r in trace_rows[c['id']]
        for a in arms if r['predictions'][a]>0 and estimates[r['query_id']][a]>0]
    bounds=protocol['tuning']['routerbench']['bounds']
    assert math.isclose(bounds['lower_bound'],min(ratios)*.5,rel_tol=1e-12)
    assert math.isclose(bounds['upper_bound'],max(ratios)*2,rel_tol=1e-12)
    proof=dict(verified=True,candidates=len(search['candidates']),selected={m:c['parameters'] for m,c in selected.items()},
        bounds=bounds,search_sha256=sha(BASE/'tuning/search/manifest.json'),verifier_sha256=sha(__file__),
        protocol_sha256=sha(BASE/'protocol.json'),preflight=preflight)
    with (BASE/'tuning-independent.json').open('x',encoding='utf-8') as out:json.dump(proof,out,indent=2)
    print(json.dumps(proof,indent=2))


def main():
    if sys.argv[1:]==['tuning']:verify_tuning();return
    reference=ROOT/'tools/research/verify_runs.py'
    if sha(reference)!=REFERENCE_SHA256:raise ValueError('Pinned independent reference verifier changed')
    source=adapt_shared_prior(reference.read_text(encoding='utf-8'))
    original="ROOT=Path(__file__).resolve().parents[2]"
    if source.count(original)!=1:raise ValueError('Reference root boundary changed')
    source=source.replace(original,"ROOT=Path(__file__).resolve().parents[1]")
    original="report=dict(verified=True,kind="
    if source.count(original)!=1:raise ValueError('Reference proof boundary changed')
    source=source.replace(original,f"report=dict(base_script_sha256={REFERENCE_SHA256!r},shared_parent_row_selection=True,verified=True,kind=")
    sys.argv=[sys.argv[0],'--base',str(BASE/'runs'),'--protocol',str(BASE/'protocol.json'),
        '--index',str(ROOT/'outputs/openai-core-20261031/prior-index.json'),
        '--cache-dir',str(ROOT/'outputs/openai-standard-20261030/imported/caches/source-0')]+sys.argv[1:]
    exec(compile(source,str(reference),'exec'),dict(__file__=__file__,__name__='__main__'))


if __name__=='__main__':main()
