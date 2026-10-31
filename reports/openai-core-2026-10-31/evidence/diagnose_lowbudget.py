"""Independently explain six already-frozen early stops; change no decisions."""
import math
from fractions import Fraction
import numpy as np
from scripts.resources import require_resources
from scripts.embedding_store import DiskEmbeddingCache
from scripts.pretraining import load_checkpoint, _read_json
from tools.paper_preflight.common import ROOT, read_json, sha256
from tools.openai_study import once

preflight=require_resources()
base=ROOT/'outputs/openai-core-20261031'
protocol=read_json(base/'protocol.json');index=read_json(base/'prior-index.json')
cache=DiskEmbeddingCache.load(ROOT/'outputs/openai-standard-20261030/imported/caches/source-0')
inputs=read_json(ROOT/'outputs/research-cost-inputs-20261029/input-counts.json')
outputs=read_json(ROOT/'outputs/research-cost-inputs-20261029/tuning-output-counts.json')
rows=[]
def unit(x):return x/np.linalg.norm(x)
for seed in (7,42,2026):
    entry=next(e for e in index['fits'] if e['seed']==seed)
    prior=load_checkpoint(ROOT/entry['prior']['path'])
    learning=base/'runs/learning/mmlu'/f'seed-{seed}'/'main/pilot'
    manifest=read_json(learning/'manifest.json')
    state=_read_json(learning/manifest['snapshots']['1']['file'])['state']
    theta=np.linalg.solve(np.asarray(state['A']),np.asarray(state['b'])[...,None])[...,0]
    for fraction in (.2,.3):
        folder=base/'runs/deployment/mmlu'/f'seed-{seed}'/'main/pilot/learn-1'/f'budget-{fraction:g}'
        value=read_json(folder/'manifest.json');s=value['summary'];terminal=s['terminal_decision']
        q=terminal['query_id'];d=terminal['decision'];meta=value['request']['estimator']
        ids=meta['tuning_query_ids'];arms=prior['arms']
        costs={a:float((inputs[q]*Fraction(str(meta['prices'][a]['input_per_million']))+
            Fraction(sum(outputs[t][a] for t in ids),len(ids))*Fraction(str(meta['prices'][a]['output_per_million'])))/1000000) for a in arms}
        x=unit(prior['W']@cache.get(q)+prior['b'])
        predictions={a:float(np.clip(unit(theta[i])@x,-1,1)) for i,a in enumerate(arms)}
        bounds=protocol['tuning']['mmlu']['bounds'];u=d['utilization_before']
        logphi=math.log(bounds['lower_bound'])-1+u*(math.log(bounds['upper_bound'])-math.log(bounds['lower_bound'])+1)
        available=d['available_before'];fair_share=available/d['remaining_queries_in_bin']
        threshold=[a for a in arms if costs[a]<=available and predictions[a]>0 and math.log(predictions[a])-math.log(costs[a])>=logphi]
        fallback=[a for a in arms if costs[a]<=fair_share]
        affordable=[a for a in arms if costs[a]<=available]
        assert not threshold and not fallback and affordable and s['budget']['estimated_total']<s['budget']['budget']
        rows.append(dict(seed=seed,budget_fraction=fraction,query_id=q,served=s['served_queries'],
            requested=s['requested_queries'],allocated_budget=s['budget']['budget'],
            globally_unspent_estimated_budget=s['budget']['budget']-s['budget']['estimated_total'],
            available_in_current_bin=available,fallback_allowance=fair_share,
            cheapest_estimate=min(costs.values()),affordable_now=affordable,
            threshold_eligible=threshold,fallback_eligible=fallback,
            source_manifest_sha256=sha256(folder/'manifest.json')))
result=dict(verified=True,scope='diagnosis only; no new policy experiment or modified result',
    explanation='Current-bin threshold rejects all arms; fallback fair share is below every model estimate. The declared policy stops although total allocation remains.',
    conditions=rows,script_sha256=sha256(__file__),preflight=preflight,additional_paid_api_calls=0)
once(ROOT/'reports/openai-core-2026-10-31/low-budget-diagnostic.json',result)
print(result)
