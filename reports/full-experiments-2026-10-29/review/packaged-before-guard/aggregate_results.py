"""Strict inventory/readback and descriptive tables, independent of runner helpers."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/research-experiments-20261029'
DEST=BASE/'aggregate'
if DEST.exists():raise FileExistsError(DEST)
import sys
sys.path.insert(0,str(ROOT))
from scripts.resources import require_resources
preflight=require_resources()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def manifest(p):
    value=read(p)
    assert value['manifest_sha256']==digest({k:v for k,v in value.items() if k!='manifest_sha256'})
    for name,sha in value['files'].items():assert filehash(Path(p).parent/name)==sha
    return value
def key(n):return format(n,'.12g')

protocol=read(ROOT/'configs/frozen_experiment_protocol.json');plan=protocol['primary_plan']
assert protocol['manifest_sha256']==digest({k:v for k,v in protocol.items() if k!='manifest_sha256'})
methods=['pilot','linucb','epoch-greedy','explore-only','random']+[f'fixed-{i:02d}' for i in range(8)]
variants=[v['id'] for v in protocol['ablation_plan']['variants']]
expected={};groups=[]
for domain in ['routerbench','mmlu']:
    for seed in [7,42,2026]:
        for variant,method in [('main',m) for m in methods]+[(v,'pilot') for v in variants]:
            groups.append((domain,seed,method,variant))
            for fraction in (plan['learning_fractions'] if variant=='main' else [1.]):
                for budget in (plan['budget_fractions'] if fraction==1 else [.5]):
                    identity=f'{domain}/seed-{seed}/{variant}/{method}/learn-{key(fraction)}/budget-{key(budget)}'
                    expected[identity]=(domain,seed,variant,method,fraction,budget)
assert len(expected)==1098 and len(groups)==108
found={p.parent.relative_to(BASE/'deployment').as_posix():p for p in (BASE/'deployment').rglob('manifest.json')}
missing=sorted(set(expected)-set(found));extra=sorted(set(found)-set(expected))
if missing or extra:raise RuntimeError(json.dumps({'incomplete':True,'missing_count':len(missing),'extra':extra,'first_missing':missing[:10]}))
verification={}
for domain,seed,method,variant in groups:
    p=BASE/'verification'/f'{domain}-seed{seed}-{method}-{variant}.json'
    if not p.exists():raise RuntimeError('Missing independent replay: '+str(p))
    v=read(p)
    assert v['verified'] and (v['domain'],v['seed'],v['method'],v['variant'])==(domain,seed,method,variant)
    for r in v['inventory']:
        assert r['id'] not in verification
        verification[r['id']]=r['manifest_sha256']
assert set(verification)==set(expected)

records=[];run_hashes={};learning_hashes={};learning_records=[]
for identity,condition in expected.items():
    domain,seed,variant,method,fraction,budget=condition
    v=manifest(found[identity]);r=v['request'];s=v['summary'];m=s['metrics'];b=s['budget']
    assert not v['synthetic'] and v['paper_equivalent'] is False
    assert v['manifest_sha256']==verification[identity]
    assert r['lineage']['protocol_sha256']==filehash(ROOT/'configs/frozen_experiment_protocol.json')
    c=r['lineage']['condition']
    assert (c['domain'],c['seed'],c['variant'],c['method'],c['learning_fraction'],c['budget_fraction'])==condition
    assert c['id']==identity
    assert s['served_queries']+s['unserved_queries']==s['requested_queries']==plan['frozen_counts'][domain]['deployment']
    assert b['actual_costs_complete'] and not b['estimated_budget_violation']
    assert math.isclose(m['reward_per_requested_query'],m['total_reward']/s['requested_queries'],abs_tol=1e-12)
    assert b['actual_budget_excess_observed']==max(0.,b['actual_total']-b['budget']) or math.isclose(b['actual_budget_excess_observed'],max(0.,b['actual_total']-b['budget']),abs_tol=1e-12)
    learn=r['lineage']['learning_metrics']
    records.append(dict(id=identity,domain=domain,seed=seed,variant=variant,method=method,
        learning_fraction=fraction,budget_fraction=budget,budget=b['budget'],requested=s['requested_queries'],
        served=s['served_queries'],total_reward=m['total_reward'],service_utility=m['reward_per_requested_query'],
        mean_reward_served=m['mean_reward_served'],coverage=m['served_fraction'],estimated_cost=b['estimated_total'],
        actual_recorded_cost=b['actual_total'],actual_budget_excess=b['actual_budget_excess_observed'],
        learning_rounds=r['lineage']['learning_rounds'],learning_total_reward=learn['total_reward'],
        learning_empirical_oracle_regret=learn['cumulative_empirical_oracle_regret']))
    run_hashes[identity]=v['manifest_sha256']
for p in (BASE/'learning').rglob('manifest.json'):
    v=manifest(p);lineage=v['request']['lineage'];identity=p.parent.relative_to(BASE/'learning').as_posix()
    learning_hashes[identity]=v['manifest_sha256']
    for fraction,snap in v['snapshots'].items():
        learning_records.append(dict(domain=lineage['domain'],seed=lineage['seed'],variant=lineage['variant'],method=lineage['method'],
            learning_fraction=float(fraction),rounds=snap['rounds'],**{k:snap['metrics'][k] for k in
            ('total_reward','total_actual_cost','cumulative_empirical_oracle_regret')}))
assert len(learning_hashes)==102 and len(learning_records)==336
full_learning=[r for r in learning_records if r['learning_fraction']==1]
assert sum(r['rounds'] for r in full_learning)==2142672

metrics=['service_utility','coverage','mean_reward_served','estimated_cost','actual_recorded_cost','actual_budget_excess',
         'learning_empirical_oracle_regret']
grouped=defaultdict(list)
for r in records:grouped[(r['domain'],r['variant'],r['method'],r['learning_fraction'],r['budget_fraction'])].append(r)
aggregated=[]
for (domain,variant,method,fraction,budget),values in grouped.items():
    assert sorted(v['seed'] for v in values)==[7,42,2026]
    row=dict(domain=domain,variant=variant,method=method,learning_fraction=fraction,budget_fraction=budget,seed_count=3)
    for metric in metrics:
        numbers=[v[metric] for v in values]
        if any(v is None for v in numbers):
            row.update({metric+'_'+stat:None for stat in ('mean','min','max')})
        else:row.update({metric+'_mean':math.fsum(numbers)/3,metric+'_min':min(numbers),metric+'_max':max(numbers)})
    aggregated.append(row)
lookup={(r['domain'],r['seed'],r['method'],r['variant'],r['learning_fraction'],r['budget_fraction']):r for r in records}
paired=[]
for r in records:
    if r['variant']=='main' and r['learning_fraction']==1:continue
    reference=lookup[(r['domain'],r['seed'],r['method'],'main',1.,r['budget_fraction'])]
    name=r['variant'] if r['variant']!='main' else 'learning-fraction-'+key(r['learning_fraction'])
    row=dict(domain=r['domain'],seed=r['seed'],method=r['method'],comparison=name,budget_fraction=r['budget_fraction'],
        run_id=r['id'],reference_id=reference['id'])
    for metric in metrics:
        row[metric+'_delta']=None if r[metric] is None or reference[metric] is None else r[metric]-reference[metric]
    paired.append(row)
assert len(paired)==474
DEST.mkdir()
for name,values in [('per-seed.csv',records),('mean-min-max.csv',aggregated),('paired-deltas.csv',paired),('learning-prefixes.csv',learning_records)]:
    with (DEST/name).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
report=dict(version=1,kind='complete-local-experiment-aggregate',complete=True,synthetic=False,paper_equivalent=False,
    protocol_sha256=filehash(ROOT/'configs/frozen_experiment_protocol.json'),primary_runs=858,ablation_runs=240,
    unique_learning_streams=102,learning_prefix_snapshots=336,unique_observed_learning_rounds=2142672,
    requested_deployment_rounds=sum(r['requested'] for r in records),served_deployment_rounds=sum(r['served'] for r in records),
    independently_replayed_groups=108,run_manifest_sha256=run_hashes,learning_manifest_sha256=learning_hashes,
    aggregation='three paired fixed seeds; descriptive mean/min/max, no confidence intervals or significance',
    missing_conditions=[],remaining_paper_equivalence=protocol['remaining_reproduction_blockers'],
    script_sha256=filehash(__file__),preflight=preflight,
    files={p.name:filehash(p) for p in DEST.iterdir() if p.is_file()})
report['manifest_sha256']=digest(report)
with (DEST/'manifest.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({k:v for k,v in report.items() if k not in ('run_manifest_sha256','learning_manifest_sha256','files')}))
