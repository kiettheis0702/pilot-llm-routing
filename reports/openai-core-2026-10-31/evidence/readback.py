"""Final offline readback and a corrupt-copy probe; preserve all source files."""
import math
import shutil
import pandas as pd
from scripts.resources import require_resources
from scripts.experiments import load_artifact, verify_pins
from tools.paper_preflight.common import ROOT, read_json, sha256, digest
from tools.openai_study import once
from tools.course.analyze import verify_package

preflight=require_resources()
base=ROOT/'outputs/openai-core-20261031'
report=ROOT/'reports/openai-core-2026-10-31'
value=read_json(report/'manifest.json')
assert value['manifest_sha256']==digest({k:v for k,v in value.items() if k!='manifest_sha256'})
for name,want in value['files'].items():assert sha256(report/name)==want
verify_pins(read_json(base/'protocol.json'),ROOT)
verify_pins(read_json(base/'prior-index.json'),ROOT)
new=pd.read_csv(report/'per-seed.csv');old=pd.read_csv(report/'bge-core-normalized.csv')
paired=pd.read_csv(report/'paired-bge.csv');learning=pd.read_csv(report/'learning-prefixes.csv')
assert len(new)==len(old)==len(paired)==858 and new.id.is_unique and set(new.id)==set(old.id)==set(paired.id)
assert len(learning)==312 and len(learning.loc[learning.learning_fraction==1])==78
assert int(learning.loc[learning.learning_fraction==1,'rounds'].sum())==1740921
for row in new.to_dict('records'):
    assert 0<=row['served']<=row['requested'] and 0<=row['coverage']<=1
    assert math.isclose(row['quality_fraction_gpt4'],row['total_reward']/row['gpt4_total_reward'],abs_tol=1e-12)
    assert math.isclose(row['cost_fraction_gpt4'],row['actual_recorded_cost']/row['gpt4_total_cost'],abs_tol=1e-12)
    assert math.isclose(row['service_utility'],row['total_reward']/row['requested'],abs_tol=1e-12)
proofs=[read_json(p) for p in (base/'verification').glob('*.json')]
assert len(proofs)==78 and sum(p['verified_deployment_runs'] for p in proofs)==858
source=next((base/'runs/deployment').rglob('manifest.json')).parent
destination=base/'corrupt-copy-probe-final'
source_value=load_artifact(source)
destination.mkdir(exist_ok=False)
for name in ['manifest.json',*source_value['files']]:shutil.copyfile(source/name,destination/name)
with (destination/'summary.json').open('ab') as stream:stream.write(b'\n')
try:load_artifact(destination)
except ValueError as error:
    assert 'checksum' in str(error)
    corrupt=dict(passed=True,error=str(error),source_unchanged=sha256(source/'summary.json')==source_value['files']['summary.json'])
    assert corrupt['source_unchanged']
else:raise AssertionError('Corrupt completed copy was accepted')
packages={name:verify_package(ROOT/'reports'/name) for name in
    ('full-experiments-2026-10-29','paper-preflight-2026-10-30','openai-workflow-2026-10-30')}
gate=(base/'full-tests-final.log').read_text(encoding='utf-8-sig')
assert '596 passed, 1 skipped' in gate
result=dict(verified=True,conditions=858,learning_streams=78,learning_snapshots=312,
    unique_observed_learning_rounds=1740921,served_deployment_rounds=int(new.served.sum()),
    actual_overspend_conditions=int((new.actual_budget_excess>0).sum()),
    actual_overspend_pilot_full=int(((new.actual_budget_excess>0)&(new.method=='pilot')&(new.learning_fraction==1)).sum()),
    independent_replays=78,changed_pin_and_partial_and_missing_inventory=read_json(base/'negative-control-results.json')['passed'],
    corrupt_copy_probe=corrupt,historical_packages=packages,full_tests='596 passed,1 skipped',
    script_sha256=sha256(__file__),preflight=preflight,additional_paid_api_calls=0)
once(base/'readback-final.json',result)
print(result)
