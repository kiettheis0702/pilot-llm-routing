from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET

root=Path.cwd();base=root/'outputs/final-reproduction-20261029'
dest=root/'reports/full-experiments-2026-10-29/reproduction'
dest.mkdir(exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
source=read(base/'source-files.json')
runtime={p:h for p,h in source.items() if p.startswith(('scripts/','tests/','configs/')) or p in ('pyproject.toml','uv.lock')}
for p,h in runtime.items():assert sha(root/p)==h,(p,'changed after clean test')
transfer=read(root/'reports/pc-transfer-manifest-2026-10-29.json');data=[]
for item in transfer['files']:
    file=root/'data'/item['relative_path'];assert file.stat().st_size==item['size_bytes'];assert sha(file)==item['sha256'];data.append(item)
pins={}
def verify(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            path=root/value['path'];assert sha(path)==value['sha256'],value;pins[value['path']]=value['sha256']
        else:
            for v in value.values():verify(v)
    elif isinstance(value,list):
        for v in value:verify(v)
for name in ('frozen_experiment_protocol.json','research_prior_index.json'):
    verify(read(root/'configs'/name))
toy=base/'toy-smoke';summary=read(toy/'summary.json')
assert summary['synthetic'] and summary['split_counts']=={'learning':50,'deployment':6,'tuning':10}
assert read(toy/'learned.json')==read(toy/'deployed.json')
with (toy/'deployment-trace.csv').open(newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==6
close(sum(float(r['reward']) for r in rows),float(rows[-1]['cumulative_reward']))
baselines=read(base/'baseline-smoke/comparison.json');assert baselines['source']=='toy'
assert len(baselines['metrics'])==7
for method,metrics in baselines['metrics'].items():
    with (base/'baseline-smoke'/method/'trace.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==100==metrics['rounds']
    # Counters treat absent and explicit-zero arms identically; dict equality does not.
    assert Counter(r['arm_id'] for r in rows)==Counter(metrics['action_counts'])
    for field,metric in [('reward','total_reward'),('actual_cost','total_actual_cost'),('empirical_oracle_regret','cumulative_empirical_oracle_regret')]:
        close(sum(float(r[field]) for r in rows),metrics[metric])
tests={}
for name in ('baseline','snapshot'):
    suite=ET.parse(base/(name+'.xml')).getroot().find('testsuite');tests[name]=dict(suite.attrib)
    assert suite.get('errors')=='0' and suite.get('failures')=='0'
report=read(base/'report.json')
report.update(tests=tests,runtime_source_files_unchanged=len(runtime),verified_source_data=data,verified_pins=pins,
    smoke_readback={'toy_learning_rounds':50,'toy_served':6,'frozen_deployed_state_unchanged':True,'baseline_methods':7,'baseline_rounds_each':100,'metrics_independently_summed':True},
    verifier_correction='Initial one-off action-count assertion compared Counter to a dict with explicit zero arms; corrected to compare two Counters. Production outputs were unchanged.',
    source_snapshot_note='Docs/status files may change after this snapshot; all runtime Python, test, config and dependency bytes matched at final check. No whole real grid or GPU embedding was rerun in this fresh environment.')
with (dest/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
for name in ('source-files.json','baseline.xml','snapshot.xml','uv-sync.log','baseline-pytest.log','snapshot-pytest.log','toy-smoke.log','baseline-smoke.log'):
    shutil.copy2(base/name,dest/name)
shutil.copy2(root/'outputs/final-reproduction-20261029-driver.py',dest/'executed-driver.py')
shutil.copy2(Path(__file__),dest/'executed-readback.py')
print({'tests':{k:{x:v[x] for x in ('tests','failures','errors','skipped')} for k,v in tests.items()},'runtime_files':len(runtime),'data_files':len(data),'pins':len(pins),'smokes':'independent readback passed'})
