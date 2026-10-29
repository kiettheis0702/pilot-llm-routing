"""Exercise real CLI failure paths on copies; preserve all research artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/research-experiments-20261029'
sys.path.insert(0,str(ROOT))
from scripts.resources import require_resources
from scripts.experiments import main
print(require_resources())
DEST=BASE/'cli-probes'
if DEST.exists():raise FileExistsError(DEST)
DEST.mkdir()
source=BASE/'learning/routerbench/seed-7/main/pilot'
def hashes(folder):return {p.relative_to(folder).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
before=hashes(source);probes=[]
def invoke(root,*,resume=False,protocol=None):
    argv=['--domain','routerbench','--seed','7','--method','pilot','--output-dir',str(root)]
    if resume:argv+=['--resume']
    if protocol is not None:argv+=['--protocol',str(protocol)]
    return main(argv)
def expected_error(name,callback,error,contains=None):
    try:callback()
    except error as e:
        if contains is not None:assert contains in str(e),(name,str(e))
        probes.append(dict(name=name,passed=True,error_type=type(e).__name__,message=str(e)))
    else:raise AssertionError('Expected rejection did not occur: '+name)
expected_error('existing output without resume',lambda:invoke(BASE),FileExistsError)
bad=json.loads((ROOT/'configs/frozen_experiment_protocol.json').read_text());bad['arms'].pop()
bad_path=DEST/'bad-protocol.json';bad_path.write_text(json.dumps(bad),encoding='utf-8')
expected_error('changed protocol',lambda:invoke(DEST/'bad-protocol-output',protocol=bad_path),ValueError,'checksum')
assert not (DEST/'bad-protocol-output').exists()
partial=DEST/'interrupted/learning/routerbench/seed-7/main/pilot';partial.mkdir(parents=True)
(partial/'trace.jsonl').write_text('partial evidence\n',encoding='utf-8');partial_before=hashes(partial)
expected_error('interrupted artifact',lambda:invoke(DEST/'interrupted',resume=True),ValueError,'Incomplete')
assert hashes(partial)==partial_before
copy=DEST/'corrupted/learning/routerbench/seed-7/main/pilot'
shutil.copytree(source,copy)
with (copy/'trace.jsonl').open('a',encoding='utf-8') as f:f.write('corrupted\n')
damaged_before=hashes(copy)
expected_error('changed completed trace',lambda:invoke(DEST/'corrupted',resume=True),ValueError,'checksum')
assert hashes(copy)==damaged_before and hashes(source)==before
report=dict(kind='real-experiment-cli-negative-probes',passed=True,probes=probes,
    original_learning_files_unchanged=True,source_sha256=before,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),synthetic=False,paper_equivalent=False)
with (DEST/'report.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps({'passed':True,'probes':len(probes),'report':str(DEST/'report.json')}))
